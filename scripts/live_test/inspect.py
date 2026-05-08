"""Compact live-test pod/run introspection."""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from typing import Any

import scripts.live_test as live_test_pkg
from scripts.live_test import config
from scripts.live_test.ssh_bootstrap import fetch_worker_logs, open_session


DEFAULT_WORKDIRS = ("/workspace/Reigh-Worker-LiveTest", "/workspace/Reigh-Worker")


def _coerce_rows(result: Any) -> list[dict[str, Any]]:
    data = getattr(result, "data", None)
    if not data:
        return []
    if isinstance(data, dict):
        return [data]
    return [row for row in data if isinstance(row, dict)]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(value: str | None, *, now: datetime | None = None) -> int | None:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    current = now or datetime.now(timezone.utc)
    return max(0, int((current - parsed).total_seconds()))


def _safe_metadata(row: dict[str, Any] | None) -> dict[str, Any]:
    metadata = (row or {}).get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _fetch_task_row(db, task_id: str | None) -> dict[str, Any] | None:
    if not task_id:
        return None
    rows = _coerce_rows(
        db.supabase.table("tasks")
        .select(
            "id, task_type, status, error_message, output_location, created_at, "
            "generation_started_at, generation_processed_at, worker_id, attempts, project_id, params"
        )
        .eq("id", task_id)
        .execute()
    )
    return rows[0] if rows else None


def _fetch_worker_row(db, worker_id: str | None) -> dict[str, Any] | None:
    if not worker_id:
        return None
    rows = _coerce_rows(
        db.supabase.table("workers")
        .select("id, status, last_heartbeat, created_at, metadata")
        .eq("id", worker_id)
        .execute()
    )
    return rows[0] if rows else None


def _worker_matches_pod(row: dict[str, Any], pod_id: str) -> bool:
    if str(row.get("id") or "") == pod_id:
        return True
    metadata = row.get("metadata")
    return isinstance(metadata, dict) and str(metadata.get("runpod_id") or "") == pod_id


def _fetch_worker_row_for_pod(db, pod_id: str | None) -> dict[str, Any] | None:
    if not pod_id:
        return None
    try:
        rows = _coerce_rows(
            db.supabase.table("workers")
            .select("id, status, last_heartbeat, created_at, metadata")
            .eq("metadata->>runpod_id", pod_id)
            .execute()
        )
        matching = [row for row in rows if _worker_matches_pod(row, pod_id)]
        if matching:
            matching.sort(
                key=lambda row: (str(row.get("last_heartbeat") or ""), str(row.get("created_at") or "")),
                reverse=True,
            )
            return matching[0]
    except Exception:
        pass

    rows = _coerce_rows(db.supabase.table("workers").select("id, status, last_heartbeat, created_at, metadata").execute())
    matching = [row for row in rows if _worker_matches_pod(row, pod_id)]
    if not matching:
        return None
    matching.sort(key=lambda row: (str(row.get("last_heartbeat") or ""), str(row.get("created_at") or "")), reverse=True)
    return matching[0]


def _runpod_summary(pod_id: str | None, api_key: str | None) -> dict[str, Any] | None:
    if not pod_id:
        return None
    summary: dict[str, Any] = {"pod_id": pod_id}
    if not api_key:
        summary["error"] = "RUNPOD_API_KEY not set"
        return summary
    try:
        status = live_test_pkg.get_pod_status(pod_id, api_key) or {}
        summary["desired_status"] = status.get("desired_status") or status.get("desiredStatus")
        summary["actual_status"] = status.get("actual_status") or status.get("actualStatus") or status.get("status")
        summary["ip"] = status.get("ip")
        summary["ports"] = _format_ports(status.get("ports"))
    except Exception as exc:
        summary["status_error"] = str(exc)
    try:
        ssh = live_test_pkg.get_pod_ssh_details(pod_id, api_key) or {}
        summary["ssh_ip"] = ssh.get("ip")
        summary["ssh_port"] = ssh.get("port")
        summary["ssh_available"] = bool(ssh.get("ip") and ssh.get("port"))
    except Exception as exc:
        summary["ssh_error"] = str(exc)
        summary["ssh_available"] = False
    return summary


def _format_ports(ports: Any) -> str | None:
    if not isinstance(ports, list):
        return None
    rendered: list[str] = []
    for port in ports:
        if not isinstance(port, dict):
            continue
        private = port.get("privatePort") or port.get("private_port") or port.get("containerPort")
        public = port.get("publicPort") or port.get("public_port") or port.get("hostPort")
        if private and public:
            rendered.append(f"{private}->{public}")
    return ",".join(rendered) if rendered else "none"


def _quote(value: str) -> str:
    return shlex.quote(str(value))


def _fetch_vibecomfy_hints(ssh, task_id: str | None, *, workdirs: tuple[str, ...] = DEFAULT_WORKDIRS) -> str:
    if not task_id:
        return ""
    task = _quote(task_id)
    workdir_tests = " ".join(_quote(path) for path in workdirs)
    script = (
        "set -euo pipefail\n"
        f"task_id={task}\n"
        f"for root in {workdir_tests}; do\n"
        "  [ -d \"$root\" ] || continue\n"
        "  run_dir=\"$root/outputs/vibecomfy_runs/$task_id\"\n"
        "  if [ -d \"$run_dir\" ]; then\n"
        "    echo \"=== vibecomfy artifacts: $run_dir ===\"\n"
        "    find \"$run_dir\" -maxdepth 3 -type f "
        "\\( -path '*/output/*' -o -name metadata.json -o -name embedded.log \\) "
        "-printf '%p\\n' | sort | tail -n 40\n"
        "  fi\n"
        "  if [ -d \"$root/logs\" ]; then\n"
        f"    grep -R --fixed-strings -n {task} \"$root/logs\" 2>/dev/null | tail -n 20 || true\n"
        "  fi\n"
        "done\n"
    )
    exit_code, stdout, stderr = ssh.execute_command(f"bash -lc {_quote(script)}", timeout=60)
    if exit_code != 0:
        return f"hint command failed exit={exit_code}: {stderr.strip()}"
    return stdout.strip()


def build_status_bundle(
    db,
    *,
    task_id: str | None = None,
    worker_id: str | None = None,
    pod_id: str | None = None,
    api_key: str | None = None,
    include_ssh: bool = True,
    ssh_wait_timeout: int = 20,
    log_lines: int = 120,
    now: datetime | None = None,
) -> dict[str, Any]:
    task = _fetch_task_row(db, task_id)
    resolved_worker_id = worker_id or (str(task.get("worker_id")) if task and task.get("worker_id") else None)
    worker = _fetch_worker_row(db, resolved_worker_id)
    if worker is None and pod_id:
        worker = _fetch_worker_row_for_pod(db, pod_id)
        if worker and not resolved_worker_id:
            resolved_worker_id = str(worker.get("id") or "")

    metadata = _safe_metadata(worker)
    resolved_pod_id = pod_id or (str(metadata.get("runpod_id")) if metadata.get("runpod_id") else None)
    runpod = _runpod_summary(resolved_pod_id, api_key)

    bundle: dict[str, Any] = {
        "input": {"task_id": task_id, "worker_id": worker_id, "pod_id": pod_id},
        "resolved": {"task_id": task_id, "worker_id": resolved_worker_id, "pod_id": resolved_pod_id},
        "task": _task_summary(task),
        "worker": _worker_summary(worker, now=now),
        "runpod": runpod,
        "ssh": {"attempted": False, "available": False, "log_tail": None, "vibecomfy_hints": None, "error": None},
    }

    if include_ssh and resolved_pod_id and api_key and (runpod or {}).get("ssh_available"):
        bundle["ssh"]["attempted"] = True
        try:
            ssh = open_session(resolved_pod_id, api_key, ssh_wait_timeout=ssh_wait_timeout, poll_interval=2)
            try:
                bundle["ssh"]["available"] = True
                bundle["ssh"]["log_tail"] = fetch_worker_logs(ssh, DEFAULT_WORKDIRS[0], lines=log_lines) or fetch_worker_logs(
                    ssh,
                    DEFAULT_WORKDIRS[1],
                    lines=log_lines,
                )
                bundle["ssh"]["vibecomfy_hints"] = _fetch_vibecomfy_hints(ssh, task_id)
            finally:
                close = getattr(ssh, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            bundle["ssh"]["error"] = str(exc)

    return bundle


def _task_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    params = row.get("params") if isinstance(row.get("params"), dict) else {}
    route_contract = params.get("route_contract") if isinstance(params.get("route_contract"), dict) else {}
    return {
        "id": row.get("id"),
        "task_type": row.get("task_type"),
        "status": row.get("status"),
        "worker_id": row.get("worker_id"),
        "attempts": row.get("attempts"),
        "error_message": row.get("error_message"),
        "output_location": row.get("output_location"),
        "created_at": row.get("created_at"),
        "generation_started_at": row.get("generation_started_at"),
        "generation_processed_at": row.get("generation_processed_at"),
        "route_key": row.get("route_key") or route_contract.get("route_key"),
        "selected_backend": row.get("selected_backend") or route_contract.get("selected_backend"),
        "selected_template_id": row.get("selected_template_id") or route_contract.get("selected_template_id"),
    }


def _worker_summary(row: dict[str, Any] | None, *, now: datetime | None = None) -> dict[str, Any] | None:
    if row is None:
        return None
    metadata = _safe_metadata(row)
    return {
        "id": row.get("id"),
        "status": row.get("status"),
        "last_heartbeat": row.get("last_heartbeat"),
        "heartbeat_age_sec": _age_seconds(row.get("last_heartbeat"), now=now),
        "created_at": row.get("created_at"),
        "runpod_id": metadata.get("runpod_id"),
        "ready_for_tasks": metadata.get("ready_for_tasks"),
        "worker_backend": metadata.get("worker_backend"),
        "worker_pool": metadata.get("worker_pool"),
        "termination_reason": metadata.get("termination_reason"),
        "error_reason": metadata.get("error_reason"),
    }


def render_status_bundle(bundle: dict[str, Any]) -> str:
    lines = ["Live inspect"]
    resolved = bundle.get("resolved") or {}
    lines.append(
        "ids: "
        f"task={resolved.get('task_id') or '-'} "
        f"worker={resolved.get('worker_id') or '-'} "
        f"pod={resolved.get('pod_id') or '-'}"
    )

    runpod = bundle.get("runpod")
    if runpod:
        lines.append(
            "runpod: "
            f"desired={runpod.get('desired_status') or '-'} "
            f"actual={runpod.get('actual_status') or '-'} "
            f"ip={runpod.get('ip') or '-'} "
            f"ports={runpod.get('ports') or '-'} "
            f"ssh={runpod.get('ssh_ip') or '-'}:{runpod.get('ssh_port') or '-'}"
        )
        for key in ("error", "status_error", "ssh_error"):
            if runpod.get(key):
                lines.append(f"runpod_{key}: {runpod[key]}")
    else:
        lines.append("runpod: no pod_id")

    worker = bundle.get("worker")
    if worker:
        age = worker.get("heartbeat_age_sec")
        age_text = f"{age}s" if age is not None else "-"
        lines.append(
            "worker: "
            f"id={worker.get('id') or '-'} "
            f"status={worker.get('status') or '-'} "
            f"heartbeat={worker.get('last_heartbeat') or '-'} "
            f"age={age_text} "
            f"backend={worker.get('worker_backend') or '-'} "
            f"ready={worker.get('ready_for_tasks')}"
        )
        if worker.get("termination_reason") or worker.get("error_reason"):
            lines.append(f"worker_reason: {worker.get('termination_reason') or worker.get('error_reason')}")
    else:
        lines.append("worker: not found")

    task = bundle.get("task")
    if task:
        lines.append(
            "task: "
            f"id={task.get('id') or '-'} "
            f"type={task.get('task_type') or '-'} "
            f"status={task.get('status') or '-'} "
            f"attempts={task.get('attempts') if task.get('attempts') is not None else '-'} "
            f"worker={task.get('worker_id') or '-'}"
        )
        if task.get("error_message"):
            lines.append(f"task_error: {task['error_message']}")
        if task.get("output_location"):
            lines.append(f"task_output: {task['output_location']}")
        route_bits = [task.get("route_key"), task.get("selected_backend"), task.get("selected_template_id")]
        if any(route_bits):
            lines.append(f"task_route: route={route_bits[0] or '-'} backend={route_bits[1] or '-'} template={route_bits[2] or '-'}")
    elif (bundle.get("input") or {}).get("task_id"):
        lines.append("task: not found")

    ssh = bundle.get("ssh") or {}
    if ssh.get("attempted"):
        lines.append(f"ssh: available={ssh.get('available')}")
        if ssh.get("error"):
            lines.append(f"ssh_error: {ssh['error']}")
        if ssh.get("vibecomfy_hints"):
            lines.append("vibecomfy_hints:")
            lines.extend(_indent_tail(str(ssh["vibecomfy_hints"]), max_lines=60))
        if ssh.get("log_tail"):
            lines.append("worker_log_tail:")
            lines.extend(_indent_tail(str(ssh["log_tail"]), max_lines=80))
    elif resolved.get("pod_id"):
        lines.append("ssh: not attempted")

    return "\n".join(lines)


def _indent_tail(text: str, *, max_lines: int) -> list[str]:
    raw_lines = text.splitlines()
    if len(raw_lines) > max_lines:
        raw_lines = [f"... truncated {len(text.splitlines()) - max_lines} earlier lines", *raw_lines[-max_lines:]]
    return [f"  {line}" for line in raw_lines]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a live-test task, worker, or RunPod pod.")
    parser.add_argument("--task-id")
    parser.add_argument("--worker-id")
    parser.add_argument("--pod-id")
    parser.add_argument("--no-ssh", action="store_true", help="Skip SSH log and artifact probes.")
    parser.add_argument("--ssh-timeout", type=int, default=20)
    parser.add_argument("--log-lines", type=int, default=120)
    parser.add_argument("--json", action="store_true", help="Emit the raw status bundle as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.task_id or args.worker_id or args.pod_id):
        parser.error("provide at least one of --task-id, --worker-id, or --pod-id")

    db = config.DatabaseClient()
    bundle = build_status_bundle(
        db,
        task_id=args.task_id,
        worker_id=args.worker_id,
        pod_id=args.pod_id,
        api_key=config.get_env("RUNPOD_API_KEY"),
        include_ssh=not args.no_ssh,
        ssh_wait_timeout=args.ssh_timeout,
        log_lines=args.log_lines,
    )
    if args.json:
        print(json.dumps(bundle, indent=2, sort_keys=True, default=str))
    else:
        print(render_status_bundle(bundle))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
