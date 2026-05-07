"""Variant B driver: update/take over an installed pod and run the live-test matrix."""

from __future__ import annotations

import asyncio
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.live_test import config
from scripts.live_test.git_ops import (
    cleanup_temp_branch,
    push_working_copy_to_temp_branch,
    restore_local_state,
    snapshot_local_state,
)
from scripts.live_test.heartbeat_waiter import wait_until_ready
from scripts.live_test.launch_command import build_direct_worker_command, build_run_worker_command
from scripts.live_test.logger import get_logger
from scripts.live_test.matrix import build_matrix, render_case_payload, run_matrix
from scripts.live_test.preflight import (
    assert_user_queue_clean,
    close_stale_live_test_tasks,
    ensure_user_cloud_generation_enabled,
    get_or_create_live_test_project,
)
from scripts.live_test.report import write_report
from scripts.live_test.safety_gate import assert_safe_to_take_over
from scripts.live_test.ssh_bootstrap import (
    WorkerProcessInfo,
    capture_current_worker_cmdline,
    clone_and_install_vibecomfy,
    export_env,
    fetch_worker_logs,
    kill_supervisor_and_worker,
    launch_worker_detached,
    open_session,
)
from scripts.live_test.terminate_guard import guarded_terminate
from scripts.live_test.token_resolver import resolve_token_to_user_id


UPDATE_VARIANT = "update"
UPDATE_WORKDIR = "/workspace/Reigh-Worker"
FRESH_LIVE_TEST_WORKDIR = "/workspace/Reigh-Worker-LiveTest"
VIBECOMFY_WORKDIR = "/workspace/vibecomfy"
VIBECOMFY_REPO_URL = "https://github.com/peteromallet/VibeComfy.git"
VIBECOMFY_PYTHON = "python3.11"
REMOTE_UV_BOOTSTRAP = (
    'export PATH="$HOME/.local/bin:$PATH" && '
    "if ! command -v uv >/dev/null 2>&1; then "
    "((python3 -m pip install --user uv) || "
    "(python3 -m ensurepip --user && python3 -m pip install --user uv) || "
    "(python -m pip install --user uv)); "
    'export PATH="$HOME/.local/bin:$PATH"; '
    "fi && "
    "command -v uv >/dev/null 2>&1"
)

log = get_logger(__name__)


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _runs_root() -> Path:
    return config.WORKER_ROOT / "scripts" / "live_test" / "runs"


def _build_matrix_cases(args) -> list:
    return build_matrix(
        anchor_image_a=args.anchor_image_a,
        anchor_image_b=args.anchor_image_b,
        timeout_image_sec=args.timeout_image,
        timeout_travel_segment_sec=args.timeout_travel_segment,
        timeout_travel_orchestrator_sec=args.timeout_travel_orchestrator,
        selected_backend=getattr(args, "backend", "wgp"),
        selector_namespace=getattr(args, "selector_namespace", "production"),
        selector_version=getattr(args, "selector_version", None),
        worker_contract_version=getattr(args, "worker_contract_version", 1),
        selected_profile=getattr(args, "worker_profile", "default"),
        case_names=getattr(args, "case", []),
        task_types=getattr(args, "task_type", []),
        route_keys=getattr(args, "route_key", []),
    )


def _build_worker_env(token: str, supabase_url: str, service_role_key: str, args=None) -> dict[str, str]:
    backend = getattr(args, "backend", "wgp")
    env = {
        "REIGH_ACCESS_TOKEN": token,
        "REIGH_BACKEND": backend,
        "REIGH_SELECTOR_NAMESPACE": getattr(args, "selector_namespace", "production"),
        "REIGH_WORKER_CONTRACT_VERSION": str(getattr(args, "worker_contract_version", 1)),
        "REIGH_WORKER_PROFILE": getattr(args, "worker_profile", "default"),
        "SUPABASE_SERVICE_ROLE_KEY": service_role_key,
        "SUPABASE_URL": supabase_url,
        "WORKER_DB_CLIENT_AUTH_MODE": "service" if backend == "vibecomfy" else "worker",
        "REIGH_CLAIM_TELEMETRY": "1",
    }
    selector_version = getattr(args, "selector_version", None)
    if selector_version:
        env["REIGH_SELECTOR_VERSION"] = str(selector_version)
    if backend == "vibecomfy":
        env.update(
            {
                "VIBECOMFY_CWD": VIBECOMFY_WORKDIR,
                "VIBECOMFY_PATH": VIBECOMFY_WORKDIR,
                "VIBECOMFY_PYTHON": VIBECOMFY_PYTHON,
            }
        )
    return env


def _prepare_context(args) -> dict[str, Any]:
    token = config.require_env("REIGH_LIVE_TEST_TOKEN")
    db = config.DatabaseClient()
    user_id = resolve_token_to_user_id(db, token)
    stale_count = close_stale_live_test_tasks(db, user_id)
    if stale_count:
        log.info("closed stale live-test tasks before update run", count=stale_count)
    if ensure_user_cloud_generation_enabled(db, user_id):
        log.info("enabled cloud generation for live-test user", user_id=user_id)
    assert_user_queue_clean(db, user_id)
    project_id = get_or_create_live_test_project(db, user_id)
    cases = _build_matrix_cases(args)
    return {
        "db": db,
        "token": token,
        "user_id": user_id,
        "project_id": project_id,
        "cases": cases,
    }


def _validate_cases(cases: list, project_id: str) -> None:
    for index, case in enumerate(cases, start=1):
        render_case_payload(case, project_id=project_id, unique_suffix=f"update-{index}")


def _ssh_execute(ssh, command: str, *, timeout: int = 1800, check: bool = True) -> tuple[str, str]:
    exit_code, stdout, stderr = ssh.execute_command(command, timeout=timeout)
    if check and exit_code != 0:
        raise RuntimeError(
            f"Remote command failed with exit {exit_code}: {command}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return stdout, stderr


def _quote(value: str) -> str:
    return shlex.quote(str(value))


def _read_remote_branch(ssh, workdir: str = UPDATE_WORKDIR) -> str:
    stdout, _ = _ssh_execute(
        ssh,
        f"bash -lc {_quote(f'cd {workdir} && git symbolic-ref --short HEAD || echo DETACHED')}",
        timeout=60,
    )
    return stdout.strip() or "DETACHED"


def _read_remote_sha(ssh, workdir: str = UPDATE_WORKDIR) -> str:
    stdout, _ = _ssh_execute(
        ssh,
        f"bash -lc {_quote(f'cd {workdir} && git rev-parse HEAD')}",
        timeout=60,
    )
    return stdout.strip()


def _extract_worker_id_from_cmdline(cmdline: list[str]) -> str | None:
    for index, item in enumerate(cmdline):
        if item == "--worker" and index + 1 < len(cmdline):
            return cmdline[index + 1]
    return None


def _worker_matches_pod(row: dict[str, Any], pod_id: str) -> bool:
    if str(row.get("id") or "") == pod_id:
        return True
    metadata = row.get("metadata")
    return isinstance(metadata, dict) and str(metadata.get("runpod_id") or "") == pod_id


def _query_workers(db) -> list[dict[str, Any]]:
    if not hasattr(db, "supabase"):
        return []
    result = db.supabase.table("workers").select("id, metadata, status, created_at, last_heartbeat").execute()
    return list(getattr(result, "data", None) or [])


def _query_workers_for_pod(db, pod_id: str) -> list[dict[str, Any]]:
    rows = _query_workers(db)
    return [row for row in rows if _worker_matches_pod(row, pod_id)]


def _fresh_live_test_worker_present(db, pod_id: str) -> bool:
    for row in _query_workers_for_pod(db, pod_id):
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and metadata.get("live_test_variant") == "fresh":
            return True
    return False


def _resolve_existing_worker_id(
    db,
    pod_id: str,
    prev_proc: WorkerProcessInfo | None,
    *,
    allow_pod_id_fallback: bool = False,
) -> str:
    matching = _query_workers_for_pod(db, pod_id)
    if matching:
        matching.sort(
            key=lambda row: (
                row.get("status") not in {"terminated", "failed", "offline"},
                str(row.get("last_heartbeat") or ""),
                str(row.get("created_at") or ""),
            ),
            reverse=True,
        )
        return str(matching[0]["id"])

    if allow_pod_id_fallback:
        return pod_id

    if prev_proc:
        worker_id = _extract_worker_id_from_cmdline(prev_proc.cmdline)
        if worker_id:
            return worker_id

    raise RuntimeError(f"Could not resolve existing worker_id for pod {pod_id}")


def _remote_dir_exists(ssh, path: str) -> bool:
    exit_code, _stdout, _stderr = ssh.execute_command(
        f"bash -lc {_quote(f'test -d {path}/.git')}",
        timeout=60,
    )
    return exit_code == 0


def _resolve_update_workdir(db, pod_id: str, ssh=None) -> str:
    if _fresh_live_test_worker_present(db, pod_id):
        return FRESH_LIVE_TEST_WORKDIR
    if ssh is not None and _remote_dir_exists(ssh, FRESH_LIVE_TEST_WORKDIR):
        return FRESH_LIVE_TEST_WORKDIR
    return UPDATE_WORKDIR


def _worker_row_exists(db, worker_id: str) -> bool:
    return any(str(row.get("id") or "") == worker_id for row in _query_workers(db))


def _create_worker_row_if_missing(db, worker_id: str, pod_id: str) -> None:
    if _worker_row_exists(db, worker_id):
        return
    create = getattr(db, "create_worker_record", None)
    if create is None:
        return
    try:
        created = asyncio.run(create(worker_id, config.RUNPOD_GPU_TYPE, runpod_id=pod_id))
    except TypeError:
        created = asyncio.run(create(worker_id, config.RUNPOD_GPU_TYPE))
    if not created:
        log.warning(
            "worker row %s was not created for pod %s; continuing because status reactivation may still succeed",
            worker_id,
            pod_id,
        )


def _build_supervisor_restore_command(workdir: str, cli_args: list[str]) -> str:
    if not cli_args:
        raise ValueError("Expected captured supervisor cmdline")
    return f"cd {_quote(workdir)} && nohup {' '.join(_quote(arg) for arg in cli_args)} > logs/startup.log 2>&1 &"


def _should_skip_restore(branch_name: str) -> bool:
    return branch_name == "DETACHED" or branch_name.startswith("live-test/")


def _remote_checkout_and_sync(ssh, branch: str, workdir: str = UPDATE_WORKDIR) -> None:
    command = (
        f"cd {shlex.quote(workdir)} && "
        f"{REMOTE_UV_BOOTSTRAP} && "
        f"git fetch origin {shlex.quote(branch)}:refs/remotes/origin/{shlex.quote(branch)} && "
        f"git checkout -B {shlex.quote(branch)} refs/remotes/origin/{shlex.quote(branch)} && "
        f"git pull --ff-only origin {shlex.quote(branch)} && "
        "uv sync --locked --extra cuda124"
    )
    _ssh_execute(ssh, f"bash -lc {_quote(command)}", timeout=3600)


def _restore_remote_state(
    ssh,
    *,
    prev_remote_branch: str,
    prev_remote_sha: str,
    prev_proc: WorkerProcessInfo | None,
    workdir: str = UPDATE_WORKDIR,
) -> None:
    if _should_skip_restore(prev_remote_branch):
        kill_supervisor_and_worker(ssh)
        log.info(
            "skipping pod restore because previous branch was scratch or detached: %s",
            prev_remote_branch,
        )
        return

    kill_supervisor_and_worker(ssh)
    restore_command = (
        f"cd {shlex.quote(workdir)} && "
        f"{REMOTE_UV_BOOTSTRAP} && "
        f"git checkout {shlex.quote(prev_remote_branch)} && "
        f"git reset --hard {shlex.quote(prev_remote_sha)} && "
        "uv sync --locked --extra cuda124"
    )
    _ssh_execute(ssh, f"bash -lc {_quote(restore_command)}", timeout=3600)

    if prev_proc is None:
        log.info("previous pod had no worker process; leaving restored checkout stopped")
        return

    if prev_proc.family == "supervisor":
        launch_worker_detached(ssh, _build_supervisor_restore_command(workdir, prev_proc.cmdline))
        return

    launch_worker_detached(ssh, build_direct_worker_command(workdir, cli_args=prev_proc.cmdline))


def _spawn_takeover_pod(db, api_key: str) -> tuple[str, str]:
    from gpu_orchestrator.worker_spawner import create_worker_spawner

    spawner = create_worker_spawner(config=None, db=db)
    worker_id = spawner.generate_worker_id()
    created = asyncio.run(db.create_worker_record(worker_id, spawner.gpu_type))
    if not created:
        raise RuntimeError(f"Failed to create worker record for {worker_id}")

    spawn_result = asyncio.run(spawner.spawn_worker(worker_id))
    if not spawn_result or not spawn_result.get("runpod_id"):
        raise RuntimeError(f"spawn_worker did not return a runpod_id for {worker_id}")

    pod_id = str(spawn_result["runpod_id"])
    metadata = {
        "runpod_id": pod_id,
        "pod_details": spawn_result.get("pod_details"),
        "ram_tier": spawn_result.get("ram_tier"),
        "storage_volume": spawn_result.get("storage_volume"),
    }
    asyncio.run(db.update_worker_status(worker_id, "spawning", metadata))

    _wait_for_spawned_pod_ssh(spawner, worker_id, pod_id)
    started = asyncio.run(spawner.start_worker_process(pod_id, worker_id, has_pending_tasks=False))
    if not started:
        raise RuntimeError(f"start_worker_process failed for worker {worker_id} on pod {pod_id}")

    return worker_id, pod_id


def _register_update_worker_record(db, worker_id: str, pod_id: str, args) -> None:
    worker_backend = getattr(args, "backend", "wgp")
    worker_profile = getattr(args, "worker_profile", "default")
    selector_namespace = getattr(args, "selector_namespace", "production")
    selector_version = getattr(args, "selector_version", None)
    worker_contract_version = int(getattr(args, "worker_contract_version", 1))
    worker_pool = f"gpu-{worker_backend}-{selector_namespace}"
    metadata = {
        "runpod_id": pod_id,
        "live_test_variant": UPDATE_VARIANT,
        "worker_backend": worker_backend,
        "worker_profile": worker_profile,
        "worker_pool": worker_pool,
        "selector_namespace": selector_namespace,
        "selector_version": selector_version,
        "worker_contract_version": worker_contract_version,
        "route_contract": {
            "selected_backend": worker_backend,
            "selected_profile": worker_profile,
            "worker_backend": worker_backend,
            "worker_profile": worker_profile,
            "worker_pool": worker_pool,
            "selector_namespace": selector_namespace,
            "selector_version": selector_version,
            "worker_contract_version": worker_contract_version,
            "route_run_id": None,
        },
    }
    updated = asyncio.run(db.update_worker_status(worker_id, "inactive", metadata))
    if not updated:
        raise RuntimeError(f"Failed to reactivate live-test worker row {worker_id} for pod {pod_id}")


def _wait_for_spawned_pod_ssh(spawner, worker_id: str, pod_id: str, *, timeout_sec: int = 300, poll_interval: int = 5) -> None:
    check = getattr(spawner, "check_and_initialize_worker", None)
    if check is None:
        return

    deadline = time.monotonic() + timeout_sec
    last_status: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_status = asyncio.run(check(worker_id, pod_id))
        if last_status.get("ready") is True:
            return
        if last_status.get("status") == "error":
            raise RuntimeError(
                f"Spawned pod {pod_id} became unhealthy before SSH was ready: "
                f"{last_status.get('error') or last_status.get('message')}"
            )
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Spawned pod {pod_id} did not expose SSH details within {timeout_sec}s; "
        f"last_status={last_status}"
    )


def _print_dry_run_plan(*, cases: list, token: str, args) -> None:
    supabase_url = config.require_env("SUPABASE_URL")
    mode = "spawn-takeover" if args.spawn_takeover else "existing"
    pod_hint = args.pod_id or "<spawned-runpod-id>"
    worker_hint = "<new-worker-id>" if args.spawn_takeover else "<existing-worker-id>"
    print("Variant: update")
    print(f"Mode: {mode}")
    print(f"Target pod: {pod_hint}")
    print("Injected env vars:")
    for key in (
        "REIGH_ACCESS_TOKEN",
        "REIGH_BACKEND",
        "REIGH_SELECTOR_NAMESPACE",
        "REIGH_SELECTOR_VERSION",
        "REIGH_WORKER_CONTRACT_VERSION",
        "REIGH_WORKER_PROFILE",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_URL",
        "WORKER_DB_CLIENT_AUTH_MODE",
    ):
        print(f"- {key}")
    print("Planned launch command:")
    print(
        build_run_worker_command(
            UPDATE_WORKDIR,
            reigh_token=token,
            supabase_url=supabase_url,
            worker_id=worker_hint,
            wgp_profile=args.wgp_profile,
            idle_release_minutes=0,
        )
    )
    print("Planned tasks:")
    for case in cases:
        route_suffix = f", route={case.route_key}, backend={case.route_runtime.selected_backend}" if case.route_key else ""
        print(f"- {case.name} ({case.task_type}{route_suffix}, timeout={case.timeout_sec}s)")


def run(args) -> int:
    context = _prepare_context(args)
    token = context["token"]
    db = context["db"]
    user_id = context["user_id"]
    project_id = context["project_id"]
    cases = context["cases"]

    _validate_cases(cases, project_id)

    if args.dry_run:
        _print_dry_run_plan(cases=cases, token=token, args=args)
        return 0

    api_key: str | None = None
    api_key = config.require_env("RUNPOD_API_KEY")
    supabase_url = config.require_env("SUPABASE_URL")
    service_role_key = config.require_env("SUPABASE_SERVICE_ROLE_KEY")
    worker_env = _build_worker_env(token, supabase_url, service_role_key, args)
    mode = "spawn-takeover" if args.spawn_takeover else "existing"
    out_dir = _runs_root() / _timestamp_label()

    ssh = None
    pod_id: str | None = None
    worker_id: str | None = None
    snapshot = None
    branch: str | None = None
    preserve_branch = True
    prev_remote_branch = "DETACHED"
    prev_remote_sha = ""
    prev_proc: WorkerProcessInfo | None = None
    workdir = UPDATE_WORKDIR

    try:
        if args.spawn_takeover:
            worker_id, pod_id = _spawn_takeover_pod(db, api_key)
        else:
            pod_id = args.pod_id

        assert pod_id
        assert_safe_to_take_over(
            db,
            pod_id,
            user_id,
            allow_fresh_heartbeat=args.spawn_takeover or getattr(args, "allow_fresh_heartbeat", False),
        )

        worker_repo_path = str(config.WORKER_ROOT)
        snapshot = snapshot_local_state(worker_repo_path)
        branch, _sha = push_working_copy_to_temp_branch(worker_repo_path, snapshot)

        ssh = open_session(pod_id, api_key)
        workdir = _resolve_update_workdir(db, pod_id, ssh)
        prev_remote_branch = _read_remote_branch(ssh, workdir)
        prev_remote_sha = _read_remote_sha(ssh, workdir)
        prev_proc = capture_current_worker_cmdline(ssh)

        if not worker_id:
            worker_id = _resolve_existing_worker_id(
                db,
                pod_id,
                prev_proc,
                allow_pod_id_fallback=workdir == FRESH_LIVE_TEST_WORKDIR,
            )
        _create_worker_row_if_missing(db, worker_id, pod_id)

        _remote_checkout_and_sync(ssh, branch, workdir)
        if getattr(args, "backend", "wgp") == "vibecomfy":
            clone_and_install_vibecomfy(
                ssh,
                repo_url=VIBECOMFY_REPO_URL,
                branch=getattr(args, "vibecomfy_ref", "megaplan/production-parity-templates"),
                workdir=VIBECOMFY_WORKDIR,
                python_path=VIBECOMFY_PYTHON,
            )
        kill_supervisor_and_worker(ssh)
        _register_update_worker_record(db, worker_id, pod_id, args)
        launch_worker_detached(
            ssh,
            export_env(worker_env)
            + " && "
            + build_run_worker_command(
                workdir,
                reigh_token=token,
                supabase_url=supabase_url,
                worker_id=worker_id,
                wgp_profile=args.wgp_profile,
                idle_release_minutes=0,
            ),
        )
        wait_until_ready(db, worker_id=worker_id, timeout_sec=900)

        results = run_matrix(db, project_id, cases)
        write_report(results, f"{UPDATE_VARIANT}-{mode}", pod_id, out_dir)
        _restore_remote_state(
            ssh,
            prev_remote_branch=prev_remote_branch,
            prev_remote_sha=prev_remote_sha,
            prev_proc=prev_proc,
            workdir=workdir,
        )
        preserve_branch = False
        return 0
    finally:
        if snapshot is not None:
            restore_local_state(str(config.WORKER_ROOT), snapshot)
        if branch:
            kept_branch = cleanup_temp_branch(branch, preserve=preserve_branch, submodule_path=str(config.WORKER_ROOT))
            if preserve_branch:
                print(f"Preserved temp branch for inspection: {kept_branch}")
        if ssh is not None:
            try:
                logs = fetch_worker_logs(ssh, workdir)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "worker_logs.txt").write_text(logs, encoding="utf-8")
            except Exception as exc:
                log.warning("failed to fetch update variant worker logs: %s", exc)
            finally:
                disconnect = getattr(ssh, "disconnect", None)
                if callable(disconnect):
                    disconnect()
        guarded_terminate(pod_id, api_key if not args.dry_run else None, no_terminate=args.no_terminate)


__all__ = [
    "UPDATE_VARIANT",
    "UPDATE_WORKDIR",
    "run",
]
