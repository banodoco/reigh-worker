"""Variant A driver: launch a fresh pod and run the live-test matrix."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import io
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time
from typing import Any

from scripts.live_test import config
from scripts.live_test.heartbeat_waiter import wait_until_ready
from scripts.live_test.launch_command import build_run_worker_command
from scripts.live_test.logger import get_logger
from scripts.live_test.matrix import build_matrix, poll_queued_matrix, queue_matrix, render_case_payload
from scripts.live_test.preflight import (
    assert_user_queue_clean,
    close_stale_live_test_tasks,
    ensure_live_test_route_selectors,
    ensure_user_cloud_generation_enabled,
    get_or_create_live_test_project,
)
from scripts.live_test.report import write_report
from scripts.live_test.ssh_bootstrap import (
    clone_and_install_vibecomfy,
    clone_repo_into,
    export_env,
    fetch_worker_logs,
    launch_worker_detached,
    open_session,
    run_install,
)
from scripts.live_test.terminate_guard import guarded_terminate, prune_stale_live_test_pods
from scripts.live_test.token_resolver import resolve_token_to_user_id


FRESH_VARIANT = "fresh"
FRESH_WORKDIR = "/workspace/Reigh-Worker-LiveTest"
FRESH_REPO_URL = "https://github.com/banodoco/Reigh-Worker.git"
VIBECOMFY_WORKDIR = "/workspace/vibecomfy"
VIBECOMFY_REPO_URL = "https://github.com/peteromallet/VibeComfy.git"
VIBECOMFY_PYTHON = "python3.11"
VIBECOMFY_DEFAULT_CASE_ORDER = {
    "z_image_turbo": 0,
    "z_image_turbo_i2i": 1,
    "qwen_image_2512": 2,
    "qwen_image_edit": 3,
    "image_inpaint": 4,
    "annotated_image_edit": 5,
    "qwen_image_style": 6,
    "wan_2_2_t2i": 7,
    "wan_2_2_i2v": 8,
    "animate_character": 9,
    "image_upscale": 10,
    "video_enhance": 11,
    "flux_klein_edit": 12,
    "individual_travel_segment_wan22_vace": 13,
}

log = get_logger(__name__)


_SENSITIVE_OUTPUT_PATTERNS = (
    re.compile(r"(?i)(['\"]?)(REIGH_ACCESS_TOKEN|SUPABASE_SERVICE_ROLE_KEY|RUNPOD_API_KEY|REIGH_LIVE_TEST_TOKEN)(['\"]?\s*[:=]\s*['\"]?)([^,'\"\s}]+)"),
    re.compile(r"(?i)(--reigh-access-token\s+)([^\s]+)"),
)


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_matrix_cases(args) -> list:
    cases = build_matrix(
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
    explicit_selection = bool(getattr(args, "case", []) or getattr(args, "task_type", []) or getattr(args, "route_key", []))
    if getattr(args, "backend", "wgp") == "vibecomfy" and not explicit_selection:
        filtered = [case for case in cases if case.support_state == "vibecomfy_supported"]
        return sorted(
            filtered,
            key=lambda case: VIBECOMFY_DEFAULT_CASE_ORDER.get(case.name, len(VIBECOMFY_DEFAULT_CASE_ORDER)),
        )
    return cases


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


def _runs_root() -> Path:
    return config.WORKER_ROOT / "scripts" / "live_test" / "runs"


def _redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in _SENSITIVE_OUTPUT_PATTERNS:
        if pattern.pattern.startswith("(?i)(--reigh-access-token"):
            redacted = pattern.sub(r"\1<redacted>", redacted)
        else:
            redacted = pattern.sub(r"\1\2\3<redacted>", redacted)
    return redacted


@contextmanager
def _capture_and_redact_noisy_lifecycle_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = stdout
        sys.stderr = stderr
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        captured_stdout = _redact_sensitive_text(stdout.getvalue()).strip()
        captured_stderr = _redact_sensitive_text(stderr.getvalue()).strip()
        if captured_stdout:
            log.info("captured runpod lifecycle stdout: %s", captured_stdout)
        if captured_stderr:
            log.warning("captured runpod lifecycle stderr: %s", captured_stderr)


@contextmanager
def _phase(name: str, **fields):
    started_at = time.monotonic()
    log.info("live test phase started", phase=name, **fields)
    try:
        yield
    except Exception:
        log.exception(
            "live test phase failed",
            phase=name,
            elapsed_sec=round(time.monotonic() - started_at, 1),
            **fields,
        )
        raise
    else:
        log.info(
            "live test phase completed",
            phase=name,
            elapsed_sec=round(time.monotonic() - started_at, 1),
            **fields,
        )


def _prepare_context(args) -> dict[str, Any]:
    token = config.require_env("REIGH_LIVE_TEST_TOKEN")
    db = config.DatabaseClient()
    user_id = resolve_token_to_user_id(db, token)
    stale_count = close_stale_live_test_tasks(db, user_id)
    if stale_count:
        log.info("closed stale live-test tasks before fresh run", count=stale_count)
    if ensure_user_cloud_generation_enabled(db, user_id):
        log.info("enabled cloud generation for live-test user", user_id=user_id)
    assert_user_queue_clean(db, user_id)
    project_id = get_or_create_live_test_project(db, user_id)
    cases = _build_matrix_cases(args)
    created_selectors = ensure_live_test_route_selectors(
        db,
        getattr(args, "selector_namespace", "production"),
        [case.route_key for case in cases if case.route_key],
        backend=getattr(args, "backend", "wgp"),
        fallback_selectors={
            str(case.route_key): {
                "selected_backend": case.route_runtime.selected_backend,
                "selector_version": case.route_runtime.selector_version,
                "support_state": case.support_state,
                "selected_template_id": case.selected_template_id,
            }
            for case in cases
            if case.route_key and case.support_state == "vibecomfy_supported"
        },
    )
    if created_selectors:
        log.info(
            "created isolated live-test route selectors",
            selector_namespace=getattr(args, "selector_namespace", "production"),
            count=created_selectors,
        )
    return {
        "db": db,
        "token": token,
        "user_id": user_id,
        "project_id": project_id,
        "cases": cases,
    }


def _validate_cases(cases: list, project_id: str) -> None:
    for index, case in enumerate(cases, start=1):
        render_case_payload(case, project_id=project_id, unique_suffix=f"fresh-{index}")


def _register_fresh_worker_record(db, pod_id: str, pod: dict[str, Any], args) -> None:
    created = asyncio.run(db.create_worker_record(pod_id, config.RUNPOD_GPU_TYPE, runpod_id=pod_id))
    if not created:
        raise RuntimeError(f"Failed to create fresh live-test worker record for pod {pod_id}")
    worker_backend = getattr(args, "backend", "wgp")
    worker_profile = getattr(args, "worker_profile", "default")
    selector_namespace = getattr(args, "selector_namespace", "production")
    selector_version = getattr(args, "selector_version", None)
    worker_contract_version = int(getattr(args, "worker_contract_version", 1))
    worker_pool = f"gpu-{worker_backend}-{selector_namespace}"
    metadata = {
        "runpod_id": pod_id,
        "pod_details": pod,
        "storage_volume": pod.get("volumeId") or pod.get("networkVolumeId"),
        "live_test_variant": FRESH_VARIANT,
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
    # Keep the row out of orchestrator spawning ownership. The live-test harness
    # owns setup and launch; the worker heartbeat will promote it to active.
    updated = asyncio.run(db.update_worker_status(pod_id, "inactive", metadata))
    if not updated:
        raise RuntimeError(f"Failed to register fresh live-test worker metadata for {pod_id}")


def _print_dry_run_plan(*, token: str, project_id: str, cases: list, args) -> None:
    supabase_url = config.get_env("SUPABASE_URL", "https://example.supabase.co")
    launch_command = build_run_worker_command(
        FRESH_WORKDIR,
        reigh_token=None,
        supabase_url=supabase_url,
        worker_id="<runpod-pod-id>",
        wgp_profile=args.wgp_profile,
        idle_release_minutes=0,
        redact_secrets=True,
    )

    print("Variant: fresh")
    print(f"Project ID: {project_id}")
    print(f"Clone target: {FRESH_WORKDIR}")
    if getattr(args, "backend", "wgp") == "vibecomfy":
        print(f"VibeComfy clone target: {VIBECOMFY_WORKDIR}")
    print(f"Terminate after run: {not args.no_terminate}")
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
        "VIBECOMFY_CWD",
        "VIBECOMFY_PATH",
        "VIBECOMFY_PYTHON",
    ):
        print(f"- {key}")
    print("Planned launch command:")
    print(launch_command)
    print("Planned tasks:")
    for case in cases:
        route_suffix = f", route={case.route_key}, backend={case.route_runtime.selected_backend}" if case.route_key else ""
        print(f"- {case.name} ({case.task_type}{route_suffix}, timeout={case.timeout_sec}s)")


def run(args) -> int:
    if args.dry_run and not config.get_env("REIGH_LIVE_TEST_TOKEN"):
        cases = _build_matrix_cases(args)
        project_id = "<live-test-project-id>"
        _validate_cases(cases, project_id)
        _print_dry_run_plan(
            token="<REIGH_LIVE_TEST_TOKEN>",
            project_id=project_id,
            cases=cases,
            args=args,
        )
        return 0

    context = _prepare_context(args)
    token = context["token"]
    db = context["db"]
    project_id = context["project_id"]
    cases = context["cases"]

    _validate_cases(cases, project_id)

    if args.dry_run:
        _print_dry_run_plan(token=token, project_id=project_id, cases=cases, args=args)
        return 0

    api_key: str | None = None
    api_key = config.require_env("RUNPOD_API_KEY")
    cleanup = prune_stale_live_test_pods(api_key)
    if cleanup.terminated:
        log.warning("terminated stale fresh live-test pods before launch: %s", ", ".join(cleanup.terminated))
    if cleanup.failed:
        failed = ", ".join(f"{pod_id}: {error}" for pod_id, error in cleanup.failed)
        raise RuntimeError(f"Failed to terminate stale fresh live-test pods before launch: {failed}")
    supabase_url = config.require_env("SUPABASE_URL")
    service_role_key = config.require_env("SUPABASE_SERVICE_ROLE_KEY")
    worker_env = _build_worker_env(token, supabase_url, service_role_key, args)
    out_dir = _runs_root() / _timestamp_label()

    pod_id: str | None = None
    ssh = None
    try:
        from runpod_lifecycle.api import create_pod as create_pod_and_wait
        from runpod_lifecycle.api import get_network_volumes

        network_volume_id: str | None = None
        selected_volume_name: str | None = None
        try:
            available = get_network_volumes(api_key)
            by_name = {v.get("name"): v.get("id") for v in available if v.get("name")}
            for candidate in config.RUNPOD_STORAGE_VOLUMES:
                if by_name.get(candidate):
                    network_volume_id = by_name[candidate]
                    selected_volume_name = candidate
                    break
        except Exception as exc:
            log.warning("could not list network volumes (%s); continuing without one", exc)

        if network_volume_id:
            log.info("attaching network volume %s (%s) at %s", selected_volume_name, network_volume_id, config.RUNPOD_VOLUME_MOUNT_PATH)
        else:
            log.warning("no network volume matched %s; pod will only have ephemeral container disk", list(config.RUNPOD_STORAGE_VOLUMES))

        with _phase("create_runpod_pod", gpu_type=config.RUNPOD_GPU_TYPE, image=config.RUNPOD_WORKER_IMAGE):
            with _capture_and_redact_noisy_lifecycle_output():
                pod = create_pod_and_wait(
                    api_key=api_key,
                    gpu_type_id=config.RUNPOD_GPU_TYPE,
                    image_name=config.RUNPOD_WORKER_IMAGE,
                    name=f"reigh-live-test-fresh-{_timestamp_label().lower()}",
                    network_volume_id=network_volume_id,
                    volume_mount_path=config.RUNPOD_VOLUME_MOUNT_PATH,
                    disk_in_gb=config.LIVE_TEST_DISK_SIZE_GB,
                    container_disk_in_gb=config.LIVE_TEST_CONTAINER_DISK_GB,
                    min_vcpu_count=config.RUNPOD_MIN_VCPU_COUNT,
                    min_memory_in_gb=config.RUNPOD_MIN_MEMORY_GB,
                    template_id=config.RUNPOD_TEMPLATE_ID,
                    env_vars=worker_env,
                )
        if not pod or not pod.get("id"):
            raise RuntimeError("create_pod_and_wait did not return a pod id")

        pod_id = str(pod["id"])
        _register_fresh_worker_record(db, pod_id, pod, args)
        with _phase("open_ssh_session", pod_id=pod_id):
            ssh = open_session(pod_id, api_key)
        with _phase("clone_reigh_worker", pod_id=pod_id, ref=args.ref or "main", workdir=FRESH_WORKDIR):
            clone_repo_into(ssh, FRESH_WORKDIR, FRESH_REPO_URL, branch=args.ref or "main")
        with _phase("install_reigh_worker", pod_id=pod_id, workdir=FRESH_WORKDIR):
            run_install(ssh, FRESH_WORKDIR)
        if args.backend == "vibecomfy":
            with _phase(
                "clone_install_vibecomfy",
                pod_id=pod_id,
                ref=args.vibecomfy_ref,
                workdir=VIBECOMFY_WORKDIR,
                python=VIBECOMFY_PYTHON,
            ):
                clone_and_install_vibecomfy(
                    ssh,
                    repo_url=VIBECOMFY_REPO_URL,
                    branch=args.vibecomfy_ref,
                    workdir=VIBECOMFY_WORKDIR,
                    python_path=VIBECOMFY_PYTHON,
                )

        command = build_run_worker_command(
            FRESH_WORKDIR,
            reigh_token=None,
            supabase_url=supabase_url,
            worker_id=pod_id,
            wgp_profile=args.wgp_profile,
            idle_release_minutes=0,
        )
        with _phase("launch_worker", pod_id=pod_id):
            launch_worker_detached(ssh, export_env(worker_env) + " && " + command)
        with _phase("wait_worker_ready", pod_id=pod_id):
            wait_until_ready(db, worker_id=pod_id, timeout_sec=900, progress_every_sec=60)
        with _phase("queue_matrix", pod_id=pod_id, cases=len(cases)):
            queued = queue_matrix(db, project_id, cases)

        with _phase("run_matrix", pod_id=pod_id, cases=len(cases)):
            results = poll_queued_matrix(db, project_id, queued, worker_id=pod_id)
        with _phase("write_report", pod_id=pod_id, out_dir=str(out_dir)):
            write_report(results, FRESH_VARIANT, pod_id, out_dir)
        return 0
    finally:
        if ssh is not None:
            try:
                logs = fetch_worker_logs(ssh, FRESH_WORKDIR)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "worker_logs.txt").write_text(logs, encoding="utf-8")
            except Exception as exc:
                log.warning("failed to fetch fresh variant worker logs: %s", exc)
            finally:
                disconnect = getattr(ssh, "disconnect", None)
                if callable(disconnect):
                    disconnect()
        guarded_terminate(pod_id, api_key if not args.dry_run else None, no_terminate=args.no_terminate)


__all__ = [
    "FRESH_REPO_URL",
    "FRESH_VARIANT",
    "FRESH_WORKDIR",
    "run",
]
