"""Shared helpers between live-test variant drivers (fresh / prebuilt / update)."""

from __future__ import annotations

import asyncio
import io
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.live_test import config
from scripts.live_test.logger import get_logger


log = get_logger(__name__)


_SENSITIVE_OUTPUT_PATTERNS = (
    re.compile(r"(?i)(['\"]?)(REIGH_ACCESS_TOKEN|SUPABASE_SERVICE_ROLE_KEY|RUNPOD_API_KEY|REIGH_LIVE_TEST_TOKEN)(['\"]?\s*[:=]\s*['\"]?)([^,'\"\s}]+)"),
    re.compile(r"(?i)(--reigh-access-token\s+)([^\s]+)"),
)


def _timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _resolve_runpod_gpu_type_id(api_key: str, requested_gpu_type: str) -> tuple[str, str]:
    from runpod_lifecycle.api import find_gpu_type

    gpu = find_gpu_type(requested_gpu_type, api_key)
    if not gpu:
        raise RuntimeError(f"RunPod GPU type not found: {requested_gpu_type!r}")

    gpu_type_id = str(gpu.get("id") or "").strip()
    gpu_display_name = str(gpu.get("displayName") or requested_gpu_type).strip()
    if not gpu_type_id:
        raise RuntimeError(f"RunPod GPU type {requested_gpu_type!r} resolved without an id")

    log.info(
        "resolved RunPod GPU type",
        requested_gpu_type=requested_gpu_type,
        gpu_type_id=gpu_type_id,
        gpu_display_name=gpu_display_name,
    )
    return gpu_type_id, gpu_display_name


@contextmanager
def _phase(name: str, **fields):
    started_at = time.monotonic()
    log.info("live test phase started", phase=name, **fields)
    try:
        yield
    except Exception as exc:
        log.error(
            "live test phase failed",
            phase=name,
            elapsed_sec=round(time.monotonic() - started_at, 1),
            error_type=type(exc).__name__,
            error=str(exc),
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


def _build_worker_env_base(
    token: str,
    supabase_url: str,
    service_role_key: str,
    args=None,
    *,
    vibecomfy_workdir: str,
    vibecomfy_python: str,
) -> dict[str, str]:
    """Base worker env shared by fresh and prebuilt variants.

    Callers may layer additional keys (e.g. HF_HOME for the prebuilt model
    cache) on top of the returned dict.
    """
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
        attention_profile = "sage" if str(getattr(args, "worker_profile", "")).strip().lower() in {"sage", "optimized"} else "portable"
        env.update(
            {
                "VIBECOMFY_CWD": vibecomfy_workdir,
                "VIBECOMFY_PATH": vibecomfy_workdir,
                "VIBECOMFY_PYTHON": vibecomfy_python,
                "VIBECOMFY_ATTENTION_PROFILE": attention_profile,
                "REIGH_VIBECOMFY_ATTENTION_PROFILE": attention_profile,
            }
        )
    return env


def select_network_volume(
    api_key: str,
    *,
    name_prefix: str,
    data_center_filter: str | None = None,
) -> tuple[str, str, str] | None:
    """Return ``(volume_id, name, data_center_id)`` of the first volume matching *name_prefix*.

    Returns ``None`` if no volume matches. When *data_center_filter* is given,
    only volumes whose ``dataCenterId`` equals (case-insensitive) the filter
    are considered.
    """
    from runpod_lifecycle import get_network_volumes

    volumes = get_network_volumes(api_key) or []
    for volume in volumes:
        name = str(volume.get("name") or "")
        volume_id = str(volume.get("id") or "")
        data_center_id = str(volume.get("dataCenterId") or "")
        if not name or not volume_id:
            continue
        if not name.startswith(name_prefix):
            continue
        if data_center_filter and data_center_id.lower() != data_center_filter.lower():
            continue
        return volume_id, name, data_center_id
    return None


def register_worker_record(
    db,
    pod_id: str,
    pod: dict[str, Any],
    args,
    *,
    variant_label: str,
) -> None:
    """Create + reactivate a workers row for a live-test pod under *variant_label*."""
    created = asyncio.run(db.create_worker_record(pod_id, config.RUNPOD_GPU_TYPE, runpod_id=pod_id))
    if not created:
        raise RuntimeError(f"Failed to create {variant_label} live-test worker record for pod {pod_id}")
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
        "live_test_variant": variant_label,
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
        raise RuntimeError(
            f"Failed to register {variant_label} live-test worker metadata for {pod_id}"
        )


__all__ = [
    "_SENSITIVE_OUTPUT_PATTERNS",
    "_build_worker_env_base",
    "_capture_and_redact_noisy_lifecycle_output",
    "_phase",
    "_redact_sensitive_text",
    "_resolve_runpod_gpu_type_id",
    "_runs_root",
    "_timestamp_label",
    "register_worker_record",
    "select_network_volume",
]
