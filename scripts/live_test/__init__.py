"""Shared package bootstrap for the live-test harness."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
WORKER_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_ROOT = WORKSPACE_ROOT / "reigh-worker-orchestrator"


def _add_path(path: Path) -> None:
    resolved = str(path.resolve())
    if path.exists() and resolved not in sys.path:
        sys.path.insert(0, resolved)


def ensure_orchestrator_imports() -> None:
    _add_path(ORCHESTRATOR_ROOT)


ensure_orchestrator_imports()

_LAZY_EXPORTS = {
    "DatabaseClient": ("scripts.live_test.db_client", "DatabaseClient"),
    "RunPodConfig": ("runpod_lifecycle", "RunPodConfig"),
    "launch": ("runpod_lifecycle", "launch"),
    "get_network_volumes": ("runpod_lifecycle", "get_network_volumes"),
}


def _call_lazy_export(name: str, *args, **kwargs):
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)(*args, **kwargs)


def get_pod_ssh_details(*args, **kwargs):
    return _call_lazy_export("get_pod_ssh_details", *args, **kwargs)


def get_pod_status(*args, **kwargs):
    return _call_lazy_export("get_pod_status", *args, **kwargs)


def terminate_pod(*args, **kwargs):
    return _call_lazy_export("terminate_pod", *args, **kwargs)


class SSHClient:
    def __new__(cls, *args, **kwargs):
        module_name, attr_name = _LAZY_EXPORTS["SSHClient"]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)(*args, **kwargs)


_LAZY_EXPORTS.update(
    {
        "SSHClient": ("runpod_lifecycle.ssh", "SSHClient"),
        "get_pod_ssh_details": ("runpod_lifecycle.api", "get_pod_ssh_details"),
        "get_pod_status": ("runpod_lifecycle.api", "get_pod_status"),
        "terminate_pod": ("runpod_lifecycle.api", "terminate_pod"),
    }
)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "DatabaseClient",
    "ORCHESTRATOR_ROOT",
    "RunPodConfig",
    "SSHClient",
    "WORKER_ROOT",
    "WORKSPACE_ROOT",
    "ensure_orchestrator_imports",
    "get_network_volumes",
    "get_pod_ssh_details",
    "get_pod_status",
    "launch",
    "terminate_pod",
]
