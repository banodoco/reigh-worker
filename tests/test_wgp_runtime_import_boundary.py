"""Contracts for late WGP runtime imports from worker processes."""

from __future__ import annotations

import os
import sys
import types

from source.runtime.wgp_ports import runtime_registry


def test_late_wgp_import_sanitizes_worker_argv_and_restores_process_state(tmp_path, monkeypatch):
    original_argv = [
        "worker.py",
        "--supabase-url",
        "https://example.invalid",
        "--reigh-access-token",
        "token",
        "--worker",
        "worker-1",
    ]
    original_cwd = os.getcwd()
    wan_root = tmp_path / "Wan2GP"
    wan_root.mkdir()
    observed: dict[str, object] = {}
    fake_module = types.SimpleNamespace(models_def={})

    def _fake_importer(module_name: str):
        observed["module_name"] = module_name
        observed["argv"] = list(sys.argv)
        observed["cwd"] = os.getcwd()
        observed["path0"] = sys.path[0]
        return fake_module

    monkeypatch.setattr(sys, "argv", list(original_argv))
    monkeypatch.setattr(runtime_registry, "_bootstrap_runtime_paths", lambda: str(wan_root))
    monkeypatch.setattr(runtime_registry, "_resolve_runtime_import_module", lambda *, prefer_bridge: _fake_importer)
    monkeypatch.setitem(runtime_registry._STATE, "module_name", "wgp_runtime_import_boundary_test")
    monkeypatch.delitem(sys.modules, "wgp_runtime_import_boundary_test", raising=False)

    try:
        runtime = runtime_registry.get_wgp_runtime_module(force_reload=True)
    finally:
        runtime_registry.reset_wgp_runtime_module()
        monkeypatch.setitem(runtime_registry._STATE, "module_name", "wgp")

    assert runtime.models_def == {}
    assert observed == {
        "module_name": "wgp_runtime_import_boundary_test",
        "argv": ["worker.py"],
        "cwd": str(wan_root),
        "path0": str(wan_root),
    }
    assert sys.argv == original_argv
    assert os.getcwd() == original_cwd
