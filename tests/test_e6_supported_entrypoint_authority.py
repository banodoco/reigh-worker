"""E6 structural gate for the supported worker entrypoint authority boundary."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

import pytest

from source.task_handlers.tasks.task_types import TASK_TYPE_CATALOG
from source.task_handlers.tasks.template_routing import (
    DIRECT_ROUTE_ALIASES,
    RouteSupportState,
    SPRINT_2_SELECTOR_MAP,
    WorkerBackend,
    parse_worker_backend,
    resolve_task_route,
)


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "source" / "runtime" / "entrypoints" / "worker.py"
SERVER = ROOT / "source" / "runtime" / "worker" / "server.py"
ROUTE_CONTRACT = ROOT / "source" / "task_handlers" / "tasks" / "template_routing.py"

FORBIDDEN_IMPORT_PREFIXES = (
    "supabase",
    "headless_model_management",
    "source.core.db",
    "source.task_handlers.queue",
    "source.task_handlers.tasks.task_registry",
    "source.task_handlers.tasks.task_conversion",
    "source.task_handlers.tasks.task_execution",
    "source.task_handlers.orchestration.finalization_service",
    "source.task_handlers.travel.chaining",
    "source.task_handlers.worker.heartbeat_utils",
    "source.task_handlers.worker.worker_utils",
)
FORBIDDEN_SERVER_SYMBOLS = (
    "create_client",
    "_initialize_db_runtime",
    "db_config",
    "SUPABASE",
    "HeadlessTaskQueue",
    "poll_next_task",
    "ClaimPollOutcome",
    "TaskRegistry",
    "process_single_task",
    "update_task_status",
    "requeue_task_for_retry",
    "task_queue",
    "guardian_process",
    "send_heartbeat_with_logs",
)


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _server_source() -> str:
    return SERVER.read_text(encoding="utf-8")


def test_supported_entrypoint_imports_without_legacy_authority() -> None:
    entrypoint_module = import_module("source.runtime.entrypoints.worker")
    server_module = import_module("source.runtime.worker.server")
    assert callable(entrypoint_module.main)
    assert callable(server_module.main)

    for path in (ENTRYPOINT, SERVER):
        imports = _imports(ast.parse(path.read_text(encoding="utf-8")))
        assert not {
            name
            for name in imports
            if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES)
        }, path


def test_supported_server_has_no_claimant_settlement_or_queue_lifecycle() -> None:
    source = _server_source()
    assert not any(symbol in source for symbol in FORBIDDEN_SERVER_SYMBOLS)
    assert "run_bootstrap_once" in source
    assert "get_bootstrap_controller" in source
    assert "start_local_http_server" in source
    assert "run_worker_preflight" in source
    assert "publish_warm_cache_state" in source
    assert "while True" not in source


def test_removed_database_and_queue_flags_are_not_supported() -> None:
    tree = ast.parse(_server_source())
    flags = {
        call.args[0].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "add_argument"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }
    assert flags.isdisjoint(
        {
            "--db-type",
            "--supabase-url",
            "--supabase-access-token",
            "--supabase-anon-key",
            "--queue-workers",
            "--poll-interval",
            "--migrate-only",
        }
    )


def test_route_contract_is_explicit_and_fail_closed() -> None:
    route_source = ROUTE_CONTRACT.read_text(encoding="utf-8")
    tree = ast.parse(route_source)
    imported_modules = _imports(tree)
    assert not imported_modules.intersection({"importlib", "pkgutil", "glob"})
    assert "SPRINT_2_SELECTOR_MAP" in route_source
    assert "SECTION3A_ROUTE_SUPPORT_MAP" in route_source

    unknown = resolve_task_route(
        task_id="e6-unknown",
        task_type="unmapped_route",
        backend=WorkerBackend.VIBECOMFY,
    )
    assert unknown.support_state is RouteSupportState.VIBECOMFY_UNSUPPORTED
    assert unknown.fail_closed_reason
    with pytest.raises(ValueError):
        parse_worker_backend("implicit-fallback")


def test_replaced_vibe_aliases_are_explicitly_disposed() -> None:
    assert DIRECT_ROUTE_ALIASES["z_image"] == "z_image_turbo"
    assert SPRINT_2_SELECTOR_MAP["z_image_turbo"].disposition == "replaced_by_astrid_d3"
    assert SPRINT_2_SELECTOR_MAP["image-upscale"].disposition == "replaced_by_astrid_d4"
    assert SPRINT_2_SELECTOR_MAP["image_upscale"].disposition == "replaced_by_astrid_d4"

    for route_key in ("z_image", "z_image_turbo", "image-upscale", "image_upscale"):
        resolved = resolve_task_route(
            task_id=f"e6-{route_key}",
            task_type=route_key,
            backend=WorkerBackend.VIBECOMFY,
        )
        assert resolved.fail_closed_reason, route_key
        assert not resolved.should_use_vibecomfy


def test_preserved_wgp_progress_artifacts_and_video_enhance() -> None:
    server_module = import_module("source.runtime.worker.server")
    assert callable(server_module.ensure_wan2gp_on_path)
    assert "wgp_bridge" in _server_source()

    conversion_source = (ROOT / "source" / "task_handlers" / "tasks" / "task_conversion.py").read_text(
        encoding="utf-8"
    )
    registry_source = (ROOT / "source" / "task_handlers" / "tasks" / "task_registry.py").read_text(
        encoding="utf-8"
    )
    orchestration_source = ROOT / "source" / "task_handlers" / "travel" / "orchestrator.py"
    assert "uni3c_start_percent" in conversion_source
    assert "uni3c_end_percent" in conversion_source
    assert "uni3c_start_percent" in registry_source
    assert "uni3c_end_percent" in registry_source
    assert "uni3c_start_percent" in orchestration_source.read_text(encoding="utf-8")
    assert "video_enhance" in TASK_TYPE_CATALOG

    for preserved in (
        ROOT / "docs" / "sprint-12-route-inventory.md",
        ROOT / "docs" / "sprint-12-route-support.md",
        ROOT / "source" / "task_handlers" / "tasks" / "template_routing.py",
        ROOT / "debug" / "diagnostics.py",
    ):
        assert preserved.is_file(), preserved
