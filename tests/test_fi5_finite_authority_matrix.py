"""Bounded F-I5 route ownership and authority convergence checks."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

from source.task_handlers.tasks.task_types import TASK_TYPE_CATALOG
from source.task_handlers.tasks.template_routing import (
    DIRECT_ROUTE_ALIASES,
    SECTION3A_ROUTE_SUPPORT_MAP,
    SPRINT_2_SELECTOR_MAP,
    RouteSupportState,
    WorkerBackend,
    resolve_task_route,
)


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "source" / "runtime" / "entrypoints" / "worker.py"
SERVER = ROOT / "source" / "runtime" / "worker" / "server.py"

ASTRID_REPLACEMENT_DISPOSITIONS = {
    "replaced_by_astrid_d3": "Astrid GenericPackHost/Astrid pack",
    "replaced_by_astrid_d4": "Astrid GenericPackHost/Astrid pack",
}
RETIRED_ROUTE_DISPOSITIONS = {
    "legacy_custom_task": "retired: direct queue fallthrough removed in E-4",
}
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
    "submit_task",
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


def _selector_entries():
    yield from SPRINT_2_SELECTOR_MAP.items()
    yield from SECTION3A_ROUTE_SUPPORT_MAP.items()


def _owner_candidates(route_key: str) -> set[str]:
    entries = dict(_selector_entries())
    entry = entries.get(route_key)
    if entry is None and route_key in DIRECT_ROUTE_ALIASES:
        return _owner_candidates(DIRECT_ROUTE_ALIASES[route_key])
    if entry is None:
        return {"native WGP"} if route_key in TASK_TYPE_CATALOG else set()
    if entry.disposition in ASTRID_REPLACEMENT_DISPOSITIONS:
        return {ASTRID_REPLACEMENT_DISPOSITIONS[entry.disposition]}
    if entry.support_state in {
        RouteSupportState.WGP_ONLY,
        RouteSupportState.VIBECOMFY_UNSUPPORTED,
    }:
        return {"native WGP"}
    if entry.support_state is RouteSupportState.VIBECOMFY_SUPPORTED:
        return {"retained Worker Vibe"}
    return set()


def test_finite_supported_routes_have_exactly_one_owner() -> None:
    selector_routes = set(SPRINT_2_SELECTOR_MAP) | set(SECTION3A_ROUTE_SUPPORT_MAP)
    catalog_only_routes = set(TASK_TYPE_CATALOG) - selector_routes
    finite_routes = selector_routes | catalog_only_routes

    assert finite_routes
    for route_key in sorted(finite_routes):
        assert len(_owner_candidates(route_key)) == 1, route_key

    for alias, canonical in DIRECT_ROUTE_ALIASES.items():
        assert canonical in finite_routes, alias
        assert _owner_candidates(alias) == _owner_candidates(canonical), alias


def test_replacements_and_unsupported_routes_are_explicit() -> None:
    entries = dict(_selector_entries())
    for route_key, disposition in (
        ("z_image_turbo", "replaced_by_astrid_d3"),
        ("image-upscale", "replaced_by_astrid_d4"),
        ("image_upscale", "replaced_by_astrid_d4"),
    ):
        entry = entries[route_key]
        assert entry.disposition == disposition
        resolved = resolve_task_route(
            task_id=f"fi5-{route_key}",
            task_type=route_key,
            params={"prompt": "finite matrix"},
            backend=WorkerBackend.VIBECOMFY,
        )
        assert resolved.should_use_vibecomfy is False
        assert resolved.fail_closed_reason
        assert "Astrid" in resolved.fail_closed_reason

    for route_key, entry in entries.items():
        if entry.support_state is RouteSupportState.VIBECOMFY_UNSUPPORTED:
            assert entry.disposition, route_key
            assert entry.blocking_reason, route_key

    assert RETIRED_ROUTE_DISPOSITIONS["legacy_custom_task"].startswith("retired:")
    unknown = resolve_task_route(
        task_id="fi5-unknown",
        task_type="fi5_unclassified_route",
        params={},
        backend=WorkerBackend.VIBECOMFY,
    )
    assert unknown.support_state is RouteSupportState.VIBECOMFY_UNSUPPORTED
    assert unknown.fail_closed_reason
    assert unknown.should_use_vibecomfy is False


def test_supported_entrypoint_has_no_second_authority() -> None:
    for path in (ENTRYPOINT, SERVER):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = _imports(tree)
        assert not {
            name
            for name in imports
            if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES)
        }, path

    server_source = SERVER.read_text(encoding="utf-8")
    assert "while True" not in server_source
    assert all(symbol not in server_source for symbol in FORBIDDEN_SERVER_SYMBOLS)
    assert "task authority remains external" in server_source


def test_preserved_authority_consumers_and_substrate_remain_present() -> None:
    for relative_path in (
        "source/task_handlers/tasks/task_conversion.py",
        "source/task_handlers/tasks/task_registry.py",
        "source/task_handlers/travel/orchestrator.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "uni3c_start_percent" in source
        assert "uni3c_end_percent" in source

    assert "video_enhance" in TASK_TYPE_CATALOG
    for relative_path in (
        "docs/sprint-12-route-inventory.md",
        "docs/sprint-12-route-support.md",
        "source/runtime/vibecomfy_profile.py",
        "debug/diagnostics.py",
    ):
        assert (ROOT / relative_path).is_file(), relative_path

    server_module = import_module("source.runtime.worker.server")
    assert callable(server_module.ensure_wan2gp_on_path)
