"""Bounded F-I5 route ownership and authority convergence checks."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

import pytest

from source.task_handlers.tasks.task_types import TASK_TYPE_CATALOG
from source.task_handlers.tasks.task_execution import execute_resolved_direct_task
from source.task_handlers.tasks.template_routing import (
    DIRECT_ROUTE_ALIASES,
    RETIRED_ASTRID_DIRECT_ROUTE_KEYS,
    RETIRED_ASTRID_DIMENSIONAL_ROUTE_KEYS,
    RETIRED_ASTRID_DIMENSIONAL_ROUTE_PREFIXES,
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
    finite_routes = (
        selector_routes | catalog_only_routes
    ) - set(RETIRED_ASTRID_DIRECT_ROUTE_KEYS)

    assert finite_routes
    for route_key in sorted(finite_routes):
        assert len(_owner_candidates(route_key)) == 1, route_key

    for alias, canonical in DIRECT_ROUTE_ALIASES.items():
        assert canonical in finite_routes, alias
        assert _owner_candidates(alias) == _owner_candidates(canonical), alias


def test_replacements_and_unsupported_routes_are_explicit() -> None:
    entries = dict(_selector_entries())
    assert not set(RETIRED_ASTRID_DIRECT_ROUTE_KEYS) & set(entries)
    assert not SECTION3A_ROUTE_SUPPORT_MAP
    for route_key in sorted(RETIRED_ASTRID_DIRECT_ROUTE_KEYS):
        for backend in (WorkerBackend.VIBECOMFY, WorkerBackend.WGP):
            resolved = resolve_task_route(
                task_id=f"fi5-{route_key}-{backend.value}",
                task_type=route_key,
                params={"prompt": "finite matrix"},
                backend=backend,
            )
            assert resolved.route_key in RETIRED_ASTRID_DIRECT_ROUTE_KEYS
            assert resolved.support_state is RouteSupportState.VIBECOMFY_UNSUPPORTED
            assert resolved.should_use_vibecomfy is False
            assert resolved.fail_closed_reason
            assert "retired after its typed Astrid replacement" in resolved.fail_closed_reason

    for route_key in sorted(RETIRED_ASTRID_DIMENSIONAL_ROUTE_KEYS) + [
        f"{RETIRED_ASTRID_DIMENSIONAL_ROUTE_PREFIXES[0]}model-ltx2__guidance-none"
    ]:
        resolved = resolve_task_route(
            task_id=f"fi5-dimensional-{route_key}",
            task_type=route_key,
            params={"prompt": "finite matrix"},
            backend=WorkerBackend.WGP,
        )
        assert resolved.support_state is RouteSupportState.VIBECOMFY_UNSUPPORTED
        assert resolved.should_use_vibecomfy is False
        assert resolved.fail_closed_reason
        assert "retired after its typed Astrid replacement" in resolved.fail_closed_reason

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


@pytest.mark.parametrize("backend", [WorkerBackend.WGP, WorkerBackend.VIBECOMFY])
@pytest.mark.parametrize(
    "route_key",
    [
        "travel_orchestrator",
        "travel_segment__model-ltx2__guidance-none",
        "join_clips_segment__model-wan22_vace__guidance-vace",
    ],
)
def test_dimensional_tombstones_stop_execution_before_either_backend(backend, route_key) -> None:
    resolved = resolve_task_route(
        task_id=f"fi5-execution-{backend.value}-{route_key}",
        task_type=route_key,
        params={"prompt": "execution boundary"},
        backend=backend,
    )
    calls = {"wgp_builder": 0, "queue_submit": 0, "vibe_handler": 0}

    def build_wgp_generation_task():
        calls["wgp_builder"] += 1
        return object()

    def vibe_handler(*_args):
        calls["vibe_handler"] += 1
        return True, "should-not-run"

    class Queue:
        def submit_task(self, _task):
            calls["queue_submit"] += 1

        def get_task_status(self, _task_id):
            raise AssertionError("retired dimensional task reached queue polling")

    ok, output = execute_resolved_direct_task(
        resolved=resolved,
        context={"task_queue": Queue(), "main_output_dir_base": ROOT},
        build_wgp_generation_task=build_wgp_generation_task,
        vibecomfy_handler=vibe_handler,
    )

    assert ok is False
    assert output is not None
    assert calls == {"wgp_builder": 0, "queue_submit": 0, "vibe_handler": 0}


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
    assert "Runtime remains the task authority" in server_source


def test_preserved_authority_consumers_and_substrate_remain_present() -> None:
    for relative_path in (
        "source/task_handlers/tasks/task_conversion.py",
        "source/task_handlers/tasks/task_registry.py",
        "source/task_handlers/travel/orchestrator.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "uni3c_start_percent" in source
        assert "uni3c_end_percent" in source

    assert "video_enhance" not in TASK_TYPE_CATALOG
    assert "video_enhance" in RETIRED_ASTRID_DIRECT_ROUTE_KEYS
    for relative_path in (
        "docs/sprint-12-route-inventory.md",
        "docs/sprint-12-route-support.md",
        "source/runtime/vibecomfy_profile.py",
        "debug/diagnostics.py",
    ):
        assert (ROOT / relative_path).is_file(), relative_path

    server_module = import_module("source.runtime.worker.server")
    assert callable(server_module.launch_generic_pack_host)
