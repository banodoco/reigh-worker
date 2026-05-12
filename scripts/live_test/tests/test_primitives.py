from __future__ import annotations

import copy
import json
import sys
import types
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_test.completion_poller import TaskResult, poll_until_complete
from scripts.live_test.heartbeat_waiter import WorkerReadyTimeoutError, wait_until_ready
from scripts.live_test.inspect import build_status_bundle, render_status_bundle
from scripts.live_test.launch_command import build_direct_worker_command, build_run_worker_command
from scripts.live_test.matrix import MATRIX, MatrixCase, build_matrix, queue_matrix, render_case_payload, run_matrix
from scripts.live_test import main as live_test_main
from scripts.live_test.preflight import (
    LIVE_TEST_PROJECT_NAME,
    UnexpectedUserWorkError,
    assert_user_queue_clean,
    close_stale_live_test_tasks,
    ensure_live_test_route_selectors,
    ensure_user_cloud_generation_enabled,
    get_or_create_live_test_project,
)
from scripts.live_test.report import all_results_passed, write_report
from scripts.live_test.safety_gate import UnsafeTakeoverError, assert_safe_to_take_over
from scripts.live_test.ssh_bootstrap import (
    KILL_COMMAND,
    WorkerProcessInfo,
    capture_current_worker_cmdline,
    clone_and_install_vibecomfy,
    kill_supervisor_and_worker,
    open_session,
)
from scripts.live_test.task_spoofer import insert_spoof_task
from scripts.live_test.terminate_guard import guarded_terminate, prune_stale_live_test_pods
from scripts.live_test.token_resolver import TokenResolutionError, resolve_token_to_user_id
from scripts.live_test import variant_fresh
from scripts.live_test.variant_fresh import run as run_variant_fresh
from scripts.live_test.variant_fresh import _build_matrix_cases as build_fresh_matrix_cases
from scripts.live_test.variant_fresh import _build_worker_env as build_fresh_worker_env
from scripts.live_test.variant_update import (
    FRESH_LIVE_TEST_WORKDIR,
    UPDATE_WORKDIR,
    _remote_checkout_and_sync,
    _resolve_existing_worker_id,
    _resolve_update_workdir,
    _spawn_takeover_pod,
    _worker_row_exists,
    run as run_variant_update,
)


def _iso_now(offset_seconds: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


def _lookup(row: dict, key: str):
    if "->>" in key:
        base, nested = key.split("->>", 1)
        current = row.get(base)
        return current.get(nested) if isinstance(current, dict) else None
    current = row
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


class FakeResult:
    def __init__(self, data):
        self.data = data


class SequenceResponder:
    def __init__(self, responses):
        self._responses = deque(copy.deepcopy(list(responses)))

    def __call__(self, _query):
        if len(self._responses) > 1:
            return self._responses.popleft()
        return copy.deepcopy(self._responses[0]) if self._responses else []


class FakeQuery:
    def __init__(self, supabase: "FakeSupabase", table_name: str):
        self.supabase = supabase
        self.table_name = table_name
        self.filters = []
        self.order_key = None
        self.order_desc = False
        self.insert_payload = None
        self.update_payload = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append(lambda row: _lookup(row, key) == value)
        return self

    def in_(self, key, values):
        allowed = set(values)
        self.filters.append(lambda row: _lookup(row, key) in allowed)
        return self

    def gte(self, key, value):
        self.filters.append(lambda row: (_lookup(row, key) or "") >= value)
        return self

    def order(self, key, desc=False):
        self.order_key = key
        self.order_desc = desc
        return self

    def insert(self, payload):
        self.insert_payload = copy.deepcopy(payload)
        return self

    def update(self, payload):
        self.update_payload = copy.deepcopy(payload)
        return self

    def single(self):
        return self

    def execute(self):
        if self.update_payload is not None:
            rows = self.supabase.tables.setdefault(self.table_name, [])
            updated = []
            for row in rows:
                if all(predicate(row) for predicate in self.filters):
                    row.update(copy.deepcopy(self.update_payload))
                    updated.append(copy.deepcopy(row))
            self.supabase.updated.setdefault(self.table_name, []).extend(updated)
            return FakeResult(updated)

        if self.insert_payload is not None:
            row = copy.deepcopy(self.insert_payload)
            row.setdefault("id", f"{self.table_name}-row-{len(self.supabase.tables.setdefault(self.table_name, [])) + 1}")
            self.supabase.tables.setdefault(self.table_name, []).append(copy.deepcopy(row))
            self.supabase.inserted.setdefault(self.table_name, []).append(copy.deepcopy(row))
            return FakeResult([row])

        source = self.supabase.sources.get(self.table_name, self.supabase.tables.get(self.table_name, []))
        if callable(source):
            rows = source(self)
        else:
            rows = copy.deepcopy(source)

        filtered = []
        for row in rows:
            if all(predicate(row) for predicate in self.filters):
                filtered.append(copy.deepcopy(row))

        if self.order_key is not None:
            filtered.sort(key=lambda row: _lookup(row, self.order_key) or "", reverse=self.order_desc)

        return FakeResult(filtered)


class FakeSupabase:
    def __init__(self, *, tables=None, sources=None):
        self.tables = copy.deepcopy(tables or {})
        self.sources = dict(sources or {})
        self.inserted = {}
        self.updated = {}

    def table(self, table_name: str) -> FakeQuery:
        return FakeQuery(self, table_name)


class FakeDB:
    def __init__(self, *, tables=None, sources=None):
        self.supabase = FakeSupabase(tables=tables, sources=sources)


class ScriptedSSH:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []

    def execute_command(self, command, timeout=600):
        self.commands.append((command, timeout))
        if not self.responses:
            return 0, "", ""
        matcher, response = self.responses.pop(0)
        if matcher is not None:
            assert matcher in command
        return response


def test_token_resolver_returns_user_id():
    db = FakeDB(tables={"user_api_tokens": [{"user_id": "user-123", "token": "secret"}]})
    assert resolve_token_to_user_id(db, "secret") == "user-123"


def test_token_resolver_raises_on_missing():
    db = FakeDB(tables={"user_api_tokens": []})
    with pytest.raises(TokenResolutionError):
        resolve_token_to_user_id(db, "missing")


def test_preflight_raises_on_stray_user_work():
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "status": "Queued",
                    "params": {"live_test": False},
                    "projects": {"user_id": "user-1"},
                }
            ]
        }
    )
    with pytest.raises(UnexpectedUserWorkError):
        assert_user_queue_clean(db, "user-1")


def test_preflight_passes_when_clean():
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "status": "Queued",
                    "params": {"live_test": True},
                    "projects": {"user_id": "user-1"},
                }
            ],
            "projects": [{"id": "project-1", "user_id": "user-1", "name": LIVE_TEST_PROJECT_NAME}],
        }
    )
    assert_user_queue_clean(db, "user-1")
    assert get_or_create_live_test_project(db, "user-1") == "project-1"


def test_preflight_closes_stale_live_test_tasks_without_touching_user_work():
    db = FakeDB(
        tables={
            "tasks": [
                {"id": "live-queued", "status": "Queued", "params": {"live_test": True}, "projects": {"user_id": "user-1"}},
                {"id": "live-active", "status": "In Progress", "params": {"live_test": True}, "projects": {"user_id": "user-1"}},
                {"id": "user-task", "status": "Queued", "params": {"live_test": False}, "projects": {"user_id": "user-1"}},
                {"id": "other-user", "status": "Queued", "params": {"live_test": True}, "projects": {"user_id": "user-2"}},
            ]
        }
    )

    assert close_stale_live_test_tasks(db, "user-1") == 2
    by_id = {row["id"]: row for row in db.supabase.tables["tasks"]}
    assert by_id["live-queued"]["status"] == "Failed"
    assert by_id["live-active"]["status"] == "Failed"
    assert by_id["live-queued"]["error_message"] == "closed stale live-test task before new live-test run"
    assert by_id["user-task"]["status"] == "Queued"
    assert by_id["other-user"]["status"] == "Queued"


def test_preflight_enables_cloud_generation_for_live_test_user():
    db = FakeDB(
        tables={
            "users": [
                {
                    "id": "user-1",
                    "settings": {
                        "ui": {
                            "generationMethods": {
                                "inCloud": False,
                                "onComputer": True,
                            },
                            "theme": {"darkMode": True},
                        }
                    },
                }
            ]
        }
    )

    assert ensure_user_cloud_generation_enabled(db, "user-1") is True
    user = db.supabase.tables["users"][0]
    assert user["settings"]["ui"]["generationMethods"]["inCloud"] is True
    assert user["settings"]["ui"]["generationMethods"]["onComputer"] is True
    assert user["settings"]["ui"]["theme"]["darkMode"] is True
    assert ensure_user_cloud_generation_enabled(db, "user-1") is False


def test_task_spoofer_stamps_live_test_and_queued_status():
    db = FakeDB(tables={"tasks": []})
    fixture_id = str(uuid.uuid4())
    task_id = insert_spoof_task(
        db,
        "project-1",
        "qwen_image",
        {"prompt": "overridden"},
        fixture_payload={"id": fixture_id, "notes": "strip me", "params": {"prompt": "base prompt"}},
    )
    inserted = db.supabase.inserted["tasks"][0]
    assert task_id == inserted["id"]
    assert inserted["id"] != fixture_id
    assert inserted["status"] == "Queued"
    assert inserted["params"]["prompt"] == "overridden"
    assert inserted["params"]["live_test"] is True
    assert "notes" not in inserted


def test_task_spoofer_promotes_route_contract_to_task_columns():
    db = FakeDB(tables={"tasks": []})
    insert_spoof_task(
        db,
        "project-1",
        "z_image_turbo",
        {
            "route_contract": {
                "selector_namespace": "production",
                "route_key": "z_image_turbo",
                "selected_backend": "vibecomfy",
                "selector_version": 10,
                "support_state": "vibecomfy_supported",
                "selected_profile": "default",
                "selected_template_id": "image/z_image",
                "worker_contract_version": 1,
                "route_selection_snapshot": {"route_key": "z_image_turbo"},
            }
        },
        fixture_payload={"params": {}},
    )

    inserted = db.supabase.inserted["tasks"][0]
    assert inserted["selector_namespace"] == "production"
    assert inserted["route_key"] == "z_image_turbo"
    assert inserted["selected_backend"] == "vibecomfy"
    assert inserted["selector_version"] == 10
    assert inserted["support_state"] == "vibecomfy_supported"
    assert inserted["selected_profile"] == "default"
    assert inserted["selected_template_id"] == "image/z_image"
    assert inserted["worker_contract_version"] == 1
    assert inserted["route_selection_snapshot"] == {"route_key": "z_image_turbo"}


def test_completion_poller_returns_on_complete(monkeypatch: pytest.MonkeyPatch):
    task_rows = SequenceResponder(
        [
            [{"id": "task-1", "project_id": "project-1", "task_type": "qwen_image", "status": "Queued", "created_at": _iso_now(-10)}],
            [{"id": "task-1", "project_id": "project-1", "task_type": "qwen_image", "status": "Complete", "created_at": _iso_now(-10), "output_location": "https://out.example/result.png"}],
        ]
    )
    generations = [
        {
            "id": "gen-1",
            "project_id": "project-1",
            "created_at": _iso_now(-5),
            "tasks": ["task-1"],
            "params": {},
            "location": "https://out.example/result.png",
        }
    ]
    db = FakeDB(sources={"tasks": task_rows}, tables={"generations": generations})
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)
    result = poll_until_complete(db, "task-1", "project-1", timeout_sec=2, interval_sec=0, case_name="case", task_type="qwen_image")
    assert result.final_status == "Complete"
    assert result.generation_ids == ["gen-1"]
    assert result.output_location == "https://out.example/result.png"
    assert result.error_summary is None


def test_completion_poller_prints_long_running_progress(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    task_rows = SequenceResponder(
        [
            [
                {
                    "id": "task-1",
                    "project_id": "project-1",
                    "task_type": "wan_animate",
                    "status": "In Progress",
                    "created_at": _iso_now(-10),
                    "worker_id": "worker-1",
                }
            ],
            [
                {
                    "id": "task-1",
                    "project_id": "project-1",
                    "task_type": "wan_animate",
                    "status": "Complete",
                    "created_at": _iso_now(-10),
                    "worker_id": "worker-1",
                    "output_location": "https://out.example/result.mp4",
                }
            ],
        ]
    )
    db = FakeDB(
        sources={"tasks": task_rows},
        tables={
            "workers": [
                {
                    "id": "worker-1",
                    "status": "active",
                    "last_heartbeat": "2026-05-08T16:00:00Z",
                    "metadata": {},
                }
            ],
            "generations": [
                {
                    "id": "gen-1",
                    "project_id": "project-1",
                    "created_at": _iso_now(-5),
                    "tasks": ["task-1"],
                    "params": {},
                    "location": "https://out.example/result.mp4",
                }
            ],
        },
    )
    monotonic_values = iter([0.0, 0.0, 61.0, 62.0, 63.0])
    monkeypatch.setattr("scripts.live_test.completion_poller.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)

    result = poll_until_complete(
        db,
        "task-1",
        "project-1",
        timeout_sec=120,
        interval_sec=0,
        progress_interval_sec=60,
        case_name="animate_character",
        task_type="wan_animate",
    )

    assert result.final_status == "Complete"
    output = capsys.readouterr().out
    assert "animate_character still running after 61s/120s" in output
    assert "task=task-1 status=In Progress" in output
    assert "worker=worker-1 status=active" in output
    assert "last_heartbeat=2026-05-08T16:00:00Z" in output


def test_completion_poller_links_orchestrator_child_generations(monkeypatch: pytest.MonkeyPatch):
    parent_created_at = _iso_now(-20)
    child_created_at = _iso_now(-10)
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "parent-1",
                    "project_id": "project-1",
                    "task_type": "travel_orchestrator",
                    "status": "Complete",
                    "created_at": parent_created_at,
                },
                {
                    "id": "child-1",
                    "project_id": "project-1",
                    "task_type": "travel_segment",
                    "status": "Complete",
                    "created_at": child_created_at,
                    "params": {"orchestrator_task_id_ref": "parent-1"},
                },
            ],
            "generations": [
                {
                    "id": "gen-child-1",
                    "project_id": "project-1",
                    "created_at": _iso_now(-5),
                    "tasks": ["child-1"],
                    "params": {},
                    "location": "https://out.example/segment.mp4",
                }
            ],
        }
    )
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)
    result = poll_until_complete(
        db,
        "parent-1",
        "project-1",
        timeout_sec=1,
        interval_sec=0,
        task_type="travel_orchestrator",
    )
    assert result.final_status == "Complete"
    assert result.generation_ids == ["gen-child-1"]
    assert result.output_location == "https://out.example/segment.mp4"
    assert result.error_summary is None


def test_completion_poller_links_orchestrator_child_generations_from_details(monkeypatch: pytest.MonkeyPatch):
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "join-parent-1",
                    "project_id": "project-1",
                    "task_type": "join_clips_orchestrator",
                    "status": "Complete",
                    "created_at": _iso_now(-20),
                },
                {
                    "id": "join-child-1",
                    "project_id": "project-1",
                    "task_type": "join_clips_segment",
                    "status": "Complete",
                    "created_at": _iso_now(-10),
                    "params": {"orchestrator_details": {"orchestrator_task_id": "join-parent-1"}},
                },
            ],
            "generations": [
                {
                    "id": "gen-join-1",
                    "project_id": "project-1",
                    "created_at": _iso_now(-5),
                    "tasks": "join-child-1",
                    "params": {},
                    "location": "https://out.example/joined.mp4",
                }
            ],
        }
    )
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)
    result = poll_until_complete(
        db,
        "join-parent-1",
        "project-1",
        timeout_sec=1,
        interval_sec=0,
        task_type="join_clips_orchestrator",
    )
    assert result.generation_ids == ["gen-join-1"]
    assert result.output_location == "https://out.example/joined.mp4"
    assert result.error_summary is None


def test_completion_poller_times_out(monkeypatch: pytest.MonkeyPatch):
    db = FakeDB(
        tables={
            "tasks": [
                {"id": "task-1", "project_id": "project-1", "task_type": "qwen_image", "status": "Queued", "created_at": _iso_now(-10)}
            ],
            "generations": [],
        }
    )
    monotonic_values = iter([0.0, 0.4, 1.2, 1.4, 1.6])
    monkeypatch.setattr("scripts.live_test.completion_poller.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)
    result = poll_until_complete(db, "task-1", "project-1", timeout_sec=1, interval_sec=0, case_name="case", task_type="qwen_image")
    assert result.final_status == "Queued"
    assert "Timed out waiting for task task-1" in (result.error_summary or "")


def test_completion_poller_records_failure(monkeypatch: pytest.MonkeyPatch):
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "project_id": "project-1",
                    "task_type": "qwen_image",
                    "status": "Failed",
                    "error_message": "backend exploded",
                    "created_at": _iso_now(-10),
                }
            ],
            "generations": [],
        }
    )
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)
    result = poll_until_complete(db, "task-1", "project-1", timeout_sec=1, interval_sec=0)
    assert result.final_status == "Failed"
    assert result.error_summary == "backend exploded"


def test_completion_poller_fails_fast_when_worker_errors(monkeypatch: pytest.MonkeyPatch):
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "project_id": "project-1",
                    "task_type": "qwen_image",
                    "status": "Queued",
                    "created_at": _iso_now(-10),
                    "worker_id": None,
                }
            ],
            "workers": [
                {
                    "id": "worker-1",
                    "status": "error",
                    "metadata": {"error_reason": "Pod externally terminated - not found in RunPod"},
                }
            ],
            "generations": [],
        }
    )
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)

    result = poll_until_complete(
        db,
        "task-1",
        "project-1",
        timeout_sec=300,
        interval_sec=0,
        case_name="case",
        task_type="qwen_image",
        worker_id="worker-1",
    )

    assert result.final_status == "Queued"
    assert "Worker worker-1 reached error status" in (result.error_summary or "")


def test_completion_poller_fails_fast_when_worker_terminates(monkeypatch: pytest.MonkeyPatch):
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "project_id": "project-1",
                    "task_type": "qwen_image",
                    "status": "In Progress",
                    "created_at": _iso_now(-10),
                }
            ],
            "workers": [
                {
                    "id": "worker-1",
                    "status": "terminated",
                    "metadata": {"termination_reason": "scale_down_idle (capacity 2 > desired 1)"},
                }
            ],
            "generations": [],
        }
    )
    monkeypatch.setattr("scripts.live_test.completion_poller.time.sleep", lambda _interval: None)

    result = poll_until_complete(
        db,
        "task-1",
        "project-1",
        timeout_sec=300,
        interval_sec=0,
        worker_id="worker-1",
    )

    assert result.final_status == "In Progress"
    assert "Worker worker-1 reached terminated status" in (result.error_summary or "")
    assert "scale_down_idle" in (result.error_summary or "")


def test_heartbeat_waiter_requires_dwell_and_ready_for_tasks(monkeypatch: pytest.MonkeyPatch):
    workers = SequenceResponder(
        [
            [{"id": "worker-1", "last_heartbeat": _iso_now(), "metadata": {"ready_for_tasks": False}}],
            [{"id": "worker-1", "last_heartbeat": _iso_now(), "metadata": {"ready_for_tasks": True}}],
            [{"id": "worker-1", "last_heartbeat": _iso_now(), "metadata": {"ready_for_tasks": True}}],
        ]
    )
    db = FakeDB(sources={"workers": workers})
    monkeypatch.setattr("scripts.live_test.heartbeat_waiter.time.sleep", lambda _interval: None)
    worker = wait_until_ready(db, "worker-1", timeout_sec=1, interval_sec=0, dwell_polls=2)
    assert worker["id"] == "worker-1"


def test_heartbeat_waiter_can_skip_ready_marker_for_queue_driven_backends(monkeypatch: pytest.MonkeyPatch):
    workers = SequenceResponder(
        [
            [{"id": "worker-1", "last_heartbeat": _iso_now(), "metadata": {"ready_for_tasks": False}}],
            [{"id": "worker-1", "last_heartbeat": _iso_now(), "metadata": {"ready_for_tasks": False}}],
        ]
    )
    db = FakeDB(sources={"workers": workers})
    monkeypatch.setattr("scripts.live_test.heartbeat_waiter.time.sleep", lambda _interval: None)

    worker = wait_until_ready(
        db,
        "worker-1",
        timeout_sec=1,
        interval_sec=0,
        dwell_polls=2,
        require_ready_for_tasks=False,
    )

    assert worker["id"] == "worker-1"


def test_heartbeat_waiter_fails_fast_on_terminal_worker_status():
    db = FakeDB(
        tables={
            "workers": [
                {
                    "id": "worker-1",
                    "status": "terminated",
                    "last_heartbeat": None,
                    "metadata": {"termination_reason": "early_termination_over_capacity (2 > 1)"},
                }
            ]
        }
    )

    with pytest.raises(WorkerReadyTimeoutError, match="early_termination_over_capacity"):
        wait_until_ready(db, "worker-1", timeout_sec=30, interval_sec=0)


def test_safety_gate_rejects_fresh_in_progress_for_user():
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "status": "In Progress",
                    "generation_started_at": _iso_now(-10),
                    "projects": {"user_id": "user-1"},
                }
            ]
        }
    )
    with pytest.raises(UnsafeTakeoverError):
        assert_safe_to_take_over(db, "pod-1", "user-1")


def test_safety_gate_rejects_fresh_heartbeat_when_not_allowed():
    db = FakeDB(
        tables={
            "tasks": [],
            "workers": [{"id": "pod-1", "last_heartbeat": _iso_now()}],
        }
    )
    with pytest.raises(UnsafeTakeoverError):
        assert_safe_to_take_over(db, "pod-1", "user-1", allow_fresh_heartbeat=False)


def test_safety_gate_permits_fresh_heartbeat_when_allowed_but_still_rejects_live_pat_work():
    clean_db = FakeDB(
        tables={
            "tasks": [],
            "workers": [{"id": "pod-1", "last_heartbeat": _iso_now()}],
        }
    )
    assert_safe_to_take_over(clean_db, "pod-1", "user-1", allow_fresh_heartbeat=True)

    busy_db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-2",
                    "status": "In Progress",
                    "generation_started_at": _iso_now(-15),
                    "projects": {"user_id": "user-1"},
                }
            ],
            "workers": [{"id": "pod-1", "last_heartbeat": _iso_now()}],
        }
    )
    with pytest.raises(UnsafeTakeoverError):
        assert_safe_to_take_over(busy_db, "pod-1", "user-1", allow_fresh_heartbeat=True)


def test_terminate_guard_respects_env_var(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setenv("REIGH_LIVE_TEST_NO_TERMINATE", "1")
    monkeypatch.setattr("scripts.live_test.terminate_guard.live_test_pkg.terminate_pod", lambda pod_id, api_key: calls.append((pod_id, api_key)))
    assert guarded_terminate("pod-1", "api-key", no_terminate=False) is False
    assert calls == []


def test_terminate_guard_respects_cli_flag(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.delenv("REIGH_LIVE_TEST_NO_TERMINATE", raising=False)
    monkeypatch.setattr("scripts.live_test.terminate_guard.live_test_pkg.terminate_pod", lambda pod_id, api_key: calls.append((pod_id, api_key)))
    assert guarded_terminate("pod-1", "api-key", no_terminate=True) is False
    assert calls == []


def test_terminate_guard_skips_on_exception_path(monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.delenv("REIGH_LIVE_TEST_NO_TERMINATE", raising=False)
    monkeypatch.setattr("scripts.live_test.terminate_guard.live_test_pkg.terminate_pod", lambda pod_id, api_key: calls.append((pod_id, api_key)))
    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("boom")
        finally:
            assert guarded_terminate(None, "api-key", no_terminate=False) is False
    assert calls == []


def test_terminate_guard_treats_missing_pod_as_already_terminated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REIGH_LIVE_TEST_NO_TERMINATE", raising=False)

    def _missing_pod(_pod_id, _api_key):
        raise RuntimeError("QueryError pod not found to terminate")

    monkeypatch.setattr("scripts.live_test.terminate_guard.live_test_pkg.terminate_pod", _missing_pod)
    assert guarded_terminate("pod-1", "api-key", no_terminate=False) is False


def test_prune_stale_live_test_pods_uses_timestamped_names_when_uptime_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("REIGH_LIVE_TEST_SKIP_STALE_POD_CLEANUP", raising=False)
    terminated: list[tuple[str, str]] = []
    pods = [
        SimpleNamespace(
            id="stale-pod",
            name="reigh-live-test-fresh-20260507t190615z",
            uptime_seconds=None,
            created_at=None,
        ),
        SimpleNamespace(
            id="current-pod",
            name="reigh-live-test-fresh-20260509t120000z",
            uptime_seconds=None,
            created_at=None,
        ),
    ]

    async def fake_list_pods(api_key: str, prefix: str):
        assert api_key == "api-key"
        assert prefix == "reigh-live-test-fresh-"
        return pods

    async def fake_terminate(api_key: str, pod_id: str):
        terminated.append((api_key, pod_id))

    result = prune_stale_live_test_pods(
        "api-key",
        max_age_seconds=6 * 60 * 60,
        list_pods_fn=fake_list_pods,
        terminate_fn=fake_terminate,
        now=datetime(2026, 5, 9, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert result.inspected == 2
    assert result.stale == ("stale-pod",)
    assert result.terminated == ("stale-pod",)
    assert result.failed == ()
    assert terminated == [("api-key", "stale-pod")]


def test_prune_stale_live_test_pods_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REIGH_LIVE_TEST_SKIP_STALE_POD_CLEANUP", "1")

    result = prune_stale_live_test_pods("api-key")

    assert result.inspected == 0
    assert result.stale == ()
    assert result.terminated == ()


def test_build_run_worker_command_uses_run_worker_py_and_idle_zero():
    command = build_run_worker_command(
        "/workspace/Reigh-Worker-LiveTest",
        reigh_token="token-1",
        supabase_url="https://supabase.example",
        worker_id="worker-1",
        wgp_profile=3,
        idle_release_minutes=0,
    )
    assert "python run_worker.py" in command
    assert "--idle-release-minutes 0" in command
    assert "--save-logging logs/worker.log" in command
    assert 'UV_PROJECT_ENVIRONMENT="/opt/reigh-worker-live-test-venv"' in command


def test_build_run_worker_command_can_redact_access_token():
    command = build_run_worker_command(
        "/workspace/Reigh-Worker-LiveTest",
        reigh_token="secret-token",
        supabase_url="https://supabase.example",
        worker_id="worker-1",
        wgp_profile=3,
        idle_release_minutes=0,
        redact_secrets=True,
    )
    assert "secret-token" not in command
    assert "<REIGH_LIVE_TEST_TOKEN>" in command


def test_build_run_worker_command_can_use_env_token_without_cli_secret():
    command = build_run_worker_command(
        "/workspace/Reigh-Worker-LiveTest",
        reigh_token=None,
        supabase_url="https://supabase.example",
        worker_id="worker-1",
        wgp_profile=3,
        idle_release_minutes=0,
    )
    assert "--reigh-access-token" not in command
    assert "python run_worker.py" in command
    assert "--worker worker-1" in command


def test_fresh_variant_redacts_noisy_runpod_lifecycle_output():
    text = (
        "raw_response {'env': {'REIGH_ACCESS_TOKEN': 'token-1', "
        "'SUPABASE_SERVICE_ROLE_KEY': 'service-key'}} --reigh-access-token token-2"
    )

    redacted = variant_fresh._redact_sensitive_text(text)

    assert "token-1" not in redacted
    assert "token-2" not in redacted
    assert "service-key" not in redacted
    assert "<redacted>" in redacted


def test_open_session_timeout_includes_latest_pod_status(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("scripts.live_test.ssh_bootstrap.time.sleep", lambda _interval: None)
    monkeypatch.setattr("scripts.live_test.ssh_bootstrap.time.monotonic", iter([0, 0.5, 1.5]).__next__)
    monkeypatch.setattr("scripts.live_test.get_pod_ssh_details", lambda _pod_id, _api_key: None)
    monkeypatch.setattr(
        "scripts.live_test.get_pod_status",
        lambda _pod_id, _api_key: {
            "desired_status": "EXITED",
            "actual_status": "EXITED",
            "ip": None,
            "ports": [],
        },
    )

    with pytest.raises(RuntimeError) as exc:
        open_session("pod-1", "api-key", ssh_wait_timeout=1, poll_interval=1)

    message = str(exc.value)
    assert "desired=EXITED" in message
    assert "actual=EXITED" in message
    assert "ports=none" in message


def test_open_session_connect_timeout_includes_latest_pod_status(monkeypatch: pytest.MonkeyPatch):
    class FailingSSH:
        def __init__(self, **_kwargs):
            pass

        def connect(self):
            raise RuntimeError("connection refused")

    monkeypatch.setattr("scripts.live_test.ssh_bootstrap.time.sleep", lambda _interval: None)
    monkeypatch.setattr(
        "scripts.live_test.ssh_bootstrap.time.monotonic",
        iter([0, 0.1, 0.2, 0.3, 1.4]).__next__,
    )
    monkeypatch.setattr(
        "scripts.live_test.get_pod_ssh_details",
        lambda _pod_id, _api_key: {"ip": "5.6.7.8", "port": 2201, "password": "runpod"},
    )
    monkeypatch.setattr("scripts.live_test.SSHClient", FailingSSH)
    monkeypatch.setattr(
        "scripts.live_test.get_pod_status",
        lambda _pod_id, _api_key: {
            "desired_status": "RUNNING",
            "actual_status": "RUNNING",
            "ip": "5.6.7.8",
            "ports": [{"privatePort": 22, "publicPort": 2201}],
        },
    )

    with pytest.raises(RuntimeError) as exc:
        open_session("pod-1", "api-key", ssh_wait_timeout=1, poll_interval=1)

    message = str(exc.value)
    assert "desired=RUNNING" in message
    assert "actual=RUNNING" in message
    assert "ip=5.6.7.8" in message
    assert "ports=22->2201" in message
    assert "connection refused" in message


def test_inspect_bundle_resolves_task_worker_pod_and_heartbeat_age(monkeypatch: pytest.MonkeyPatch):
    now = datetime(2026, 5, 8, 16, 2, tzinfo=timezone.utc)
    db = FakeDB(
        tables={
            "tasks": [
                {
                    "id": "task-1",
                    "task_type": "z_image_turbo",
                    "status": "Complete",
                    "worker_id": "worker-1",
                    "attempts": 1,
                    "output_location": "https://out.example/result.png",
                    "error_message": None,
                    "created_at": "2026-05-08T16:00:00Z",
                    "params": {
                        "route_contract": {
                            "route_key": "z_image_turbo",
                            "selected_backend": "vibecomfy",
                            "selected_template_id": "image/z_image",
                        }
                    },
                }
            ],
            "workers": [
                {
                    "id": "worker-1",
                    "status": "active",
                    "last_heartbeat": "2026-05-08T16:01:30Z",
                    "created_at": "2026-05-08T15:55:00Z",
                    "metadata": {
                        "runpod_id": "pod-1",
                        "ready_for_tasks": True,
                        "worker_backend": "vibecomfy",
                        "worker_pool": "gpu-vibecomfy-live",
                    },
                }
            ],
        }
    )
    monkeypatch.setattr(
        "scripts.live_test.get_pod_status",
        lambda _pod_id, _api_key: {
            "desired_status": "RUNNING",
            "actual_status": "RUNNING",
            "ip": "1.2.3.4",
            "ports": [{"privatePort": 22, "publicPort": 31022}],
        },
    )
    monkeypatch.setattr(
        "scripts.live_test.get_pod_ssh_details",
        lambda _pod_id, _api_key: {"ip": "1.2.3.4", "port": 31022},
    )

    bundle = build_status_bundle(
        db,
        task_id="task-1",
        api_key="api-key",
        include_ssh=False,
        now=now,
    )

    assert bundle["resolved"] == {"task_id": "task-1", "worker_id": "worker-1", "pod_id": "pod-1"}
    assert bundle["worker"]["heartbeat_age_sec"] == 30
    assert bundle["runpod"]["desired_status"] == "RUNNING"
    assert bundle["runpod"]["ports"] == "22->31022"
    rendered = render_status_bundle(bundle)
    assert "ids: task=task-1 worker=worker-1 pod=pod-1" in rendered
    assert "worker: id=worker-1 status=active" in rendered
    assert "task_output: https://out.example/result.png" in rendered
    assert "task_route: route=z_image_turbo backend=vibecomfy template=image/z_image" in rendered


def test_inspect_bundle_uses_pod_worker_lookup_and_records_ssh_hints(monkeypatch: pytest.MonkeyPatch):
    db = FakeDB(
        tables={
            "tasks": [{"id": "task-1", "status": "In Progress", "worker_id": None, "params": {}}],
            "workers": [
                {
                    "id": "worker-from-pod",
                    "status": "active",
                    "last_heartbeat": "2026-05-08T16:00:00Z",
                    "metadata": {"runpod_id": "pod-1", "worker_backend": "vibecomfy"},
                }
            ],
        }
    )

    class InspectSSH:
        def __init__(self):
            self.commands = []

        def execute_command(self, command, timeout=600):
            self.commands.append((command, timeout))
            if "logs/startup.log" in command:
                return 0, "=== startup.log ===\nbooted\n", ""
            if "logs/worker.log" in command:
                return 0, "=== worker.log ===\nTask task-1 running\n", ""
            if "vibecomfy_runs" in command:
                return 0, "=== vibecomfy artifacts: /workspace/Reigh-Worker-LiveTest/outputs/vibecomfy_runs/task-1 ===\noutput/result.png\nmetadata.json\n", ""
            return 0, "", ""

        def close(self):
            pass

    monkeypatch.setattr(
        "scripts.live_test.get_pod_status",
        lambda _pod_id, _api_key: {"desired_status": "RUNNING", "actual_status": "RUNNING", "ports": []},
    )
    monkeypatch.setattr(
        "scripts.live_test.get_pod_ssh_details",
        lambda _pod_id, _api_key: {"ip": "1.2.3.4", "port": 31022},
    )
    monkeypatch.setattr("scripts.live_test.inspect.open_session", lambda *_args, **_kwargs: InspectSSH())

    bundle = build_status_bundle(db, task_id="task-1", pod_id="pod-1", api_key="api-key", log_lines=10)

    assert bundle["resolved"]["worker_id"] == "worker-from-pod"
    assert bundle["ssh"]["available"] is True
    assert "Task task-1 running" in bundle["ssh"]["log_tail"]
    assert "metadata.json" in bundle["ssh"]["vibecomfy_hints"]
    rendered = render_status_bundle(bundle)
    assert "vibecomfy_hints:" in rendered
    assert "worker_log_tail:" in rendered


def test_build_direct_worker_command_roundtrips_template_cmdline():
    command = build_direct_worker_command(
        "/workspace/Reigh-Worker",
        cli_args=["python", "worker.py", "--task-id", "task-1", "--gpu-id", "0"],
    )
    assert command.startswith("cd /workspace/Reigh-Worker && nohup python worker.py")
    assert "> logs/startup.log 2>&1 &" in command


def test_capture_cmdline_detects_supervisor_family_from_run_worker_ps():
    ssh = ScriptedSSH(
        [
            (
                "ps -eo pid=,args=",
                (
                    0,
                    "123 python run_worker.py --worker worker-1\n124 python worker.py --task-id task-1\n",
                    "",
                ),
            )
        ]
    )
    info = capture_current_worker_cmdline(ssh)
    assert info == WorkerProcessInfo(
        family="supervisor",
        cmdline=["python", "run_worker.py", "--worker", "worker-1"],
        pid=123,
    )


def test_capture_cmdline_detects_direct_family_from_template_ps():
    ssh = ScriptedSSH(
        [
            (
                "ps -eo pid=,args=",
                (0, "222 python worker.py --task-id task-1 --gpu-id 0\n", ""),
            )
        ]
    )
    info = capture_current_worker_cmdline(ssh)
    assert info == WorkerProcessInfo(
        family="direct",
        cmdline=["python", "worker.py", "--task-id", "task-1", "--gpu-id", "0"],
        pid=222,
    )


def test_capture_cmdline_detects_direct_family_from_absolute_worker_path():
    ssh = ScriptedSSH(
        [
            (
                "ps -eo pid=,args=",
                (
                    0,
                    "333 /workspace/Reigh-Worker-LiveTest/.venv/bin/python3 -u "
                    "/workspace/Reigh-Worker-LiveTest/worker.py --worker pod-1\n",
                    "",
                ),
            )
        ]
    )
    info = capture_current_worker_cmdline(ssh)
    assert info == WorkerProcessInfo(
        family="direct",
        cmdline=[
            "/workspace/Reigh-Worker-LiveTest/.venv/bin/python3",
            "-u",
            "/workspace/Reigh-Worker-LiveTest/worker.py",
            "--worker",
            "pod-1",
        ],
        pid=333,
    )


def test_kill_supervisor_and_worker_patterns_cover_both_families(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("scripts.live_test.ssh_bootstrap.time.sleep", lambda _interval: None)
    ssh = ScriptedSSH(
        [
            (None, (0, "", "")),
            ("ps -eo pid=,args=", (0, "123 python run_worker.py\n", "")),
            ("ps -eo pid=,args=", (0, "", "")),
        ]
    )
    kill_supervisor_and_worker(ssh)
    assert ssh.commands[0][0] == KILL_COMMAND
    assert "run_worker[.]py" in ssh.commands[0][0]
    assert "python worker[.]py" in ssh.commands[0][0]
    assert "python[^ ]* .*worker[.]py" in ssh.commands[0][0]
    assert "source[.]runtime[.]worker" in ssh.commands[0][0]
    assert "awk" in ssh.commands[1][0]
    assert "python[^ ]* .*worker[.]py" in ssh.commands[1][0]
    assert "source[.]runtime[.]worker" in ssh.commands[1][0]


def test_matrix_contains_route_specific_z_image_turbo_case():
    assert [case.name for case in MATRIX].count("z_image_turbo") == 1
    assert [case.name for case in MATRIX].count("z_image_turbo_i2i") == 1
    z_image_case = next(case for case in MATRIX if case.name == "z_image_turbo")
    z_image_i2i_case = next(case for case in MATRIX if case.name == "z_image_turbo_i2i")
    assert z_image_case.task_type == "z_image_turbo"
    assert z_image_case.route_key == "z_image_turbo"
    assert z_image_case.support_state == "vibecomfy_supported"
    assert z_image_case.selected_template_id == "image/z_image"
    assert z_image_i2i_case.task_type == "z_image_turbo_i2i"
    assert z_image_i2i_case.route_key == "z_image_turbo_i2i"
    assert z_image_i2i_case.support_state == "vibecomfy_supported"
    assert z_image_i2i_case.selected_template_id == "image/z_image_img2img"


def test_route_specific_matrix_stamps_selector_contract():
    cases = build_matrix(
        selected_backend="vibecomfy",
        selector_namespace="canary-a",
        selector_version="2026-05-06",
        worker_contract_version=1,
        selected_profile="z-image-default",
        route_keys=["z_image_turbo"],
    )

    assert [case.name for case in cases] == ["z_image_turbo"]
    payload = render_case_payload(cases[0], project_id="project-1", unique_suffix="abc123")
    params = payload["params"]
    contract = params["route_contract"]
    snapshot = contract["route_selection_snapshot"]
    assert payload["task_type"] == "z_image_turbo"
    assert contract["route_key"] == "z_image_turbo"
    assert contract["selected_backend"] == "vibecomfy"
    assert contract["selector_namespace"] == "canary-a"
    assert contract["selector_version"] == "2026-05-06"
    assert contract["worker_contract_version"] == 1
    assert contract["selected_profile"] == "z-image-default"
    assert contract["support_state"] == "vibecomfy_supported"
    assert contract["selected_template_id"] == "image/z_image"
    assert snapshot["live_test_run_id"] == "live-test-z_image_turbo-abc123"
    assert params["live_test"] is True


def test_vibecomfy_live_test_defaults_to_isolated_selector_namespace(monkeypatch):
    monkeypatch.setattr(live_test_main, "_live_test_selector_namespace", lambda: "livet-20260507215500")
    parser = live_test_main.build_parser()
    args = live_test_main._finalize_args(
        parser.parse_args(["--variant", "update", "--pod-id", "pod-1", "--backend", "vibecomfy"]),
        parser,
    )

    assert args.selector_namespace == "livet-20260507215500"
    assert args.no_terminate is True


def test_wgp_rollback_keeps_production_selector_namespace(monkeypatch):
    monkeypatch.setattr(live_test_main, "_live_test_selector_namespace", lambda: "livet-20260507215500")
    parser = live_test_main.build_parser()
    args = live_test_main._finalize_args(
        parser.parse_args(
            ["--variant", "update", "--pod-id", "pod-1", "--backend", "vibecomfy", "--wgp-rollback"]
        ),
        parser,
    )

    assert args.backend == "wgp"
    assert args.selector_namespace == "production"


def test_ensure_live_test_route_selectors_clones_production_rows():
    db = FakeDB(
        tables={
            "route_backend_selectors": [
                {
                    "selector_namespace": "production",
                    "route_key": "z_image_turbo_i2i",
                    "selected_backend": "vibecomfy",
                    "selector_version": 3,
                    "enabled": True,
                    "expires_at": None,
                    "min_worker_version": None,
                    "reason": "production",
                    "metadata": {"template": "image/z_image_img2img"},
                }
            ]
        }
    )

    created = ensure_live_test_route_selectors(
        db,
        "livet-20260507215500",
        ["z_image_turbo_i2i", "z_image_turbo_i2i"],
        backend="vibecomfy",
    )

    assert created == 1
    inserted = db.supabase.inserted["route_backend_selectors"][0]
    assert inserted["selector_namespace"] == "livet-20260507215500"
    assert inserted["route_key"] == "z_image_turbo_i2i"
    assert inserted["selected_backend"] == "vibecomfy"
    assert inserted["selector_version"] == 3
    assert inserted["metadata"]["live_test"] is True


def test_ensure_live_test_route_selectors_fails_when_production_selector_missing():
    db = FakeDB(tables={"route_backend_selectors": []})

    with pytest.raises(RuntimeError, match="production route selectors are missing"):
        ensure_live_test_route_selectors(
            db,
            "livet-20260507215500",
            ["missing_route"],
            backend="vibecomfy",
        )


def test_ensure_live_test_route_selectors_synthesizes_matrix_fallback_rows():
    db = FakeDB(tables={"route_backend_selectors": []})

    created = ensure_live_test_route_selectors(
        db,
        "livet-20260507215500",
        ["branch_only_route"],
        backend="vibecomfy",
        fallback_selectors={
            "branch_only_route": {
                "selected_backend": "vibecomfy",
                "selector_version": None,
                "support_state": "vibecomfy_supported",
                "selected_template_id": "image/qwen_image_2512",
            }
        },
    )

    assert created == 2
    inserted = db.supabase.inserted["route_backend_selectors"][0]
    assert inserted["selector_namespace"] == "livet-20260507215500"
    assert inserted["route_key"] == "branch_only_route"
    assert inserted["selected_backend"] == "vibecomfy"
    assert inserted["selector_version"] == 1
    assert inserted["metadata"]["source_selector_namespace"] == "matrix"
    assert inserted["metadata"]["support_state"] == "vibecomfy_supported"
    assert inserted["metadata"]["selected_template_id"] == "image/qwen_image_2512"
    capability = db.supabase.inserted["route_backend_capabilities"][0]
    assert capability["backend"] == "vibecomfy"
    assert capability["route_key"] == "branch_only_route"
    assert capability["supports_route"] is True
    assert capability["metadata"]["source"] == "matrix"
    assert capability["metadata"]["selected_template_id"] == "image/qwen_image_2512"


def test_ensure_live_test_route_selectors_updates_stale_fallback_capability_rows():
    db = FakeDB(
        tables={
            "route_backend_selectors": [],
            "route_backend_capabilities": [
                {
                    "id": "cap-1",
                    "backend": "vibecomfy",
                    "route_key": "branch_only_route",
                    "supports_route": False,
                    "supports_missing_selector": False,
                    "enabled": True,
                    "capability_version": 9,
                    "metadata": {"support_state": "vibecomfy_unsupported"},
                }
            ],
        }
    )

    changed = ensure_live_test_route_selectors(
        db,
        "livet-20260507215500",
        ["branch_only_route"],
        backend="vibecomfy",
        fallback_selectors={
            "branch_only_route": {
                "selected_backend": "vibecomfy",
                "selector_version": None,
                "support_state": "vibecomfy_supported",
                "selected_template_id": "image/qwen_image_2512",
            }
        },
    )

    assert changed == 2
    updated = db.supabase.updated["route_backend_capabilities"][0]
    assert updated["supports_route"] is True
    assert updated["metadata"]["support_state"] == "vibecomfy_supported"
    assert updated["metadata"]["selected_template_id"] == "image/qwen_image_2512"


def test_travel_live_matrix_disables_prompt_enhancement_download():
    cases = build_matrix(case_names=["travel_orchestrator_wan2_1seg"])
    payload = render_case_payload(cases[0], project_id="project-1", unique_suffix="abc123")
    details = payload["params"]["orchestrator_details"]

    assert details["enhance_prompt"] is False
    assert details["enhanced_prompts_expanded"] == [""]


def test_travel_orchestrator_live_matrix_stamps_parent_route_contract():
    cases = build_matrix(case_names=["travel_orchestrator_wan2_1seg"])
    payload = render_case_payload(cases[0], project_id="project-1", unique_suffix="abc123")
    contract = payload["params"]["route_contract"]

    assert contract["route_key"] == "travel_orchestrator"
    assert contract["selected_backend"] == "wgp"
    assert contract["support_state"] == "wgp_only"
    assert contract["route_selection_snapshot"]["route_key"] == "travel_orchestrator"
    assert contract["route_selection_snapshot"]["live_test_run_id"] == "live-test-travel_orchestrator_wan2_1seg-abc123"


def test_live_matrix_includes_documented_wgp_only_direct_routes():
    cases = {case.name: case for case in build_matrix()}
    for name, task_type in {
        "travel_stitch": "travel_stitch",
        "join_clips_orchestrator": "join_clips_orchestrator",
        "edit_video_orchestrator": "edit_video_orchestrator",
    }.items():
        case = cases[name]
        assert case.task_type == task_type
        assert case.route_key == task_type
        assert case.support_state == "wgp_only"


def test_live_matrix_includes_promoted_vibecomfy_direct_routes():
    cases = {case.name: case for case in build_matrix()}
    expected = {
        "qwen_image_t2i": ("qwen_image", "image/qwen_image_2512"),
        "qwen_image_2512": ("qwen_image_2512", "image/qwen_image_2512"),
        "qwen_image_edit": ("qwen_image_edit", "edit/qwen_image_edit"),
        "qwen_image_style": ("qwen_image_style", "edit/qwen_image_edit"),
        "image_inpaint": ("image_inpaint", "edit/qwen_image_edit"),
        "annotated_image_edit": ("annotated_image_edit", "edit/qwen_image_edit"),
        "z_image_turbo_i2i": ("z_image_turbo_i2i", "image/z_image_img2img"),
        "wan_2_2_t2i": ("wan_2_2_t2i", "video/wanvideo_wrapper_22_14b_t2i"),
        "wan_2_2_i2v": ("wan_2_2_i2v", "video/wanvideo_wrapper_22_14b_i2v_kijai"),
        "travel_segment_wan22_i2v_first_last": ("travel_segment", "video/wanvideo_wrapper_22_14b_i2v_kijai"),
        "travel_segment_ltx2_first_last": ("travel_segment", "video/ltx2_3_runexx_first_last_frame"),
        "travel_segment_ltx2_distilled_first_last": ("travel_segment", "video/ltx2_3_runexx_first_last_frame"),
        "travel_segment_ltx2_control_pose_first_last": (
            "travel_segment",
            "video/ltx2_3_first_last_frame_travel_iclora_control",
        ),
        "travel_segment_ltx2_control_depth_first_last": (
            "travel_segment",
            "video/ltx2_3_first_last_frame_travel_iclora_control",
        ),
        "travel_segment_ltx2_control_canny_first_last": (
            "travel_segment",
            "video/ltx2_3_first_last_frame_travel_iclora_control",
        ),
        "travel_segment_ltx2_control_cameraman_first_last": (
            "travel_segment",
            "video/ltx2_3_first_last_frame_travel_iclora_control",
        ),
        "animate_character": (
            "animate_character",
            "video/wan22_animate_native_first_stage",
        ),
        "image_upscale": ("image-upscale", "image/basic_image_upscale"),
        "video_enhance": ("video_enhance", "video/basic_video_enhance"),
        "flux_klein_edit": ("flux_klein_edit", "edit/flux2_klein_4b_image_edit_distilled"),
        "join_clips_segment_wan22_vace": (
            "join_clips_segment",
            "video/wanvideo_wrapper_22_14b_vace_cocktail",
        ),
        "travel_segment_ltx2_first_last": (
            "travel_segment",
            "video/ltx2_3_runexx_first_last_frame",
        ),
        "travel_segment_ltx2_distilled_first_last": (
            "travel_segment",
            "video/ltx2_3_runexx_first_last_frame",
        ),
    }
    for name, (task_type, template_id) in expected.items():
        case = cases[name]
        assert case.task_type == task_type
        assert case.route_key
        assert case.support_state == "vibecomfy_supported"
        assert case.selected_template_id == template_id


@pytest.mark.parametrize(
    ("case_name", "route_key"),
    [
        ("qwen_image_t2i", "qwen_image"),
        ("qwen_image_2512", "qwen_image_2512"),
        ("qwen_image_edit", "qwen_image_edit"),
        ("qwen_image_style", "qwen_image_style"),
        ("z_image_turbo_i2i", "z_image_turbo_i2i"),
        ("wan_2_2_t2i", "wan_2_2_t2i"),
        ("wan_2_2_i2v", "wan_2_2_i2v"),
        (
            "travel_segment_wan22_i2v_first_last",
            "travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default",
        ),
        (
            "travel_segment_ltx2_distilled_first_last",
            "travel_segment__model-ltx2_distilled__guidance-none__continuity-first_last__profile-default",
        ),
        (
            "travel_segment_ltx2_control_pose_first_last",
            "travel_segment__model-ltx2_distilled__guidance-ltx_control_pose__continuity-first_last__profile-default",
        ),
        (
            "travel_segment_ltx2_control_depth_first_last",
            "travel_segment__model-ltx2_distilled__guidance-ltx_control_depth__continuity-first_last__profile-default",
        ),
        (
            "travel_segment_ltx2_control_canny_first_last",
            "travel_segment__model-ltx2_distilled__guidance-ltx_control_canny__continuity-first_last__profile-default",
        ),
        (
            "travel_segment_ltx2_control_cameraman_first_last",
            "travel_segment__model-ltx2_distilled__guidance-ltx_control_cameraman__continuity-first_last__profile-default",
        ),
        ("animate_character", "animate_character"),
        ("image_upscale", "image-upscale"),
        ("video_enhance", "video_enhance"),
        ("flux_klein_edit", "flux_klein_edit"),
        (
            "individual_travel_segment_wan22_vace_flow",
            "individual_travel_segment__model-wan22_vace__guidance-vace_flow__continuity-first_last__profile-default",
        ),
        (
            "individual_travel_segment_wan22_vace_canny",
            "individual_travel_segment__model-wan22_vace__guidance-vace_canny__continuity-first_last__profile-default",
        ),
        (
            "individual_travel_segment_wan22_vace_depth",
            "individual_travel_segment__model-wan22_vace__guidance-vace_depth__continuity-first_last__profile-default",
        ),
        (
            "travel_segment_wan22_vace_raw_video_source",
            "travel_segment__model-wan22_vace__guidance-vace_raw__continuity-video_source__profile-default",
        ),
        (
            "travel_segment_wan22_vace_flow_video_source",
            "travel_segment__model-wan22_vace__guidance-vace_flow__continuity-video_source__profile-default",
        ),
        (
            "travel_segment_wan22_vace_canny_video_source",
            "travel_segment__model-wan22_vace__guidance-vace_canny__continuity-video_source__profile-default",
        ),
        (
            "travel_segment_wan22_vace_depth_video_source",
            "travel_segment__model-wan22_vace__guidance-vace_depth__continuity-video_source__profile-default",
        ),
        (
            "join_clips_segment_wan22_vace",
            "join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default",
        ),
        (
            "travel_segment_ltx2_first_last",
            "travel_segment__model-ltx2__guidance-none__continuity-first_last__profile-default",
        ),
        (
            "travel_segment_ltx2_distilled_first_last",
            "travel_segment__model-ltx2_distilled__guidance-none__continuity-first_last__profile-default",
        ),
        ("image_inpaint", "image_inpaint"),
        ("annotated_image_edit", "annotated_image_edit"),
        ("travel_stitch", "travel_stitch"),
        ("join_clips_orchestrator", "join_clips_orchestrator"),
        ("edit_video_orchestrator", "edit_video_orchestrator"),
    ],
)
def test_live_matrix_stamps_direct_route_contracts(case_name: str, route_key: str):
    cases = build_matrix(case_names=[case_name])
    payload = render_case_payload(cases[0], project_id="project-1", unique_suffix="abc123")
    contract = payload["params"]["route_contract"]
    assert contract["route_key"] == route_key
    assert contract["support_state"] == cases[0].support_state
    assert payload["params"]["selected_backend"] == "wgp"


def test_live_matrix_vace_video_source_routes_use_real_travel_segment_contract():
    cases = build_matrix(case_names=["travel_segment_wan22_vace_flow_video_source"])
    payload = render_case_payload(cases[0], project_id="project-1", unique_suffix="abc123")
    details = payload["params"]["orchestrator_details"]

    for key in (
        "model_name",
        "parsed_resolution_wh",
        "segment_frames_expanded",
        "num_new_segments_to_generate",
        "base_prompts_expanded",
        "negative_prompts_expanded",
        "frame_overlap_expanded",
        "input_image_paths_resolved",
    ):
        assert key in details

    assert payload["task_type"] == "travel_segment"
    assert payload["params"]["segment_index"] == 0
    assert payload["params"]["video_source"]
    assert payload["params"]["travel_guidance"]["mode"] == "flow"
    assert details["continuation_config"] == {"type": "video_source"}


def test_live_matrix_wan_first_last_direct_segment_includes_child_identity():
    cases = build_matrix(case_names=["travel_segment_wan22_i2v_first_last"])
    payload = render_case_payload(cases[0], project_id="project-1", unique_suffix="abc123")
    params = payload["params"]
    details = params["orchestrator_details"]

    assert payload["task_type"] == "travel_segment"
    assert params["segment_index"] == 0
    assert params["orchestrator_run_id"]
    assert params["orchestrator_task_id_ref"]
    assert details["run_id"]
    assert details["orchestrator_task_id"]
    for key in (
        "base_prompts_expanded",
        "frame_overlap_expanded",
        "negative_prompts_expanded",
        "num_new_segments_to_generate",
        "segment_frames_expanded",
    ):
        assert key in details


def test_live_matrix_ltx_direct_segments_include_orchestrator_child_identity():
    cases = build_matrix(case_names=["travel_segment_ltx2_control_video_first_last"])
    payload = render_case_payload(cases[0], project_id="project-1", unique_suffix="abc123")
    params = payload["params"]
    details = params["orchestrator_details"]

    assert payload["task_type"] == "travel_segment"
    assert params["segment_index"] == 0
    assert params["orchestrator_run_id"]
    assert params["orchestrator_task_id_ref"]
    assert details["run_id"]
    assert details["orchestrator_task_id"]


def test_live_matrix_join_segment_vace_case_has_adapter_inputs():
    cases = build_matrix(case_names=["join_clips_segment_wan22_vace"])
    payload = render_case_payload(cases[0], project_id="project-1", unique_suffix="abc123")
    params = payload["params"]
    details = params["orchestrator_details"]

    assert payload["task_type"] == "join_clips_segment"
    assert params["model_family"] == "wan22_vace"
    assert params["continuity_case"] == "join_bridge"
    assert params["video_source"]
    assert params["travel_guidance"]["kind"] == "vace"
    assert "mode" not in params["travel_guidance"]
    assert params["input_image_paths_resolved"]
    assert details["clip_list"]
    assert details["run_id"] == "live-test-join_clips_segment_wan22_vace-abc123"


@pytest.mark.parametrize("case_name", ["join_clips_orchestrator", "edit_video_orchestrator"])
def test_live_matrix_orchestrator_cases_use_unique_run_ids(case_name: str):
    cases = build_matrix(case_names=[case_name])
    payload = render_case_payload(cases[0], project_id="project-1", unique_suffix="abc123")
    details = payload["params"]["orchestrator_details"]
    assert details["run_id"] == f"live-test-{case_name}-abc123"
    assert details["orchestrator_task_id"] == f"live-test-{case_name}-abc123"


def test_run_matrix_continues_after_individual_case_failures(monkeypatch: pytest.MonkeyPatch):
    cases = [
        MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=5),
        MatrixCase(name="case-b", task_type="qwen_image_style", fixture_key="qwen_image_style_db_task", timeout_sec=5),
    ]

    inserted = []
    polled = []

    def fake_insert(_db, _project_id, task_type, _params_overrides, **_kwargs):
        task_id = f"{task_type}-task-{len(inserted) + 1}"
        inserted.append(task_id)
        return task_id

    def fake_poll(_db, task_id, _project_id, **kwargs):
        polled.append(task_id)
        if task_id.startswith("qwen_image-task"):
            return TaskResult(
                task_id=task_id,
                case_name=kwargs["case_name"],
                task_type=kwargs["task_type"],
                final_status="Failed",
                output_location=None,
                generation_ids=[],
                elapsed_sec=1.0,
                error_summary="backend exploded",
            )
        return TaskResult(
            task_id=task_id,
            case_name=kwargs["case_name"],
            task_type=kwargs["task_type"],
            final_status="Complete",
            output_location="https://out.example/image.png",
            generation_ids=["gen-2"],
            elapsed_sec=1.0,
            error_summary=None,
        )

    monkeypatch.setattr("scripts.live_test.matrix.insert_spoof_task", fake_insert)
    monkeypatch.setattr("scripts.live_test.matrix.poll_until_complete", fake_poll)

    results = run_matrix(object(), "project-1", cases)
    assert [result.case_name for result in results] == ["case-a", "case-b"]
    assert results[0].final_status == "Failed"
    assert results[1].final_status == "Complete"
    assert inserted == ["qwen_image-task-1", "qwen_image_style-task-2"]
    assert polled == inserted


def test_run_matrix_serial_queues_next_case_after_previous_poll(monkeypatch: pytest.MonkeyPatch):
    cases = [
        MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=5),
        MatrixCase(name="case-b", task_type="qwen_image_style", fixture_key="qwen_image_style_db_task", timeout_sec=5),
    ]
    events = []

    def fake_insert(_db, _project_id, task_type, _params_overrides, **_kwargs):
        task_id = f"{task_type}-task"
        events.append(("insert", task_id))
        return task_id

    def fake_poll(_db, task_id, _project_id, **kwargs):
        events.append(("poll", task_id, kwargs.get("worker_id")))
        return TaskResult(
            task_id=task_id,
            case_name=kwargs["case_name"],
            task_type=kwargs["task_type"],
            final_status="Complete",
            output_location="https://out.example/image.png",
            generation_ids=["gen-1"],
            elapsed_sec=1.0,
            error_summary=None,
        )

    monkeypatch.setattr("scripts.live_test.matrix.insert_spoof_task", fake_insert)
    monkeypatch.setattr("scripts.live_test.matrix.poll_until_complete", fake_poll)

    results = run_matrix(object(), "project-1", cases, worker_id="worker-target", serial=True)

    assert [result.case_name for result in results] == ["case-a", "case-b"]
    assert events == [
        ("insert", "qwen_image-task"),
        ("poll", "qwen_image-task", "worker-target"),
        ("insert", "qwen_image_style-task"),
        ("poll", "qwen_image_style-task", "worker-target"),
    ]


def test_queue_matrix_inserts_all_cases_before_polling(monkeypatch: pytest.MonkeyPatch):
    cases = [
        MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=5),
        MatrixCase(name="case-b", task_type="qwen_image_style", fixture_key="qwen_image_style_db_task", timeout_sec=5),
    ]
    inserted = []

    def fake_insert(_db, _project_id, task_type, _params_overrides, **_kwargs):
        task_id = f"{task_type}-task-{len(inserted) + 1}"
        inserted.append(task_id)
        return task_id

    monkeypatch.setattr("scripts.live_test.matrix.insert_spoof_task", fake_insert)
    queued = queue_matrix(object(), "project-1", cases)
    assert [case.name for case, _task_id in queued] == ["case-a", "case-b"]
    assert [task_id for _case, task_id in queued] == ["qwen_image-task-1", "qwen_image_style-task-2"]


def test_fresh_vibecomfy_default_matrix_excludes_wgp_only_cases():
    args = SimpleNamespace(
        anchor_image_a="https://example.test/a.jpg",
        anchor_image_b="https://example.test/b.jpg",
        timeout_image=60,
        timeout_travel_segment=60,
        timeout_travel_orchestrator=60,
        backend="vibecomfy",
        selector_namespace="production",
        selector_version=None,
        worker_contract_version=1,
        worker_profile="default",
        case=[],
        task_type=[],
        route_key=[],
    )

    cases = build_fresh_matrix_cases(args)

    assert cases
    assert {case.support_state for case in cases} == {"vibecomfy_supported"}
    assert [case.name for case in cases[:3]] == ["z_image_turbo", "z_image_turbo_i2i", "qwen_image_2512"]
    assert "travel_orchestrator_wan2_1seg" not in {case.name for case in cases}


def test_fresh_vibecomfy_explicit_matrix_selection_can_include_wgp_only_case():
    args = SimpleNamespace(
        anchor_image_a="https://example.test/a.jpg",
        anchor_image_b="https://example.test/b.jpg",
        timeout_image=60,
        timeout_travel_segment=60,
        timeout_travel_orchestrator=60,
        backend="vibecomfy",
        selector_namespace="production",
        selector_version=None,
        worker_contract_version=1,
        worker_profile="default",
        case=["travel_orchestrator_wan2_1seg"],
        task_type=[],
        route_key=[],
    )

    cases = build_fresh_matrix_cases(args)

    assert [case.name for case in cases] == ["travel_orchestrator_wan2_1seg"]
    assert cases[0].support_state == "wgp_only"


def test_fresh_vibecomfy_worker_env_uses_service_claim_auth():
    env = build_fresh_worker_env(
        "pat-token",
        "https://supabase.example",
        "service-key",
        SimpleNamespace(
            backend="vibecomfy",
            selector_namespace="production",
            selector_version=None,
            worker_contract_version=1,
            worker_profile="default",
        ),
    )

    assert env["WORKER_DB_CLIENT_AUTH_MODE"] == "service"
    assert env["REIGH_ACCESS_TOKEN"] == "pat-token"
    assert env["SUPABASE_SERVICE_ROLE_KEY"] == "service-key"
    assert env["REIGH_BACKEND"] == "vibecomfy"
    assert env["VIBECOMFY_ATTENTION_PROFILE"] == "portable"


def test_fresh_vibecomfy_worker_env_maps_sage_worker_profile_to_attention_profile():
    env = build_fresh_worker_env(
        "pat-token",
        "https://supabase.example",
        "service-key",
        SimpleNamespace(
            backend="vibecomfy",
            selector_namespace="production",
            selector_version=None,
            worker_contract_version=1,
            worker_profile="sage",
        ),
    )

    assert env["VIBECOMFY_ATTENTION_PROFILE"] == "sage"
    assert env["REIGH_VIBECOMFY_ATTENTION_PROFILE"] == "sage"


def test_fresh_wgp_worker_env_keeps_pat_claim_auth():
    env = build_fresh_worker_env(
        "pat-token",
        "https://supabase.example",
        "service-key",
        SimpleNamespace(
            backend="wgp",
            selector_namespace="production",
            selector_version=None,
            worker_contract_version=1,
            worker_profile="default",
        ),
    )

    assert env["WORKER_DB_CLIENT_AUTH_MODE"] == "worker"


def test_write_report_outputs_json_and_markdown(tmp_path: Path):
    results = [
        TaskResult(
            task_id="task-1",
            case_name="case-a",
            task_type="qwen_image",
            final_status="Complete",
            output_location="https://out.example/a.png",
            generation_ids=["gen-1"],
            elapsed_sec=1.234,
            error_summary=None,
        ),
        TaskResult(
            task_id="task-2",
            case_name="case-b",
            task_type="qwen_image_style",
            final_status="Failed",
            output_location=None,
            generation_ids=[],
            elapsed_sec=2.345,
            error_summary="backend exploded",
        ),
    ]
    out_dir = write_report(results, "fresh", "pod-1", tmp_path / "runs" / "case-1")
    report_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert report_json["passed"] == 1
    assert report_json["total"] == 2
    assert "Summary: `1/2 passed`" in report_md
    assert all_results_passed(results) is False
    assert all_results_passed([results[0]]) is True


def test_variant_fresh_dry_run_uses_livetest_workspace_and_env_exports(capsys, monkeypatch: pytest.MonkeyPatch):
    cases = [MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=900)]
    monkeypatch.setattr(
        "scripts.live_test.variant_fresh._prepare_context",
        lambda _args: {
            "db": object(),
            "token": "token-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "cases": cases,
        },
    )
    monkeypatch.setattr("scripts.live_test.variant_fresh._validate_cases", lambda _cases, _project_id: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_fresh.config.require_env",
        lambda name: {
            "SUPABASE_URL": "https://supabase.example",
        }[name],
    )
    args = SimpleNamespace(
        dry_run=True,
        no_terminate=False,
        wgp_profile=3,
        timeout_image=900,
        timeout_travel_segment=1500,
        timeout_travel_orchestrator=2400,
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        ref="main",
        backend="vibecomfy",
        selector_namespace="canary-a",
        selector_version="2026-05-06",
        worker_contract_version=1,
        worker_profile="z-image-default",
        case=[],
        task_type=[],
        route_key=[],
        wgp_rollback=False,
    )
    assert run_variant_fresh(args) == 0
    output = capsys.readouterr().out
    assert "/workspace/Reigh-Worker-LiveTest" in output
    assert "Terminate after run: True" in output
    assert "REIGH_ACCESS_TOKEN" in output
    assert "SUPABASE_SERVICE_ROLE_KEY" in output
    assert "SUPABASE_URL" in output
    assert "WORKER_DB_CLIENT_AUTH_MODE" in output
    assert "REIGH_BACKEND" in output
    assert "REIGH_SELECTOR_NAMESPACE" in output
    assert "REIGH_WORKER_CONTRACT_VERSION" in output
    assert "VibeComfy clone target: /workspace/vibecomfy" in output
    assert "VIBECOMFY_PATH" in output


def test_variant_fresh_dry_run_without_token_still_validates_static_plan(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
):
    cases = [MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=900)]
    monkeypatch.setattr(
        "scripts.live_test.variant_fresh.config.get_env",
        lambda name, default=None: None if name == "REIGH_LIVE_TEST_TOKEN" else default,
    )
    monkeypatch.setattr("scripts.live_test.variant_fresh._build_matrix_cases", lambda _args: cases)
    validated = []
    monkeypatch.setattr(
        "scripts.live_test.variant_fresh._validate_cases",
        lambda cases_arg, project_id: validated.append((cases_arg, project_id)),
    )
    args = SimpleNamespace(
        dry_run=True,
        no_terminate=False,
        wgp_profile=3,
        backend="vibecomfy",
        selector_namespace="canary-a",
        selector_version=None,
        worker_contract_version=1,
        worker_profile="default",
        ref="main",
    )

    assert run_variant_fresh(args) == 0

    output = capsys.readouterr().out
    assert validated == [(cases, "<live-test-project-id>")]
    assert "--reigh-access-token" not in output
    assert "https://example.supabase.co" in output


def test_variant_update_dry_run_without_token_still_validates_static_plan(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
):
    cases = [MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=900)]
    monkeypatch.setattr(
        "scripts.live_test.variant_update.config.get_env",
        lambda name, default=None: None if name == "REIGH_LIVE_TEST_TOKEN" else default,
    )
    monkeypatch.setattr("scripts.live_test.variant_update._build_matrix_cases", lambda _args: cases)
    validated = []
    monkeypatch.setattr(
        "scripts.live_test.variant_update._validate_cases",
        lambda cases_arg, project_id: validated.append((cases_arg, project_id)),
    )
    args = SimpleNamespace(
        dry_run=True,
        spawn_takeover=False,
        pod_id="pod-1",
        no_terminate=True,
        wgp_profile=3,
        backend="vibecomfy",
        selector_namespace="canary-a",
        selector_version=None,
        worker_contract_version=1,
        worker_profile="default",
    )

    assert run_variant_update(args) == 0

    output = capsys.readouterr().out
    assert validated == [(cases, "<live-test-project-id>")]
    assert "<REIGH_LIVE_TEST_TOKEN>" in output
    assert "https://example.supabase.co" in output


def test_variant_fresh_registers_pod_worker_row_before_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts.live_test import variant_fresh

    events = []
    cases = [MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=900)]

    class FakeDB:
        async def create_worker_record(self, worker_id, instance_type, runpod_id=None):
            events.append(("create_worker_record", worker_id, instance_type, runpod_id))
            return True

        async def update_worker_status(self, worker_id, status, metadata):
            events.append(("update_worker_status", worker_id, status, metadata.get("runpod_id")))
            return True

    class DummySSH:
        def execute_command(self, _command, timeout=600):
            return 0, "", ""

        def disconnect(self):
            return None

    monkeypatch.setattr(
        "scripts.live_test.variant_fresh._prepare_context",
        lambda _args: {
            "db": FakeDB(),
            "token": "token-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "cases": cases,
        },
    )
    monkeypatch.setattr("scripts.live_test.variant_fresh._validate_cases", lambda _cases, _project_id: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_fresh.config.require_env",
        lambda name: {
            "RUNPOD_API_KEY": "api-key",
            "SUPABASE_URL": "https://supabase.example",
            "SUPABASE_SERVICE_ROLE_KEY": "service-key",
        }[name],
    )
    monkeypatch.setattr("scripts.live_test.variant_fresh._runs_root", lambda: tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "runpod_lifecycle.api",
        types.SimpleNamespace(
            create_pod=lambda **_kwargs: {"id": "pod-123", "networkVolumeId": "volume-1"},
            get_network_volumes=lambda _api_key: [],
        ),
    )
    monkeypatch.setattr("scripts.live_test.variant_fresh.open_session", lambda _pod_id, _api_key: DummySSH())
    monkeypatch.setattr("scripts.live_test.variant_fresh.clone_repo_into", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.live_test.variant_fresh.run_install", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.live_test.variant_fresh.clone_and_install_vibecomfy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.live_test.variant_fresh.launch_worker_detached", lambda *_args, **_kwargs: events.append(("launch_worker",)))
    monkeypatch.setattr("scripts.live_test.variant_fresh.wait_until_ready", lambda *_args, **_kwargs: events.append(("wait_until_ready",)))
    monkeypatch.setattr(
        "scripts.live_test.variant_fresh.queue_matrix",
        lambda _db, _project_id, _cases: events.append(("queue_matrix",)) or [(cases[0], "task-1")],
    )
    monkeypatch.setattr("scripts.live_test.variant_fresh.poll_queued_matrix", lambda _db, _project_id, _queued, **_kwargs: [])
    monkeypatch.setattr("scripts.live_test.variant_fresh.write_report", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr("scripts.live_test.variant_fresh.fetch_worker_logs", lambda *_args, **_kwargs: "logs")
    monkeypatch.setattr("scripts.live_test.variant_fresh.guarded_terminate", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        "scripts.live_test.variant_fresh.prune_stale_live_test_pods",
        lambda _api_key: SimpleNamespace(terminated=(), failed=()),
    )

    args = SimpleNamespace(
        dry_run=False,
        no_terminate=False,
        wgp_profile=3,
        timeout_image=900,
        timeout_travel_segment=1500,
        timeout_travel_orchestrator=2400,
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        ref="main",
        backend="vibecomfy",
        vibecomfy_ref="branch-a",
        selector_namespace="production",
        selector_version=None,
        worker_contract_version=1,
        worker_profile="default",
        case=[],
        task_type=[],
        route_key=[],
    )
    assert run_variant_fresh(args) == 0
    assert ("create_worker_record", "pod-123", variant_fresh.config.RUNPOD_GPU_TYPE, "pod-123") in events
    assert ("update_worker_status", "pod-123", "inactive", "pod-123") in events
    assert events.index(("launch_worker",)) < events.index(("wait_until_ready",)) < events.index(("queue_matrix",))


def test_clone_and_install_vibecomfy_validates_required_manifests():
    calls = []

    class DummySSH:
        def execute_command(self, command, timeout=600):
            calls.append((command, timeout))
            return 0, "", ""

    clone_and_install_vibecomfy(
        DummySSH(),
        repo_url="https://github.com/peteromallet/VibeComfy.git",
        branch="branch-a",
        workdir="/workspace/vibecomfy",
        python_path="python3.11",
    )

    command, timeout = calls[0]
    assert timeout == 3600
    assert "git clone --branch branch-a --single-branch https://github.com/peteromallet/VibeComfy.git /workspace/vibecomfy" in command
    assert "git -C /workspace/vibecomfy fetch origin branch-a" in command
    assert "git -C /workspace/vibecomfy reset --hard FETCH_HEAD" in command
    assert 'echo "VibeComfy checkout: $(git -C /workspace/vibecomfy rev-parse --short HEAD)"' in command
    assert "python3.11 -m pip install -e /workspace/vibecomfy" in command
    assert "python3.11 -m pip install" in command
    assert "comfyui@git+https://github.com/peteromallet/ComfyUI.git@fix/latentupscale-model-mmap-residency" in command
    assert "export VIBECOMFY_ATTENTION_PROFILE=portable" in command
    assert "SageAttention" not in command
    assert "cd /workspace/vibecomfy" in command
    assert "test -f custom_nodes.lock" in command
    assert "python3.11 -m vibecomfy.cli nodes restore --lockfile custom_nodes.lock" in command
    assert "test -f /workspace/vibecomfy/template_index.json" in command
    assert "test -f /workspace/vibecomfy/workflow_corpus/manifests/coverage.json" in command


def test_clone_and_install_vibecomfy_installs_and_verifies_sageattention_when_profile_sage():
    calls = []

    class DummySSH:
        def execute_command(self, command, timeout=600):
            calls.append((command, timeout))
            return 0, "", ""

    clone_and_install_vibecomfy(
        DummySSH(),
        repo_url="https://github.com/peteromallet/VibeComfy.git",
        branch="branch-a",
        workdir="/workspace/vibecomfy",
        python_path="python3.11",
        attention_profile="sage",
    )

    command, _timeout = calls[0]
    assert "export VIBECOMFY_ATTENTION_PROFILE=sage" in command
    assert "git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/sageattention" in command
    assert "python3.11 -m pip install --no-build-isolation /tmp/sageattention" in command
    assert "import sageattention" in command
    assert "sageattention verified" in command


def test_spawn_takeover_pod_calls_create_record_and_waits_for_ssh(monkeypatch: pytest.MonkeyPatch):
    events = []

    class FakeRunpodConfig:
        def merge(self, **overrides):
            events.append(("runpod_config_merge", overrides))
            return self

    class FakeDB:
        async def create_worker_record(self, worker_id, instance_type):
            events.append(("create_worker_record", worker_id, instance_type))
            return True

        async def update_worker_status(self, worker_id, status, metadata):
            events.append((
                "update_worker_status",
                worker_id,
                status,
                metadata["runpod_id"],
                metadata.get("live_test_variant"),
                metadata.get("worker_pool"),
            ))
            return True

    class FakeSpawner:
        def __init__(self):
            events.append(("init",))
            self.gpu_type = "NVIDIA GeForce RTX 4090"
            self.runpod_config = FakeRunpodConfig()

        def generate_worker_id(self):
            events.append(("generate_worker_id",))
            return "worker-123"

        async def spawn_worker(self, worker_id):
            events.append(("spawn_worker", worker_id))
            return {"runpod_id": "pod-456", "pod_details": {"id": "pod-456"}}

    def _fake_factory(config, db):
        return FakeSpawner()

    monkeypatch.setitem(
        sys.modules,
        "gpu_orchestrator.worker_spawner",
        types.SimpleNamespace(create_worker_spawner=_fake_factory),
    )

    worker_id, pod_id = _spawn_takeover_pod(FakeDB(), "api-key")
    assert (worker_id, pod_id) == ("worker-123", "pod-456")
    assert events == [
        ("init",),
        (
            "runpod_config_merge",
            {
                "disk_size_gb": 200,
                "container_disk_gb": 200,
                "min_memory_gb": 32,
                "ram_tiers": (32, 24, 16),
            },
        ),
        ("generate_worker_id",),
        ("create_worker_record", "worker-123", "NVIDIA GeForce RTX 4090"),
        ("spawn_worker", "worker-123"),
        ("update_worker_status", "worker-123", "spawning", "pod-456", "update", "gpu-live-test"),
    ]


def test_spawn_takeover_waits_for_ssh_without_starting_production_worker(monkeypatch: pytest.MonkeyPatch):
    from scripts.live_test import variant_update

    events = []

    class FakeDB:
        async def create_worker_record(self, worker_id, instance_type):
            return True

        async def update_worker_status(self, worker_id, status, metadata):
            return True

    class FakeSpawner:
        gpu_type = "NVIDIA GeForce RTX 4090"

        def generate_worker_id(self):
            return "worker-123"

        async def spawn_worker(self, worker_id):
            return {"runpod_id": "pod-456", "pod_details": {"id": "pod-456"}}

        async def check_and_initialize_worker(self, worker_id, pod_id):
            events.append(("check", worker_id, pod_id))
            return {"status": "spawning"} if len(events) == 1 else {"status": "spawning", "ready": True}

    monkeypatch.setitem(
        sys.modules,
        "gpu_orchestrator.worker_spawner",
        types.SimpleNamespace(create_worker_spawner=lambda config, db: FakeSpawner()),
    )
    monkeypatch.setattr(variant_update.time, "sleep", lambda _interval: None)

    assert _spawn_takeover_pod(FakeDB(), "api-key") == ("worker-123", "pod-456")
    assert events == [
        ("check", "worker-123", "pod-456"),
        ("check", "worker-123", "pod-456"),
    ]


def test_remote_checkout_and_sync_bootstraps_uv_before_sync():
    ssh = ScriptedSSH([(None, (0, "", ""))])

    _remote_checkout_and_sync(ssh, "live-test/branch-a")

    command, timeout = ssh.commands[0]
    assert timeout == 3600
    assert "cd /workspace/Reigh-Worker" in command
    assert 'export PATH="$HOME/.local/bin:$PATH"' in command
    assert "python3 -m pip install --user uv" in command
    assert "command -v uv >/dev/null 2>&1" in command
    assert "git fetch origin live-test/branch-a:refs/remotes/origin/live-test/branch-a" in command
    assert "git checkout -B live-test/branch-a refs/remotes/origin/live-test/branch-a" in command
    assert 'export UV_CACHE_DIR="/root/.cache/uv-live-test"' in command
    assert 'UV_PROJECT_ENVIRONMENT="/opt/reigh-worker-live-test-venv"' in command
    assert "UV_LINK_MODE=copy" in command
    assert 'rm -rf .venv "$UV_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT"' in command
    assert "git gc --prune=now" in command
    assert "uv sync --locked --extra cuda124" in command
    assert "uv sync attempt $attempt failed; cleaning partial venv/cache and retrying" in command


def test_variant_update_spawn_takeover_threads_worker_id_not_pod_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    safety_calls = []
    wait_calls = []
    launched = []
    cleanup_calls = []
    restore_calls = []
    status_updates = []
    finally_events = []

    cases = [MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=900)]
    class FakeDB:
        async def update_worker_status(self, worker_id, status, metadata):
            status_updates.append((worker_id, status, metadata["runpod_id"]))
            return True

    monkeypatch.setattr(
        "scripts.live_test.variant_update._prepare_context",
        lambda _args: {
            "db": FakeDB(),
            "token": "token-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "cases": cases,
        },
    )
    monkeypatch.setattr("scripts.live_test.variant_update._validate_cases", lambda _cases, _project_id: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.config.require_env",
        lambda name: {
            "RUNPOD_API_KEY": "api-key",
            "SUPABASE_URL": "https://supabase.example",
            "SUPABASE_SERVICE_ROLE_KEY": "service-key",
        }[name],
    )
    monkeypatch.setattr("scripts.live_test.variant_update._runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        "scripts.live_test.variant_update._spawn_takeover_pod",
        lambda _db, _api_key: ("worker-123", "pod-456"),
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update.assert_safe_to_take_over",
        lambda _db, pod_id, user_id, allow_fresh_heartbeat=False: safety_calls.append(
            (pod_id, user_id, allow_fresh_heartbeat)
        ),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.snapshot_local_state", lambda _path: "snapshot")
    monkeypatch.setattr(
        "scripts.live_test.variant_update.push_working_copy_to_temp_branch",
        lambda _path, _snapshot: ("live-test/branch", "sha-1"),
    )

    class DummySSH:
        def execute_command(self, _command, timeout=600):
            return 0, "", ""

        def disconnect(self):
            return None

    monkeypatch.setattr("scripts.live_test.variant_update.open_session", lambda _pod_id, _api_key: DummySSH())
    monkeypatch.setattr("scripts.live_test.variant_update._read_remote_branch", lambda _ssh, _workdir=None: "main")
    monkeypatch.setattr("scripts.live_test.variant_update._read_remote_sha", lambda _ssh, _workdir=None: "sha-prev")
    monkeypatch.setattr(
        "scripts.live_test.variant_update.capture_current_worker_cmdline",
        lambda _ssh: WorkerProcessInfo(
            family="supervisor",
            cmdline=["python", "run_worker.py", "--worker", "old-worker"],
            pid=123,
        ),
    )
    monkeypatch.setattr("scripts.live_test.variant_update._remote_checkout_and_sync", lambda _ssh, _branch, _workdir=None: None)
    monkeypatch.setattr("scripts.live_test.variant_update.kill_supervisor_and_worker", lambda _ssh: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.launch_worker_detached",
        lambda _ssh, command: launched.append(command),
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update.wait_until_ready",
        lambda _db, worker_id, timeout_sec=900, **kwargs: wait_calls.append((worker_id, timeout_sec, kwargs)),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.run_matrix", lambda _db, _project_id, _cases, **_kwargs: [])
    monkeypatch.setattr(
        "scripts.live_test.variant_update.write_report",
        lambda _results, _variant, _pod_id, _out_dir: tmp_path,
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update._restore_remote_state",
        lambda _ssh, **kwargs: restore_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update.cleanup_temp_branch",
        lambda branch, preserve, submodule_path="reigh-worker": cleanup_calls.append((branch, preserve, submodule_path))
        or branch,
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update.restore_local_state",
        lambda _path, _snapshot: finally_events.append("local_restore") or restore_calls.append({"local_restore": True}),
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update.fetch_worker_logs",
        lambda _ssh, _workdir: finally_events.append("fetch_logs") or "logs",
    )
    monkeypatch.setattr("scripts.live_test.variant_update.guarded_terminate", lambda *_args, **_kwargs: False)

    args = SimpleNamespace(
        dry_run=False,
        spawn_takeover=True,
        pod_id=None,
        no_terminate=True,
        wgp_profile=3,
        timeout_image=900,
        timeout_travel_segment=1500,
        timeout_travel_orchestrator=2400,
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        ref="main",
    )
    assert run_variant_update(args) == 0
    assert safety_calls == [("pod-456", "user-1", True)]
    assert wait_calls == [("worker-123", 900, {"require_ready_for_tasks": True})]
    assert any("--worker worker-123" in command for command in launched)
    assert any('UV_PROJECT_ENVIRONMENT="/opt/reigh-worker-live-test-venv"' in command for command in launched)
    assert all("--worker pod-456" not in command for command in launched)
    assert status_updates == [("worker-123", "inactive", "pod-456")]
    assert cleanup_calls == [("live-test/branch", False, str(ROOT))]
    assert finally_events == ["fetch_logs", "local_restore"]


def test_variant_update_existing_mode_uses_stale_heartbeat_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    safety_calls = []
    wait_calls = []
    status_updates = []

    cases = [MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=900)]
    class FakeDB:
        async def update_worker_status(self, worker_id, status, metadata):
            status_updates.append((worker_id, status, metadata["runpod_id"], metadata["worker_backend"]))
            return True

    monkeypatch.setattr(
        "scripts.live_test.variant_update._prepare_context",
        lambda _args: {
            "db": FakeDB(),
            "token": "token-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "cases": cases,
        },
    )
    monkeypatch.setattr("scripts.live_test.variant_update._validate_cases", lambda _cases, _project_id: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.config.require_env",
        lambda name: {
            "RUNPOD_API_KEY": "api-key",
            "SUPABASE_URL": "https://supabase.example",
            "SUPABASE_SERVICE_ROLE_KEY": "service-key",
        }[name],
    )
    monkeypatch.setattr("scripts.live_test.variant_update._runs_root", lambda: tmp_path)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.assert_safe_to_take_over",
        lambda _db, pod_id, user_id, allow_fresh_heartbeat=False: safety_calls.append(
            (pod_id, user_id, allow_fresh_heartbeat)
        ),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.snapshot_local_state", lambda _path: "snapshot")
    monkeypatch.setattr(
        "scripts.live_test.variant_update.push_working_copy_to_temp_branch",
        lambda _path, _snapshot: ("live-test/branch", "sha-1"),
    )

    class DummySSH:
        def execute_command(self, _command, timeout=600):
            return 0, "", ""

        def disconnect(self):
            return None

    monkeypatch.setattr("scripts.live_test.variant_update.open_session", lambda _pod_id, _api_key: DummySSH())
    monkeypatch.setattr("scripts.live_test.variant_update._read_remote_branch", lambda _ssh, _workdir=None: "main")
    monkeypatch.setattr("scripts.live_test.variant_update._read_remote_sha", lambda _ssh, _workdir=None: "sha-prev")
    monkeypatch.setattr(
        "scripts.live_test.variant_update.capture_current_worker_cmdline",
        lambda _ssh: WorkerProcessInfo(
            family="supervisor",
            cmdline=["python", "run_worker.py", "--worker", "worker-prev"],
            pid=123,
        ),
    )
    monkeypatch.setattr(
        "scripts.live_test.variant_update._resolve_existing_worker_id",
        lambda _db, _pod_id, _prev_proc, **_kwargs: "worker-prev",
    )
    monkeypatch.setattr("scripts.live_test.variant_update._remote_checkout_and_sync", lambda _ssh, _branch, _workdir=None: None)
    monkeypatch.setattr("scripts.live_test.variant_update.clone_and_install_vibecomfy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.live_test.variant_update.kill_supervisor_and_worker", lambda _ssh: None)
    monkeypatch.setattr("scripts.live_test.variant_update.launch_worker_detached", lambda _ssh, _command: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.wait_until_ready",
        lambda _db, worker_id, timeout_sec=900, **kwargs: wait_calls.append((worker_id, timeout_sec, kwargs)),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.run_matrix", lambda _db, _project_id, _cases, **_kwargs: [])
    monkeypatch.setattr("scripts.live_test.variant_update.write_report", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr("scripts.live_test.variant_update._restore_remote_state", lambda _ssh, **_kwargs: None)
    monkeypatch.setattr("scripts.live_test.variant_update.cleanup_temp_branch", lambda branch, preserve, submodule_path='reigh-worker': branch)
    monkeypatch.setattr("scripts.live_test.variant_update.restore_local_state", lambda _path, _snapshot: None)
    monkeypatch.setattr("scripts.live_test.variant_update.fetch_worker_logs", lambda _ssh, _workdir: "logs")
    monkeypatch.setattr("scripts.live_test.variant_update.guarded_terminate", lambda *_args, **_kwargs: False)

    args = SimpleNamespace(
        dry_run=False,
        spawn_takeover=False,
        pod_id="pod-existing",
        no_terminate=True,
        wgp_profile=3,
        timeout_image=900,
        timeout_travel_segment=1500,
        timeout_travel_orchestrator=2400,
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        ref="main",
        backend="wgp",
    )
    assert run_variant_update(args) == 0
    assert safety_calls == [("pod-existing", "user-1", False)]
    assert status_updates == [("worker-prev", "inactive", "pod-existing", "wgp")]
    assert wait_calls == [("worker-prev", 900, {"require_ready_for_tasks": True})]


def test_variant_update_prefers_pod_worker_row_over_stale_process_cmdline():
    db = FakeDB(
        tables={
            "workers": [
                {
                    "id": "old-worker",
                    "metadata": {"runpod_id": "old-pod"},
                    "status": "active",
                    "created_at": "2026-05-07T10:00:00Z",
                    "last_heartbeat": "2026-05-07T10:01:00Z",
                },
                {
                    "id": "pod-existing",
                    "metadata": {"runpod_id": "pod-existing", "live_test_variant": "fresh"},
                    "status": "terminated",
                    "created_at": "2026-05-07T11:00:00Z",
                    "last_heartbeat": "2026-05-07T11:01:00Z",
                },
            ]
        }
    )
    prev_proc = WorkerProcessInfo(
        family="supervisor",
        cmdline=["python", "run_worker.py", "--worker", "old-worker"],
        pid=123,
    )

    assert _resolve_existing_worker_id(db, "pod-existing", prev_proc) == "pod-existing"
    assert _resolve_existing_worker_id(
        FakeDB(tables={"workers": []}),
        "pod-missing",
        prev_proc,
        allow_pod_id_fallback=True,
    ) == "pod-missing"


def test_variant_update_uses_targeted_pod_worker_lookup_when_broad_scan_misses_row():
    target_row = {
        "id": "worker-target",
        "metadata": {"runpod_id": "pod-target"},
        "status": "active",
        "created_at": "2026-05-08T10:00:00Z",
        "last_heartbeat": "2026-05-08T10:01:00Z",
    }

    def workers_source(query):
        return [target_row] if query.filters else []

    db = FakeDB(sources={"workers": workers_source})

    assert _resolve_existing_worker_id(db, "pod-target", prev_proc=None) == "worker-target"


def test_variant_update_uses_targeted_worker_row_lookup_when_broad_scan_misses_row():
    target_row = {"id": "worker-target"}

    def workers_source(query):
        return [target_row] if query.filters else []

    db = FakeDB(sources={"workers": workers_source})

    assert _worker_row_exists(db, "worker-target")


def test_variant_update_reuses_fresh_live_test_workdir_for_fresh_pods():
    fresh_db = FakeDB(
        tables={
            "workers": [
                {
                    "id": "pod-existing",
                    "metadata": {"runpod_id": "pod-existing", "live_test_variant": "fresh"},
                    "status": "terminated",
                }
            ]
        }
    )
    normal_db = FakeDB(tables={"workers": [{"id": "worker-1", "metadata": {"runpod_id": "pod-existing"}}]})

    assert _resolve_update_workdir(fresh_db, "pod-existing") == FRESH_LIVE_TEST_WORKDIR
    assert _resolve_update_workdir(normal_db, "pod-existing") == UPDATE_WORKDIR


def test_variant_update_vibecomfy_refreshes_vibecomfy_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    refresh_calls = []
    launched = []
    wait_calls = []
    status_updates = []

    cases = [MatrixCase(name="case-a", task_type="qwen_image", fixture_key="qwen_image_basic", timeout_sec=900)]
    class FakeDB:
        async def update_worker_status(self, worker_id, status, metadata):
            status_updates.append((worker_id, status, metadata["runpod_id"], metadata["worker_backend"]))
            return True

    monkeypatch.setattr(
        "scripts.live_test.variant_update._prepare_context",
        lambda _args: {
            "db": FakeDB(),
            "token": "token-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "cases": cases,
        },
    )
    monkeypatch.setattr("scripts.live_test.variant_update._validate_cases", lambda _cases, _project_id: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.config.require_env",
        lambda name: {
            "RUNPOD_API_KEY": "api-key",
            "SUPABASE_URL": "https://supabase.example",
            "SUPABASE_SERVICE_ROLE_KEY": "service-key",
        }[name],
    )
    monkeypatch.setattr("scripts.live_test.variant_update._runs_root", lambda: tmp_path)
    monkeypatch.setattr("scripts.live_test.variant_update.assert_safe_to_take_over", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("scripts.live_test.variant_update.snapshot_local_state", lambda _path: "snapshot")
    monkeypatch.setattr(
        "scripts.live_test.variant_update.push_working_copy_to_temp_branch",
        lambda _path, _snapshot: ("live-test/branch", "sha-1"),
    )

    class DummySSH:
        def execute_command(self, _command, timeout=600):
            return 0, "", ""

        def disconnect(self):
            return None

    monkeypatch.setattr("scripts.live_test.variant_update.open_session", lambda _pod_id, _api_key: DummySSH())
    monkeypatch.setattr("scripts.live_test.variant_update._read_remote_branch", lambda _ssh, _workdir=None: "main")
    monkeypatch.setattr("scripts.live_test.variant_update._read_remote_sha", lambda _ssh, _workdir=None: "sha-prev")
    monkeypatch.setattr("scripts.live_test.variant_update.capture_current_worker_cmdline", lambda _ssh: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update._resolve_existing_worker_id",
        lambda _db, _pod_id, _prev_proc, **_kwargs: "worker-prev",
    )
    monkeypatch.setattr("scripts.live_test.variant_update._remote_checkout_and_sync", lambda _ssh, _branch, _workdir=None: None)
    monkeypatch.setattr(
        "scripts.live_test.variant_update.clone_and_install_vibecomfy",
        lambda _ssh, **kwargs: refresh_calls.append(kwargs),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.kill_supervisor_and_worker", lambda _ssh: None)
    monkeypatch.setattr("scripts.live_test.variant_update.launch_worker_detached", lambda _ssh, command: launched.append(command))
    monkeypatch.setattr(
        "scripts.live_test.variant_update.wait_until_ready",
        lambda _db, worker_id, timeout_sec=900, **kwargs: wait_calls.append((worker_id, timeout_sec, kwargs)),
    )
    monkeypatch.setattr("scripts.live_test.variant_update.run_matrix", lambda _db, _project_id, _cases, **_kwargs: [])
    monkeypatch.setattr("scripts.live_test.variant_update.write_report", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr("scripts.live_test.variant_update._restore_remote_state", lambda _ssh, **_kwargs: None)
    monkeypatch.setattr("scripts.live_test.variant_update.cleanup_temp_branch", lambda branch, preserve, submodule_path='reigh-worker': branch)
    monkeypatch.setattr("scripts.live_test.variant_update.restore_local_state", lambda _path, _snapshot: None)
    monkeypatch.setattr("scripts.live_test.variant_update.fetch_worker_logs", lambda _ssh, _workdir: "logs")
    monkeypatch.setattr("scripts.live_test.variant_update.guarded_terminate", lambda *_args, **_kwargs: False)

    args = SimpleNamespace(
        dry_run=False,
        spawn_takeover=False,
        pod_id="pod-existing",
        no_terminate=True,
        wgp_profile=3,
        timeout_image=900,
        timeout_travel_segment=1500,
        timeout_travel_orchestrator=2400,
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        ref="main",
        backend="vibecomfy",
        vibecomfy_ref="vibe-branch",
        selector_namespace="production",
        selector_version=None,
        worker_contract_version=1,
        worker_profile="default",
    )
    assert run_variant_update(args) == 0
    assert refresh_calls == [
        {
            "repo_url": "https://github.com/peteromallet/VibeComfy.git",
            "branch": "vibe-branch",
            "workdir": "/workspace/vibecomfy",
            "python_path": "python3.11",
            "attention_profile": "portable",
        }
    ]
    assert launched
    assert wait_calls == [("worker-prev", 900, {"require_ready_for_tasks": False})]
    assert status_updates == [("worker-prev", "inactive", "pod-existing", "vibecomfy")]
    assert "WORKER_DB_CLIENT_AUTH_MODE=service" in launched[0]
    assert "VIBECOMFY_CWD=/workspace/vibecomfy" in launched[0]


def test_variant_update_reconnects_when_vibecomfy_install_loses_ssh(monkeypatch: pytest.MonkeyPatch):
    from scripts.live_test import variant_update

    calls = []
    reopened = []

    class DummySSH:
        def __init__(self, name):
            self.name = name
            self.disconnected = False

        def disconnect(self):
            self.disconnected = True

    first = DummySSH("first")
    second = DummySSH("second")

    def fake_clone(ssh, **kwargs):
        calls.append((ssh.name, kwargs))
        if ssh is first:
            raise RuntimeError("Remote command failed with exit -1: SSH session not active")

    monkeypatch.setattr(variant_update, "clone_and_install_vibecomfy", fake_clone)
    monkeypatch.setattr(
        variant_update,
        "open_session",
        lambda pod_id, api_key, ssh_wait_timeout=180: reopened.append((pod_id, api_key, ssh_wait_timeout)) or second,
    )

    result = variant_update._clone_and_install_vibecomfy_with_reconnect(
        first,
        pod_id="pod-1",
        api_key="api-key",
        repo_url="repo",
        branch="branch",
        workdir="/workspace/vibecomfy",
        python_path="python3.11",
    )

    assert result is second
    assert first.disconnected is True
    assert reopened == [("pod-1", "api-key", 180)]
    assert [name for name, _kwargs in calls] == ["first", "second"]


def test_main_defaults_terminate_for_fresh_and_no_terminate_for_update(monkeypatch: pytest.MonkeyPatch):
    seen = []

    monkeypatch.setattr(
        "scripts.live_test.main.run_variant_fresh",
        lambda args: seen.append(("fresh", args.no_terminate, args.backend, args.route_key)) or 0,
    )
    monkeypatch.setattr(
        "scripts.live_test.main.run_variant_update",
        lambda args: seen.append(("update", args.no_terminate, args.backend, args.route_key)) or 0,
    )

    assert live_test_main.main(["--variant", "fresh", "--dry-run", "--backend", "vibecomfy", "--route-key", "z_image_turbo"]) == 0
    assert live_test_main.main(["--variant", "update", "--pod-id", "pod-1", "--dry-run", "--wgp-rollback", "--route-key", "z_image_turbo"]) == 0
    assert seen == [
        ("fresh", False, "vibecomfy", ["z_image_turbo"]),
        ("update", True, "wgp", ["z_image_turbo"]),
    ]


def test_main_defaults_to_production_parity_vibecomfy_ref() -> None:
    parser = live_test_main.build_parser()
    args = parser.parse_args(["--variant", "update", "--pod-id", "pod-1", "--backend", "vibecomfy"])

    assert args.vibecomfy_ref == "megaplan/production-parity-templates"
