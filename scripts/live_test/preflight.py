"""User-safety checks before the live harness mutates queue state."""

from __future__ import annotations

from typing import Any


LIVE_TEST_PROJECT_NAME = "live-test"


class UnexpectedUserWorkError(RuntimeError):
    """Raised when the supposedly isolated test user still has live work queued."""


def _coerce_rows(result: Any) -> list[dict[str, Any]]:
    data = getattr(result, "data", None)
    if not data:
        return []
    if isinstance(data, dict):
        return [data]
    return [row for row in data if isinstance(row, dict)]


def _is_live_test_task(task: dict[str, Any]) -> bool:
    params = task.get("params")
    if not isinstance(params, dict):
        return False
    return str(params.get("live_test", "false")).lower() == "true"


def assert_user_queue_clean(db, user_id: str) -> None:
    """Abort if the target user has non-live-test queued or in-progress work."""
    result = (
        db.supabase.table("tasks")
        .select("id, status, params, project_id, projects!inner(user_id)")
        .in_("status", ["Queued", "In Progress"])
        .eq("projects.user_id", user_id)
        .execute()
    )
    offending_rows = [row for row in _coerce_rows(result) if not _is_live_test_task(row)]
    if offending_rows:
        task_ids = ", ".join(str(row.get("id")) for row in offending_rows if row.get("id"))
        raise UnexpectedUserWorkError(
            "Unexpected queued or in-progress non-live-test work exists for this user: "
            f"{task_ids}"
        )


def close_stale_live_test_tasks(db, user_id: str) -> int:
    """Mark live-test queued/in-progress rows failed before starting a new run."""
    result = (
        db.supabase.table("tasks")
        .select("id, status, params, project_id, projects!inner(user_id)")
        .in_("status", ["Queued", "In Progress"])
        .eq("projects.user_id", user_id)
        .execute()
    )
    live_test_ids = [
        str(row["id"])
        for row in _coerce_rows(result)
        if row.get("id") and _is_live_test_task(row)
    ]
    if not live_test_ids:
        return 0

    (
        db.supabase.table("tasks")
        .update(
            {
                "status": "Failed",
                "error_message": "closed stale live-test task before new live-test run",
            }
        )
        .in_("id", live_test_ids)
        .execute()
    )
    return len(live_test_ids)


def get_or_create_live_test_project(db, user_id: str) -> str:
    """Return the dedicated live-test project ID for the target user."""
    existing = (
        db.supabase.table("projects")
        .select("id, created_at")
        .eq("user_id", user_id)
        .eq("name", LIVE_TEST_PROJECT_NAME)
        .order("created_at")
        .execute()
    )
    rows = _coerce_rows(existing)
    if rows:
        project_id = rows[0].get("id")
        if not project_id:
            raise RuntimeError("Existing live-test project row is missing id")
        return str(project_id)

    created = (
        db.supabase.table("projects")
        .insert({"user_id": user_id, "name": LIVE_TEST_PROJECT_NAME})
        .execute()
    )
    created_rows = _coerce_rows(created)
    if len(created_rows) != 1 or not created_rows[0].get("id"):
        raise RuntimeError("Failed to create live-test project")
    return str(created_rows[0]["id"])


__all__ = [
    "LIVE_TEST_PROJECT_NAME",
    "UnexpectedUserWorkError",
    "assert_user_queue_clean",
    "close_stale_live_test_tasks",
    "get_or_create_live_test_project",
]
