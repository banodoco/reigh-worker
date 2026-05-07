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


def ensure_user_cloud_generation_enabled(db, user_id: str) -> bool:
    """Ensure the live-test user can be claimed by cloud workers."""
    result = db.supabase.table("users").select("id, settings").eq("id", user_id).execute()
    rows = _coerce_rows(result)
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one live-test user row for {user_id}, found {len(rows)}")

    settings = rows[0].get("settings")
    if not isinstance(settings, dict):
        settings = {}
    ui = settings.get("ui")
    if not isinstance(ui, dict):
        ui = {}
    generation_methods = ui.get("generationMethods")
    if not isinstance(generation_methods, dict):
        generation_methods = {}

    if generation_methods.get("inCloud") is True:
        return False

    generation_methods["inCloud"] = True
    ui["generationMethods"] = generation_methods
    settings["ui"] = ui
    db.supabase.table("users").update({"settings": settings}).eq("id", user_id).execute()
    return True


def ensure_live_test_route_selectors(
    db,
    selector_namespace: str,
    route_keys: list[str],
    *,
    backend: str,
    fallback_selectors: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Clone production route selectors into an isolated live-test namespace.

    Branch-only route promotions may not have production selector rows yet. In
    that case the live matrix can provide a fallback selector contract so the
    isolated namespace remains testable without mutating production selectors.
    """
    unique_route_keys = sorted({route_key for route_key in route_keys if route_key})
    if backend != "vibecomfy" or selector_namespace == "production" or not unique_route_keys:
        return 0

    existing = (
        db.supabase.table("route_backend_selectors")
        .select("route_key")
        .eq("selector_namespace", selector_namespace)
        .in_("route_key", unique_route_keys)
        .execute()
    )
    existing_keys = {str(row.get("route_key")) for row in _coerce_rows(existing) if row.get("route_key")}
    missing_keys = [route_key for route_key in unique_route_keys if route_key not in existing_keys]
    created = 0

    production = (
        db.supabase.table("route_backend_selectors")
        .select("route_key, selected_backend, selector_version, enabled, expires_at, min_worker_version, reason, metadata")
        .eq("selector_namespace", "production")
        .in_("route_key", missing_keys or ["__none__"])
        .execute()
    )
    production_by_key = {str(row.get("route_key")): row for row in _coerce_rows(production) if row.get("route_key")}
    still_missing = [route_key for route_key in missing_keys if route_key not in production_by_key]
    fallback_selectors = fallback_selectors or {}
    unhandled_missing = [route_key for route_key in still_missing if route_key not in fallback_selectors]
    if unhandled_missing:
        raise RuntimeError(
            "Cannot isolate live-test selector namespace; production route selectors are missing for: "
            + ", ".join(unhandled_missing)
        )

    for route_key in missing_keys:
        source = production_by_key.get(route_key)
        fallback = fallback_selectors.get(route_key, {})
        if source is None:
            source = {
                "route_key": route_key,
                "selected_backend": fallback.get("selected_backend") or backend,
                "selector_version": fallback.get("selector_version") or 1,
                "enabled": True,
                "expires_at": None,
                "min_worker_version": None,
                "reason": f"live-test isolation synthesized from matrix for {route_key}",
                "metadata": {
                    "support_state": fallback.get("support_state"),
                    "selected_template_id": fallback.get("selected_template_id"),
                },
            }
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        payload = {
            "selector_namespace": selector_namespace,
            "route_key": route_key,
            "selected_backend": source.get("selected_backend") or backend,
            "selector_version": source.get("selector_version") or 1,
            "enabled": source.get("enabled") is not False,
            "expires_at": source.get("expires_at"),
            "min_worker_version": source.get("min_worker_version"),
            "reason": str(source.get("reason") or f"live-test isolation cloned from production for {route_key}"),
            "metadata": {
                **metadata,
                "live_test": True,
                "source_selector_namespace": "production" if route_key in production_by_key else "matrix",
            },
        }
        db.supabase.table("route_backend_selectors").insert(payload).execute()
        created += 1

    fallback_capability_keys = [
        route_key
        for route_key in unique_route_keys
        if route_key in fallback_selectors
    ]
    if fallback_capability_keys:
        existing_capabilities = (
            db.supabase.table("route_backend_capabilities")
            .select("route_key")
            .eq("backend", backend)
            .in_("route_key", fallback_capability_keys)
            .execute()
        )
        existing_capability_keys = {
            str(row.get("route_key"))
            for row in _coerce_rows(existing_capabilities)
            if row.get("route_key")
        }
        for route_key in fallback_capability_keys:
            if route_key in existing_capability_keys:
                continue
            fallback = fallback_selectors.get(route_key, {})
            db.supabase.table("route_backend_capabilities").insert(
                {
                    "backend": backend,
                    "route_key": route_key,
                    "supports_route": True,
                    "supports_missing_selector": False,
                    "enabled": True,
                    "capability_version": 1,
                    "metadata": {
                        "live_test": True,
                        "source": "matrix",
                        "support_state": fallback.get("support_state"),
                        "selected_template_id": fallback.get("selected_template_id"),
                    },
                }
            ).execute()
            created += 1
    return created


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
    "ensure_user_cloud_generation_enabled",
    "get_or_create_live_test_project",
]
