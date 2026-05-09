"""Small Supabase DB adapter for the live-test harness.

This keeps local smoke/fresh runs independent from the orchestrator checkout.
The harness only needs a Supabase table client plus two worker-row helpers.
"""

from __future__ import annotations

import os
import logging
from typing import Any

from supabase import create_client


for _logger_name in ("httpx", "httpcore", "postgrest"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)


class DatabaseClient:
    """Supabase-backed subset used by live-test preflight and fresh runs."""

    def __init__(self, *, supabase_url: str | None = None, service_role_key: str | None = None):
        url = supabase_url or os.environ.get("SUPABASE_URL")
        key = service_role_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
        self.supabase = create_client(url, key)

    async def create_worker_record(
        self,
        worker_id: str,
        instance_type: str,
        runpod_id: str | None = None,
    ) -> bool:
        payload: dict[str, Any] = {
            "id": worker_id,
            "instance_type": instance_type,
            "status": "inactive",
        }
        if runpod_id:
            payload["metadata"] = {"runpod_id": runpod_id}
        result = self.supabase.table("workers").insert(payload).execute()
        return bool(getattr(result, "data", None))

    async def update_worker_status(self, worker_id: str, status: str, metadata: dict[str, Any]) -> bool:
        result = (
            self.supabase.table("workers")
            .update({"status": status, "metadata": metadata})
            .eq("id", worker_id)
            .execute()
        )
        return bool(getattr(result, "data", None))


__all__ = ["DatabaseClient"]
