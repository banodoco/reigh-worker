"""Termination safety wrappers for live RunPod tests."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Iterable

import scripts.live_test as live_test_pkg

LIVE_TEST_FRESH_POD_PREFIX = "reigh-live-test-fresh-"
DEFAULT_STALE_LIVE_TEST_POD_AGE_SEC = 6 * 60 * 60
_FRESH_POD_NAME_RE = re.compile(r"^reigh-live-test-fresh-(\d{8})t(\d{6})z$")


def guarded_terminate(pod_id: str | None, api_key: str | None, *, no_terminate: bool) -> bool:
    """Terminate the pod only when neither the CLI flag nor env opt-out is set."""
    if not pod_id or not api_key:
        return False
    if no_terminate or os.getenv("REIGH_LIVE_TEST_NO_TERMINATE") == "1":
        return False
    try:
        live_test_pkg.terminate_pod(pod_id, api_key)
    except Exception as exc:
        if "pod not found" not in str(exc).lower():
            raise
        return False
    return True


@dataclass(frozen=True)
class StalePodCleanupResult:
    inspected: int
    stale: tuple[str, ...]
    terminated: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]


def prune_stale_live_test_pods(
    api_key: str | None,
    *,
    max_age_seconds: int | None = None,
    prefix: str = LIVE_TEST_FRESH_POD_PREFIX,
    list_pods_fn: Callable[[str, str], Awaitable[Iterable[object]]] | None = None,
    terminate_fn: Callable[[str, str], Awaitable[None]] | None = None,
    now: datetime | None = None,
) -> StalePodCleanupResult:
    """Terminate old Reigh live-test pods that survived a previous interrupted run."""
    if not api_key:
        return StalePodCleanupResult(0, (), (), ())
    if os.getenv("REIGH_LIVE_TEST_SKIP_STALE_POD_CLEANUP") == "1":
        return StalePodCleanupResult(0, (), (), ())
    age_seconds = max_age_seconds
    if age_seconds is None:
        age_seconds = int(os.getenv("REIGH_LIVE_TEST_STALE_POD_AGE_SEC", str(DEFAULT_STALE_LIVE_TEST_POD_AGE_SEC)))
    return asyncio.run(
        _prune_stale_live_test_pods_async(
            api_key,
            max_age_seconds=age_seconds,
            prefix=prefix,
            list_pods_fn=list_pods_fn,
            terminate_fn=terminate_fn,
            now=now,
        )
    )


async def _prune_stale_live_test_pods_async(
    api_key: str,
    *,
    max_age_seconds: int,
    prefix: str,
    list_pods_fn: Callable[[str, str], Awaitable[Iterable[object]]] | None,
    terminate_fn: Callable[[str, str], Awaitable[None]] | None,
    now: datetime | None,
) -> StalePodCleanupResult:
    list_pods = list_pods_fn or _list_pods
    terminate = terminate_fn or _terminate
    current_time = now or datetime.now(timezone.utc)
    pods = list(await list_pods(api_key, prefix))
    stale = tuple(_pod_id(pod) for pod in pods if _pod_id(pod) and _pod_age_seconds(pod, current_time) >= max_age_seconds)
    terminated: list[str] = []
    failed: list[tuple[str, str]] = []
    for pod_id in stale:
        try:
            await terminate(api_key, pod_id)
            terminated.append(pod_id)
        except Exception as exc:
            failed.append((pod_id, str(exc)))
    return StalePodCleanupResult(len(pods), stale, tuple(terminated), tuple(failed))


async def _list_pods(api_key: str, prefix: str) -> Iterable[object]:
    from runpod_lifecycle import discovery

    return await discovery.list_pods(api_key, name_prefix=prefix)


async def _terminate(api_key: str, pod_id: str) -> None:
    from runpod_lifecycle import discovery

    await discovery.terminate(pod_id, api_key)


def _pod_id(pod: object) -> str:
    return str(getattr(pod, "id", "") or "")


def _pod_age_seconds(pod: object, now: datetime) -> int:
    uptime = getattr(pod, "uptime_seconds", None)
    if isinstance(uptime, int) and uptime >= 0:
        return uptime
    name_age = _age_from_live_test_name(getattr(pod, "name", None), now)
    if name_age is not None:
        return name_age
    created_age = _age_from_iso_timestamp(getattr(pod, "created_at", None), now)
    return created_age if created_age is not None else 0


def _age_from_live_test_name(name: str | None, now: datetime) -> int | None:
    if not name:
        return None
    match = _FRESH_POD_NAME_RE.fullmatch(name)
    if not match:
        return None
    created = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return max(0, int((now - created).total_seconds()))


def _age_from_iso_timestamp(value: str | None, now: datetime) -> int | None:
    if not value:
        return None
    try:
        created = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0, int((now - created).total_seconds()))


__all__ = [
    "DEFAULT_STALE_LIVE_TEST_POD_AGE_SEC",
    "LIVE_TEST_FRESH_POD_PREFIX",
    "StalePodCleanupResult",
    "guarded_terminate",
    "prune_stale_live_test_pods",
]
