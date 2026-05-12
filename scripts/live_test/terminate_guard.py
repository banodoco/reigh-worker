"""Termination safety wrappers for live RunPod tests."""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Awaitable, Callable, Iterable

import scripts.live_test as live_test_pkg
from runpod_lifecycle.guard import (
    StalePodCleanupResult,
    prune_pods_by_prefix,
)

# All three prefixes follow the same `(\d{8})t(\d{6})z` lowercase timestamp
# convention because the live-test pod-create call lowercases the strftime
# output before constructing the pod name (variant_fresh.py uses
# `_timestamp_label().lower()`); the builder and prebuilt variants follow the
# same convention so the regex tuple catches all three uniformly.
LIVE_TEST_POD_PREFIXES: tuple[str, ...] = (
    "reigh-live-test-fresh-",
    "reigh-livetest-prebuilt-",
    "reigh-livetest-builder-",
)
LIVE_TEST_FRESH_POD_PREFIX = LIVE_TEST_POD_PREFIXES[0]
DEFAULT_STALE_LIVE_TEST_POD_AGE_SEC = 6 * 60 * 60

_LIVE_TEST_POD_NAME_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"^{re.escape(prefix)}(\d{{8}})t(\d{{6}})z$")
    for prefix in LIVE_TEST_POD_PREFIXES
)
_FRESH_POD_NAME_RE = _LIVE_TEST_POD_NAME_RES[0]


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


def prune_stale_live_test_pods(
    api_key: str | None,
    *,
    max_age_seconds: int | None = None,
    prefix: str | None = None,
    list_pods_fn: Callable[[str, str], Awaitable[Iterable[object]]] | None = None,
    terminate_fn: Callable[[str, str], Awaitable[None]] | None = None,
    now: datetime | None = None,
) -> StalePodCleanupResult:
    """Terminate old Reigh live-test pods that survived a previous interrupted run.

    Delegates to ``runpod_lifecycle.guard.prune_pods_by_prefix``. The optional
    *prefix* parameter remains for backward compatibility with callers that
    want to constrain pruning to a single prefix; when omitted, all three
    live-test prefixes (fresh / prebuilt / builder) are pruned.
    """
    if not api_key:
        return StalePodCleanupResult(0, (), (), ())
    age_seconds = max_age_seconds
    if age_seconds is None:
        age_seconds = int(
            os.getenv("REIGH_LIVE_TEST_STALE_POD_AGE_SEC", str(DEFAULT_STALE_LIVE_TEST_POD_AGE_SEC))
        )
    prefixes: tuple[str, ...] = (prefix,) if prefix else LIVE_TEST_POD_PREFIXES
    return prune_pods_by_prefix(
        prefixes,
        api_key,
        stale_age_sec=age_seconds,
        list_pods_fn=list_pods_fn,
        terminate_fn=terminate_fn,
        now=now,
    )


__all__ = [
    "DEFAULT_STALE_LIVE_TEST_POD_AGE_SEC",
    "LIVE_TEST_FRESH_POD_PREFIX",
    "LIVE_TEST_POD_PREFIXES",
    "StalePodCleanupResult",
    "guarded_terminate",
    "prune_stale_live_test_pods",
]
