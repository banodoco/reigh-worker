"""Parity test for public.derive_route_key.

Asserts the live Postgres function returns the expected route_key for each of
the six route families plus an orchestrator parent. Establishes a single
source of truth between the worker live-test write path and the DB trigger
that gates ``tasks.status='Queued'`` inserts.

Skipped when SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not in the
environment so unit-test runs without DB credentials remain green.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


pytestmark = pytest.mark.skipif(
    not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY")),
    reason="SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY required for derive_route_key parity test",
)


# (case_label, task_type, params, expected_route_key)
DERIVE_PARITY_CASES: tuple[tuple[str, str, dict, str], ...] = (
    (
        "wan22_i2v",
        "travel_segment",
        {"model_name": "wan_2_2_i2v"},
        "travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default",
    ),
    (
        "wan22_vace",
        "travel_segment",
        {"model_name": "wan_2_2_vace_lightning_baseline_2_2_2", "video_source": "anchor.mp4"},
        "travel_segment__model-wan22_vace__guidance-none__continuity-video_source__profile-default",
    ),
    (
        "ltx2",
        "travel_segment",
        {"model_name": "ltx2_22B"},
        "travel_segment__model-ltx2__guidance-none__continuity-first_last__profile-default",
    ),
    (
        "ltx2_distilled",
        "travel_segment",
        {"model_name": "ltx2_22B_distilled_1_1"},
        "travel_segment__model-ltx2_distilled__guidance-none__continuity-first_last__profile-default",
    ),
    (
        "qwen",
        "individual_travel_segment",
        {"model_name": "qwen_image"},
        "individual_travel_segment__model-qwen__guidance-none__continuity-first_last__profile-default",
    ),
    (
        "z_image",
        "z_image",
        {},
        "z_image_turbo",
    ),
    (
        "orchestrator_parent",
        "travel_orchestrator",
        {},
        "travel_orchestrator",
    ),
)


@pytest.fixture(scope="module")
def supabase_client():
    from scripts.live_test import config

    return config.DatabaseClient().supabase


@pytest.mark.parametrize(
    "case_label,task_type,params,expected_route_key",
    DERIVE_PARITY_CASES,
    ids=[entry[0] for entry in DERIVE_PARITY_CASES],
)
def test_derive_route_key_matches_hand_table(
    supabase_client,
    case_label: str,
    task_type: str,
    params: dict,
    expected_route_key: str,
) -> None:
    response = supabase_client.rpc(
        "derive_route_key",
        {"p_task_type": task_type, "p_params": params},
    ).execute()
    derived = getattr(response, "data", None)
    assert derived == expected_route_key, (
        f"derive_route_key parity drift for {case_label!r} "
        f"(task_type={task_type!r}, params={params!r}): "
        f"db returned {derived!r}, hand-table expects {expected_route_key!r}"
    )
