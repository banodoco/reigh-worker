"""Regression tests for task-type catalog consistency."""

from source.task_handlers.tasks.task_types import (
    DIRECT_QUEUE_TASK_TYPES,
    TASK_TYPE_CATALOG,
    TASK_TYPE_TO_MODEL,
    WGP_TASK_TYPES,
    allows_empty_prompt,
    forced_video_length_for_task,
    get_default_model,
    is_direct_queue_task,
    is_wgp_task,
)
from source.task_handlers.tasks.template_routing import RETIRED_ASTRID_DIRECT_ROUTE_KEYS


def test_derived_sets_match_catalog_flags():
    expected_direct = {
        task_type
        for task_type, meta in TASK_TYPE_CATALOG.items()
        if meta.is_direct_queue
    }
    expected_wgp = {
        task_type
        for task_type, meta in TASK_TYPE_CATALOG.items()
        if meta.is_wgp_output
    }

    assert DIRECT_QUEUE_TASK_TYPES == expected_direct
    assert WGP_TASK_TYPES == expected_wgp


def test_direct_queue_tasks_always_have_wgp_output_routing():
    missing = sorted(
        task_type
        for task_type, meta in TASK_TYPE_CATALOG.items()
        if meta.is_direct_queue and not meta.is_wgp_output
    )
    assert not missing, f"Direct queue tasks missing output routing metadata: {missing}"


def test_retired_direct_families_have_no_worker_catalog_owner():
    for task_type in RETIRED_ASTRID_DIRECT_ROUTE_KEYS:
        assert task_type not in TASK_TYPE_CATALOG
        assert not is_wgp_task(task_type)
        assert not is_direct_queue_task(task_type)

    assert is_direct_queue_task("wan_2_2_i2v")
    assert is_wgp_task("wan_2_2_i2v")
    assert "qwen_image_hires" not in TASK_TYPE_CATALOG
    assert "inpaint_frames" not in TASK_TYPE_CATALOG
    assert not is_wgp_task("qwen_image_hires")
    assert not is_direct_queue_task("qwen_image_hires")
    assert not is_wgp_task("inpaint_frames")
    assert not is_direct_queue_task("inpaint_frames")


def test_legacy_catalog_only_task_types_are_removed():
    for task_type in (
        "flux",
        "hunyuan",
        "i2v",
        "i2v_22",
        "ltx2",
        "ltxv",
        "t2v",
        "t2v_22",
        "vace",
        "vace_21",
        "vace_22",
    ):
        assert task_type not in TASK_TYPE_CATALOG
        assert not is_wgp_task(task_type)
        assert not is_direct_queue_task(task_type)


def test_retired_direct_families_use_unknown_task_fallback_model():
    assert get_default_model("qwen_image") == "t2v"
    assert get_default_model("qwen_image_2512") == "t2v"


def test_default_model_is_projected_from_catalog():
    for task_type, meta in TASK_TYPE_CATALOG.items():
        assert TASK_TYPE_TO_MODEL[task_type] == meta.default_model
        assert get_default_model(task_type) == meta.default_model


def test_catalog_behavior_helpers_match_metadata_contracts():
    assert allows_empty_prompt("qwen_image_edit") is False
    assert allows_empty_prompt("z_image_turbo_i2i") is False
    assert allows_empty_prompt("t2v") is False

    assert forced_video_length_for_task("wan_2_2_t2i") == 1
    assert forced_video_length_for_task("t2v") is None
