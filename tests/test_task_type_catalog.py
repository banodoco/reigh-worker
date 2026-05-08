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


def test_known_drift_cases_now_resolve_to_wgp_output():
    for task_type in (
        "qwen_image",
        "qwen_image_2512",
        "qwen_image_hires",
        "wan_2_2_i2v",
        "animate_character",
        "image-upscale",
        "image_upscale",
        "video_enhance",
        "flux_klein_edit",
    ):
        assert is_direct_queue_task(task_type)
        assert is_wgp_task(task_type)
    assert is_wgp_task("inpaint_frames")
    assert not is_direct_queue_task("inpaint_frames")


def test_qwen_image_catalog_models_remain_distinct():
    assert get_default_model("qwen_image") == "qwen_image_20B"
    assert get_default_model("qwen_image_2512") == "qwen_image_2512_20B"


def test_default_model_is_projected_from_catalog():
    for task_type, meta in TASK_TYPE_CATALOG.items():
        assert TASK_TYPE_TO_MODEL[task_type] == meta.default_model
        assert get_default_model(task_type) == meta.default_model


def test_catalog_behavior_helpers_match_metadata_contracts():
    assert allows_empty_prompt("qwen_image_edit") is True
    assert allows_empty_prompt("z_image_turbo_i2i") is True
    assert allows_empty_prompt("t2v") is False

    assert forced_video_length_for_task("wan_2_2_t2i") == 1
    assert forced_video_length_for_task("t2v") is None
