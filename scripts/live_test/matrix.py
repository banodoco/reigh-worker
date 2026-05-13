"""Live-test task matrix definitions and execution helpers."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from scripts.create_test_task import TEST_TASKS
from scripts.live_test import config
from scripts.live_test.completion_poller import TaskResult, poll_until_complete
from scripts.live_test.task_spoofer import insert_spoof_task, load_fixture


TRAVEL_WAN_FIXTURE_KEY = "travel_orchestrator_wan2_1seg"
TRAVEL_LTX_FIXTURE_KEY = "travel_orchestrator_ltx"
Z_IMAGE_TURBO_FIXTURE_KEY = "z_image_turbo"
WAN_2_2_T2I_FIXTURE_KEY = "wan_2_2_t2i"
IMAGE_INPAINT_FIXTURE_KEY = "image_inpaint"
ANNOTATED_IMAGE_EDIT_FIXTURE_KEY = "annotated_image_edit"
JOIN_CLIPS_ORCHESTRATOR_FIXTURE_KEY = "join_clips_orchestrator"
JOIN_CLIPS_SEGMENT_VACE_FIXTURE_KEY = "join_clips_segment_wan22_vace"
EDIT_VIDEO_ORCHESTRATOR_FIXTURE_KEY = "edit_video_orchestrator"
TRAVEL_STITCH_FIXTURE_KEY = "travel_stitch"

LIVE_TEST_VIDEO_URL = (
    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/"
    "guidance-videos/onboarding/structure_video_optimized.mp4"
)


@dataclass(frozen=True)
class RouteRuntimeOptions:
    selected_backend: str = "wgp"
    selector_namespace: str = "production"
    selector_version: str | None = None
    worker_contract_version: int = 1
    selected_profile: str = "default"


@dataclass(frozen=True)
class MatrixCase:
    name: str
    task_type: str
    fixture_key: str
    param_overrides: dict[str, Any] = field(default_factory=dict)
    timeout_sec: int = 0
    route_key: str | None = None
    support_state: str | None = None
    selected_template_id: str | None = None
    route_runtime: RouteRuntimeOptions = field(default_factory=RouteRuntimeOptions)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _anchor_pair() -> list[str]:
    return [config.ANCHOR_IMAGE_A_URL, config.ANCHOR_IMAGE_B_URL]


def _build_wan_travel_fixture() -> dict[str, Any]:
    template = copy.deepcopy(TEST_TASKS["travel_orchestrator"])
    orchestrator_details = template["params"]["orchestrator_details"]
    orchestrator_details["input_image_paths_resolved"] = _anchor_pair()
    orchestrator_details["input_image_generation_ids"] = [
        "live-test-anchor-a",
        "live-test-anchor-b",
    ]
    orchestrator_details["num_new_segments_to_generate"] = 1
    orchestrator_details["segment_frames_expanded"] = [65]
    orchestrator_details["frame_overlap_expanded"] = [10]
    orchestrator_details["base_prompt"] = "A smooth cinematic move bridging two high-contrast scenes"
    orchestrator_details["base_prompts_expanded"] = [orchestrator_details["base_prompt"]]
    orchestrator_details["negative_prompts_expanded"] = [""]
    orchestrator_details["enhanced_prompts_expanded"] = [""]
    orchestrator_details["enhance_prompt"] = False
    return template


def _build_ltx_travel_fixture() -> dict[str, Any]:
    template = _build_wan_travel_fixture()
    orchestrator_details = template["params"]["orchestrator_details"]
    orchestrator_details["model_name"] = config.LTX_MODEL_ID
    orchestrator_details["model_type"] = "ltx2"
    orchestrator_details["parsed_resolution_wh"] = "768x512"
    orchestrator_details["steps"] = 8
    orchestrator_details["fps_helpers"] = 24
    orchestrator_details["guidance_scale"] = 3.0
    orchestrator_details["num_inference_steps"] = 8
    orchestrator_details["flow_shift"] = 5
    orchestrator_details["enhance_prompt"] = False
    orchestrator_details["frame_overlap_expanded"] = [25]
    orchestrator_details["travel_guidance"] = {"kind": "none"}
    orchestrator_details.pop("phase_config", None)
    orchestrator_details.pop("selected_phase_preset_id", None)
    return template


def _build_z_image_turbo_fixture() -> dict[str, Any]:
    return {
        "task_type": "z_image_turbo",
        "status": "Queued",
        "params": {
            "prompt": "A compact red cube on a clean white tabletop, product-photo lighting.",
            "resolution": "1024x1024",
            "seed": 1732,
            "steps": 4,
            "num_inference_steps": 4,
            "model": "z_image_turbo",
        },
    }


def _build_wan_2_2_t2i_fixture() -> dict[str, Any]:
    return {
        "task_type": "wan_2_2_t2i",
        "status": "Queued",
        "params": {
            "prompt": "A compact cinematic still of a red cube on a clean white tabletop.",
            "resolution": "832x480",
            "seed": 20260507,
            "num_inference_steps": 4,
            "guidance_scale": 1,
        },
    }


def _build_wan_2_2_i2v_fixture() -> dict[str, Any]:
    return {
        "task_type": "wan_2_2_i2v",
        "status": "Queued",
        "params": {
            "prompt": "A slow cinematic push-in on the subject with natural motion.",
            "negative_prompt": "flicker, blur, warped anatomy, heavy compression artifacts",
            "image_url": config.ANCHOR_IMAGE_A_URL,
            "start_image_url": config.ANCHOR_IMAGE_A_URL,
            "resolution": "832x480",
            "parsed_resolution_wh": "832x480",
            "seed": 20260508,
            "num_frames": 81,
            "video_length": 81,
            "fps": 16,
            "num_inference_steps": 4,
            "steps": 4,
            "high_noise_end_step": 2,
        },
    }


def _build_wan_2_2_i2v_first_last_fixture() -> dict[str, Any]:
    fixture = _build_wan_2_2_i2v_fixture()
    fixture["task_type"] = "travel_segment"
    params = fixture["params"]
    params.update(
        {
            "model": "wan_2_2_i2v",
            "model_name": "wan_2_2_i2v",
            "model_family": "wan22_i2v",
            "start_image_url": config.ANCHOR_IMAGE_A_URL,
            "end_image_url": config.ANCHOR_IMAGE_B_URL,
            "input_image_paths_resolved": _anchor_pair(),
            "continuity_case": "first_last",
            "orchestrator_details": {
                "model": "wan_2_2_i2v",
                "model_name": "wan_2_2_i2v",
                "model_family": "wan22_i2v",
                "parsed_resolution_wh": "832x480",
                "input_image_paths_resolved": _anchor_pair(),
                "continuity_case": "first_last",
                "base_prompt": params.get("prompt", ""),
                "base_prompts_expanded": [params.get("prompt", "")],
                "negative_prompts_expanded": [params.get("negative_prompt", "")],
                "segment_frames_expanded": [int(params.get("num_frames", 81))],
                "frame_overlap_expanded": [0],
                "num_new_segments_to_generate": 1,
                "enhance_prompt": False,
                "enhanced_prompts_expanded": [""],
            },
        }
    )
    return fixture


def _build_qwen_image_fixture(task_type: str = "qwen_image") -> dict[str, Any]:
    return {
        "task_type": task_type,
        "status": "Queued",
        "params": {
            "prompt": "A compact red cube on a clean white tabletop, product-photo lighting.",
            "resolution": "1024x1024",
            "seed": 20260508,
            "num_inference_steps": 4,
            "steps": 4,
        },
    }


def _build_qwen_image_edit_fixture(task_type: str = "qwen_image_edit") -> dict[str, Any]:
    return {
        "task_type": task_type,
        "status": "Queued",
        "params": {
            "prompt": "Make the image crisp and editorial while preserving the subject.",
            "image_url": config.ANCHOR_IMAGE_A_URL,
            "image": config.ANCHOR_IMAGE_A_URL,
            "resolution": "1024x1024",
            "seed": 20260508,
            "num_inference_steps": 4,
            "steps": 4,
        },
    }


def _build_qwen_image_style_fixture() -> dict[str, Any]:
    return {
        "task_type": "qwen_image_style",
        "status": "Queued",
        "params": {
            "prompt": "Apply the style reference while preserving the subject.",
            "style_reference_image": config.ANCHOR_IMAGE_A_URL,
            "subject_reference_image": config.ANCHOR_IMAGE_B_URL,
            "resolution": "1024x1024",
            "seed": 20260508,
            "num_inference_steps": 4,
            "steps": 4,
        },
    }


def _build_z_image_turbo_i2i_fixture() -> dict[str, Any]:
    fixture = _build_z_image_turbo_fixture()
    fixture["task_type"] = "z_image_turbo_i2i"
    fixture["params"].update({"image_url": config.ANCHOR_IMAGE_A_URL, "image": config.ANCHOR_IMAGE_A_URL})
    return fixture


def _build_wan_i2v_individual_fixture() -> dict[str, Any]:
    return {
        "task_type": "individual_travel_segment",
        "status": "Queued",
        "params": {
            "prompt": "A coherent short camera move between two anchor frames.",
            "negative_prompt": "fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted",
            "model": "wan_2_2_i2v",
            "model_name": "wan_2_2_i2v",
            "model_family": "wan22_i2v",
            "resolution": "832x480",
            "parsed_resolution_wh": "832x480",
            "fps": 16,
            "num_frames": 81,
            "video_length": 81,
            "seed": 20260508,
            "start_image_url": config.ANCHOR_IMAGE_A_URL,
            "end_image_url": config.ANCHOR_IMAGE_B_URL,
            "input_image_paths_resolved": _anchor_pair(),
            "orchestrator_details": {
                "model": "wan_2_2_i2v",
                "model_name": "wan_2_2_i2v",
                "model_family": "wan22_i2v",
                "parsed_resolution_wh": "832x480",
                "input_image_paths_resolved": _anchor_pair(),
            },
            "individual_segment_params": {
                "start_image_url": config.ANCHOR_IMAGE_A_URL,
                "end_image_url": config.ANCHOR_IMAGE_B_URL,
                "input_image_paths_resolved": _anchor_pair(),
            },
        },
    }


def _build_animate_character_fixture() -> dict[str, Any]:
    return {
        "task_type": "animate_character",
        "status": "Queued",
        "params": {
            "prompt": "Animate the character following the motion video while preserving identity.",
            "negative_prompt": "flicker, blur, distorted face, extra limbs",
            "character_image_url": config.ANCHOR_IMAGE_A_URL,
            "reference_image_url": config.ANCHOR_IMAGE_A_URL,
            "motion_video_url": LIVE_TEST_VIDEO_URL,
            "resolution": "832x480",
            "parsed_resolution_wh": "832x480",
            "seed": 20260508,
            "num_frames": 49,
            "video_length": 49,
            "fps": 16,
            "num_inference_steps": 4,
            "steps": 4,
        },
    }


def _build_image_upscale_fixture() -> dict[str, Any]:
    return {
        "task_type": "image-upscale",
        "status": "Queued",
        "params": {
            "image_url": config.ANCHOR_IMAGE_A_URL,
            "image": config.ANCHOR_IMAGE_A_URL,
            "scale_factor": 2,
            "upscale_factor": 2,
        },
    }


def _build_video_enhance_fixture() -> dict[str, Any]:
    return {
        "task_type": "video_enhance",
        "status": "Queued",
        "params": {
            "video_url": LIVE_TEST_VIDEO_URL,
            "video": LIVE_TEST_VIDEO_URL,
            "fps": 16,
            "enable_interpolation": True,
            "enable_upscale": True,
            "interpolation": {"num_frames": 1},
            "upscale": {"upscale_factor": 1.25},
        },
    }


def _build_flux_klein_edit_fixture() -> dict[str, Any]:
    return {
        "task_type": "flux_klein_edit",
        "status": "Queued",
        "params": {
            "prompt": "Turn the scene into a crisp editorial product photograph while preserving layout.",
            "image_url": config.ANCHOR_IMAGE_A_URL,
            "image": config.ANCHOR_IMAGE_A_URL,
            "resolution": "1024x1024",
            "seed": 20260508,
            "num_inference_steps": 4,
            "steps": 4,
            "klein_model": "flux-klein-4b",
        },
    }


def _build_masked_qwen_fixture(task_type: str) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "status": "Queued",
        "params": {
            "prompt": "Repair the masked area while preserving the scene.",
            "image_url": config.ANCHOR_IMAGE_A_URL,
            "image": config.ANCHOR_IMAGE_A_URL,
            "mask_url": config.ANCHOR_IMAGE_B_URL,
            "resolution": "1024x1024",
            "seed": 20260507,
            "num_inference_steps": 4,
        },
    }


def _build_join_clips_orchestrator_fixture() -> dict[str, Any]:
    return {
        "task_type": "join_clips_orchestrator",
        "status": "Queued",
        "params": {
            "orchestrator_details": {
                "run_id": "live-test-join",
                "clip_list": [
                    {"url": LIVE_TEST_VIDEO_URL, "name": "clip_a"},
                    {"url": LIVE_TEST_VIDEO_URL, "name": "clip_b"},
                ],
                "loop_first_clip": False,
                "context_frame_count": 4,
                "gap_frame_count": 8,
                "replace_mode": False,
                "prompt": "A simple coherent bridge between the clips.",
                "negative_prompt": "",
                "model": "wan_2_2_vace_lightning_baseline_2_2_2",
                "seed": 20260507,
                "resolution": "902x508",
                "fps": 16,
                "use_input_video_resolution": True,
                "use_input_video_fps": True,
                "use_parallel_joins": False,
                "skip_frame_validation": True,
                "enhance_prompt": False,
            },
        },
    }


def _build_join_clips_segment_vace_fixture() -> dict[str, Any]:
    model_name = "wan_2_2_vace_lightning_baseline_2_2_2"
    return {
        "task_type": "join_clips_segment",
        "status": "Queued",
        "params": {
            "prompt": "A coherent short bridge between two matching clips.",
            "negative_prompt": "fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted",
            "model": model_name,
            "model_name": model_name,
            "model_family": "wan22_vace",
            "resolution": "832x480",
            "parsed_resolution_wh": "832x480",
            "fps": 16,
            "num_frames": 81,
            "video_length": 81,
            "seed": 20260508,
            "continuity_case": "join_bridge",
            "video_source": LIVE_TEST_VIDEO_URL,
            "start_image_url": config.ANCHOR_IMAGE_A_URL,
            "end_image_url": config.ANCHOR_IMAGE_B_URL,
            "input_image_paths_resolved": _anchor_pair(),
            "travel_guidance": {
                "kind": "vace",
                "videos": [{"url": LIVE_TEST_VIDEO_URL}],
            },
            "orchestrator_details": {
                "run_id": "live-test-join-segment",
                "orchestrator_task_id": "live-test-join-segment-parent",
                "clip_list": [
                    {"url": LIVE_TEST_VIDEO_URL, "name": "clip_a"},
                    {"url": LIVE_TEST_VIDEO_URL, "name": "clip_b"},
                ],
                "context_frame_count": 4,
                "gap_frame_count": 8,
                "model": model_name,
                "model_name": model_name,
                "model_family": "wan22_vace",
                "parsed_resolution_wh": "832x480",
                "fps": 16,
                "input_image_paths_resolved": _anchor_pair(),
                "travel_guidance": {
                    "kind": "vace",
                    "videos": [{"url": LIVE_TEST_VIDEO_URL}],
                },
            },
        },
    }


def _build_edit_video_orchestrator_fixture() -> dict[str, Any]:
    return {
        "task_type": "edit_video_orchestrator",
        "status": "Queued",
        "params": {
            "orchestrator_details": {
                "run_id": "live-test-edit-video",
                "source_video_url": LIVE_TEST_VIDEO_URL,
                "source_video_fps": 16,
                "source_video_total_frames": 80,
                "portions_to_regenerate": [
                    {
                        "start_frame": 16,
                        "end_frame": 24,
                        "prompt": "A clean transition through the edited region.",
                    }
                ],
                "context_frame_count": 4,
                "gap_frame_count": 8,
                "replace_mode": False,
                "prompt": "A simple coherent edit.",
                "negative_prompt": "",
                "model": "wan_2_2_vace_lightning_baseline_2_2_2",
                "seed": 20260507,
                "resolution": "902x508",
                "fps": 16,
                "use_input_video_resolution": True,
                "use_input_video_fps": True,
                "enhance_prompt": False,
            },
        },
    }


def _build_travel_stitch_fixture() -> dict[str, Any]:
    return {
        "task_type": "travel_stitch",
        "status": "Queued",
        "params": {
            "clip_urls": [LIVE_TEST_VIDEO_URL, LIVE_TEST_VIDEO_URL],
            "frame_overlap_settings_expanded": [4],
            "fps": 16,
            "crossfade_sharp_amt": 0.3,
        },
    }


def _ltx_first_last_overrides(
    *,
    mode: str | None,
    anchor_image_a: str,
    anchor_image_b: str,
) -> dict[str, Any]:
    travel_guidance = {"kind": "none"} if mode is None else {
        "kind": "ltx_control",
        "mode": mode,
        "videos": [{"url": LIVE_TEST_VIDEO_URL}],
        "strength": 0.5 if mode in {"pose", "depth", "canny"} else 1.0,
    }
    details = {
        "model_name": config.LTX_MODEL_ID,
        "model": config.LTX_MODEL_ID,
        "model_family": "ltx2_distilled",
        "run_id": "live-test-ltx-first-last",
        "orchestrator_task_id": "live-test-ltx-first-last-parent",
        "parsed_resolution_wh": "768x512",
        "segment_frames_expanded": [81],
        "num_new_segments_to_generate": 1,
        "base_prompts_expanded": ["A smooth cinematic move between two anchor frames"],
        "negative_prompts_expanded": ["blurry, oversaturated, pixelated, low resolution, grainy, distorted"],
        "frame_overlap_expanded": [0],
        "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
        "travel_guidance": travel_guidance,
        "fps_helpers": 24,
        "seed_base": 20260508,
        "continuity_case": "first_last",
    }
    overrides = {
        "segment_index": 0,
        "orchestrator_run_id": "live-test-ltx-first-last",
        "orchestrator_task_id_ref": "live-test-ltx-first-last-parent",
        "prompt": "A smooth cinematic move between two anchor frames.",
        "negative_prompt": "blurry, oversaturated, pixelated, low resolution, grainy, distorted",
        "model_name": config.LTX_MODEL_ID,
        "model": config.LTX_MODEL_ID,
        "model_family": "ltx2_distilled",
        "resolution": "768x512",
        "parsed_resolution_wh": "768x512",
        "fps": 24,
        "num_frames": 81,
        "video_length": 81,
        "seed": 20260508,
        "start_image_url": anchor_image_a,
        "end_image_url": anchor_image_b,
        "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
        "continuity_case": "first_last",
        "travel_guidance": travel_guidance,
        "orchestrator_details": details,
    }
    if mode is not None:
        overrides["control_video_url"] = LIVE_TEST_VIDEO_URL
        details["control_video_url"] = LIVE_TEST_VIDEO_URL
    return overrides


def resolve_case_fixture(case: MatrixCase) -> dict[str, Any]:
    if case.fixture_key == TRAVEL_WAN_FIXTURE_KEY:
        return _build_wan_travel_fixture()
    if case.fixture_key == TRAVEL_LTX_FIXTURE_KEY:
        return _build_ltx_travel_fixture()
    if case.fixture_key == Z_IMAGE_TURBO_FIXTURE_KEY:
        return _build_z_image_turbo_fixture()
    if case.fixture_key == WAN_2_2_T2I_FIXTURE_KEY:
        return _build_wan_2_2_t2i_fixture()
    if case.fixture_key == "wan_2_2_i2v":
        return _build_wan_2_2_i2v_fixture()
    if case.fixture_key == "wan_2_2_i2v_first_last":
        return _build_wan_2_2_i2v_first_last_fixture()
    if case.fixture_key == "qwen_image_basic":
        return _build_qwen_image_fixture("qwen_image")
    if case.fixture_key == "qwen_image_edit_basic":
        return _build_qwen_image_edit_fixture("qwen_image_edit")
    if case.fixture_key == "qwen_image_style_db_task":
        return _build_qwen_image_style_fixture()
    if case.fixture_key == "z_image_turbo_i2i_basic":
        return _build_z_image_turbo_i2i_fixture()
    if case.fixture_key == "wan22_i2v_individual_segment":
        return _build_wan_i2v_individual_fixture()
    if case.fixture_key == "animate_character":
        return _build_animate_character_fixture()
    if case.fixture_key == "image-upscale":
        return _build_image_upscale_fixture()
    if case.fixture_key == "video_enhance":
        return _build_video_enhance_fixture()
    if case.fixture_key == "flux_klein_edit":
        return _build_flux_klein_edit_fixture()
    if case.fixture_key == IMAGE_INPAINT_FIXTURE_KEY:
        return _build_masked_qwen_fixture("image_inpaint")
    if case.fixture_key == ANNOTATED_IMAGE_EDIT_FIXTURE_KEY:
        return _build_masked_qwen_fixture("annotated_image_edit")
    if case.fixture_key == JOIN_CLIPS_ORCHESTRATOR_FIXTURE_KEY:
        return _build_join_clips_orchestrator_fixture()
    if case.fixture_key == JOIN_CLIPS_SEGMENT_VACE_FIXTURE_KEY:
        return _build_join_clips_segment_vace_fixture()
    if case.fixture_key == EDIT_VIDEO_ORCHESTRATOR_FIXTURE_KEY:
        return _build_edit_video_orchestrator_fixture()
    if case.fixture_key == TRAVEL_STITCH_FIXTURE_KEY:
        return _build_travel_stitch_fixture()
    return load_fixture(case.fixture_key)


_DERIVE_ROUTE_KEY_CLIENT: Any = None


def _derive_route_key_client() -> Any:
    global _DERIVE_ROUTE_KEY_CLIENT
    if _DERIVE_ROUTE_KEY_CLIENT is None:
        _DERIVE_ROUTE_KEY_CLIENT = config.DatabaseClient().supabase
    return _DERIVE_ROUTE_KEY_CLIENT


def _derive_route_key_for_case(case: MatrixCase) -> str:
    """Resolve route_key via public.derive_route_key, the single source of truth.

    Trusts the DB function rather than the fixture-encoded ``case.route_key`` so
    a divergence between the fixture and the registered routes surfaces loudly
    instead of silently shipping an unclaimable task contract.
    """
    fixture_params: dict[str, Any]
    try:
        fixture = resolve_case_fixture(case) or {}
        fixture_params = dict(fixture.get("params") or {})
    except Exception:
        fixture_params = {}
    merged_params = _deep_merge(fixture_params, case.param_overrides)
    response = _derive_route_key_client().rpc(
        "derive_route_key",
        {"p_task_type": case.task_type, "p_params": merged_params},
    ).execute()
    derived = getattr(response, "data", None)
    if not isinstance(derived, str) or not derived:
        raise RuntimeError(
            "derive_route_key returned no route_key for case "
            f"{case.name!r} (task_type={case.task_type!r}); fixture must include "
            "a model_name resolvable via public.model_family_for_model"
        )
    return derived


def _route_contract(case: MatrixCase, task_marker: str) -> dict[str, Any] | None:
    if not case.route_key:
        return None

    derived_route_key = _derive_route_key_for_case(case)

    runtime = case.route_runtime
    snapshot = {
        "route_key": derived_route_key,
        "task_type": case.task_type,
        "selected_backend": runtime.selected_backend,
        "selector_namespace": runtime.selector_namespace,
        "selector_version": runtime.selector_version,
        "worker_contract_version": runtime.worker_contract_version,
        "selected_profile": runtime.selected_profile,
        "support_state": case.support_state,
        "selected_template_id": case.selected_template_id,
        "live_test_run_id": task_marker,
    }
    return {
        **snapshot,
        "route_selection_snapshot": snapshot,
    }


def build_case_params_overrides(
    case: MatrixCase,
    *,
    unique_suffix: str | None = None,
) -> dict[str, Any]:
    suffix = unique_suffix or uuid.uuid4().hex[:12]
    task_marker = f"live-test-{case.name}-{suffix}"
    runtime: dict[str, Any] = {"task_id": task_marker}

    if case.task_type == "travel_orchestrator":
        runtime["orchestrator_details"] = {
            "run_id": task_marker,
            "orchestrator_task_id": task_marker,
            "input_image_generation_ids": [
                f"{task_marker}-anchor-a",
                f"{task_marker}-anchor-b",
            ],
        }

    if case.task_type in {"join_clips_orchestrator", "edit_video_orchestrator"}:
        runtime["orchestrator_details"] = {
            "run_id": task_marker,
            "orchestrator_task_id": task_marker,
            "orchestrator_task_id_ref": task_marker,
        }

    if case.task_type == "join_clips_segment":
        runtime["orchestrator_details"] = {
            "run_id": task_marker,
            "orchestrator_task_id": f"{task_marker}-parent",
            "orchestrator_task_id_ref": f"{task_marker}-parent",
        }

    if case.task_type == "individual_travel_segment":
        runtime["orchestrator_details"] = {
            "orchestrator_task_id": f"{task_marker}-parent",
            "input_image_paths_resolved": _anchor_pair(),
        }
        runtime["individual_segment_params"] = {
            "start_image_url": config.ANCHOR_IMAGE_A_URL,
            "end_image_url": config.ANCHOR_IMAGE_B_URL,
            "input_image_paths_resolved": _anchor_pair(),
        }
        runtime["start_image_url"] = config.ANCHOR_IMAGE_A_URL
        runtime["end_image_url"] = config.ANCHOR_IMAGE_B_URL
        runtime["input_image_paths_resolved"] = _anchor_pair()

    route_contract = _route_contract(case, task_marker)
    if route_contract is not None:
        runtime["route_contract"] = route_contract
        runtime["route_key"] = case.route_key
        runtime["selected_backend"] = case.route_runtime.selected_backend
        runtime["selector_namespace"] = case.route_runtime.selector_namespace
        runtime["selector_version"] = case.route_runtime.selector_version
        runtime["worker_contract_version"] = case.route_runtime.worker_contract_version
        runtime["selected_profile"] = case.route_runtime.selected_profile

    return _deep_merge(case.param_overrides, runtime)


def filter_matrix(
    cases: list[MatrixCase],
    *,
    case_names: list[str] | None = None,
    task_types: list[str] | None = None,
    route_keys: list[str] | None = None,
) -> list[MatrixCase]:
    case_filter = set(case_names or [])
    task_filter = set(task_types or [])
    route_filter = set(route_keys or [])
    if not case_filter and not task_filter and not route_filter:
        return cases
    selected = [
        case
        for case in cases
        if (case_filter and case.name in case_filter)
        or (task_filter and case.task_type in task_filter)
        or (route_filter and case.route_key in route_filter)
    ]
    missing_cases = case_filter - {case.name for case in selected}
    missing_tasks = task_filter - {case.task_type for case in selected}
    missing_routes = route_filter - {case.route_key for case in selected if case.route_key}
    missing = sorted(missing_cases | missing_tasks | missing_routes)
    if missing:
        raise ValueError(f"Unknown live-test case/task/route selection: {', '.join(missing)}")
    return selected


def render_case_payload(
    case: MatrixCase,
    *,
    project_id: str,
    unique_suffix: str | None = None,
) -> dict[str, Any]:
    payload = copy.deepcopy(resolve_case_fixture(case))
    payload.pop("notes", None)
    payload.pop("description", None)
    payload["project_id"] = project_id
    payload["task_type"] = case.task_type
    payload["status"] = "Queued"

    params = payload.get("params")
    if not isinstance(params, dict):
        params = {}
    payload["params"] = _deep_merge(
        params,
        build_case_params_overrides(case, unique_suffix=unique_suffix),
    )
    payload["params"]["live_test"] = True
    return payload


def _wan_vace_individual_overrides(
    *,
    mode: str,
    anchor_image_a: str,
    anchor_image_b: str,
) -> dict[str, Any]:
    model_name = "wan_2_2_vace_lightning_baseline_2_2_2"
    travel_guidance = {
        "kind": "vace",
        "mode": mode,
        "videos": [{"url": LIVE_TEST_VIDEO_URL}],
    }
    return {
        "model_name": model_name,
        "model": model_name,
        "resolution": "832x480",
        "parsed_resolution_wh": "832x480",
        "num_frames": 81,
        "video_length": 81,
        "fps": 16,
        "start_image_url": anchor_image_a,
        "end_image_url": anchor_image_b,
        "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
        "travel_guidance": travel_guidance,
        "orchestrator_details": {
            "model_name": model_name,
            "model": model_name,
            "parsed_resolution_wh": "832x480",
            "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
            "travel_guidance": travel_guidance,
        },
        "individual_segment_params": {
            "model_name": model_name,
            "model": model_name,
            "start_image_url": anchor_image_a,
            "end_image_url": anchor_image_b,
            "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
            "num_frames": 81,
            "travel_guidance": travel_guidance,
        },
    }


def _wan_vace_travel_video_source_overrides(
    *,
    mode: str,
    anchor_image_a: str,
    anchor_image_b: str,
) -> dict[str, Any]:
    model_name = "wan_2_2_vace_lightning_baseline_2_2_2"
    travel_guidance = {
        "kind": "vace",
        "mode": mode,
        "videos": [{"url": LIVE_TEST_VIDEO_URL}],
    }
    orchestrator_details = {
        "model_name": model_name,
        "model": model_name,
        "model_family": "wan22_vace",
        "model_type": "vace",
        "parsed_resolution_wh": "832x480",
        "segment_frames_expanded": [81],
        "num_new_segments_to_generate": 1,
        "base_prompts_expanded": ["A person standing in a dynamic pose"],
        "negative_prompts_expanded": ["fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted"],
        "frame_overlap_expanded": [0],
        "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
        "travel_guidance": travel_guidance,
        "continuation_config": {"type": "video_source"},
        "fps_helpers": 16,
        "seed_base": 42,
    }
    return {
        "model_name": model_name,
        "model": model_name,
        "model_family": "wan22_vace",
        "model_type": "vace",
        "segment_index": 0,
        "resolution": "832x480",
        "parsed_resolution_wh": "832x480",
        "num_frames": 81,
        "video_length": 81,
        "fps": 16,
        "start_image_url": anchor_image_a,
        "end_image_url": anchor_image_b,
        "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
        "video_source": LIVE_TEST_VIDEO_URL,
        "travel_guidance": travel_guidance,
        "orchestrator_details": orchestrator_details,
        "individual_segment_params": {
            "model_name": model_name,
            "model": model_name,
            "start_image_url": anchor_image_a,
            "end_image_url": anchor_image_b,
            "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
            "num_frames": 81,
            "travel_guidance": travel_guidance,
        },
    }


def build_matrix(
    *,
    anchor_image_a: str = config.ANCHOR_IMAGE_A_URL,
    anchor_image_b: str = config.ANCHOR_IMAGE_B_URL,
    timeout_image_sec: int = config.TIMEOUT_IMAGE_SEC,
    timeout_travel_segment_sec: int = config.TIMEOUT_INDIVIDUAL_TRAVEL_SEGMENT_SEC,
    timeout_travel_orchestrator_sec: int = config.TIMEOUT_TRAVEL_ORCHESTRATOR_SEC,
    selected_backend: str = "wgp",
    selector_namespace: str = "production",
    selector_version: str | None = None,
    worker_contract_version: int = 1,
    selected_profile: str = "default",
    case_names: list[str] | None = None,
    task_types: list[str] | None = None,
    route_keys: list[str] | None = None,
) -> list[MatrixCase]:
    route_runtime = RouteRuntimeOptions(
        selected_backend=selected_backend,
        selector_namespace=selector_namespace,
        selector_version=selector_version,
        worker_contract_version=worker_contract_version,
        selected_profile=selected_profile,
    )
    cases = [
        MatrixCase(
            name="travel_orchestrator_wan2_1seg",
            task_type="travel_orchestrator",
            fixture_key=TRAVEL_WAN_FIXTURE_KEY,
            timeout_sec=timeout_travel_orchestrator_sec,
            route_key="travel_orchestrator",
            support_state="wgp_only",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="travel_orchestrator_ltx",
            task_type="travel_orchestrator",
            fixture_key=TRAVEL_LTX_FIXTURE_KEY,
            timeout_sec=timeout_travel_orchestrator_sec,
            route_key="travel_orchestrator",
            support_state="wgp_only",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="individual_travel_segment",
            task_type="individual_travel_segment",
            fixture_key="wan22_i2v_individual_segment",
            param_overrides={
                "start_image_url": anchor_image_a,
                "end_image_url": anchor_image_b,
                "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
                "orchestrator_details": {
                    "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
                },
                "individual_segment_params": {
                    "start_image_url": anchor_image_a,
                    "end_image_url": anchor_image_b,
                    "input_image_paths_resolved": [anchor_image_a, anchor_image_b],
                },
            },
            timeout_sec=timeout_travel_segment_sec,
        ),
        MatrixCase(
            name="travel_segment_wan22_i2v_first_last",
            task_type="travel_segment",
            fixture_key="wan_2_2_i2v_first_last",
            param_overrides={
                "segment_index": 0,
                "orchestrator_run_id": "live-test-wan22-i2v-first-last",
                "orchestrator_task_id_ref": "live-test-wan22-i2v-first-last-parent",
                "orchestrator_details": {
                    "run_id": "live-test-wan22-i2v-first-last",
                    "orchestrator_task_id": "live-test-wan22-i2v-first-last-parent",
                },
            },
            timeout_sec=timeout_travel_segment_sec,
            route_key="travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default",
            support_state="vibecomfy_supported",
            selected_template_id="video/wanvideo_wrapper_22_14b_i2v_kijai",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="individual_travel_segment_wan22_vace",
            task_type="individual_travel_segment",
            fixture_key="wan22_i2v_individual_segment",
            param_overrides=_wan_vace_individual_overrides(
                mode="raw",
                anchor_image_a=anchor_image_a,
                anchor_image_b=anchor_image_b,
            ),
            timeout_sec=timeout_travel_segment_sec,
            route_key="individual_travel_segment__model-wan22_vace__guidance-vace_raw__continuity-first_last__profile-default",
            support_state="vibecomfy_supported",
            selected_template_id="video/wanvideo_wrapper_22_14b_vace_cocktail",
            route_runtime=route_runtime,
        ),
        *[
            MatrixCase(
                name=f"individual_travel_segment_wan22_vace_{mode}",
                task_type="individual_travel_segment",
                fixture_key="wan22_i2v_individual_segment",
                param_overrides=_wan_vace_individual_overrides(
                    mode=mode,
                    anchor_image_a=anchor_image_a,
                    anchor_image_b=anchor_image_b,
                ),
                timeout_sec=timeout_travel_segment_sec,
                route_key=(
                    "individual_travel_segment__model-wan22_vace__"
                    f"guidance-vace_{mode}__continuity-first_last__profile-default"
                ),
                support_state="vibecomfy_supported",
                selected_template_id="video/wanvideo_wrapper_22_14b_vace_cocktail",
                route_runtime=route_runtime,
            )
            for mode in ("flow", "canny", "depth")
        ],
        *[
            MatrixCase(
                name=f"travel_segment_wan22_vace_{mode}_video_source",
                task_type="travel_segment",
                fixture_key="wan22_i2v_individual_segment",
                param_overrides=_wan_vace_travel_video_source_overrides(
                    mode=mode,
                    anchor_image_a=anchor_image_a,
                    anchor_image_b=anchor_image_b,
                ),
                timeout_sec=timeout_travel_segment_sec,
                route_key=(
                    "travel_segment__model-wan22_vace__"
                    f"guidance-vace_{mode}__continuity-video_source__profile-default"
                ),
                support_state="vibecomfy_supported",
                selected_template_id="video/wanvideo_wrapper_22_14b_vace_cocktail",
                route_runtime=route_runtime,
            )
            for mode in ("raw", "flow", "canny", "depth")
        ],
        MatrixCase(
            name="travel_segment_ltx2_first_last",
            task_type="travel_segment",
            fixture_key=TRAVEL_LTX_FIXTURE_KEY,
            param_overrides={
                **_ltx_first_last_overrides(
                    mode=None,
                    anchor_image_a=anchor_image_a,
                    anchor_image_b=anchor_image_b,
                ),
                "model": "ltx2_22B",
                "model_name": "ltx2_22B",
                "model_family": "ltx2",
            },
            timeout_sec=timeout_travel_segment_sec,
            route_key="travel_segment__model-ltx2__guidance-none__continuity-first_last__profile-default",
            support_state="vibecomfy_supported",
            selected_template_id="video/ltx2_3_runexx_first_last_frame",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="travel_segment_ltx2_distilled_first_last",
            task_type="travel_segment",
            fixture_key=TRAVEL_LTX_FIXTURE_KEY,
            param_overrides=_ltx_first_last_overrides(
                mode=None,
                anchor_image_a=anchor_image_a,
                anchor_image_b=anchor_image_b,
            ),
            timeout_sec=timeout_travel_segment_sec,
            route_key="travel_segment__model-ltx2_distilled__guidance-none__continuity-first_last__profile-default",
            support_state="vibecomfy_supported",
            selected_template_id="video/ltx2_3_runexx_first_last_frame",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="travel_segment_ltx2_control_video_first_last",
            task_type="travel_segment",
            fixture_key=TRAVEL_LTX_FIXTURE_KEY,
            param_overrides=_ltx_first_last_overrides(
                mode="video",
                anchor_image_a=anchor_image_a,
                anchor_image_b=anchor_image_b,
            ),
            timeout_sec=timeout_travel_segment_sec,
            route_key=(
                "travel_segment__model-ltx2_distilled__"
                "guidance-ltx_control_video__continuity-first_last__profile-default"
            ),
            support_state="vibecomfy_supported",
            selected_template_id="video/ltx2_3_runexx_first_last_raw_video_guide",
            route_runtime=route_runtime,
        ),
        *[
            MatrixCase(
                name=f"travel_segment_ltx2_control_{mode}_first_last",
                task_type="travel_segment",
                fixture_key=TRAVEL_LTX_FIXTURE_KEY,
                param_overrides=_ltx_first_last_overrides(
                    mode=mode,
                    anchor_image_a=anchor_image_a,
                    anchor_image_b=anchor_image_b,
                ),
                timeout_sec=timeout_travel_segment_sec,
                route_key=(
                    "travel_segment__model-ltx2_distilled__"
                    f"guidance-ltx_control_{mode}__continuity-first_last__profile-default"
                ),
                support_state="vibecomfy_supported",
                selected_template_id="video/ltx2_3_first_last_frame_travel_iclora_control",
                route_runtime=route_runtime,
            )
            for mode in ("pose", "depth", "canny", "cameraman")
        ],
        MatrixCase(
            name="travel_stitch",
            task_type="travel_stitch",
            fixture_key=TRAVEL_STITCH_FIXTURE_KEY,
            timeout_sec=timeout_image_sec,
            route_key="travel_stitch",
            support_state="wgp_only",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="join_clips_orchestrator",
            task_type="join_clips_orchestrator",
            fixture_key=JOIN_CLIPS_ORCHESTRATOR_FIXTURE_KEY,
            timeout_sec=timeout_travel_orchestrator_sec,
            route_key="join_clips_orchestrator",
            support_state="wgp_only",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="join_clips_segment_wan22_vace",
            task_type="join_clips_segment",
            fixture_key=JOIN_CLIPS_SEGMENT_VACE_FIXTURE_KEY,
            timeout_sec=timeout_travel_segment_sec,
            route_key="join_clips_segment__model-wan22_vace__guidance-vace__continuity-join_bridge__profile-default",
            support_state="vibecomfy_supported",
            selected_template_id="video/wanvideo_wrapper_22_14b_vace_cocktail",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="edit_video_orchestrator",
            task_type="edit_video_orchestrator",
            fixture_key=EDIT_VIDEO_ORCHESTRATOR_FIXTURE_KEY,
            timeout_sec=timeout_travel_orchestrator_sec,
            route_key="edit_video_orchestrator",
            support_state="wgp_only",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="qwen_image_style",
            task_type="qwen_image_style",
            fixture_key="qwen_image_style_db_task",
            param_overrides={
                "style_reference_image": anchor_image_a,
                "subject_reference_image": anchor_image_a,
            },
            timeout_sec=timeout_image_sec,
            route_key="qwen_image_style",
            support_state="vibecomfy_supported",
            selected_template_id="edit/qwen_image_edit",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="qwen_image_t2i",
            task_type="qwen_image",
            fixture_key="qwen_image_basic",
            timeout_sec=timeout_image_sec,
            route_key="qwen_image",
            support_state="vibecomfy_supported",
            selected_template_id="image/qwen_image_2512",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="qwen_image_2512",
            task_type="qwen_image_2512",
            fixture_key="qwen_image_basic",
            param_overrides={"resolution": "1536x864"},
            timeout_sec=timeout_image_sec,
            route_key="qwen_image_2512",
            support_state="vibecomfy_supported",
            selected_template_id="image/qwen_image_2512",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="wan_2_2_t2i",
            task_type="wan_2_2_t2i",
            fixture_key=WAN_2_2_T2I_FIXTURE_KEY,
            timeout_sec=timeout_image_sec,
            route_key="wan_2_2_t2i",
            support_state="vibecomfy_supported",
            selected_template_id="video/wanvideo_wrapper_22_14b_t2i",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="wan_2_2_i2v",
            task_type="wan_2_2_i2v",
            fixture_key="wan_2_2_i2v",
            timeout_sec=timeout_travel_segment_sec,
            route_key="wan_2_2_i2v",
            support_state="vibecomfy_supported",
            selected_template_id="video/wanvideo_wrapper_22_14b_i2v_kijai",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="animate_character",
            task_type="animate_character",
            fixture_key="animate_character",
            timeout_sec=timeout_travel_segment_sec,
            route_key="animate_character",
            support_state="vibecomfy_supported",
            selected_template_id="video/wan22_animate_native_first_stage",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="image_upscale",
            task_type="image-upscale",
            fixture_key="image-upscale",
            timeout_sec=timeout_image_sec,
            route_key="image-upscale",
            support_state="vibecomfy_supported",
            selected_template_id="image/basic_image_upscale",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="video_enhance",
            task_type="video_enhance",
            fixture_key="video_enhance",
            timeout_sec=timeout_travel_segment_sec,
            route_key="video_enhance",
            support_state="vibecomfy_supported",
            selected_template_id="video/basic_video_enhance",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="flux_klein_edit",
            task_type="flux_klein_edit",
            fixture_key="flux_klein_edit",
            timeout_sec=timeout_image_sec,
            route_key="flux_klein_edit",
            support_state="vibecomfy_supported",
            selected_template_id="edit/flux2_klein_4b_image_edit_distilled",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="qwen_image_edit",
            task_type="qwen_image_edit",
            fixture_key="qwen_image_edit_basic",
            param_overrides={
                "image": anchor_image_b,
                "image_url": anchor_image_b,
            },
            timeout_sec=timeout_image_sec,
            route_key="qwen_image_edit",
            support_state="vibecomfy_supported",
            selected_template_id="edit/qwen_image_edit",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="image_inpaint",
            task_type="image_inpaint",
            fixture_key=IMAGE_INPAINT_FIXTURE_KEY,
            timeout_sec=timeout_image_sec,
            route_key="image_inpaint",
            support_state="vibecomfy_supported",
            selected_template_id="edit/qwen_image_edit",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="annotated_image_edit",
            task_type="annotated_image_edit",
            fixture_key=ANNOTATED_IMAGE_EDIT_FIXTURE_KEY,
            timeout_sec=timeout_image_sec,
            route_key="annotated_image_edit",
            support_state="vibecomfy_supported",
            selected_template_id="edit/qwen_image_edit",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="z_image_turbo",
            task_type="z_image_turbo",
            fixture_key=Z_IMAGE_TURBO_FIXTURE_KEY,
            timeout_sec=timeout_image_sec,
            route_key="z_image_turbo",
            support_state="vibecomfy_supported",
            selected_template_id="image/z_image",
            route_runtime=route_runtime,
        ),
        MatrixCase(
            name="z_image_turbo_i2i",
            task_type="z_image_turbo_i2i",
            fixture_key="z_image_turbo_i2i_basic",
            param_overrides={"image_url": anchor_image_a},
            timeout_sec=timeout_image_sec,
            route_key="z_image_turbo_i2i",
            support_state="vibecomfy_supported",
            selected_template_id="image/z_image_img2img",
            route_runtime=route_runtime,
        ),
    ]
    return filter_matrix(cases, case_names=case_names, task_types=task_types, route_keys=route_keys)


def build_target_manifest(
    cases: list[MatrixCase],
    *,
    selected_backend: str,
    selector_namespace: str,
    selector_version: str | None,
    worker_contract_version: int,
    selected_profile: str,
    selection: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Return a Reigh-owned target manifest for downstream VibeComfy enrichment.

    This deliberately serializes selected live-test case metadata only. It does
    not import VibeComfy and it does not enumerate every ready template.
    """
    targets: list[dict[str, Any]] = []
    template_ids: list[str] = []
    seen_templates: set[str] = set()
    for case in cases:
        template_id = case.selected_template_id
        if template_id and template_id not in seen_templates:
            seen_templates.add(template_id)
            template_ids.append(template_id)
        targets.append(
            {
                "case_name": case.name,
                "task_type": case.task_type,
                "route_key": case.route_key,
                "support_state": case.support_state,
                "template_id": template_id,
                "fixture_key": case.fixture_key,
                "timeout_sec": case.timeout_sec,
                "route_runtime": asdict(case.route_runtime),
            }
        )
    return {
        "schema_version": 1,
        "producer": "reigh-worker.scripts.live_test",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": selected_backend,
        "selector": {
            "namespace": selector_namespace,
            "version": selector_version,
            "worker_contract_version": worker_contract_version,
            "profile": selected_profile,
        },
        "selection": selection or {"case_names": [], "task_types": [], "route_keys": []},
        "target_count": len(targets),
        "template_count": len(template_ids),
        "templates": template_ids,
        "targets": targets,
    }


MATRIX = build_matrix()


def queue_matrix(db, project_id: str, cases: list[MatrixCase]) -> list[tuple[MatrixCase, str]]:
    queued: list[tuple[MatrixCase, str]] = []
    for case in cases:
        suffix = uuid.uuid4().hex[:12]
        fixture_payload = resolve_case_fixture(case)
        overrides = build_case_params_overrides(case, unique_suffix=suffix)
        task_id = insert_spoof_task(
            db,
            project_id,
            case.task_type,
            overrides,
            fixture_payload=fixture_payload,
        )
        queued.append((case, task_id))
    return queued


def poll_queued_matrix(db, project_id: str, queued: list[tuple[MatrixCase, str]], *, worker_id: str | None = None) -> list[TaskResult]:
    results: list[TaskResult] = []
    for case, task_id in queued:
        result = poll_until_complete(
            db,
            task_id,
            project_id,
            timeout_sec=case.timeout_sec,
            case_name=case.name,
            task_type=case.task_type,
            worker_id=worker_id,
        )
        results.append(result)
    return results


def run_matrix(
    db,
    project_id: str,
    cases: list[MatrixCase],
    *,
    worker_id: str | None = None,
    serial: bool = False,
) -> list[TaskResult]:
    try:
        if serial:
            results: list[TaskResult] = []
            for case in cases:
                queued = queue_matrix(db, project_id, [case])
                results.extend(poll_queued_matrix(db, project_id, queued, worker_id=worker_id))
            return results

        queued = queue_matrix(db, project_id, cases)
    except Exception as exc:
        return [
            TaskResult(
                task_id="insert-failed:matrix",
                case_name="matrix",
                task_type="",
                final_status="Insert Failed",
                output_location=None,
                generation_ids=[],
                elapsed_sec=0,
                error_summary=str(exc),
            )
        ]
    return poll_queued_matrix(db, project_id, queued, worker_id=worker_id)


__all__ = [
    "MATRIX",
    "MatrixCase",
    "RouteRuntimeOptions",
    "TRAVEL_LTX_FIXTURE_KEY",
    "TRAVEL_WAN_FIXTURE_KEY",
    "Z_IMAGE_TURBO_FIXTURE_KEY",
    "build_case_params_overrides",
    "build_matrix",
    "build_target_manifest",
    "filter_matrix",
    "poll_queued_matrix",
    "queue_matrix",
    "render_case_payload",
    "resolve_case_fixture",
    "run_matrix",
]
