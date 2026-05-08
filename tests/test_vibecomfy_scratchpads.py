from __future__ import annotations

import importlib
from pathlib import Path

from source.task_handlers.tasks.template_routing import ResolvedTask, RouteSupportState, WorkerBackend


def _resolved(route_key: str, params: dict) -> ResolvedTask:
    return ResolvedTask(
        task_id=f"{route_key}-task",
        task_type=route_key,
        route_key=route_key,
        backend=WorkerBackend.VIBECOMFY,
        support_state=RouteSupportState.VIBECOMFY_SUPPORTED,
        params=params,
        template_id="template",
    )


def test_wan_2_2_i2v_scratchpad_patches_kijai_template_inputs(tmp_path: Path, monkeypatch) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    monkeypatch.setattr(adapter, "download_image_if_url", lambda value, *_args, **_kwargs: value)

    scratchpad = adapter._write_wan_2_2_i2v_scratchpad(
        _resolved(
            "wan_2_2_i2v",
            {
                "image": str(source),
                "prompt": "make the subject walk forward",
                "negative_prompt": "bad",
                "resolution": "640x360",
                "num_frames": 17,
                "fps": 12,
                "seed": 123,
                "steps": 4,
            },
        ),
        tmp_path,
    )

    text = scratchpad.read_text(encoding="utf-8")
    assert "video/wanvideo_wrapper_22_14b_i2v_kijai" in text
    assert "make the subject walk forward" in text
    assert "workflow.nodes['60'].inputs['save_output'] = True" in text
    assert "workflow.nodes['89'].inputs['num_frames'] = 17" in text
    assert "workflow.nodes['68'].inputs['width'] = 640" in text
    assert (tmp_path / "input" / "wan_2_2_i2v_wan_2_2_i2v-task.png").exists()


def test_animate_character_scratchpad_patches_reference_and_motion_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    image = tmp_path / "character.png"
    video = tmp_path / "motion.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    monkeypatch.setattr(adapter, "download_image_if_url", lambda value, *_args, **_kwargs: value)
    monkeypatch.setattr(adapter, "download_video_if_url", lambda value, *_args, **_kwargs: value)

    scratchpad = adapter._write_animate_character_scratchpad(
        _resolved(
            "animate_character",
            {
                "character_image_url": str(image),
                "motion_video_url": str(video),
                "prompt": "dance naturally",
                "resolution": "512x512",
                "fps": 8,
                "seed": 456,
                "steps": 3,
            },
        ),
        tmp_path,
    )

    text = scratchpad.read_text(encoding="utf-8")
    assert "video/wanvideo_wrapper_22_wan_animate_preprocess_kijai" in text
    assert "dance naturally" in text
    assert "workflow.nodes['63'].inputs['video']" in text
    assert "workflow.nodes['30'].inputs['save_output'] = True" in text
    assert "workflow.nodes['150'].inputs['widget_0'] = 512" in text
    assert (tmp_path / "input" / "animate_character_reference_animate_character-task.png").exists()
    assert (tmp_path / "input" / "animate_character_motion_animate_character-task.mp4").exists()
