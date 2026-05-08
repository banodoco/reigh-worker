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
    assert "WanVideoEmptyEmbeds" in text
    assert "workflow.replace_edge('27.image_embeds'" in text
    assert "workflow.replace_edge('90.image_embeds'" in text
    assert "blocks_to_swap'] = max" in text
    assert "offload_img_emb'] = True" in text
    assert "offload_txt_emb'] = True" in text
    assert (tmp_path / "input" / "wan_2_2_i2v_wan_2_2_i2v-task.png").exists()


def test_vibecomfy_command_omits_ensure_flags_when_cli_does_not_support_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    monkeypatch.setattr(adapter, "_workflow_reference_for_resolved_task", lambda *_args: ("scratch.py", False))
    monkeypatch.setattr(adapter, "_vibecomfy_run_help_text", lambda *_args: "usage: vibecomfy run")

    command = adapter._build_vibecomfy_command(_resolved("qwen_image", {}), tmp_path)

    assert "--ensure-packs" not in command
    assert "--ensure-models" not in command


def test_vibecomfy_command_keeps_ensure_flags_when_cli_supports_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    monkeypatch.setattr(adapter, "_workflow_reference_for_resolved_task", lambda *_args: ("scratch.py", False))
    monkeypatch.setattr(adapter, "_vibecomfy_run_help_text", lambda *_args: "--ensure-packs --ensure-models")

    command = adapter._build_vibecomfy_command(_resolved("qwen_image", {}), tmp_path)

    assert "--ensure-packs" in command
    assert "--ensure-models" in command


def test_wan_video_scratchpads_wire_dynamic_loras_as_vibecomfy_assets(tmp_path: Path, monkeypatch) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    monkeypatch.setattr(adapter, "download_image_if_url", lambda value, *_args, **_kwargs: value)

    scratchpad = adapter._write_wan_2_2_i2v_scratchpad(
        _resolved(
            "wan_2_2_i2v",
            {
                "image": str(source),
                "prompt": "move with style",
                "model_name": "wan_2_2_i2v",
                "additional_loras": {
                    "https://huggingface.co/acme/wan-style/resolve/main/style.safetensors": 0.7
                },
            },
        ),
        tmp_path,
    )

    text = scratchpad.read_text(encoding="utf-8")
    assert "_append_model_assets(workflow" in text
    assert "_chain_wanvideo_select_loras(workflow" in text
    assert '"directory": "loras/WanVideo/Reigh"' in text
    assert '"name": "WanVideo\\\\Reigh\\\\style.safetensors"' in text
    assert '"strength": 0.7' in text


def test_wan_2_2_i2v_first_last_scratchpad_preserves_image_embeds(tmp_path: Path, monkeypatch) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    start = tmp_path / "start.png"
    end = tmp_path / "end.png"
    start.write_bytes(b"start")
    end.write_bytes(b"end")
    monkeypatch.setattr(adapter, "download_image_if_url", lambda value, *_args, **_kwargs: value)

    scratchpad = adapter._write_wan_2_2_i2v_first_last_scratchpad(
        _resolved(
            "travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default",
            {
                "input_image_paths_resolved": [str(start), str(end)],
                "prompt": "bridge the two frames",
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
    assert "WanVideoEmptyEmbeds" not in text
    assert "workflow.replace_edge('27.image_embeds'" not in text
    assert "workflow.replace_edge('90.image_embeds'" not in text
    assert "workflow.connect(f'{end_resized.id}.0', '89.end_image')" in text
    assert "workflow.nodes['89'].inputs['fun_or_fl2v_model'] = False" in text
    assert "WanVideo2_2_I2V_FirstLast" in text
    assert (tmp_path / "input" / "wan_2_2_i2v_start_travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default-task.png").exists()
    assert (tmp_path / "input" / "wan_2_2_i2v_end_travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default-task.png").exists()


def test_animate_character_scratchpad_chains_dynamic_loras(tmp_path: Path, monkeypatch) -> None:
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
                "additional_loras": {
                    "https://huggingface.co/acme/wan-animate/resolve/main/pose-style.safetensors": 0.5
                },
            },
        ),
        tmp_path,
    )

    text = scratchpad.read_text(encoding="utf-8")
    assert "_chain_lora_loader_model_only(workflow" in text
    assert '"name": "WanVideo\\\\Reigh\\\\pose-style.safetensors"' in text
    assert '"strength": 0.5' in text


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
                "num_frames": 49,
                "fps": 8,
                "seed": 456,
                "steps": 3,
            },
        ),
        tmp_path,
    )

    text = scratchpad.read_text(encoding="utf-8")
    assert "video/wan22_animate_native_first_stage" in text
    assert "dance naturally" in text
    assert "workflow.nodes['145'].inputs['file']" in text
    assert "workflow.nodes['19'].inputs['format'] = 'mp4'" in text
    assert "workflow.nodes['159'].inputs['value'] = 512" in text
    assert "workflow.nodes['232:62'].inputs['length'] = 49" in text
    assert "workflow.nodes['232:230'].inputs['length'] = 49" in text
    assert (tmp_path / "input" / "animate_character_reference_animate_character-task.png").exists()
    assert (tmp_path / "input" / "animate_character_motion_animate_character-task.mp4").exists()


def test_image_upscale_scratchpad_patches_core_scale_template(tmp_path: Path, monkeypatch) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    image = tmp_path / "upscale.png"
    image.write_bytes(b"image")
    monkeypatch.setattr(adapter, "download_image_if_url", lambda value, *_args, **_kwargs: value)

    scratchpad = adapter._write_image_upscale_scratchpad(
        _resolved("image-upscale", {"image": str(image), "scale_factor": 3}),
        tmp_path,
    )

    text = scratchpad.read_text(encoding="utf-8")
    assert "image/basic_image_upscale" in text
    assert "workflow.nodes['2'].inputs['scale_by'] = 3.0" in text
    assert (tmp_path / "input" / "image_upscale_image-upscale-task.png").exists()


def test_video_enhance_scratchpad_can_bypass_disabled_stages(tmp_path: Path, monkeypatch) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(adapter, "download_video_if_url", lambda value, *_args, **_kwargs: value)

    scratchpad = adapter._write_video_enhance_scratchpad(
        _resolved(
            "video_enhance",
            {
                "video_url": str(video),
                "enable_interpolation": False,
                "enable_upscale": True,
                "upscale": {"upscale_factor": 4},
            },
        ),
        tmp_path,
    )

    text = scratchpad.read_text(encoding="utf-8")
    assert "video/basic_video_enhance" in text
    assert "workflow.replace_edge('5.images', '1.0')" in text
    assert "workflow.nodes['4'].inputs['scale_by'] = 4.0" in text
    assert (tmp_path / "input" / "video_enhance_video_enhance-task.mp4").exists()


def test_video_enhance_postprocess_runs_rife_when_requested(tmp_path: Path, monkeypatch) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    expected = tmp_path / "output" / "video-enhance-rife-x2.mp4"

    calls = []

    def fake_rife(input_path, output_path, *, fps, exp):
        calls.append((input_path, output_path, fps, exp))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"interpolated")
        return output_path

    monkeypatch.setattr(adapter, "_rife_interpolate_video", fake_rife)
    result = adapter._maybe_postprocess_vibecomfy_output(
        resolved=_resolved(
            "video_enhance",
            {"enable_interpolation": True, "fps": 12, "interpolation": {"num_frames": 1}},
        ),
        output_path=source,
        run_workspace=tmp_path,
    )

    assert result == expected
    assert calls == [(source, expected, 12, 1)]


def test_video_enhance_interpolation_num_frames_maps_to_rife_exp() -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")

    assert adapter._rife_exp_from_interpolation_params({"num_frames": 1}) == 1
    assert adapter._rife_exp_from_interpolation_params({"num_frames": 3}) == 2
    assert adapter._rife_exp_from_interpolation_params({"rife_exp": 2, "num_frames": 1}) == 2


def test_video_enhance_contract_expects_interpolated_fps() -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")

    contract = adapter._video_contract_for_resolved_task(
        _resolved(
            "video_enhance",
            {"enable_interpolation": True, "fps": 16, "interpolation": {"num_frames": 1}},
        )
    )

    assert contract.expected_fps == 32


def test_rife_checkpoint_path_downloads_missing_checkpoint(tmp_path: Path, monkeypatch) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    fake_root = tmp_path / "repo" / "source" / "models" / "comfy" / "vibecomfy_adapter.py"
    fake_root.parent.mkdir(parents=True)
    fake_root.write_text("", encoding="utf-8")
    target = tmp_path / "repo" / "Wan2GP" / "ckpts" / "rife4.26.pkl"

    monkeypatch.setattr(adapter, "__file__", str(fake_root))

    def fake_download(filename: str, ckpt_dir: Path) -> Path:
        assert filename == "rife4.26.pkl"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"checkpoint")
        return target

    monkeypatch.setattr(adapter, "_download_rife_checkpoint", fake_download)

    assert adapter._rife_checkpoint_path(prefer_v4=True) == target


def test_flux_klein_edit_scratchpad_uses_expanded_4b_template(tmp_path: Path, monkeypatch) -> None:
    adapter = importlib.import_module("source.models.comfy.vibecomfy_adapter")
    image = tmp_path / "klein.png"
    image.write_bytes(b"image")
    monkeypatch.setattr(adapter, "download_image_if_url", lambda value, *_args, **_kwargs: value)

    scratchpad = adapter._write_flux_klein_edit_scratchpad(
        _resolved(
            "flux_klein_edit",
            {
                "image": str(image),
                "prompt": "turn the jacket red",
                "seed": 321,
                "num_inference_steps": 5,
                "klein_model": "flux-klein-4b",
            },
        ),
        tmp_path,
    )

    text = scratchpad.read_text(encoding="utf-8")
    assert "edit/flux2_klein_4b_image_edit_distilled" in text
    assert "turn the jacket red" in text
    assert "workflow.nodes['75:73'].inputs['noise_seed'] = 321" in text
    assert "workflow.nodes['75:62'].inputs['steps'] = 5" in text
    assert (tmp_path / "input" / "flux_klein_edit_flux_klein_edit-task.png").exists()
