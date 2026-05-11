"""Subprocess adapter for Sprint 2 direct VibeComfy routes.

The worker must not import VibeComfy directly.  This module crosses the Python
3.11 VibeComfy boundary with ``subprocess`` only.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from source.core.log import headless_logger
from source.core.params.lora import LoRAConfig, LoRAEntry
from source.models.lora.module_manifest import LoRAModuleManifestError
from source.models.model_handlers.qwen_compositor import create_qwen_masked_composite
from source.runtime.vibecomfy_profile import (
    PROCESS_DEFAULT_PROFILE,
    build_memory_profile_cli_args,
)
from source.utils.download_utils import download_image_if_url, download_video_if_url
from source.media.video_contract import (
    VIDEO_EXTENSIONS,
    VideoArtifactContract,
    VideoContractError,
    validate_video_artifact,
)
from source.task_handlers.tasks.template_routing import (
    ResolvedTask,
    RouteSupportState,
    WorkerBackend,
)


_MAX_CAPTURE_CHARS = 4000
_OUTPUT_EXTENSIONS = {
    ".apng",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".png",
    ".webm",
    ".webp",
}

_VIBECOMFY_WAN_USER_LORA_DIR = "loras/WanVideo/Reigh"
_VIBECOMFY_WAN_USER_LORA_PREFIX = "WanVideo\\Reigh"


_WANVIDEO_DEFAULTS_HELPER = """
def _patch_wanvideo_defaults(workflow, *, steps, cfg, shift, seed):
    for node in workflow.nodes.values():
        if node.class_type == 'WanVideoSampler':
            node.inputs['steps'] = steps
            node.inputs['cfg'] = node.inputs.get('cfg', node.inputs.get('widget_1', cfg))
            node.inputs['shift'] = node.inputs.get('shift', node.inputs.get('widget_2', shift))
            node.inputs['seed'] = node.inputs.get('seed', node.inputs.get('widget_3', seed))
            node.inputs['scheduler'] = node.inputs.get('scheduler', node.inputs.get('widget_6', 'unipc'))
            node.inputs['force_offload'] = node.inputs.get('force_offload', node.inputs.get('widget_5', True))
            node.inputs['riflex_freq_index'] = node.inputs.get('riflex_freq_index', node.inputs.get('widget_7', 0))
        elif node.class_type == 'WanVideoTextEncodeCached':
            node.inputs['model_name'] = node.inputs.get('model_name', node.inputs.get('widget_0', 'umt5-xxl-enc-bf16.safetensors'))
            node.inputs['precision'] = node.inputs.get('precision', node.inputs.get('widget_1', 'bf16'))
            node.inputs['positive_prompt'] = node.inputs.get('positive_prompt', node.inputs.get('widget_2', ''))
            node.inputs['negative_prompt'] = node.inputs.get('negative_prompt', node.inputs.get('widget_3', ''))
            node.inputs['quantization'] = node.inputs.get('quantization', node.inputs.get('widget_4', 'disabled'))
            node.inputs['use_disk_cache'] = node.inputs.get('use_disk_cache', node.inputs.get('widget_5', True))
            node.inputs['device'] = node.inputs.get('device', node.inputs.get('widget_6', 'gpu'))
        elif node.class_type == 'WanVideoModelLoader':
            node.inputs['model'] = node.inputs.get('model', node.inputs.get('widget_0', 'WanVideo\\\\2_2\\\\Wan2_2-T2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors'))
            node.inputs['base_precision'] = node.inputs.get('base_precision', node.inputs.get('widget_1', 'fp16'))
            node.inputs['quantization'] = node.inputs.get('quantization', node.inputs.get('widget_2', 'fp8_e4m3fn_scaled'))
            node.inputs['load_device'] = node.inputs.get('load_device', node.inputs.get('widget_3', 'offload_device'))
            node.inputs['attention_mode'] = node.inputs.get('attention_mode', node.inputs.get('widget_4', 'sdpa'))
        elif node.class_type == 'WanVideoLoraSelectMulti':
            for index in range(5):
                node.inputs[f'lora_{index}'] = node.inputs.get(f'lora_{index}', node.inputs.get(f'widget_{index * 2}', 'none'))
                node.inputs[f'strength_{index}'] = node.inputs.get(f'strength_{index}', node.inputs.get(f'widget_{index * 2 + 1}', 1.0))
            node.inputs['low_mem_load'] = node.inputs.get('low_mem_load', node.inputs.get('widget_10', False))
            node.inputs['merge_loras'] = node.inputs.get('merge_loras', node.inputs.get('widget_11', False))
        elif node.class_type == 'WanVideoBlockSwap':
            node.inputs['blocks_to_swap'] = node.inputs.get('blocks_to_swap', node.inputs.get('widget_0', 0))
            node.inputs['offload_img_emb'] = node.inputs.get('offload_img_emb', node.inputs.get('widget_1', False))
            node.inputs['offload_txt_emb'] = node.inputs.get('offload_txt_emb', node.inputs.get('widget_2', False))
            node.inputs['use_non_blocking'] = node.inputs.get('use_non_blocking', node.inputs.get('widget_3', False))
            node.inputs['vace_blocks_to_swap'] = node.inputs.get('vace_blocks_to_swap', node.inputs.get('widget_4', 0))
            node.inputs['prefetch_blocks'] = node.inputs.get('prefetch_blocks', node.inputs.get('widget_5', 0))
            node.inputs['blocks_to_keep'] = node.inputs.get('blocks_to_keep', node.inputs.get('widget_6', False))
        elif node.class_type == 'WanVideoVACEModelSelect':
            node.inputs['vace_model'] = node.inputs.get('vace_model', node.inputs.get('widget_0', 'WanVideo\\\\Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors'))
        elif node.class_type == 'WanVideoVACEStartToEndFrame':
            node.inputs['num_frames'] = node.inputs.get('num_frames', node.inputs.get('widget_0', 81))
            node.inputs['empty_frame_level'] = node.inputs.get('empty_frame_level', node.inputs.get('widget_1', 0.0))
        elif node.class_type == 'WanVideoVAELoader':
            node.inputs['model_name'] = node.inputs.get('model_name', node.inputs.get('widget_0', 'wanvideo\\\\Wan2_1_VAE_bf16.safetensors'))
            node.inputs['precision'] = node.inputs.get('precision', node.inputs.get('widget_1', 'bf16'))
        elif node.class_type == 'WanVideoDecode':
            node.inputs['enable_vae_tiling'] = node.inputs.get('enable_vae_tiling', node.inputs.get('widget_0', False))
            node.inputs['tile_x'] = node.inputs.get('tile_x', node.inputs.get('widget_1', 272))
            node.inputs['tile_y'] = node.inputs.get('tile_y', node.inputs.get('widget_2', 272))
            node.inputs['tile_stride_x'] = node.inputs.get('tile_stride_x', node.inputs.get('widget_3', 144))
            node.inputs['tile_stride_y'] = node.inputs.get('tile_stride_y', node.inputs.get('widget_4', 128))
            node.inputs['normalization'] = node.inputs.get('normalization', node.inputs.get('widget_5', 'default'))
        elif node.class_type == 'VHS_VideoCombine':
            node.inputs['loop_count'] = node.inputs.get('loop_count', node.inputs.get('widget_1', 0))
            node.inputs['filename_prefix'] = node.inputs.get('filename_prefix', node.inputs.get('widget_2', 'Wan-2-2-VACE'))
            node.inputs['format'] = node.inputs.get('format', node.inputs.get('widget_3', 'video/h264-mp4'))
            node.inputs['pingpong'] = node.inputs.get('pingpong', node.inputs.get('widget_8', False))
            node.inputs['save_output'] = node.inputs.get('save_output', True)
""".strip()


_WANVIDEO_DYNAMIC_LORA_HELPER = """
def _append_model_assets(workflow, assets):
    if not assets:
        return
    model_assets = workflow.metadata.setdefault('model_assets', [])
    seen = {
        (asset.get('name'), asset.get('directory') or asset.get('subdir'))
        for asset in model_assets
        if isinstance(asset, dict)
    }
    for asset in assets:
        key = (asset.get('name'), asset.get('directory') or asset.get('subdir'))
        if key not in seen:
            model_assets.append(dict(asset))
            seen.add(key)


def _patch_wanvideo_dynamic_loras(workflow, loras, *, node_ids=('98', '93'), first_user_slot=1):
    if not loras:
        return
    for node_id in node_ids:
        if node_id not in workflow.nodes:
            continue
        inputs = workflow.nodes[node_id].inputs
        for offset, lora in enumerate(loras):
            slot = first_user_slot + offset
            if slot > 4:
                raise ValueError('WanVideoLoraSelectMulti supports at most four dynamic user LoRAs while preserving slot 0')
            lora_key = f'lora_{slot}'
            strength_key = f'strength_{slot}'
            widget_lora_key = f'widget_{slot * 2}'
            widget_strength_key = f'widget_{slot * 2 + 1}'
            inputs[lora_key] = lora['name']
            inputs[strength_key] = lora['strength']
            inputs[widget_lora_key] = lora['name']
            inputs[widget_strength_key] = lora['strength']


def _chain_wanvideo_select_loras(workflow, loras, *, chains=(('97', '79'), ('56', '80'))):
    if not loras:
        return
    for base_lora_node_id, set_loras_node_id in chains:
        if base_lora_node_id not in workflow.nodes or set_loras_node_id not in workflow.nodes:
            continue
        previous_node_id = base_lora_node_id
        for lora in loras:
            node = workflow.add_node(
                'WanVideoLoraSelect',
                widget_0=lora['name'],
                widget_1=lora['strength'],
                widget_2=False,
                widget_3=False,
            )
            workflow.connect(f'{previous_node_id}.0', f'{node.id}.prev_lora')
            previous_node_id = node.id
        workflow.replace_edge(f'{set_loras_node_id}.lora', f'{previous_node_id}.0')


def _chain_lora_loader_model_only(workflow, loras, *, source_node_id='99', target_ref='60.model'):
    if not loras or source_node_id not in workflow.nodes:
        return
    previous_node_id = source_node_id
    for lora in loras:
        node = workflow.add_node(
            'LoraLoaderModelOnly',
            lora_name=lora['name'],
            model=[previous_node_id, 0],
            strength_model=lora['strength'],
        )
        previous_node_id = node.id
    workflow.replace_edge(target_ref, f'{previous_node_id}.0')
""".strip()


def handle_vibecomfy_resolved_task(
    resolved: ResolvedTask,
    main_output_dir_base: str | Path,
) -> tuple[bool, str | None]:
    """Run a supported direct VibeComfy route and return the discovered output."""

    validation_error = _validate_supported_resolved_task(resolved)
    if validation_error:
        return False, validation_error

    run_workspace = _prepare_run_workspace(main_output_dir_base, resolved.task_id)
    command = _build_vibecomfy_command(resolved, run_workspace)
    env = _build_subprocess_env(run_workspace)

    headless_logger.debug_block(
        "VIBECOMFY_ROUTE",
        {
            "task_id": resolved.task_id,
            "task_type": resolved.task_type,
            "route_key": resolved.route_key,
            "backend": resolved.backend.value,
            "template_id": resolved.template_id,
            "support_state": resolved.support_state.value,
            "memory_profile": _memory_profile_for_log(resolved),
        },
        task_id=resolved.task_id,
    )

    command_cwd = _vibecomfy_cwd(run_workspace)
    try:
        completed = subprocess.run(
            command,
            cwd=command_cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        message = _failure_message(
            resolved=resolved,
            exit_code=None,
            stderr=str(exc),
            stdout="",
        )
        _log_failure(
            resolved=resolved,
            exit_code=None,
            stderr=str(exc),
            stdout="",
        )
        headless_logger.error(message, task_id=resolved.task_id)
        return False, message

    stdout = _bounded(completed.stdout)
    stderr = _bounded(completed.stderr)
    if completed.returncode != 0:
        message = _failure_message(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr,
            stdout=stdout,
        )
        _log_failure(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr,
            stdout=stdout,
        )
        headless_logger.error(message, task_id=resolved.task_id)
        return False, message

    output_path = _discover_output_path(
        stdout=stdout,
        run_workspace=run_workspace,
        command_cwd=command_cwd,
    )
    if output_path is None:
        message = _failure_message(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr or "no output path discovered",
            stdout=stdout,
        )
        _log_failure(
            resolved=resolved,
            exit_code=completed.returncode,
            stderr=stderr or "no output path discovered",
            stdout=stdout,
        )
        headless_logger.error(message, task_id=resolved.task_id)
        return False, message

    postprocessed_output = _maybe_postprocess_vibecomfy_output(
        resolved=resolved,
        output_path=Path(output_path),
        run_workspace=run_workspace,
    )
    if postprocessed_output is not None:
        output_path = str(postprocessed_output)

    media_metadata = None
    if Path(output_path).suffix.lower() in VIDEO_EXTENSIONS:
        try:
            media_metadata = validate_video_artifact(
                output_path,
                _video_contract_for_resolved_task(resolved),
            )
        except VideoContractError as exc:
            message = _failure_message(
                resolved=resolved,
                exit_code=completed.returncode,
                stderr=f"media contract violation: {exc}",
                stdout=stdout,
            )
            _log_failure(
                resolved=resolved,
                exit_code=completed.returncode,
                stderr=f"media contract violation: {exc}",
                stdout=stdout,
            )
            headless_logger.error(message, task_id=resolved.task_id)
            return False, message

    headless_logger.debug_block(
        "VIBECOMFY_COMPLETE",
        {
            "task_id": resolved.task_id,
            "route_key": resolved.route_key,
            "backend": resolved.backend.value,
            "template_id": resolved.template_id,
            "memory_profile": _memory_profile_for_log(resolved),
            "exit_code": completed.returncode,
            "output_path": output_path,
            "media_metadata": _video_metadata_for_log(media_metadata),
        },
        task_id=resolved.task_id,
    )
    return True, output_path


def _validate_supported_resolved_task(resolved: ResolvedTask) -> str | None:
    if resolved.backend != WorkerBackend.VIBECOMFY:
        return f"VibeComfy adapter received non-VibeComfy backend {resolved.backend.value}"
    if resolved.fail_closed_reason:
        return (
            f"VibeComfy backend fail-closed for task {resolved.task_id} "
            f"({resolved.route_key}): {resolved.fail_closed_reason}"
        )
    if resolved.support_state != RouteSupportState.VIBECOMFY_SUPPORTED:
        return (
            f"VibeComfy route {resolved.route_key!r} is "
            f"{resolved.support_state.value}; adapter will not execute it"
        )
    if not resolved.template_id:
        return f"VibeComfy route {resolved.route_key!r} has no template_id"
    return None


def _build_vibecomfy_command(resolved: ResolvedTask, run_workspace: Path) -> list[str]:
    workflow_ref, ready = _workflow_reference_for_resolved_task(resolved, run_workspace)
    command = [
        _vibecomfy_python(),
        "-m",
        "vibecomfy.cli",
        "run",
        workflow_ref,
        "--runtime",
        "embedded",
    ]
    if _vibecomfy_run_supports_ensure_flags(run_workspace):
        command.extend(["--ensure-packs", "--ensure-models"])
    if ready:
        command.append("--ready")
        prompt = resolved.params.get("prompt")
        if prompt is not None:
            command.extend(["--prompt", str(prompt)])

        seed = resolved.params.get("seed")
        if seed is not None:
            command.extend(["--seed", str(int(seed))])

        steps = resolved.params.get("steps", resolved.params.get("num_inference_steps"))
        if steps is not None:
            command.extend(["--steps", str(int(steps))])

    command.extend(
        build_memory_profile_cli_args(
            process_default=_process_default_memory_profile(),
            override_profile=_override_memory_profile(resolved),
        )
    )
    return command


def _vibecomfy_run_supports_ensure_flags(run_workspace: Path) -> bool:
    override = os.environ.get("VIBECOMFY_RUN_ENSURE_FLAGS", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False

    cwd = _vibecomfy_cwd(run_workspace)
    env = _build_subprocess_env(run_workspace)
    help_text = _vibecomfy_run_help_text(
        _vibecomfy_python(),
        str(cwd),
        env.get("PYTHONPATH", ""),
    )
    return "--ensure-packs" in help_text and "--ensure-models" in help_text


@lru_cache(maxsize=8)
def _vibecomfy_run_help_text(python_executable: str, cwd: str, pythonpath: str) -> str:
    env = os.environ.copy()
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    try:
        completed = subprocess.run(
            [python_executable, "-m", "vibecomfy.cli", "run", "--help"],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except TypeError:
        return ""
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return f"{completed.stdout}\n{completed.stderr}"


def _workflow_reference_for_resolved_task(resolved: ResolvedTask, run_workspace: Path) -> tuple[str, bool]:
    if resolved.route_key == "z_image_turbo":
        return str(_write_z_image_scratchpad(resolved, run_workspace)), False
    if resolved.route_key == "z_image_turbo_i2i":
        return str(_write_z_image_img2img_scratchpad(resolved, run_workspace)), False
    if resolved.route_key in {"qwen_image", "qwen_image_2512"}:
        return str(_write_qwen_image_2512_scratchpad(resolved, run_workspace)), False
    if resolved.route_key in {"image-upscale", "image_upscale"}:
        return str(_write_image_upscale_scratchpad(resolved, run_workspace)), False
    if resolved.route_key in {
        "qwen_image_edit",
        "qwen_image_style",
        "image_inpaint",
        "annotated_image_edit",
    }:
        return str(_write_qwen_image_edit_scratchpad(resolved, run_workspace)), False
    if resolved.route_key == "wan_2_2_t2i":
        return str(_write_wan_2_2_t2i_scratchpad(resolved, run_workspace)), False
    if resolved.route_key == "wan_2_2_i2v":
        return str(_write_wan_2_2_i2v_scratchpad(resolved, run_workspace)), False
    if _is_wan_i2v_first_last_route(resolved.route_key):
        return str(_write_wan_2_2_i2v_first_last_scratchpad(resolved, run_workspace)), False
    if resolved.route_key == "animate_character":
        return str(_write_animate_character_scratchpad(resolved, run_workspace)), False
    if resolved.route_key == "video_enhance":
        return str(_write_video_enhance_scratchpad(resolved, run_workspace)), False
    if resolved.route_key == "flux_klein_edit":
        return str(_write_flux_klein_edit_scratchpad(resolved, run_workspace)), False
    if _is_wan_vace_route(resolved.route_key):
        return str(_write_wan_2_2_vace_scratchpad(resolved, run_workspace)), False
    if _is_ltx_first_last_control_route(resolved.route_key):
        return str(_write_ltx_first_last_control_scratchpad(resolved, run_workspace)), False
    if _is_ltx_first_last_route(resolved.route_key):
        return str(_write_ltx_first_last_scratchpad(resolved, run_workspace)), False
    return str(resolved.template_id), True


def _maybe_postprocess_vibecomfy_output(
    *,
    resolved: ResolvedTask,
    output_path: Path,
    run_workspace: Path,
) -> Path | None:
    if resolved.route_key != "video_enhance" or not _bool_param(resolved.params, "enable_interpolation"):
        return None
    interpolation = resolved.params.get("interpolation")
    interpolation_params = interpolation if isinstance(interpolation, Mapping) else {}
    exp = _rife_exp_from_interpolation_params(interpolation_params)
    exp = max(1, min(exp, 2))
    fps = int(float(resolved.params.get("fps") or 16))
    target = run_workspace / "output" / f"video-enhance-rife-x{2 ** exp}.mp4"
    return _rife_interpolate_video(output_path, target, fps=fps, exp=exp)


def _rife_exp_from_interpolation_params(interpolation_params: Mapping[str, Any]) -> int:
    if "exp" in interpolation_params or "rife_exp" in interpolation_params:
        return int(interpolation_params.get("exp") or interpolation_params.get("rife_exp") or 1)
    inserted_frames = interpolation_params.get("num_frames")
    if inserted_frames is None:
        return 1
    inserted = max(1, int(inserted_frames))
    return 2 if inserted >= 3 else 1


def _rife_interpolate_video(input_path: Path, output_path: Path, *, fps: int, exp: int) -> Path:
    import cv2
    import numpy as np
    import torch

    from source.media.video.ffmpeg_ops import create_video_from_frames_list
    from source.media.video.frame_extraction import extract_frames_from_video
    from source.runtime.wgp_bridge import run_rife_temporal_interpolation

    frames_bgr = extract_frames_from_video(input_path)
    if len(frames_bgr) < 2:
        raise ValueError(f"RIFE interpolation requires at least two frames, got {len(frames_bgr)} from {input_path}")

    height, width = frames_bgr[0].shape[:2]
    headless_logger.info(
        "VibeComfy video_enhance RIFE postprocess starting: "
        f"input={input_path} output={output_path} frames={len(frames_bgr)} "
        f"size={width}x{height} fps={fps} exp={exp}"
    )
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    sample_np = np.stack(frames_rgb, axis=0).astype(np.float32) / 127.5 - 1.0
    sample = torch.from_numpy(sample_np).permute(3, 0, 1, 2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = sample.to(device)

    ckpt = _rife_checkpoint_path(prefer_v4=True)
    rife_version = "v4" if ckpt.name == "rife4.26.pkl" else "v3"
    headless_logger.info(
        "VibeComfy video_enhance RIFE invoking model: "
        f"checkpoint={ckpt} rife_version={rife_version} device={device} sample_shape={tuple(sample.shape)}"
    )
    output = run_rife_temporal_interpolation(str(ckpt), sample, exp, device=device, rife_version=rife_version)
    if output is None:
        raise RuntimeError("RIFE interpolation returned no frames")

    output = output.to("cpu")
    frames_out: list[np.ndarray] = []
    for index in range(output.shape[1]):
        frame = output[:, index]
        frame_rgb = ((frame.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        frames_out.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = create_video_from_frames_list(frames_out, output_path, fps * (2 ** exp), (width, height))
    headless_logger.info(
        "VibeComfy video_enhance RIFE postprocess finished: "
        f"output={written} output_frames={len(frames_out)} output_fps={fps * (2 ** exp)}"
    )
    return written


def _rife_checkpoint_path(*, prefer_v4: bool) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    ckpt_dir = repo_root / "Wan2GP" / "ckpts"
    candidates = []
    if prefer_v4:
        candidates.append(ckpt_dir / "rife4.26.pkl")
    candidates.append(ckpt_dir / "flownet.pkl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    download_errors = []
    for candidate in candidates:
        try:
            downloaded = _download_rife_checkpoint(candidate.name, ckpt_dir)
        except Exception as exc:
            download_errors.append(f"{candidate.name}: {exc}")
            continue
        if downloaded.is_file():
            return downloaded
    detail = "; ".join(download_errors) if download_errors else "no candidates attempted"
    raise FileNotFoundError(f"RIFE checkpoint missing; failed to download rife4.26.pkl or flownet.pkl ({detail})")


def _download_rife_checkpoint(filename: str, ckpt_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id="DeepBeepMeep/Wan2.1",
            filename=filename,
            local_dir=str(ckpt_dir),
        )
    )


def _write_z_image_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    width, height = _parse_resolution(resolved.params.get("resolution") or "1024x1024")
    prompt = str(resolved.params.get("prompt") or "")
    seed = int(resolved.params.get("seed", -1))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 8)))
    scratchpad = run_workspace / "z_image_turbo_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "from vibecomfy.patches.resolution import resolution",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('image/z_image')",
                f"    resolution({width}, {height}).apply(workflow)",
                f"    workflow.set_prompt({json.dumps(prompt)})",
                f"    workflow.set_seed({seed})",
                f"    workflow.set_steps({steps})",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_qwen_image_2512_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    width, height = _parse_resolution(resolved.params.get("resolution") or "768x768")
    prompt = str(resolved.params.get("prompt") or "")
    seed = int(resolved.params.get("seed", -1))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 4)))
    scratchpad = run_workspace / "qwen_image_2512_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "from vibecomfy.patches.resolution import resolution",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('image/qwen_image_2512')",
                f"    resolution({width}, {height}).apply(workflow)",
                f"    workflow.set_prompt({json.dumps(prompt)})",
                f"    workflow.set_seed({seed})",
                f"    workflow.nodes['238:224'].inputs['value'] = {steps}",
                f"    workflow.nodes['238:225'].inputs['value'] = {steps}",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_qwen_image_edit_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    input_name = _materialize_qwen_edit_input(resolved, run_workspace)
    prompt = str(resolved.params.get("prompt") or _default_qwen_edit_prompt(resolved.route_key))
    seed = int(resolved.params.get("seed", -1))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 4)))
    scratchpad = run_workspace / f"{resolved.route_key}_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('edit/qwen_image_edit')",
                f"    workflow.nodes['78'].inputs['image'] = {json.dumps(input_name)}",
                f"    workflow.nodes['102:76'].inputs['image'] = ['78', 0]",
                f"    workflow.nodes['102:77'].inputs['image'] = ['78', 0]",
                f"    workflow.nodes['102:88'].inputs['pixels'] = ['78', 0]",
                f"    workflow.set_prompt({json.dumps(prompt)})",
                f"    workflow.set_seed({seed})",
                f"    workflow.nodes['102:103'].inputs['value'] = {steps}",
                f"    workflow.nodes['102:106'].inputs['value'] = {steps}",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_z_image_img2img_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    input_name = _materialize_image_input(
        resolved,
        run_workspace,
        "image",
        "image_url",
        "image_guide",
        fallback_filename=f"z_image_img2img_{resolved.task_id}.png",
    )
    width, height = _parse_resolution(
        resolved.params.get("resolution")
        or resolved.params.get("parsed_resolution_wh")
        or "1024x1024"
    )
    prompt = str(resolved.params.get("prompt") or "")
    seed = int(resolved.params.get("seed", resolved.params.get("seed_to_use", -1)))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 8)))
    denoise = float(resolved.params.get("denoising_strength", resolved.params.get("denoise", 0.7)))
    scratchpad = run_workspace / "z_image_img2img_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('image/z_image_img2img')",
                f"    input_name = {json.dumps(input_name)}",
                "    for node in workflow.nodes.values():",
                "        if node.class_type == 'LoadImage':",
                "            node.inputs['image'] = input_name",
                "        elif node.class_type == 'ImageScale':",
                f"            node.inputs['width'] = {width}",
                f"            node.inputs['height'] = {height}",
                "        elif node.class_type == 'KSampler':",
                f"            node.inputs['seed'] = {seed}",
                f"            node.inputs['steps'] = {steps}",
                f"            node.inputs['denoise'] = {denoise}",
                f"    workflow.set_prompt({json.dumps(prompt)})",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_image_upscale_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    input_name = _materialize_image_input(
        resolved,
        run_workspace,
        "image",
        "image_url",
        fallback_filename=f"image_upscale_{resolved.task_id}.png",
    )
    scale = float(resolved.params.get("scale_factor") or resolved.params.get("upscale_factor") or 2)
    scratchpad = run_workspace / "image_upscale_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('image/basic_image_upscale')",
                f"    workflow.nodes['1'].inputs['image'] = {json.dumps(input_name)}",
                f"    workflow.nodes['2'].inputs['scale_by'] = {scale}",
                "    workflow.nodes['2'].inputs['upscale_method'] = 'lanczos'",
                "    workflow.nodes['3'].inputs['filename_prefix'] = 'image-upscale'",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_wan_2_2_t2i_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    width, height = _parse_resolution(
        resolved.params.get("resolution")
        or resolved.params.get("parsed_resolution_wh")
        or "832x480"
    )
    prompt = str(resolved.params.get("prompt") or "")
    negative = str(resolved.params.get("negative_prompt") or "fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted")
    seed = int(resolved.params.get("seed", resolved.params.get("seed_to_use", -1)))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 6)))
    cfg_1 = float(resolved.params.get("guidance_scale", 3))
    cfg_2 = float(resolved.params.get("guidance2_scale", 1))
    shift = float(resolved.params.get("flow_shift", 5))
    loras, lora_assets = _wanvideo_dynamic_lora_payloads(resolved)
    scratchpad = run_workspace / "wan_2_2_t2i_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                _WANVIDEO_DEFAULTS_HELPER,
                "",
                _WANVIDEO_DYNAMIC_LORA_HELPER,
                "",
                "def build():",
                "    workflow = load_workflow_any('video/wanvideo_wrapper_22_14b_t2i')",
                f"    _append_model_assets(workflow, {json.dumps(lora_assets)})",
                f"    _patch_wanvideo_dynamic_loras(workflow, {json.dumps(loras)})",
                f"    _patch_wanvideo_defaults(workflow, steps={steps}, cfg={cfg_1}, shift={shift}, seed={seed})",
                f"    workflow.nodes['78'].inputs['widget_0'] = {width}",
                f"    workflow.nodes['78'].inputs['widget_1'] = {height}",
                "    workflow.nodes['78'].inputs['widget_2'] = 1",
                f"    workflow.nodes['78'].inputs['width'] = {width}",
                f"    workflow.nodes['78'].inputs['height'] = {height}",
                "    workflow.nodes['78'].inputs['num_frames'] = 1",
                f"    workflow.nodes['16'].inputs['widget_0'] = {json.dumps(prompt)}",
                f"    workflow.nodes['16'].inputs['widget_1'] = {json.dumps(negative)}",
                f"    workflow.nodes['16'].inputs['positive_prompt'] = {json.dumps(prompt)}",
                f"    workflow.nodes['16'].inputs['negative_prompt'] = {json.dumps(negative)}",
                f"    workflow.nodes['27'].inputs['steps'] = {steps}",
                f"    workflow.nodes['27'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['27'].inputs['cfg'] = {cfg_1}",
                f"    workflow.nodes['27'].inputs['shift'] = {shift}",
                f"    workflow.nodes['27'].inputs['seed'] = {seed}",
                f"    workflow.nodes['27'].inputs['widget_1'] = {cfg_1}",
                f"    workflow.nodes['27'].inputs['widget_2'] = {shift}",
                f"    workflow.nodes['27'].inputs['widget_3'] = {seed}",
                f"    workflow.nodes['87'].inputs['steps'] = {steps}",
                f"    workflow.nodes['87'].inputs['cfg'] = {cfg_2}",
                f"    workflow.nodes['87'].inputs['shift'] = {shift}",
                f"    workflow.nodes['87'].inputs['seed'] = {seed}",
                f"    workflow.nodes['87'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['87'].inputs['widget_1'] = {cfg_2}",
                f"    workflow.nodes['87'].inputs['widget_2'] = {shift}",
                f"    workflow.nodes['87'].inputs['widget_3'] = {seed}",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_wan_2_2_vace_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    start_name = _materialize_image_input(
        resolved,
        run_workspace,
        "start_image_url",
        "start_image",
        "image",
        "image_url",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        list_keys=("input_image_paths_resolved",),
        list_index=0,
        fallback_filename=f"vace_start_{resolved.task_id}.png",
    )
    end_name = _materialize_image_input(
        resolved,
        run_workspace,
        "end_image_url",
        "end_image",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        list_keys=("input_image_paths_resolved",),
        list_index=1,
        fallback_filename=f"vace_end_{resolved.task_id}.png",
    )
    control_name = _materialize_optional_video_input(
        resolved,
        run_workspace,
        "video_source",
        "video_guide",
        "structure_video_path",
        "source_video_url",
        "vid2vid_source_video_path",
        nested_keys=("travel_guidance", "individual_segment_params", "segment_params", "orchestrator_details"),
        fallback_filename=f"vace_control_{resolved.task_id}.mp4",
    )
    width, height = _parse_resolution(
        resolved.params.get("resolution")
        or resolved.params.get("parsed_resolution_wh")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "parsed_resolution_wh")
        or "832x480"
    )
    frames = int(
        resolved.params.get("num_frames")
        or resolved.params.get("video_length")
        or _first_nested_value(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "num_frames", "video_length")
        or 81
    )
    fps = int(float(resolved.params.get("fps") or resolved.params.get("fps_helpers") or 16))
    prompt = str(
        resolved.params.get("prompt")
        or resolved.params.get("base_prompt")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "prompt", "base_prompt")
        or ""
    )
    negative = str(
        resolved.params.get("negative_prompt")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "negative_prompt")
        or "fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted"
    )
    seed = int(
        resolved.params.get("seed")
        or resolved.params.get("seed_to_use")
        or _first_nested_value(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "seed", "seed_to_use", "seed_base")
        or -1
    )
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 6)))
    cfg = float(resolved.params.get("guidance_scale", 3))
    shift = float(resolved.params.get("flow_shift", 5))
    loras, lora_assets = _wanvideo_dynamic_lora_payloads(resolved)
    scratchpad = run_workspace / "wan_2_2_vace_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                _WANVIDEO_DEFAULTS_HELPER,
                "",
                _WANVIDEO_DYNAMIC_LORA_HELPER,
                "",
                "def build():",
                "    workflow = load_workflow_any('video/wanvideo_wrapper_22_14b_vace_cocktail')",
                f"    _append_model_assets(workflow, {json.dumps(lora_assets)})",
                f"    _patch_wanvideo_dynamic_loras(workflow, {json.dumps(loras)})",
                f"    _patch_wanvideo_defaults(workflow, steps={steps}, cfg={cfg}, shift={shift}, seed={seed})",
                "    block_swap = workflow.nodes['39'].inputs",
                "    block_swap['blocks_to_swap'] = max(int(block_swap.get('blocks_to_swap', block_swap.get('widget_0', 0)) or 0), 30)",
                "    block_swap['widget_0'] = block_swap['blocks_to_swap']",
                "    block_swap['offload_txt_emb'] = True",
                "    block_swap['offload_img_emb'] = True",
                "    block_swap['offload_img_emb_nonblock'] = True",
                "    block_swap['widget_1'] = True",
                "    block_swap['widget_2'] = True",
                "    block_swap['widget_3'] = True",
                "    block_swap['vace_blocks_to_swap'] = max(int(block_swap.get('vace_blocks_to_swap', block_swap.get('widget_4', 0)) or 0), 8)",
                "    block_swap['widget_4'] = block_swap['vace_blocks_to_swap']",
                f"    workflow.nodes['64'].inputs['image'] = {json.dumps(start_name)}",
                f"    workflow.nodes['112'].inputs['image'] = {json.dumps(end_name)}",
                f"    control_name = {json.dumps(control_name)}",
                "    if control_name:",
                "        workflow.nodes['199'].inputs['video'] = control_name",
                f"        workflow.nodes['199'].inputs['force_rate'] = {fps}",
                f"        workflow.nodes['199'].inputs['custom_width'] = {width}",
                f"        workflow.nodes['199'].inputs['custom_height'] = {height}",
                f"        workflow.nodes['199'].inputs['frame_load_cap'] = {frames}",
                "    else:",
                "        workflow.disconnect('111.control_images')",
                f"    workflow.nodes['111'].inputs['widget_0'] = {frames}",
                f"    workflow.nodes['56'].inputs['widget_0'] = {width}",
                f"    workflow.nodes['56'].inputs['widget_1'] = {height}",
                f"    workflow.nodes['56'].inputs['widget_2'] = {frames}",
                f"    workflow.nodes['56'].inputs['width'] = {width}",
                f"    workflow.nodes['56'].inputs['height'] = {height}",
                f"    workflow.nodes['56'].inputs['num_frames'] = {frames}",
                f"    workflow.nodes['16'].inputs['widget_0'] = {json.dumps(prompt)}",
                f"    workflow.nodes['16'].inputs['widget_1'] = {json.dumps(negative)}",
                f"    workflow.nodes['16'].inputs['positive_prompt'] = {json.dumps(prompt)}",
                f"    workflow.nodes['16'].inputs['negative_prompt'] = {json.dumps(negative)}",
                f"    workflow.nodes['139'].inputs['frame_rate'] = {fps}",
                "    workflow.nodes['139'].inputs['loop_count'] = 0",
                "    workflow.nodes['139'].inputs['filename_prefix'] = 'Wan-2-2-VACE'",
                "    workflow.nodes['139'].inputs['format'] = 'video/h264-mp4'",
                "    workflow.nodes['139'].inputs['pingpong'] = False",
                "    workflow.nodes['139'].inputs['save_output'] = True",
                f"    workflow.nodes['27'].inputs['steps'] = {steps}",
                f"    workflow.nodes['27'].inputs['cfg'] = {cfg}",
                f"    workflow.nodes['27'].inputs['shift'] = {shift}",
                f"    workflow.nodes['27'].inputs['seed'] = {seed}",
                f"    workflow.nodes['27'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['27'].inputs['widget_3'] = {seed}",
                f"    workflow.nodes['87'].inputs['steps'] = {steps}",
                "    workflow.nodes['87'].inputs['cfg'] = 1.0",
                f"    workflow.nodes['87'].inputs['shift'] = {shift}",
                f"    workflow.nodes['87'].inputs['seed'] = {seed}",
                f"    workflow.nodes['87'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['87'].inputs['widget_3'] = {seed}",
                f"    workflow.nodes['197'].inputs['steps'] = {steps}",
                "    workflow.nodes['197'].inputs['cfg'] = 1.0",
                f"    workflow.nodes['197'].inputs['shift'] = {shift}",
                f"    workflow.nodes['197'].inputs['seed'] = {seed}",
                f"    workflow.nodes['197'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['197'].inputs['widget_3'] = {seed}",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_wan_2_2_i2v_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    input_name = _materialize_image_input(
        resolved,
        run_workspace,
        "image",
        "image_url",
        "input_image",
        "start_image",
        "start_image_url",
        "first_frame",
        "first_frame_url",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        list_keys=("input_image_paths_resolved",),
        list_index=0,
        fallback_filename=f"wan_2_2_i2v_{resolved.task_id}.png",
    )
    width, height = _parse_resolution(
        resolved.params.get("resolution")
        or resolved.params.get("parsed_resolution_wh")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "parsed_resolution_wh")
        or "832x480"
    )
    frames = int(
        resolved.params.get("num_frames")
        or resolved.params.get("video_length")
        or _first_nested_value(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "num_frames", "video_length")
        or 81
    )
    fps = int(float(resolved.params.get("fps") or resolved.params.get("fps_helpers") or 16))
    prompt = str(
        resolved.params.get("prompt")
        or resolved.params.get("base_prompt")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "prompt", "base_prompt")
        or ""
    )
    negative = str(
        resolved.params.get("negative_prompt")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "negative_prompt")
        or "fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted"
    )
    seed = int(
        resolved.params.get("seed")
        or resolved.params.get("seed_to_use")
        or _first_nested_value(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "seed", "seed_to_use", "seed_base")
        or -1
    )
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 6)))
    end_step = max(1, min(steps - 1, int(resolved.params.get("high_noise_end_step", 3))))
    loras, lora_assets = _wanvideo_dynamic_lora_payloads(resolved)
    scratchpad = run_workspace / "wan_2_2_i2v_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                _WANVIDEO_DEFAULTS_HELPER,
                "",
                _WANVIDEO_DYNAMIC_LORA_HELPER,
                "",
                "def build():",
                "    workflow = load_workflow_any('video/wanvideo_wrapper_22_14b_i2v_kijai')",
                f"    _append_model_assets(workflow, {json.dumps(lora_assets)})",
                f"    _chain_wanvideo_select_loras(workflow, {json.dumps(loras)})",
                f"    _patch_wanvideo_defaults(workflow, steps={steps}, cfg=1.0, shift=8.0, seed={seed})",
                "    workflow.nodes['39'].inputs['blocks_to_swap'] = max(int(workflow.nodes['39'].inputs.get('blocks_to_swap', 0) or 0), 35)",
                "    workflow.nodes['39'].inputs['widget_0'] = workflow.nodes['39'].inputs['blocks_to_swap']",
                "    workflow.nodes['39'].inputs['offload_img_emb'] = True",
                "    workflow.nodes['39'].inputs['offload_txt_emb'] = True",
                "    workflow.nodes['39'].inputs['use_non_blocking'] = False",
                "    workflow.nodes['39'].inputs['widget_1'] = True",
                "    workflow.nodes['39'].inputs['widget_2'] = True",
                "    workflow.nodes['39'].inputs['widget_3'] = False",
                f"    workflow.nodes['67'].inputs['widget_0'] = {json.dumps(input_name)}",
                f"    workflow.nodes['67'].inputs['image'] = {json.dumps(input_name)}",
                f"    workflow.nodes['68'].inputs['widget_0'] = {width}",
                f"    workflow.nodes['68'].inputs['widget_1'] = {height}",
                f"    workflow.nodes['68'].inputs['width'] = {width}",
                f"    workflow.nodes['68'].inputs['height'] = {height}",
                f"    workflow.nodes['89'].inputs['widget_0'] = {width}",
                f"    workflow.nodes['89'].inputs['widget_1'] = {height}",
                f"    workflow.nodes['89'].inputs['widget_2'] = {frames}",
                f"    workflow.nodes['89'].inputs['width'] = {width}",
                f"    workflow.nodes['89'].inputs['height'] = {height}",
                f"    workflow.nodes['89'].inputs['num_frames'] = {frames}",
                f"    empty_embeds = workflow.add_node('WanVideoEmptyEmbeds', widget_0={width}, widget_1={height}, widget_2={frames}, width={width}, height={height}, num_frames={frames})",
                "    # The Kijai Wan 2.2 A14B HIGH/LOW pair reports in_dim=16 in the sampler.",
                "    # Feed empty embeds to both samplers and keep node 89 for image-side conditioning.",
                "    workflow.replace_edge('27.image_embeds', f'{empty_embeds.id}.0')",
                "    workflow.replace_edge('90.image_embeds', f'{empty_embeds.id}.0')",
                f"    workflow.nodes['16'].inputs['widget_0'] = {json.dumps(prompt)}",
                f"    workflow.nodes['16'].inputs['widget_1'] = {json.dumps(negative)}",
                f"    workflow.nodes['16'].inputs['positive_prompt'] = {json.dumps(prompt)}",
                f"    workflow.nodes['16'].inputs['negative_prompt'] = {json.dumps(negative)}",
                f"    workflow.nodes['27'].inputs['steps'] = {steps}",
                f"    workflow.nodes['27'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['27'].inputs['widget_3'] = {seed}",
                f"    workflow.nodes['27'].inputs['seed'] = {seed}",
                f"    workflow.nodes['27'].inputs['end_step'] = {end_step}",
                f"    workflow.nodes['27'].inputs['widget_12'] = {end_step}",
                f"    workflow.nodes['90'].inputs['steps'] = {steps}",
                f"    workflow.nodes['90'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['90'].inputs['widget_3'] = {seed}",
                f"    workflow.nodes['90'].inputs['seed'] = {seed}",
                f"    workflow.nodes['90'].inputs['start_step'] = {end_step}",
                f"    workflow.nodes['90'].inputs['widget_11'] = {end_step}",
                f"    workflow.nodes['91'].inputs['widget_0'] = {end_step}",
                f"    workflow.nodes['94'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['60'].inputs['frame_rate'] = {fps}",
                "    workflow.nodes['60'].inputs['filename_prefix'] = 'WanVideo2_2_I2V'",
                "    workflow.nodes['60'].inputs['format'] = 'video/h264-mp4'",
                "    workflow.nodes['60'].inputs['save_output'] = True",
                "    workflow.nodes['60'].inputs['trim_to_audio'] = False",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _is_wan_i2v_first_last_route(route_key: str) -> bool:
    return route_key == "travel_segment__model-wan22_i2v__guidance-none__continuity-first_last__profile-default"


def _write_wan_2_2_i2v_first_last_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    start_name = _materialize_image_input(
        resolved,
        run_workspace,
        "start_image_url",
        "start_image",
        "image",
        "image_url",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        list_keys=("input_image_paths_resolved",),
        list_index=0,
        fallback_filename=f"wan_2_2_i2v_start_{resolved.task_id}.png",
    )
    end_name = _materialize_image_input(
        resolved,
        run_workspace,
        "end_image_url",
        "end_image",
        "image_end",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        list_keys=("input_image_paths_resolved",),
        list_index=1,
        fallback_filename=f"wan_2_2_i2v_end_{resolved.task_id}.png",
    )
    width, height = _parse_resolution(
        resolved.params.get("resolution")
        or resolved.params.get("parsed_resolution_wh")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "parsed_resolution_wh")
        or "832x480"
    )
    frames = int(
        resolved.params.get("num_frames")
        or resolved.params.get("video_length")
        or _first_nested_value(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "num_frames", "video_length")
        or 81
    )
    fps = int(float(resolved.params.get("fps") or resolved.params.get("fps_helpers") or 16))
    prompt = str(
        resolved.params.get("prompt")
        or resolved.params.get("base_prompt")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "prompt", "base_prompt")
        or ""
    )
    negative = str(
        resolved.params.get("negative_prompt")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "negative_prompt")
        or "fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted"
    )
    seed = int(
        resolved.params.get("seed")
        or resolved.params.get("seed_to_use")
        or _first_nested_value(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "seed", "seed_to_use", "seed_base")
        or -1
    )
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 6)))
    end_step = max(1, min(steps - 1, int(resolved.params.get("high_noise_end_step", 3))))
    loras, lora_assets = _wanvideo_dynamic_lora_payloads(resolved)
    scratchpad = run_workspace / "wan_2_2_i2v_first_last_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                _WANVIDEO_DEFAULTS_HELPER,
                "",
                _WANVIDEO_DYNAMIC_LORA_HELPER,
                "",
                "def build():",
                "    workflow = load_workflow_any('video/wanvideo_wrapper_22_14b_i2v_kijai')",
                f"    _append_model_assets(workflow, {json.dumps(lora_assets)})",
                f"    _chain_wanvideo_select_loras(workflow, {json.dumps(loras)})",
                f"    _patch_wanvideo_defaults(workflow, steps={steps}, cfg=1.0, shift=8.0, seed={seed})",
                "    workflow.nodes['39'].inputs['blocks_to_swap'] = max(int(workflow.nodes['39'].inputs.get('blocks_to_swap', 0) or 0), 35)",
                "    workflow.nodes['39'].inputs['widget_0'] = workflow.nodes['39'].inputs['blocks_to_swap']",
                "    workflow.nodes['39'].inputs['offload_img_emb'] = True",
                "    workflow.nodes['39'].inputs['offload_txt_emb'] = True",
                "    workflow.nodes['39'].inputs['use_non_blocking'] = False",
                "    workflow.nodes['39'].inputs['widget_1'] = True",
                "    workflow.nodes['39'].inputs['widget_2'] = True",
                "    workflow.nodes['39'].inputs['widget_3'] = False",
                f"    workflow.nodes['67'].inputs['widget_0'] = {json.dumps(start_name)}",
                f"    workflow.nodes['67'].inputs['image'] = {json.dumps(start_name)}",
                f"    end_image = workflow.add_node('LoadImage', image={json.dumps(end_name)}, widget_0={json.dumps(end_name)}, widget_1='image')",
                f"    end_resized = workflow.add_node('ImageResizeKJv2', widget_0={width}, widget_1={height}, width={width}, height={height}, upscale_method='lanczos', keep_proportion='crop', pad_color='0, 0, 0', crop_position='center', divisible_by=32, device='cpu')",
                "    workflow.connect(f'{end_image.id}.0', f'{end_resized.id}.image')",
                "    workflow.connect(f'{end_resized.id}.0', '89.end_image')",
                f"    workflow.nodes['68'].inputs['widget_0'] = {width}",
                f"    workflow.nodes['68'].inputs['widget_1'] = {height}",
                f"    workflow.nodes['68'].inputs['width'] = {width}",
                f"    workflow.nodes['68'].inputs['height'] = {height}",
                f"    workflow.nodes['89'].inputs['widget_0'] = {width}",
                f"    workflow.nodes['89'].inputs['widget_1'] = {height}",
                f"    workflow.nodes['89'].inputs['widget_2'] = {frames}",
                f"    workflow.nodes['89'].inputs['width'] = {width}",
                f"    workflow.nodes['89'].inputs['height'] = {height}",
                f"    workflow.nodes['89'].inputs['num_frames'] = {frames}",
                "    workflow.nodes['89'].inputs['fun_or_fl2v_model'] = False",
                f"    workflow.nodes['16'].inputs['widget_0'] = {json.dumps(prompt)}",
                f"    workflow.nodes['16'].inputs['widget_1'] = {json.dumps(negative)}",
                f"    workflow.nodes['16'].inputs['positive_prompt'] = {json.dumps(prompt)}",
                f"    workflow.nodes['16'].inputs['negative_prompt'] = {json.dumps(negative)}",
                f"    workflow.nodes['27'].inputs['steps'] = {steps}",
                f"    workflow.nodes['27'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['27'].inputs['widget_3'] = {seed}",
                f"    workflow.nodes['27'].inputs['seed'] = {seed}",
                f"    workflow.nodes['27'].inputs['end_step'] = {end_step}",
                f"    workflow.nodes['27'].inputs['widget_12'] = {end_step}",
                f"    workflow.nodes['90'].inputs['steps'] = {steps}",
                f"    workflow.nodes['90'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['90'].inputs['widget_3'] = {seed}",
                f"    workflow.nodes['90'].inputs['seed'] = {seed}",
                f"    workflow.nodes['90'].inputs['start_step'] = {end_step}",
                f"    workflow.nodes['90'].inputs['widget_11'] = {end_step}",
                f"    workflow.nodes['91'].inputs['widget_0'] = {end_step}",
                f"    workflow.nodes['94'].inputs['widget_0'] = {steps}",
                f"    workflow.nodes['60'].inputs['frame_rate'] = {fps}",
                "    workflow.nodes['60'].inputs['filename_prefix'] = 'WanVideo2_2_I2V_FirstLast'",
                "    workflow.nodes['60'].inputs['format'] = 'video/h264-mp4'",
                "    workflow.nodes['60'].inputs['save_output'] = True",
                "    workflow.nodes['60'].inputs['trim_to_audio'] = False",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_animate_character_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    reference_name = _materialize_image_input(
        resolved,
        run_workspace,
        "character_image_url",
        "reference_image",
        "reference_image_url",
        "image",
        "image_url",
        fallback_filename=f"animate_character_reference_{resolved.task_id}.png",
    )
    motion_name = _materialize_optional_video_input(
        resolved,
        run_workspace,
        "motion_video_url",
        "motion_video",
        "pose_video_url",
        "pose_video",
        "video",
        "video_url",
        fallback_filename=f"animate_character_motion_{resolved.task_id}.mp4",
    )
    if not motion_name:
        raise ValueError("VibeComfy route 'animate_character' requires motion_video_url or motion_video")
    width, height = _parse_resolution(resolved.params.get("resolution") or "832x480")
    prompt = str(resolved.params.get("prompt") or "Animate the character following the motion reference.")
    negative = str(
        resolved.params.get("negative_prompt")
        or "fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted"
    )
    seed = int(resolved.params.get("seed", resolved.params.get("seed_to_use", -1)))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 4)))
    fps = int(float(resolved.params.get("fps") or 16))
    frames = int(resolved.params.get("num_frames") or resolved.params.get("video_length") or 49)
    loras, lora_assets = _wanvideo_dynamic_lora_payloads(resolved)
    scratchpad = run_workspace / "animate_character_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                _WANVIDEO_DYNAMIC_LORA_HELPER,
                "",
                "def build():",
                "    workflow = load_workflow_any('video/wan22_animate_native_first_stage')",
                f"    _append_model_assets(workflow, {json.dumps(lora_assets)})",
                f"    _chain_lora_loader_model_only(workflow, {json.dumps(loras)})",
                f"    workflow.nodes['10'].inputs['image'] = {json.dumps(reference_name)}",
                f"    workflow.nodes['145'].inputs['file'] = {json.dumps(motion_name)}",
                f"    workflow.nodes['159'].inputs['value'] = {width}",
                f"    workflow.nodes['160'].inputs['value'] = {height}",
                f"    workflow.nodes['229'].inputs['width'] = {width}",
                f"    workflow.nodes['229'].inputs['height'] = {height}",
                f"    workflow.nodes['21'].inputs['text'] = {json.dumps(prompt)}",
                f"    workflow.nodes['1'].inputs['text'] = {json.dumps(negative)}",
                f"    workflow.nodes['232:63'].inputs['steps'] = {steps}",
                f"    workflow.nodes['232:63'].inputs['seed'] = {seed}",
                f"    workflow.nodes['232:62'].inputs['length'] = {frames}",
                f"    workflow.nodes['232:230'].inputs['length'] = {frames}",
                f"    workflow.nodes['232:15'].inputs['fps'] = {fps}",
                "    workflow.nodes['19'].inputs['filename_prefix'] = 'Wanimate'",
                "    workflow.nodes['19'].inputs['format'] = 'mp4'",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_video_enhance_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    video_name = _materialize_optional_video_input(
        resolved,
        run_workspace,
        "video_url",
        "video",
        fallback_filename=f"video_enhance_{resolved.task_id}.mp4",
    )
    if not video_name:
        raise ValueError("VibeComfy route 'video_enhance' requires video_url or video")
    upscale = resolved.params.get("upscale")
    upscale_params = upscale if isinstance(upscale, Mapping) else {}
    enable_upscale = _bool_param(resolved.params, "enable_upscale")
    scale = float(upscale_params.get("upscale_factor") or resolved.params.get("upscale_factor") or 2)
    fps = int(float(resolved.params.get("fps") or 16))
    scratchpad = run_workspace / "video_enhance_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('video/basic_video_enhance')",
                f"    workflow.nodes['1'].inputs['video'] = {json.dumps(video_name)}",
                f"    workflow.nodes['1'].inputs['force_rate'] = {fps}",
                f"    workflow.nodes['4'].inputs['scale_by'] = {scale}",
                f"    workflow.nodes['5'].inputs['frame_rate'] = {fps}",
                "    workflow.nodes['5'].inputs['filename_prefix'] = 'video-enhance'",
                "    workflow.nodes['5'].inputs['format'] = 'video/h264-mp4'",
                "    workflow.nodes['5'].inputs['save_output'] = True",
                f"    enable_upscale = {enable_upscale!r}",
                "    if not enable_upscale:",
                "        workflow.replace_edge('5.images', '1.0')",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_flux_klein_edit_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    input_name = _materialize_image_input(
        resolved,
        run_workspace,
        "image",
        "image_url",
        fallback_filename=f"flux_klein_edit_{resolved.task_id}.png",
    )
    prompt = str(resolved.params.get("prompt") or "")
    seed = int(resolved.params.get("seed", resolved.params.get("seed_to_use", -1)))
    steps = int(resolved.params.get("steps", resolved.params.get("num_inference_steps", 8)))
    scratchpad = run_workspace / "flux_klein_edit_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('edit/flux2_klein_4b_image_edit_distilled')",
                f"    workflow.nodes['76'].inputs['image'] = {json.dumps(input_name)}",
                "    if '81' in workflow.nodes:",
                f"        workflow.nodes['81'].inputs['image'] = {json.dumps(input_name)}",
                f"    workflow.nodes['75:74'].inputs['text'] = {json.dumps(prompt)}",
                f"    workflow.nodes['75:73'].inputs['noise_seed'] = {seed}",
                f"    workflow.nodes['75:62'].inputs['steps'] = {steps}",
                "    workflow.nodes['9'].inputs['filename_prefix'] = 'Flux2-Klein-Edit'",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _materialize_qwen_edit_input(resolved: ResolvedTask, run_workspace: Path) -> str:
    input_dir = run_workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if resolved.route_key in {"image_inpaint", "annotated_image_edit"}:
        image_source = _first_string_param(resolved.params, "image_guide", "image_url", "image")
        mask_source = _first_string_param(resolved.params, "mask_url", "mask")
        if image_source and mask_source:
            composite = create_qwen_masked_composite(
                image_source,
                mask_source,
                input_dir,
                task_id=resolved.task_id,
            )
            return _copy_to_input_dir(composite, input_dir, f"{resolved.route_key}_{resolved.task_id}.jpg")

    image_source = _first_string_param(
        resolved.params,
        "image_guide",
        "image_url",
        "image",
        "style_reference_image",
        "subject_reference_image",
    )
    if not image_source:
        raise ValueError(f"VibeComfy route {resolved.route_key!r} requires an input image")
    return _copy_to_input_dir(
        download_image_if_url(image_source, input_dir, resolved.task_id),
        input_dir,
        f"{resolved.route_key}_{resolved.task_id}.png",
    )


def _is_wan_vace_route(route_key: str) -> bool:
    return (
        "model-wan22_vace" in route_key
        or route_key.startswith("join_clips_segment__model-wan22_vace__")
    )


def _is_ltx_first_last_route(route_key: str) -> bool:
    return (
        route_key.startswith("travel_segment__model-ltx2")
        and "__guidance-none__continuity-first_last__" in route_key
    )


def _is_ltx_first_last_control_route(route_key: str) -> bool:
    return (
        route_key.startswith("travel_segment__model-ltx2_distilled__")
        and "__guidance-ltx_control_" in route_key
        and "__continuity-first_last__" in route_key
    )


def _ltx_control_mode_from_route(route_key: str, params: Mapping[str, Any]) -> str:
    guidance = params.get("travel_guidance")
    if isinstance(guidance, Mapping):
        mode = guidance.get("mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip()
    marker = "__guidance-ltx_control_"
    if marker in route_key:
        return route_key.split(marker, 1)[1].split("__", 1)[0]
    return "video"


def _ltx_first_last_inputs(resolved: ResolvedTask, run_workspace: Path) -> dict[str, Any]:
    first_name = _materialize_image_input(
        resolved,
        run_workspace,
        "start_image_url",
        "start_image",
        "first_frame",
        "first_frame_url",
        "image",
        "image_url",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        list_keys=("input_image_paths_resolved",),
        list_index=0,
        fallback_filename=f"ltx_first_{resolved.task_id}.png",
    )
    last_name = _materialize_image_input(
        resolved,
        run_workspace,
        "end_image_url",
        "end_image",
        "last_frame",
        "last_frame_url",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        list_keys=("input_image_paths_resolved",),
        list_index=1,
        fallback_filename=f"ltx_last_{resolved.task_id}.png",
    )
    width, height = _parse_resolution(
        resolved.params.get("resolution")
        or resolved.params.get("parsed_resolution_wh")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "parsed_resolution_wh")
        or "1280x720"
    )
    frames = int(
        resolved.params.get("num_frames")
        or resolved.params.get("video_length")
        or _first_nested_value(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "num_frames", "video_length")
        or 121
    )
    fps = int(float(resolved.params.get("fps") or resolved.params.get("fps_helpers") or 24))
    prompt = str(
        resolved.params.get("prompt")
        or resolved.params.get("base_prompt")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "prompt", "base_prompt")
        or ""
    )
    negative = str(
        resolved.params.get("negative_prompt")
        or _first_nested_string(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "negative_prompt")
        or "blurry, oversaturated, pixelated, low resolution, grainy, distorted, noise, compression artifacts"
    )
    seed = int(
        resolved.params.get("seed")
        or resolved.params.get("seed_to_use")
        or _first_nested_value(resolved.params, ("individual_segment_params", "segment_params", "orchestrator_details"), "seed", "seed_to_use", "seed_base")
        or 42
    )
    return {
        "first_name": first_name,
        "last_name": last_name,
        "width": width,
        "height": height,
        "template_width": width * 2,
        "template_height": height * 2,
        "frames": frames,
        "fps": fps,
        "prompt": prompt,
        "negative": negative,
        "seed": seed,
        "first_strength": float(resolved.params.get("first_frame_strength", resolved.params.get("start_strength", 8))),
        "last_strength": float(resolved.params.get("last_frame_strength", resolved.params.get("end_strength", 8))),
    }


def _write_ltx_first_last_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    values = _ltx_first_last_inputs(resolved, run_workspace)
    scratchpad = run_workspace / "ltx_first_last_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('video/ltx2_3_runexx_first_last_frame')",
                f"    workflow.nodes['45'].inputs['image'] = {json.dumps(values['first_name'])}",
                f"    workflow.nodes['47'].inputs['image'] = {json.dumps(values['last_name'])}",
                f"    workflow.nodes['2103'].inputs['value'] = {json.dumps(values['prompt'])}",
                f"    workflow.nodes['11'].inputs['text'] = {json.dumps(values['negative'])}",
                f"    workflow.nodes['14'].inputs['noise_seed'] = {values['seed']}",
                f"    workflow.nodes['15'].inputs['noise_seed'] = {values['seed']}",
                *_ltx_exact_frame_count_scratchpad_lines(values),
                *_ltx_template_dimension_scratchpad_lines(values),
                f"    workflow.nodes['2076'].inputs['value'] = {values['fps']}",
                f"    workflow.nodes['2110'].inputs['value'] = {values['first_strength']}",
                f"    workflow.nodes['2108'].inputs['value'] = {values['last_strength']}",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _ltx_exact_frame_count_scratchpad_lines(values: Mapping[str, Any]) -> list[str]:
    frames = int(values["frames"])
    return [
        f"    workflow.nodes['2078'].inputs['widget_0'] = {frames}",
        f"    workflow.nodes['2078'].inputs['value'] = {frames}",
        "    workflow.nodes['2077'].inputs['widget_0'] = 'a'",
    ]


def _ltx_template_dimension_scratchpad_lines(values: Mapping[str, Any]) -> list[str]:
    # The Runexx LTX first/last templates downscale the configured canvas by
    # 0.5 before sampling. Worker params are final artifact dimensions.
    height = int(values["template_height"])
    width = int(values["template_width"])
    return [
        f"    workflow.nodes['2079'].inputs['widget_0'] = {height}",
        f"    workflow.nodes['2079'].inputs['value'] = {height}",
        f"    workflow.nodes['2080'].inputs['widget_0'] = {width}",
        f"    workflow.nodes['2080'].inputs['value'] = {width}",
    ]


def _write_ltx_first_last_control_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    mode = _ltx_control_mode_from_route(resolved.route_key, resolved.params)
    if mode == "video":
        return _write_ltx_first_last_raw_video_control_scratchpad(resolved, run_workspace)

    values = _ltx_first_last_inputs(resolved, run_workspace)
    control_name = _materialize_optional_video_input(
        resolved,
        run_workspace,
        "control_video",
        "control_video_url",
        "video_guide",
        "guide_video",
        "guide_video_path",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        fallback_filename=f"ltx_control_{resolved.task_id}.mp4",
    )
    if not control_name:
        raise ValueError(f"VibeComfy route {resolved.route_key!r} requires a materialized LTX control guide video")

    guidance = resolved.params.get("travel_guidance")
    guidance_strength = 1.0
    if isinstance(guidance, Mapping) and guidance.get("strength") is not None:
        guidance_strength = float(guidance["strength"])
    elif mode in {"pose", "depth", "canny"}:
        guidance_strength = 0.5
    lora_name = (
        "LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors"
        if mode == "cameraman"
        else "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
    )
    guide_source_ref = {
        "pose": "6102.0",
        "depth": "6103.0",
        "canny": "5028.0",
        "cameraman": "6101.0",
    }.get(mode)
    if guide_source_ref is None:
        raise ValueError(f"unsupported LTX control mode for VibeComfy first/last route: {mode!r}")

    scratchpad = run_workspace / "ltx_first_last_control_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('video/ltx2_3_first_last_frame_travel_iclora_control')",
                f"    workflow.nodes['45'].inputs['image'] = {json.dumps(values['first_name'])}",
                f"    workflow.nodes['47'].inputs['image'] = {json.dumps(values['last_name'])}",
                f"    workflow.nodes['5001'].inputs['file'] = {json.dumps(control_name)}",
                f"    workflow.nodes['5001'].inputs['video'] = {json.dumps(control_name)}",
                f"    workflow.nodes['5001'].inputs['widget_0'] = {json.dumps(control_name)}",
                f"    workflow.nodes['6000'].inputs['value'] = {json.dumps(mode)}",
                f"    workflow.nodes['16'].inputs['text'] = {json.dumps(values['prompt'])}",
                f"    workflow.nodes['11'].inputs['text'] = {json.dumps(values['negative'])}",
                f"    workflow.nodes['14'].inputs['noise_seed'] = {values['seed']}",
                f"    workflow.nodes['15'].inputs['noise_seed'] = {values['seed']}",
                *_ltx_exact_frame_count_scratchpad_lines(values),
                *_ltx_template_dimension_scratchpad_lines(values),
                f"    workflow.nodes['2076'].inputs['value'] = {values['fps']}",
                f"    workflow.nodes['2110'].inputs['value'] = {values['first_strength']}",
                f"    workflow.nodes['2108'].inputs['value'] = {values['last_strength']}",
                f"    workflow.nodes['5011'].inputs['lora_name'] = {json.dumps(lora_name)}",
                f"    workflow.nodes['5011'].inputs['widget_0'] = {json.dumps(lora_name)}",
                f"    workflow.nodes['5011'].inputs['widget_1'] = {guidance_strength}",
                f"    workflow.nodes['5012'].inputs['widget_1'] = {guidance_strength}",
                f"    workflow.replace_edge('5012.image', {json.dumps(guide_source_ref)})",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _write_ltx_first_last_raw_video_control_scratchpad(resolved: ResolvedTask, run_workspace: Path) -> Path:
    values = _ltx_first_last_inputs(resolved, run_workspace)
    control_name = _materialize_optional_video_input(
        resolved,
        run_workspace,
        "control_video",
        "control_video_url",
        "video_guide",
        "guide_video",
        "guide_video_path",
        nested_keys=("individual_segment_params", "segment_params", "orchestrator_details"),
        fallback_filename=f"ltx_control_{resolved.task_id}.mp4",
    )
    if not control_name:
        raise ValueError(f"VibeComfy route {resolved.route_key!r} requires a materialized LTX raw video guide")

    guidance = resolved.params.get("travel_guidance")
    guidance_strength = 1.0
    if isinstance(guidance, Mapping) and guidance.get("strength") is not None:
        guidance_strength = float(guidance["strength"])

    scratchpad = run_workspace / "ltx_first_last_raw_video_control_scratchpad.py"
    scratchpad.write_text(
        "\n".join(
            [
                "from vibecomfy.cli_loader import load_workflow_any",
                "",
                "",
                "def build():",
                "    workflow = load_workflow_any('video/ltx2_3_runexx_first_last_raw_video_guide')",
                f"    workflow.nodes['45'].inputs['image'] = {json.dumps(values['first_name'])}",
                f"    workflow.nodes['47'].inputs['image'] = {json.dumps(values['last_name'])}",
                f"    workflow.nodes['5001'].inputs['file'] = {json.dumps(control_name)}",
                f"    workflow.nodes['5001'].inputs['video'] = {json.dumps(control_name)}",
                f"    workflow.nodes['5001'].inputs['widget_0'] = {json.dumps(control_name)}",
                f"    workflow.nodes['2103'].inputs['value'] = {json.dumps(values['prompt'])}",
                f"    workflow.nodes['11'].inputs['text'] = {json.dumps(values['negative'])}",
                f"    workflow.nodes['14'].inputs['noise_seed'] = {values['seed']}",
                f"    workflow.nodes['15'].inputs['noise_seed'] = {values['seed']}",
                *_ltx_exact_frame_count_scratchpad_lines(values),
                *_ltx_template_dimension_scratchpad_lines(values),
                f"    workflow.nodes['2076'].inputs['value'] = {values['fps']}",
                f"    workflow.nodes['2110'].inputs['value'] = {values['first_strength']}",
                f"    workflow.nodes['2108'].inputs['value'] = {values['last_strength']}",
                f"    workflow.nodes['6102'].inputs['value'] = {guidance_strength}",
                "    return workflow.finalize_metadata()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return scratchpad


def _materialize_image_input(
    resolved: ResolvedTask,
    run_workspace: Path,
    *keys: str,
    nested_keys: tuple[str, ...] = (),
    list_keys: tuple[str, ...] = (),
    list_index: int = 0,
    fallback_filename: str,
) -> str:
    input_dir = run_workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    image_source = (
        _first_string_param(resolved.params, *keys)
        or _first_nested_string(resolved.params, nested_keys, *keys)
        or _first_list_string(resolved.params, list_keys, list_index)
        or _first_nested_list_string(resolved.params, nested_keys, list_keys, list_index)
    )
    if not image_source:
        raise ValueError(f"VibeComfy route {resolved.route_key!r} requires image input {keys!r}")
    return _copy_to_input_dir(
        download_image_if_url(image_source, input_dir, resolved.task_id),
        input_dir,
        fallback_filename,
    )


def _materialize_optional_video_input(
    resolved: ResolvedTask,
    run_workspace: Path,
    *keys: str,
    nested_keys: tuple[str, ...] = (),
    fallback_filename: str,
) -> str | None:
    input_dir = run_workspace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    video_source = (
        _first_string_param(resolved.params, *keys)
        or _first_nested_string(resolved.params, nested_keys, *keys)
        or _first_travel_guidance_video(resolved.params)
    )
    if not video_source:
        return None
    return _copy_to_input_dir(
        download_video_if_url(video_source, input_dir, resolved.task_id),
        input_dir,
        fallback_filename,
    )


def _copy_to_input_dir(source: str | Path, input_dir: Path, filename: str) -> str:
    source_path = Path(source)
    suffix = source_path.suffix or Path(filename).suffix or ".png"
    target = input_dir / (Path(filename).stem + suffix)
    if source_path.resolve() != target.resolve():
        shutil.copy2(source_path, target)
    return target.name


def _wanvideo_dynamic_lora_payloads(resolved: ResolvedTask) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return VibeComfy LoRA selector payloads and downloadable model assets.

    WanVideo templates keep the built-in LightX2V LoRA in slot 0. User LoRAs are
    staged under a Reigh-owned subfolder so downloaded URL basenames do not
    collide with template assets or Wan2GP's separate LoRA cache.
    """

    entries = _lora_entries_for_params(resolved.params, resolved=resolved)
    if not entries:
        return [], []
    if len(entries) > 4:
        raise ValueError("VibeComfy WanVideo dynamic LoRA support currently accepts at most four user LoRAs")

    loras: list[dict[str, Any]] = []
    assets: list[dict[str, str]] = []
    seen_assets: set[tuple[str, str]] = set()
    for entry in entries:
        filename = _lora_filename(entry)
        if not filename:
            continue
        loras.append(
            {
                "name": f"{_VIBECOMFY_WAN_USER_LORA_PREFIX}\\{filename}",
                "strength": _simple_lora_strength(entry.multiplier),
            }
        )
        if entry.url:
            key = (filename, _VIBECOMFY_WAN_USER_LORA_DIR)
            if key not in seen_assets:
                assets.append(
                    {
                        "name": filename,
                        "url": entry.url,
                        "directory": _VIBECOMFY_WAN_USER_LORA_DIR,
                    }
                )
                seen_assets.add(key)
    return loras, assets


def _lora_entries_for_params(params: Mapping[str, Any], *, resolved: ResolvedTask) -> list[LoRAEntry]:
    context = {
        "task_id": resolved.task_id,
        "model": params.get("model") or params.get("model_name"),
        "model_name": params.get("model_name") or params.get("model"),
    }
    try:
        config = LoRAConfig.from_params(dict(params), **context)
    except LoRAModuleManifestError:
        unsanitized_params = dict(params)
        unsanitized_params.pop("model", None)
        unsanitized_params.pop("model_name", None)
        config = LoRAConfig.from_params(unsanitized_params)
    segment_loras = params.get("loras")
    if isinstance(segment_loras, Sequence) and not isinstance(segment_loras, (str, bytes, bytearray)):
        try:
            segment_config = LoRAConfig.from_segment_loras(list(segment_loras), **context)
        except LoRAModuleManifestError:
            segment_config = LoRAConfig.from_segment_loras(list(segment_loras))
        config = config.merge(segment_config)
    return [entry for entry in config.entries if _lora_filename(entry)]


def _lora_filename(entry: LoRAEntry) -> str | None:
    raw = entry.filename or entry.local_path or entry.url
    if not raw:
        return None
    return Path(str(raw).split("?", 1)[0]).name


def _simple_lora_strength(value: Any) -> float:
    if isinstance(value, str):
        first = value.replace(",", ";").split(";", 1)[0].strip()
        if not first:
            return 1.0
        return float(first)
    return float(value)


def _default_qwen_edit_prompt(route_key: str) -> str:
    if route_key == "image_inpaint":
        return "Repair the highlighted green mask area while preserving the original scene."
    if route_key == "annotated_image_edit":
        return "Apply the requested edit indicated by the annotation while preserving the original scene."
    if route_key == "qwen_image_style":
        return "Restyle the subject using the reference image while preserving the subject identity."
    return "Apply the requested image edit while preserving the main subject and scene."


def _first_string_param(params: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _first_nested_string(params: Mapping[str, Any], nested_keys: tuple[str, ...], *keys: str) -> str | None:
    value = _first_nested_value(params, nested_keys, *keys)
    return value if isinstance(value, str) and value.strip() else None


def _first_nested_value(params: Mapping[str, Any], nested_keys: tuple[str, ...], *keys: str) -> Any:
    for nested_key in nested_keys:
        nested = params.get(nested_key)
        if isinstance(nested, Mapping):
            for key in keys:
                value = nested.get(key)
                if value is not None and value != "":
                    return value
    return None


def _first_list_string(params: Mapping[str, Any], list_keys: tuple[str, ...], index: int) -> str | None:
    for key in list_keys:
        value = params.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) > index:
            item = value[index]
            if isinstance(item, str) and item.strip():
                return item
    return None


def _first_nested_list_string(
    params: Mapping[str, Any],
    nested_keys: tuple[str, ...],
    list_keys: tuple[str, ...],
    index: int,
) -> str | None:
    for nested_key in nested_keys:
        nested = params.get(nested_key)
        if isinstance(nested, Mapping):
            found = _first_list_string(nested, list_keys, index)
            if found:
                return found
    return None


def _first_travel_guidance_video(params: Mapping[str, Any]) -> str | None:
    guidance = params.get("travel_guidance")
    if isinstance(guidance, Mapping):
        videos = guidance.get("videos")
        if isinstance(videos, Sequence) and not isinstance(videos, (str, bytes, bytearray)):
            for entry in videos:
                if isinstance(entry, str) and entry.strip():
                    return entry
                if isinstance(entry, Mapping):
                    for key in ("url", "path", "video_url", "video_path"):
                        value = entry.get(key)
                        if isinstance(value, str) and value.strip():
                            return value
    return None


def _parse_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    text = str(value).strip().lower().replace(" ", "").replace("×", "x")
    if "x" not in text:
        raise ValueError(f"invalid VibeComfy resolution {value!r}")
    width_raw, height_raw = text.split("x", 1)
    width, height = int(width_raw), int(height_raw)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid VibeComfy resolution {value!r}")
    return width, height


def _prepare_run_workspace(main_output_dir_base: str | Path, task_id: str) -> Path:
    safe_task_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in task_id)
    run_workspace = Path(main_output_dir_base) / "vibecomfy_runs" / safe_task_id
    run_workspace.mkdir(parents=True, exist_ok=True)
    return run_workspace


def _build_subprocess_env(run_workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["VIBECOMFY_WORKER_RUN_DIR"] = str(run_workspace)
    input_dir = run_workspace / "input"
    output_dir = run_workspace / "output"
    temp_dir = run_workspace / "temp"
    for path in (input_dir, output_dir, temp_dir):
        path.mkdir(parents=True, exist_ok=True)
    comfy_config = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "temp_directory": str(temp_dir),
    }
    existing_config = env.get("VIBECOMFY_COMFY_CONFIGURATION")
    if existing_config:
        try:
            parsed = json.loads(existing_config)
            if isinstance(parsed, dict):
                comfy_config = {**parsed, **comfy_config}
        except json.JSONDecodeError:
            pass
    env["VIBECOMFY_COMFY_CONFIGURATION"] = json.dumps(comfy_config)

    vibecomfy_cwd = os.environ.get("VIBECOMFY_CWD")
    if vibecomfy_cwd:
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{vibecomfy_cwd}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else vibecomfy_cwd
        )

    return env


def _vibecomfy_python() -> str:
    return os.environ.get("VIBECOMFY_PYTHON") or "python3.11"


def _vibecomfy_cwd(run_workspace: Path) -> Path:
    configured = os.environ.get("VIBECOMFY_CWD") or os.environ.get("VIBECOMFY_PATH")
    if configured and Path(configured).exists():
        return Path(configured)
    return run_workspace


def _process_default_memory_profile() -> int | None:
    raw_profile = os.environ.get("VIBECOMFY_MEMORY_PROFILE")
    if raw_profile is None or raw_profile.strip() == "":
        return None
    return int(raw_profile)


def _override_memory_profile(resolved: ResolvedTask) -> int | None:
    raw_profile = resolved.params.get("override_profile")
    if raw_profile is None:
        if resolved.route_key in {
            "wan_2_2_t2i",
            "wan_2_2_i2v",
            "wan_2_2_vace",
            "animate_character",
        }:
            return 5
        return PROCESS_DEFAULT_PROFILE
    return int(raw_profile)


def _memory_profile_for_log(resolved: ResolvedTask) -> int | None:
    try:
        args = build_memory_profile_cli_args(
            process_default=_process_default_memory_profile(),
            override_profile=_override_memory_profile(resolved),
        )
    except ValueError:
        return None
    if "--memory-profile" not in args:
        return None
    return int(args[args.index("--memory-profile") + 1])


def _video_contract_for_resolved_task(resolved: ResolvedTask) -> VideoArtifactContract:
    width, height = _expected_dimensions(resolved.params)
    return VideoArtifactContract(
        expected_frame_count=_int_param(resolved.params, "expected_frame_count", "num_frames", "video_length"),
        expected_fps=_expected_fps_for_resolved_task(resolved),
        expected_duration_seconds=_float_param(resolved.params, "expected_duration_seconds", "duration_seconds"),
        require_audio=_bool_param(resolved.params, "require_audio", "audio_required", "requires_audio"),
        expected_width=width,
        expected_height=height,
        require_thumbnail=_bool_param(resolved.params, "require_thumbnail", "thumbnail_required", "requires_thumbnail"),
        thumbnail_path=_string_param(resolved.params, "thumbnail_path", "thumbnail_storage_path"),
    )


def _expected_fps_for_resolved_task(resolved: ResolvedTask) -> float | None:
    explicit = _float_param(resolved.params, "expected_fps")
    if explicit is not None:
        return explicit
    fps = _float_param(resolved.params, "fps", "fps_helpers")
    if fps is None:
        return None
    if resolved.route_key == "video_enhance" and _bool_param(resolved.params, "enable_interpolation"):
        interpolation = resolved.params.get("interpolation")
        interpolation_params = interpolation if isinstance(interpolation, Mapping) else {}
        return fps * (2 ** _rife_exp_from_interpolation_params(interpolation_params))
    return fps


def _expected_dimensions(params: dict[str, Any]) -> tuple[int | None, int | None]:
    width = _int_param(params, "expected_width", "width")
    height = _int_param(params, "expected_height", "height")
    if width is not None or height is not None:
        return width, height
    resolution = params.get("resolution") or params.get("parsed_resolution_wh")
    if resolution is None:
        return None, None
    try:
        return _parse_resolution(resolution)
    except (TypeError, ValueError):
        return None, None


def _int_param(params: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = params.get(key)
        if value is not None and value != "":
            return int(value)
    return None


def _float_param(params: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = params.get(key)
        if value is not None and value != "":
            return float(value)
    return None


def _bool_param(params: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = params.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return False


def _string_param(params: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _video_metadata_for_log(metadata: Any) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return {
        "content_type": metadata.content_type,
        "frame_count": metadata.frame_count,
        "fps": metadata.fps,
        "duration_seconds": metadata.duration_seconds,
        "has_audio": metadata.has_audio,
        "audio_duration_seconds": metadata.audio_duration_seconds,
        "width": metadata.width,
        "height": metadata.height,
    }


def _discover_output_path(*, stdout: str, run_workspace: Path, command_cwd: Path | None = None) -> str | None:
    output_lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("output: "):
            output_lines.append(line.removeprefix("output: ").strip())
    for raw_output in output_lines:
        for resolved in _resolve_output_candidates(raw_output, run_workspace, command_cwd):
            if resolved.is_file():
                return str(resolved)
    for raw_output in reversed(output_lines):
        if Path(raw_output).is_absolute():
            return str(Path(raw_output))

    metadata_path = _metadata_path_from_stdout(stdout, run_workspace, command_cwd)
    if metadata_path and metadata_path.exists():
        output = _output_from_metadata(metadata_path, run_workspace, command_cwd)
        if output:
            return output

    if output_lines:
        return str(_resolve_output_candidates(output_lines[-1], run_workspace, command_cwd)[0])

    for output_path in _candidate_output_files(run_workspace):
        return str(output_path)

    return None


def _metadata_path_from_stdout(stdout: str, run_workspace: Path, command_cwd: Path | None = None) -> Path | None:
    for line in stdout.splitlines():
        if line.startswith("metadata: "):
            raw_path = line.removeprefix("metadata: ").strip()
            for candidate in _resolve_output_candidates(raw_path, run_workspace, command_cwd):
                if candidate.exists():
                    return candidate
            return _resolve_output_candidates(raw_path, run_workspace, command_cwd)[0]
    return None


def _output_from_metadata(metadata_path: Path, run_workspace: Path, command_cwd: Path | None = None) -> str | None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    for output in _flatten_outputs(metadata.get("outputs")):
        for resolved in _resolve_output_candidates(str(output), run_workspace, command_cwd):
            if resolved.is_file():
                return str(resolved)
        return str(_resolve_output_candidates(str(output), run_workspace, command_cwd)[0])
    return None


def _flatten_outputs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        outputs: list[str] = []
        for item in value.values():
            outputs.extend(_flatten_outputs(item))
        return outputs
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        outputs = []
        for item in value:
            outputs.extend(_flatten_outputs(item))
        return outputs
    return []


def _resolve_output_path(value: str, run_workspace: Path) -> str:
    return str(_resolve_output_candidates(value, run_workspace)[0])


def _resolve_output_candidates(value: str, run_workspace: Path, command_cwd: Path | None = None) -> list[Path]:
    path = Path(value)
    if path.is_absolute():
        return [path]

    candidates = [
        run_workspace / "output" / path,
        run_workspace / path,
    ]
    if command_cwd is not None:
        candidates.extend(
            [
                command_cwd / path,
                command_cwd / "output" / path,
            ]
        )
    return _dedupe_paths(candidates)


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _candidate_output_files(run_workspace: Path) -> list[Path]:
    candidates = [
        path
        for path in run_workspace.rglob("*")
        if path.is_file() and path.suffix.lower() in _OUTPUT_EXTENSIONS
    ]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _failure_message(
    *,
    resolved: ResolvedTask,
    exit_code: int | None,
    stderr: str,
    stdout: str,
) -> str:
    return (
        "VibeComfy task failed "
        f"task_id={resolved.task_id} "
        f"backend={resolved.backend.value} "
        f"template={resolved.template_id} "
        f"profile={_memory_profile_for_log(resolved)} "
        f"exit_code={exit_code} "
        f"stderr={_bounded(stderr)!r} "
        f"stdout={_bounded(stdout)!r}"
    )


def _log_failure(
    *,
    resolved: ResolvedTask,
    exit_code: int | None,
    stderr: str,
    stdout: str,
) -> None:
    headless_logger.debug_block(
        "VIBECOMFY_FAILURE",
        {
            "task_id": resolved.task_id,
            "task_type": resolved.task_type,
            "route_key": resolved.route_key,
            "backend": resolved.backend.value,
            "template_id": resolved.template_id,
            "memory_profile": _memory_profile_for_log(resolved),
            "exit_code": exit_code,
            "stderr": _bounded(stderr),
            "stdout": _bounded(stdout),
        },
        task_id=resolved.task_id,
    )


def _bounded(value: str | None, limit: int = _MAX_CAPTURE_CHARS) -> str:
    text = value or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


__all__ = ["handle_vibecomfy_resolved_task"]
