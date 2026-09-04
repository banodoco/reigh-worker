"""Runtime worker server helpers and canonical boundaries."""

from __future__ import annotations

import os
import argparse
import signal
from pathlib import Path

from source.core.log import headless_logger
from source.core.platform_utils import suppress_alsa_errors
from source.core.runtime_paths import get_repo_root
from source.runtime import wgp_bridge
from source.runtime.process_globals import get_bootstrap_controller, run_bootstrap_once

repo_root = str(get_repo_root())
wan2gp_path = str((Path(repo_root) / "Wan2GP").resolve())
WORKER_BOOTSTRAP_CONTROLLER = get_bootstrap_controller("worker.server")
_ensure_runtime_bridge_path = wgp_bridge.ensure_wan2gp_on_path
ensure_wan2gp_on_path = wgp_bridge.ensure_wan2gp_on_path




def bootstrap_runtime_environment() -> dict[str, object]:
    def _initializer() -> None:
        os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
        os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/runtime-root")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        globals()["ensure_wan2gp_on_path"]()
        suppress_alsa_errors()

    return run_bootstrap_once(
        "worker.runtime_environment",
        _initializer,
        version="2026-02-27",
        controller=WORKER_BOOTSTRAP_CONTROLLER,
    )




def _worker_backend_name() -> str:
    return os.environ.get("REIGH_BACKEND", os.environ.get("WORKER_BACKEND", "wgp")).strip().lower()


def _uses_vibecomfy_backend() -> bool:
    return _worker_backend_name() == "vibecomfy"


def parse_args():
    parser = argparse.ArgumentParser("WanGP Worker Server")
    parser.add_argument("--main-output-dir", type=str, default="./outputs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--worker", type=str, default=None)
    parser.add_argument("--save-logging", type=str, nargs="?", const="logs/worker.log", default=None)
    parser.add_argument("--colour-match-videos", action="store_true")
    parser.add_argument("--mask-active-frames", dest="mask_active_frames", action="store_true", default=True)
    parser.add_argument("--no-mask-active-frames", dest="mask_active_frames", action="store_false")
    parser.add_argument("--preload-model", type=str, default="")
    parser.add_argument("--reigh-access-token", type=str, default=None, help="Access token for Reigh API")

    # WGP Globals
    parser.add_argument("--wgp-attention-mode", type=str, default=None)
    parser.add_argument("--wgp-compile", type=str, default=None)
    parser.add_argument("--wgp-profile", type=int, default=None)
    parser.add_argument("--wgp-vae-config", type=int, default=None)
    parser.add_argument("--wgp-boost", type=int, default=None)
    parser.add_argument("--wgp-transformer-quantization", type=str, default=None)
    parser.add_argument("--wgp-transformer-dtype-policy", type=str, default=None)
    parser.add_argument("--wgp-text-encoder-quantization", type=str, default=None)
    parser.add_argument("--wgp-vae-precision", type=str, default=None)
    parser.add_argument("--wgp-mixed-precision", type=str, default=None)
    parser.add_argument("--wgp-preload-policy", type=str, default=None)
    parser.add_argument("--wgp-preload", type=int, default=None)

    return parser.parse_args()


def _wait_for_shutdown() -> None:
    """Keep the passive worker substrate alive until its supervisor stops it."""
    if os.name == "nt":
        import time

        time.sleep(3600)
    else:
        signal.pause()


def main():
    import datetime
    import logging
    import sys

    from dotenv import load_dotenv

    headless_logger.essential("main() entered")
    load_dotenv()
    bootstrap_runtime_environment()

    def _request_shutdown(signum, _frame):
        try:
            signal_name = signal.Signals(signum).name
        except (ValueError, AttributeError):
            signal_name = f"signum={signum}"
        headless_logger.essential(f"[SHUTDOWN] {signal_name} received")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    headless_logger.essential("bootstrap done")
    cli_args = parse_args()
    headless_logger.essential(f"args parsed: worker={cli_args.worker}, debug={cli_args.debug}")

    access_token = cli_args.reigh_access_token or os.environ.get("REIGH_ACCESS_TOKEN")
    if not access_token:
        print("Error: Worker authentication credential is required", file=sys.stderr)
        return 1

    if not cli_args.worker:
        cli_args.worker = os.environ.get("RUNPOD_POD_ID") or "local-worker"
    os.environ["WORKER_ID"] = cli_args.worker
    os.environ["WAN2GP_WORKER_MODE"] = "true"

    from source.core.log import set_log_file
    from source.core.log.core import (
        _is_env_debug,
        disable_debug_mode,
        enable_debug_mode,
        install_stdout_filter,
        suppress_library_logging,
    )

    debug_mode = cli_args.debug or _is_env_debug()
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        enable_debug_mode()
        try:
            from mmgp import offload

            offload.default_verboseLevel = 2
        except ImportError:
            pass
        if not cli_args.save_logging:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            set_log_file(os.path.join(log_dir, f"debug_{timestamp}.log"))
    else:
        disable_debug_mode()
        suppress_library_logging()
        install_stdout_filter()
    if cli_args.save_logging:
        set_log_file(cli_args.save_logging)

    local_http_server = None
    from source.runtime.worker.local_http import start_local_http_server

    port = int(os.environ.get("REIGH_LOCAL_WORKER_PORT", "8765"))
    materialization_dir = Path(
        os.environ.get("REIGH_LOCAL_WORKER_DIR", str(Path.home() / ".reigh-local-files"))
    ).expanduser()
    try:
        local_http_server = start_local_http_server(
            materialization_dir=materialization_dir,
            port=port,
            worker_id=cli_args.worker,
            version="worker",
            auth_optional=os.environ.get("REIGH_LOCAL_WORKER_AUTH_OPTIONAL") in ("1", "true", "yes"),
            file_ttl_seconds=int(os.environ.get("REIGH_LOCAL_WORKER_FILE_TTL_SECONDS", "21600")),
            janitor_interval_seconds=int(os.environ.get("REIGH_LOCAL_WORKER_JANITOR_INTERVAL_SECONDS", "1800")),
        )
        headless_logger.essential(
            f"Local HTTP server listening on 127.0.0.1:{port}, materializing to {materialization_dir}"
        )
    except OSError as exc:
        headless_logger.warning(f"Local HTTP server could not bind to 127.0.0.1:{port}: {exc}")

    def _shutdown_local_http() -> None:
        if local_http_server is None:
            return
        try:
            local_http_server.shutdown()
        except BaseException as exc:
            headless_logger.debug_anomaly("SHUTDOWN", f"local_http_server.shutdown failed: {exc}")

    main_output_dir = Path(cli_args.main_output_dir).resolve()
    main_output_dir.mkdir(parents=True, exist_ok=True)

    from source.runtime.worker.preflight import (
        PREFLIGHT_STATUS_RUNNING,
        PreflightCheck,
        WorkerPreflightResult,
        finalize_preflight_result,
        run_worker_preflight,
        write_preflight_state,
    )
    from source.runtime.worker.warm_cache import publish_warm_cache_state, resolve_warm_cache_plan

    def _publish_preflight(result: WorkerPreflightResult, *, ready_for_tasks: bool) -> None:
        if cli_args.worker:
            write_preflight_state(
                cli_args.worker,
                {
                    **result.to_metadata(),
                    "ready_for_tasks": bool(ready_for_tasks and result.ok),
                },
            )

    worker_backend = _worker_backend_name()
    static_preflight = run_worker_preflight(
        repo_root=Path(repo_root),
        wan2gp_path=Path(wan2gp_path),
        main_output_dir=main_output_dir,
        backend=worker_backend,
    )
    if not static_preflight.ok:
        _publish_preflight(static_preflight, ready_for_tasks=False)
        _shutdown_local_http()
        return 1
    _publish_preflight(
        WorkerPreflightResult(
            status=PREFLIGHT_STATUS_RUNNING,
            checks=static_preflight.checks,
            started_at=static_preflight.started_at,
            completed_at=static_preflight.completed_at,
            phase="startup",
        ),
        ready_for_tasks=False,
    )

    if _uses_vibecomfy_backend():
        headless_logger.essential("VibeComfy backend active; skipping WGP import")
        wgp_import_check = PreflightCheck("wgp_import", True, "skipped for vibecomfy backend", required=False)
    else:
        original_cwd = os.getcwd()
        original_argv = sys.argv[:]
        try:
            os.chdir(wan2gp_path)
            sys.path.insert(0, wan2gp_path)
            sys.argv = ["worker.py"]
            import wgp as wgp_mod

            if cli_args.wgp_attention_mode:
                wgp_mod.attention_mode = cli_args.wgp_attention_mode
            if cli_args.wgp_compile:
                wgp_mod.compile = cli_args.wgp_compile
            if cli_args.wgp_profile:
                wgp_mod.force_profile_no = cli_args.wgp_profile
                wgp_mod.default_profile = cli_args.wgp_profile
            if cli_args.wgp_vae_config:
                wgp_mod.vae_config = cli_args.wgp_vae_config
            if cli_args.wgp_boost:
                wgp_mod.boost = cli_args.wgp_boost
            if cli_args.wgp_transformer_quantization:
                wgp_mod.transformer_quantization = cli_args.wgp_transformer_quantization
            if cli_args.wgp_transformer_dtype_policy:
                wgp_mod.transformer_dtype_policy = cli_args.wgp_transformer_dtype_policy
            if cli_args.wgp_text_encoder_quantization:
                wgp_mod.text_encoder_quantization = cli_args.wgp_text_encoder_quantization
            if cli_args.wgp_vae_precision:
                wgp_mod.server_config["vae_precision"] = cli_args.wgp_vae_precision
            if cli_args.wgp_mixed_precision:
                wgp_mod.server_config["mixed_precision"] = cli_args.wgp_mixed_precision
            if cli_args.wgp_preload_policy:
                wgp_mod.server_config["preload_model_policy"] = [
                    item.strip() for item in cli_args.wgp_preload_policy.split(",")
                ]
            if cli_args.wgp_preload:
                wgp_mod.server_config["preload_in_VRAM"] = cli_args.wgp_preload
            if "transformer_types" not in wgp_mod.server_config:
                wgp_mod.server_config["transformer_types"] = []
            headless_logger.essential("WGP imported OK")
            wgp_import_check = PreflightCheck("wgp_import", True, "wgp imported", required=True)
        except (ImportError, RuntimeError, AttributeError, KeyError) as exc:
            headless_logger.essential(f"WGP import failed: {exc}")
            _publish_preflight(
                finalize_preflight_result(
                    static_preflight,
                    extra_checks=[PreflightCheck("wgp_import", False, str(exc), required=True)],
                ),
                ready_for_tasks=False,
            )
            _shutdown_local_http()
            return 1
        finally:
            sys.argv = original_argv
            os.chdir(original_cwd)

    from source.models.lora.lora_utils import cleanup_legacy_lora_collisions, sweep_lora_cache_from_env

    cleanup_legacy_lora_collisions()
    sweep_lora_cache_from_env(task_id="worker_startup")

    worker_profile = os.environ.get("REIGH_WORKER_PROFILE", os.environ.get("WGP_PROFILE", str(cli_args.wgp_profile or "1")))
    warm_cache_plan = resolve_warm_cache_plan(
        backend=worker_backend,
        profile=worker_profile,
        cli_preload_model=cli_args.preload_model or None,
    )
    publish_warm_cache_state(cli_args.worker, warm_cache_plan, status="planned")
    _publish_preflight(
        finalize_preflight_result(static_preflight, extra_checks=[wgp_import_check]),
        ready_for_tasks=False,
    )
    headless_logger.essential(
        f"Worker substrate ready backend={worker_backend} worker_id={cli_args.worker}; "
        "task authority remains external"
    )

    try:
        _wait_for_shutdown()
    except KeyboardInterrupt:
        headless_logger.essential("Shutting down...")
    finally:
        _shutdown_local_http()
    return 0
