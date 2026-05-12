"""Variant C driver: launch a consumer pod against a prebuilt validation environment.

The prebuilt variant boots a RunPod pod with a pre-baked network volume
attached at ``/workspace``. The bootstrap pipeline extracts the volume's
zstd-compressed venv and VibeComfy bundles to ``/opt`` on the container disk,
syncs the worker tree from the operator-selected branch, applies any delta
between the bundled state and the live ref, runs preflight probes, and
launches the worker against the prebuilt env.

The argparse default for ``--variant`` remains ``fresh`` (see main.py); this
driver is opt-in via ``--variant prebuilt`` or ``--variant auto``.
"""

from __future__ import annotations

from typing import Any

from runpod_lifecycle.prebuilt import (
    PrebuiltEnvContract,
    PrebuiltManifest,
    read_manifest,
    verify_extracted_env,
)

from scripts.live_test import config
from scripts.live_test._shared import (
    _build_worker_env_base,
    _capture_and_redact_noisy_lifecycle_output,
    _phase,
    _resolve_runpod_gpu_type_id,
    _runs_root,
    _timestamp_label,
    register_worker_record,
    select_network_volume,
)
from scripts.live_test.heartbeat_waiter import wait_until_ready
from scripts.live_test.launch_command import build_run_worker_command
from scripts.live_test.logger import get_logger
from scripts.live_test.matrix import build_matrix, poll_queued_matrix, queue_matrix, render_case_payload
from scripts.live_test.preflight import (
    assert_user_queue_clean,
    close_stale_live_test_tasks,
    ensure_live_test_route_selectors,
    ensure_user_cloud_generation_enabled,
    get_or_create_live_test_project,
)
from scripts.live_test.report import all_results_passed, write_report
from scripts.live_test.ssh_bootstrap import (
    _uv_sync_shell,
    _vibecomfy_install_shell,
    ensure_git_ref_synced,
    export_env,
    extract_bundle_to_container_disk,
    fetch_worker_logs,
    launch_worker_detached,
    open_session,
)
from scripts.live_test.terminate_guard import guarded_terminate, prune_stale_live_test_pods
from scripts.live_test.token_resolver import resolve_token_to_user_id


PREBUILT_VARIANT = "prebuilt"
PREBUILT_POD_PREFIX = "reigh-livetest-prebuilt-"
VIBECOMFY_PYTHON_DEFAULT = "python3.11"
WORKER_REPO_URL = "https://github.com/banodoco/Reigh-Worker.git"
VIBECOMFY_REPO_URL = "https://github.com/peteromallet/VibeComfy.git"
_VENV_BUNDLE_NAME = "venv.cuda124.tar.zst"
_VIBECOMFY_BUNDLE_NAME = "vibecomfy.tar.zst"

# Drift fields that require a full rebuild — never delta-syncable.
_HARD_FAIL_FIELDS = (
    "schema_version",
    "bundle_format_version",
    "python_version",
    "cuda_extra",
)

log = get_logger(__name__)


def _contract_from_volume(name: str, data_center_id: str, manifest: PrebuiltManifest) -> PrebuiltEnvContract:
    return PrebuiltEnvContract(
        volume_name=name,
        data_center_id=data_center_id,
        attention_profile=manifest.attention_profile,
        comfyui_pin=manifest.comfyui_pin,
        python_version=manifest.python_version,
        bundle_format_version=manifest.bundle_format_version,
    )


def _expected_hard_fail_values(args) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "bundle_format_version": 1,
        "python_version": getattr(args, "python_version", None) or "3.10",
        "cuda_extra": "cuda124",
    }


def _attention_profile_from_args(args) -> str:
    profile = str(getattr(args, "worker_profile", "")).strip().lower()
    return "sage" if profile in {"sage", "optimized"} else "portable"


def _recommended_volume_name(profile: str, data_center_id: str = "eu-no-1") -> str:
    return config.prebuilt_name_for_profile(profile, data_center_id)


def _check_hard_fail_drift(manifest: PrebuiltManifest, args) -> None:
    expected = _expected_hard_fail_values(args)
    for field in _HARD_FAIL_FIELDS:
        observed = getattr(manifest, field)
        if field == "python_version" and getattr(args, "python_version", None) is None:
            # When the consumer doesn't set --python-version explicitly, accept
            # whatever the manifest baked in; the worker launch will follow suit.
            continue
        if field == "schema_version" or field == "bundle_format_version":
            if observed != expected[field]:
                raise RuntimeError(
                    f"Hard-fail drift on {field}: manifest={observed} expected={expected[field]}. "
                    f"Run `rl prebuilt build --volume-name {args.prebuilt_volume_name or '<volume>'} "
                    f"--data-center <dc> --attention-profile {manifest.attention_profile}` to rebuild."
                )
            continue
        if observed != expected[field]:
            raise RuntimeError(
                f"Hard-fail drift on {field}: manifest={observed} expected={expected[field]}. "
                f"Run `rl prebuilt build --volume-name {args.prebuilt_volume_name or '<volume>'} "
                f"--data-center <dc> --attention-profile {manifest.attention_profile}` to rebuild."
            )


def _build_worker_env(token: str, supabase_url: str, service_role_key: str, args, contract: PrebuiltEnvContract) -> dict[str, str]:
    env = _build_worker_env_base(
        token,
        supabase_url,
        service_role_key,
        args,
        vibecomfy_workdir=contract.runtime_vibecomfy_path,
        vibecomfy_python=f"{contract.runtime_venv_path}/bin/python",
    )
    # Bind models cache onto the volume so workflows reuse weights across runs.
    models_root = contract.models_path
    hf_home = f"{models_root}/huggingface"
    env.update(
        {
            "HF_HOME": hf_home,
            "HF_HUB_CACHE": f"{hf_home}/hub",
            "COMFYUI_EXTRA_MODEL_PATHS_PATH": models_root,
        }
    )
    return env


def _ensure_mounted(ssh, mount_path: str) -> None:
    exit_code, _stdout, stderr = ssh.execute_command(
        f"mountpoint -q {mount_path}", timeout=30
    )
    if exit_code != 0:
        raise RuntimeError(
            f"Network volume is not mounted at {mount_path}; stderr={stderr!r}"
        )


def _build_matrix_cases(args) -> list:
    cases = build_matrix(
        anchor_image_a=args.anchor_image_a,
        anchor_image_b=args.anchor_image_b,
        timeout_image_sec=args.timeout_image,
        timeout_travel_segment_sec=args.timeout_travel_segment,
        timeout_travel_orchestrator_sec=args.timeout_travel_orchestrator,
        selected_backend=getattr(args, "backend", "wgp"),
        selector_namespace=getattr(args, "selector_namespace", "production"),
        selector_version=getattr(args, "selector_version", None),
        worker_contract_version=getattr(args, "worker_contract_version", 1),
        selected_profile=getattr(args, "worker_profile", "default"),
        case_names=getattr(args, "case", []),
        task_types=getattr(args, "task_type", []),
        route_keys=getattr(args, "route_key", []),
    )
    return cases


def _validate_cases(cases: list, project_id: str) -> None:
    for index, case in enumerate(cases, start=1):
        render_case_payload(case, project_id=project_id, unique_suffix=f"prebuilt-{index}")


def _write_extra_model_paths_yaml(ssh, contract: PrebuiltEnvContract) -> None:
    """Clobber {runtime_vibecomfy_path}/extra_model_paths.yaml with consumer paths."""
    import shlex as _sh

    body = (
        "comfyui:\n"
        f"  base_path: {contract.models_path}/\n"
        "  checkpoints: checkpoints\n"
        "  vae: vae\n"
        "  loras: loras\n"
        "  embeddings: embeddings\n"
        "  diffusion_models: diffusion_models\n"
        "  clip: clip\n"
        "  upscale_models: upscale_models\n"
    )
    target = f"{contract.runtime_vibecomfy_path}/extra_model_paths.yaml"
    script = (
        "set -euo pipefail\n"
        f"mkdir -p {_sh.quote(contract.runtime_vibecomfy_path)}\n"
        f"cat > {_sh.quote(target)} <<'EXTRA_MODEL_PATHS_EOF'\n"
        f"{body}"
        "EXTRA_MODEL_PATHS_EOF\n"
    )
    exit_code, _stdout, stderr = ssh.execute_command("bash -lc " + _sh.quote(script), timeout=60)
    if exit_code != 0:
        raise RuntimeError(
            f"failed to write extra_model_paths.yaml to {target}; stderr={stderr!r}"
        )


def _print_dry_run_plan(
    *,
    token: str,
    project_id: str,
    cases: list,
    args,
    contract: PrebuiltEnvContract,
) -> None:
    supabase_url = config.get_env("SUPABASE_URL", "https://example.supabase.co")
    launch_command = build_run_worker_command(
        contract.runtime_worker_path,
        reigh_token=None,
        supabase_url=supabase_url,
        worker_id="<runpod-pod-id>",
        wgp_profile=args.wgp_profile,
        idle_release_minutes=0,
        redact_secrets=True,
        venv_path=contract.runtime_venv_path,
        python_version=contract.python_version,
    )
    print("Variant: prebuilt")
    print(f"Project ID: {project_id}")
    print(f"Prebuilt volume name: {args.prebuilt_volume_name or '<resolved at launch>'}")
    print(f"Runtime venv path: {contract.runtime_venv_path}")
    print(f"Runtime worker path: {contract.runtime_worker_path}")
    print(f"Runtime VibeComfy path: {contract.runtime_vibecomfy_path}")
    print(f"Models path: {contract.models_path}")
    print(f"Python version (from manifest): {contract.python_version}")
    print(f"Terminate after run: {not args.no_terminate}")
    print("Planned launch command:")
    print(launch_command)
    print("Planned tasks:")
    for case in cases:
        route_suffix = (
            f", route={case.route_key}, backend={case.route_runtime.selected_backend}"
            if case.route_key
            else ""
        )
        print(f"- {case.name} ({case.task_type}{route_suffix}, timeout={case.timeout_sec}s)")


def _resolve_volume(args, api_key: str) -> tuple[str, str]:
    profile = _attention_profile_from_args(args)
    if args.prebuilt_volume_name:
        prefix = args.prebuilt_volume_name
    else:
        prefix = f"{config.PREBUILT_VOLUME_NAME_PREFIX}{profile}-"
    selection = select_network_volume(api_key, name_prefix=prefix)
    if selection is None:
        recommended = _recommended_volume_name(profile)
        raise RuntimeError(
            f"No prebuilt volume matches prefix {prefix!r}. "
            f"Build one with: rl prebuilt build --volume-name {recommended} "
            f"--data-center <dc> --attention-profile {profile}"
        )
    _, name, data_center_id = selection
    return name, data_center_id


def _prepare_context(args) -> dict[str, Any]:
    token = config.require_env("REIGH_LIVE_TEST_TOKEN")
    db = config.DatabaseClient()
    user_id = resolve_token_to_user_id(db, token)
    stale_count = close_stale_live_test_tasks(db, user_id)
    if stale_count:
        log.info("closed stale live-test tasks before prebuilt run", count=stale_count)
    if ensure_user_cloud_generation_enabled(db, user_id):
        log.info("enabled cloud generation for live-test user", user_id=user_id)
    assert_user_queue_clean(db, user_id)
    project_id = get_or_create_live_test_project(db, user_id)
    cases = _build_matrix_cases(args)
    created_selectors = ensure_live_test_route_selectors(
        db,
        getattr(args, "selector_namespace", "production"),
        [case.route_key for case in cases if case.route_key],
        backend=getattr(args, "backend", "wgp"),
        fallback_selectors={
            str(case.route_key): {
                "selected_backend": case.route_runtime.selected_backend,
                "selector_version": case.route_runtime.selector_version,
                "support_state": case.support_state,
                "selected_template_id": case.selected_template_id,
            }
            for case in cases
            if case.route_key and case.support_state == "vibecomfy_supported"
        },
    )
    if created_selectors:
        log.info(
            "created isolated live-test route selectors",
            selector_namespace=getattr(args, "selector_namespace", "production"),
            count=created_selectors,
        )
    return {
        "db": db,
        "token": token,
        "user_id": user_id,
        "project_id": project_id,
        "cases": cases,
    }


def run(args) -> int:
    if args.container_disk_gb is not None and args.container_disk_gb < 100:
        raise RuntimeError(
            f"--container-disk-gb must be >= 100 (got {args.container_disk_gb})"
        )

    if args.dry_run:
        cases = _build_matrix_cases(args)
        project_id = "<live-test-project-id>"
        _validate_cases(cases, project_id)
        manifest_python = getattr(args, "python_version", None) or "3.10"
        contract = PrebuiltEnvContract(
            volume_name=args.prebuilt_volume_name or "<resolved-at-launch>",
            data_center_id="<resolved-at-launch>",
            attention_profile=_attention_profile_from_args(args),
            comfyui_pin="fix/latentupscale-model-mmap-residency",
            python_version=manifest_python,
            bundle_format_version=1,
        )
        _print_dry_run_plan(
            token="<REIGH_LIVE_TEST_TOKEN>",
            project_id=project_id,
            cases=cases,
            args=args,
            contract=contract,
        )
        return 0

    context = _prepare_context(args)
    token = context["token"]
    db = context["db"]
    project_id = context["project_id"]
    cases = context["cases"]
    _validate_cases(cases, project_id)

    api_key = config.require_env("RUNPOD_API_KEY")
    cleanup = prune_stale_live_test_pods(api_key)
    if cleanup.terminated:
        log.warning("terminated stale live-test pods before launch: %s", ", ".join(cleanup.terminated))

    volume_name, data_center_id = _resolve_volume(args, api_key)
    supabase_url = config.require_env("SUPABASE_URL")
    service_role_key = config.require_env("SUPABASE_SERVICE_ROLE_KEY")
    out_dir = _runs_root() / _timestamp_label()

    pod_id: str | None = None
    ssh = None
    try:
        from runpod_lifecycle.api import create_pod as create_pod_and_wait
        from runpod_lifecycle.api import get_network_volumes

        resolved_gpu_type_id, resolved_gpu_display_name = _resolve_runpod_gpu_type_id(
            api_key, config.RUNPOD_GPU_TYPE
        )
        # Look up the volume id explicitly (we know name + datacenter).
        network_volume_id: str | None = None
        for vol in get_network_volumes(api_key) or []:
            if str(vol.get("name") or "") == volume_name:
                network_volume_id = str(vol.get("id") or "")
                break
        if not network_volume_id:
            raise RuntimeError(f"Could not resolve volume id for {volume_name!r}")

        container_disk_gb = args.container_disk_gb if args.container_disk_gb else 200

        with _phase(
            "create_runpod_pod",
            gpu_type=config.RUNPOD_GPU_TYPE,
            gpu_type_id=resolved_gpu_type_id,
            gpu_display_name=resolved_gpu_display_name,
            volume=volume_name,
            data_center_id=data_center_id,
        ):
            with _capture_and_redact_noisy_lifecycle_output():
                pod = create_pod_and_wait(
                    api_key=api_key,
                    gpu_type_id=resolved_gpu_type_id,
                    image_name=config.RUNPOD_WORKER_IMAGE,
                    name=f"{PREBUILT_POD_PREFIX}{_timestamp_label().lower()}",
                    network_volume_id=network_volume_id,
                    volume_mount_path=config.RUNPOD_VOLUME_MOUNT_PATH,
                    disk_in_gb=container_disk_gb,
                    container_disk_in_gb=container_disk_gb,
                    min_vcpu_count=config.RUNPOD_MIN_VCPU_COUNT,
                    min_memory_in_gb=config.RUNPOD_MIN_MEMORY_GB,
                    template_id=config.RUNPOD_TEMPLATE_ID,
                    env_vars={},
                )
        if not pod or not pod.get("id"):
            raise RuntimeError("create_pod_and_wait did not return a pod id")
        pod_id = str(pod["id"])

        with _phase("open_ssh_session", pod_id=pod_id):
            ssh = open_session(pod_id, api_key)

        with _phase("attach_prebuilt_volume", mount=config.RUNPOD_VOLUME_MOUNT_PATH):
            _ensure_mounted(ssh, config.RUNPOD_VOLUME_MOUNT_PATH)

        # Read the manifest BEFORE building a contract; the manifest's
        # python_version + bundle_format_version drive the contract paths
        # and the python the worker is launched against.
        bootstrap_contract = PrebuiltEnvContract(
            volume_name=volume_name,
            data_center_id=data_center_id,
            attention_profile=_attention_profile_from_args(args),
            comfyui_pin="fix/latentupscale-model-mmap-residency",
            python_version=getattr(args, "python_version", None) or "3.10",
            bundle_format_version=1,
        )
        with _phase("read_prebuilt_manifest"):
            manifest = read_manifest(ssh, bootstrap_contract)
            if manifest is None:
                raise RuntimeError(
                    f"No prebuilt manifest at {bootstrap_contract.cache_root}/env.manifest.json. "
                    f"Build one with: rl prebuilt build --volume-name {volume_name} "
                    f"--data-center {data_center_id} --attention-profile {bootstrap_contract.attention_profile}"
                )
        contract = _contract_from_volume(volume_name, data_center_id, manifest)

        with _phase("check_hard_fail_drift"):
            _check_hard_fail_drift(manifest, args)

        with _phase("extract_venv_bundle", target=contract.runtime_venv_path):
            extract_bundle_to_container_disk(
                ssh,
                bundle_path=f"{contract.cache_root}/{_VENV_BUNDLE_NAME}",
                target_path=contract.runtime_venv_path,
                expected_sha256=manifest.venv_bundle_sha256,
            )

        with _phase("extract_vibecomfy_bundle", target=contract.runtime_vibecomfy_path):
            extract_bundle_to_container_disk(
                ssh,
                bundle_path=f"{contract.cache_root}/{_VIBECOMFY_BUNDLE_NAME}",
                target_path=contract.runtime_vibecomfy_path,
                expected_sha256=manifest.vibecomfy_bundle_sha256,
            )

        worker_ref = args.ref or "main"
        with _phase("sync_worker_ref", ref=worker_ref, workdir=contract.runtime_worker_path):
            ensure_git_ref_synced(
                ssh,
                workdir=contract.runtime_worker_path,
                repo_url=WORKER_REPO_URL,
                ref=worker_ref,
                force_clone=True,
            )
            if args.allow_delta and not args.strict_prebuilt:
                # Probe pyproject hash for delta sync.
                import shlex as _sh

                exit_code, stdout, _ = ssh.execute_command(
                    f"cat {_sh.quote(contract.runtime_worker_path)}/pyproject.toml", timeout=60
                )
                if exit_code == 0:
                    from runpod_lifecycle.prebuilt import compute_pyproject_hash

                    observed = compute_pyproject_hash(stdout)
                    if observed != manifest.pyproject_hash:
                        log.info(
                            "pyproject_hash drift; running uv sync to reconcile",
                            manifest=manifest.pyproject_hash[:12],
                            observed=observed[:12],
                        )
                        sync_body = _uv_sync_shell(
                            contract.runtime_worker_path,
                            env_path=contract.runtime_venv_path,
                            extras=("cuda124",),
                        )
                        exit_code, _stdout, stderr = ssh.execute_command(
                            "bash -lc " + _sh.quote("set -euo pipefail\n" + sync_body),
                            timeout=3600,
                        )
                        if exit_code != 0:
                            raise RuntimeError(
                                f"delta uv sync failed: stderr={stderr!r}"
                            )
            elif args.strict_prebuilt:
                # Strict mode: any pyproject drift becomes an abort. We still
                # don't run uv sync; we raise.
                import shlex as _sh

                exit_code, stdout, _ = ssh.execute_command(
                    f"cat {_sh.quote(contract.runtime_worker_path)}/pyproject.toml", timeout=60
                )
                if exit_code == 0:
                    from runpod_lifecycle.prebuilt import compute_pyproject_hash

                    if compute_pyproject_hash(stdout) != manifest.pyproject_hash:
                        raise RuntimeError(
                            "strict-prebuilt: pyproject_hash drift detected. "
                            f"Run rl prebuilt invalidate --volume-name {volume_name} && rl prebuilt build."
                        )

        vibecomfy_ref = args.vibecomfy_ref
        with _phase("sync_vibecomfy_ref", ref=vibecomfy_ref):
            ensure_git_ref_synced(
                ssh,
                workdir=contract.runtime_vibecomfy_path,
                repo_url=VIBECOMFY_REPO_URL,
                ref=vibecomfy_ref,
                force_clone=False,
            )
            if args.allow_delta and not args.strict_prebuilt:
                import shlex as _sh

                exit_code, stdout, _ = ssh.execute_command(
                    f"cat {_sh.quote(contract.runtime_vibecomfy_path)}/custom_nodes.lock",
                    timeout=60,
                )
                if exit_code == 0:
                    from runpod_lifecycle.prebuilt import compute_lockfile_hash

                    observed = compute_lockfile_hash(stdout)
                    if observed != manifest.custom_nodes_lock_hash:
                        log.info(
                            "custom_nodes_lock drift; running nodes restore",
                            manifest=manifest.custom_nodes_lock_hash[:12],
                            observed=observed[:12],
                        )
                        install_body = _vibecomfy_install_shell(
                            contract.runtime_vibecomfy_path,
                            python_path=f"{contract.runtime_venv_path}/bin/python",
                            attention_profile=manifest.attention_profile,
                            run_nodes_restore=True,
                        )
                    else:
                        install_body = (
                            f"{contract.runtime_venv_path}/bin/python -m pip install -e "
                            f"{contract.runtime_vibecomfy_path}\n"
                        )
                    exit_code, _stdout, stderr = ssh.execute_command(
                        "bash -lc " + _sh.quote("set -euo pipefail\n" + install_body),
                        timeout=3600,
                    )
                    if exit_code != 0:
                        raise RuntimeError(f"delta vibecomfy install failed: stderr={stderr!r}")

        with _phase("verify_extracted_env"):
            issues = verify_extracted_env(ssh, contract, manifest)
            if issues:
                numbered = "\n".join(f"  {i+1}. {msg}" for i, msg in enumerate(issues))
                raise RuntimeError(
                    "Prebuilt env verification failed:\n"
                    f"{numbered}\n"
                    f"Run `rl prebuilt invalidate --volume-name {volume_name}` "
                    f"&& `rl prebuilt build --volume-name {volume_name} --data-center {data_center_id} "
                    f"--attention-profile {manifest.attention_profile}` to rebuild."
                )

        with _phase("bind_models_dir", models=contract.models_path):
            _write_extra_model_paths_yaml(ssh, contract)
        worker_env = _build_worker_env(token, supabase_url, service_role_key, args, contract)

        register_worker_record(db, pod_id, pod, args, variant_label=PREBUILT_VARIANT)

        command = build_run_worker_command(
            contract.runtime_worker_path,
            reigh_token=None,
            supabase_url=supabase_url,
            worker_id=pod_id,
            wgp_profile=args.wgp_profile,
            idle_release_minutes=0,
            venv_path=contract.runtime_venv_path,
            python_version=contract.python_version,
        )
        with _phase("launch_worker", pod_id=pod_id):
            launch_worker_detached(ssh, export_env(worker_env) + " && " + command)
        with _phase("wait_worker_ready", pod_id=pod_id):
            wait_until_ready(db, worker_id=pod_id, timeout_sec=900, progress_every_sec=60)
        with _phase("queue_matrix", pod_id=pod_id, cases=len(cases)):
            queued = queue_matrix(db, project_id, cases)
        with _phase("run_matrix", pod_id=pod_id, cases=len(cases)):
            results = poll_queued_matrix(db, project_id, queued, worker_id=pod_id)
        with _phase("write_report", pod_id=pod_id, out_dir=str(out_dir)):
            write_report(results, PREBUILT_VARIANT, pod_id, out_dir)
        return 0 if all_results_passed(results) else 1
    finally:
        if ssh is not None:
            try:
                logs = fetch_worker_logs(ssh, "/opt/reigh-livetest-prebuilt/worker")
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "worker_logs.txt").write_text(logs, encoding="utf-8")
            except Exception as exc:
                log.warning("failed to fetch prebuilt variant worker logs: %s", exc)
            finally:
                disconnect = getattr(ssh, "disconnect", None)
                if callable(disconnect):
                    disconnect()
        guarded_terminate(pod_id, api_key if not args.dry_run else None, no_terminate=args.no_terminate)


__all__ = [
    "PREBUILT_POD_PREFIX",
    "PREBUILT_VARIANT",
    "run",
]
