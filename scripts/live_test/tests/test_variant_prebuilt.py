"""Harness tests for variant_prebuilt + variant_update guard + terminate_guard (T15).

These exercise unit-level pieces of the prebuilt path that don't require a
real RunPod pod. Where a full end-to-end bootstrap would be needed, we cover
the relevant decision-point function in isolation (e.g. `_check_hard_fail_drift`,
`_build_worker_env`, `_resolve_volume`, the variant_update guard).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runpod_lifecycle.prebuilt import (
    PrebuiltEnvContract,
    PrebuiltHealthIssue,
    PrebuiltHealthReport,
    PrebuiltManifest,
    PrebuiltPythonEnvReport,
)

from scripts.live_test import variant_prebuilt
from scripts.live_test import variant_update
from scripts.live_test.terminate_guard import LIVE_TEST_POD_PREFIXES, _LIVE_TEST_POD_NAME_RES
from scripts.live_test.variant_prebuilt import (
    _build_worker_env,
    _check_hard_fail_drift,
    _install_prebuilt_system_tools,
    _resolve_volume,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_manifest(**overrides) -> PrebuiltManifest:
    defaults = dict(
        schema_version=1,
        bundle_format_version=1,
        built_at_utc="2026-05-13T12:00:00+00:00",
        built_by="pod-abc",
        pyproject_hash="a" * 64,
        custom_nodes_lock_hash="b" * 64,
        comfyui_pin="fix/latentupscale-model-mmap-residency",
        attention_profile="portable",
        python_version="3.11",
        cuda_extra="cuda124",
        vibecomfy_commit="c" * 40,
        reigh_worker_commit="d" * 40,
        uv_version="uv 0.4.10",
        venv_bundle_sha256="e" * 64,
        vibecomfy_bundle_sha256="f" * 64,
        models_index_sha256="",
        venv_size_bytes=12_000_000_000,
        notes="",
    )
    defaults.update(overrides)
    return PrebuiltManifest(**defaults)


def _make_contract() -> PrebuiltEnvContract:
    return PrebuiltEnvContract(
        volume_name="reigh-livetest-prebuilt-portable-eu-no-1",
        data_center_id="eu-no-1",
        attention_profile="portable",
        comfyui_pin="fix/latentupscale-model-mmap-residency",
        python_version="3.11",
        bundle_format_version=1,
    )


def _base_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        backend="vibecomfy",
        worker_profile="portable",
        selector_namespace="production",
        selector_version=None,
        worker_contract_version=1,
        prebuilt_volume_name=None,
        python_version=None,
        strict_prebuilt=False,
        allow_delta=True,
        update_manifest_on_sync=False,
        container_disk_gb=200,
        dry_run=False,
        no_terminate=False,
        wgp_profile=3,
        ref="main",
        vibecomfy_ref="main",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# --------------------------------------------------------------------------- #
# (e) python_version drift HARD FAIL — raises, no extract/sync attempted
# --------------------------------------------------------------------------- #


def test_python_version_drift_hard_fails_with_rl_prebuilt_build_text():
    manifest = _make_manifest(python_version="3.11")
    args = _base_args(python_version="3.10", prebuilt_volume_name="vol-x")
    with pytest.raises(RuntimeError) as info:
        _check_hard_fail_drift(manifest, args)
    msg = str(info.value)
    assert "python_version" in msg
    assert "rl prebuilt build" in msg
    assert "vol-x" in msg


# --------------------------------------------------------------------------- #
# (f) schema_version drift HARD FAIL
# --------------------------------------------------------------------------- #


def test_schema_version_drift_hard_fails():
    manifest = _make_manifest(schema_version=999)
    args = _base_args(prebuilt_volume_name="vol-y")
    with pytest.raises(RuntimeError) as info:
        _check_hard_fail_drift(manifest, args)
    msg = str(info.value)
    assert "schema_version" in msg
    assert "rl prebuilt build" in msg


def test_bundle_format_version_drift_hard_fails():
    manifest = _make_manifest(bundle_format_version=99)
    args = _base_args(prebuilt_volume_name="vol-z")
    with pytest.raises(RuntimeError) as info:
        _check_hard_fail_drift(manifest, args)
    assert "bundle_format_version" in str(info.value)
    assert "rl prebuilt build" in str(info.value)


def test_cuda_extra_drift_hard_fails():
    manifest = _make_manifest(cuda_extra="cuda128")
    args = _base_args(prebuilt_volume_name="vol-q")
    with pytest.raises(RuntimeError) as info:
        _check_hard_fail_drift(manifest, args)
    assert "cuda_extra" in str(info.value)
    assert "rl prebuilt build" in str(info.value)


def test_hard_fail_drift_passes_when_manifest_aligns():
    manifest = _make_manifest(
        schema_version=1, bundle_format_version=1, python_version="3.10", cuda_extra="cuda124"
    )
    args = _base_args(python_version="3.10")
    # Should not raise.
    _check_hard_fail_drift(manifest, args)


# --------------------------------------------------------------------------- #
# (g) missing manifest → raises with `rl prebuilt build --volume-name` literal
# --------------------------------------------------------------------------- #


def test_resolve_volume_raises_with_rl_prebuilt_build_when_no_volume(monkeypatch):
    args = _base_args(prebuilt_volume_name=None, worker_profile="portable")
    monkeypatch.setattr(
        "scripts.live_test.variant_prebuilt.select_network_volume",
        lambda api_key, *, name_prefix, data_center_filter=None: None,
    )
    with pytest.raises(RuntimeError) as info:
        _resolve_volume(args, "api-key")
    msg = str(info.value)
    assert "rl prebuilt build --volume-name" in msg
    assert "--attention-profile portable" in msg
    assert "reigh-livetest-prebuilt-portable-" in msg


# --------------------------------------------------------------------------- #
# (i) worker-env test — HF_HOME / HF_HUB_CACHE / ComfyUI models-path key
# --------------------------------------------------------------------------- #


def test_build_worker_env_layers_hf_and_comfy_models_path_keys():
    contract = _make_contract()
    args = _base_args(backend="vibecomfy", worker_profile="portable")
    env = _build_worker_env(
        "token-xyz",
        "https://supabase.example",
        "service-key",
        args,
        contract,
    )
    assert env["HF_HOME"] == f"{contract.models_path}/huggingface"
    assert env["HF_HUB_CACHE"] == f"{contract.models_path}/huggingface/hub"
    assert env["COMFYUI_EXTRA_MODEL_PATHS_PATH"] == contract.models_path
    assert env["VIBECOMFY_PYTHON"] == f"{contract.runtime_vibecomfy_path}/.venv/bin/python"
    # Fresh-variant env never sets these three.
    from scripts.live_test import variant_fresh

    fresh_env = variant_fresh._build_worker_env(
        "token-xyz", "https://supabase.example", "service-key", args
    )
    assert "HF_HOME" not in fresh_env
    assert "HF_HUB_CACHE" not in fresh_env
    assert "COMFYUI_EXTRA_MODEL_PATHS_PATH" not in fresh_env


def test_prebuilt_dry_run_is_side_effect_free(monkeypatch, capsys):
    args = _base_args(
        dry_run=True,
        case=["z_image_turbo"],
        task_type=[],
        route_key=[],
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        timeout_image=60,
        timeout_travel_segment=60,
        timeout_travel_orchestrator=60,
    )

    def _boom(_args):
        raise AssertionError("dry-run must not prepare DB/RunPod context")

    monkeypatch.setattr(variant_prebuilt, "_prepare_context", _boom)
    assert variant_prebuilt.run(args) == 0
    out = capsys.readouterr().out
    assert "Variant: prebuilt" in out
    assert "z_image_turbo" in out


def test_target_manifest_for_cases_excludes_unselected_ready_templates():
    args = _base_args(
        backend="vibecomfy",
        selector_namespace="livet-test",
        selector_version="v1",
        worker_contract_version=3,
        worker_profile="portable",
        case=["z_image_turbo"],
        task_type=[],
        route_key=[],
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        timeout_image=60,
        timeout_travel_segment=60,
        timeout_travel_orchestrator=60,
    )
    cases = variant_prebuilt._build_matrix_cases(args)
    manifest = variant_prebuilt._target_manifest_for_cases(cases, args)

    assert [target["case_name"] for target in manifest["targets"]] == ["z_image_turbo"]
    assert manifest["templates"] == ["image/z_image"]
    assert manifest["selection"] == {
        "case_names": ["z_image_turbo"],
        "task_types": [],
        "route_keys": [],
    }
    assert manifest["selector"] == {
        "namespace": "livet-test",
        "version": "v1",
        "worker_contract_version": 3,
        "profile": "portable",
    }
    # This is a selected-target manifest, not a dump of every VibeComfy ready template.
    assert "image/qwen_image_2512" not in manifest["templates"]
    assert "edit/qwen_image_edit" not in manifest["templates"]


def test_prebuilt_health_failure_aborts_before_register_launch_or_queue(monkeypatch, tmp_path):
    events: list[str] = []
    args = _base_args(
        dry_run=False,
        no_terminate=False,
        prebuilt_volume_name="reigh-livetest-prebuilt-portable-us-tx-1",
        backend="vibecomfy",
        selector_namespace="livet-test",
        case=["z_image_turbo"],
        task_type=[],
        route_key=[],
        anchor_image_a="https://example.com/a.png",
        anchor_image_b="https://example.com/b.png",
        timeout_image=60,
        timeout_travel_segment=60,
        timeout_travel_orchestrator=60,
    )
    manifest = _make_manifest(python_version="3.11")

    class FakeSSH:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def execute_command(self, command: str, timeout: int = 600):
            self.commands.append(command)
            return 0, "{}", ""

        def disconnect(self):
            events.append("disconnect")

    fake_ssh = FakeSSH()

    def forbidden(name):
        def _inner(*_args, **_kwargs):
            raise AssertionError(f"{name} must not run after failed health")

        return _inner

    health_report = PrebuiltHealthReport(
        schema_version=1,
        generated_at_utc="2026-05-13T00:00:00+00:00",
        volume_name=args.prebuilt_volume_name,
        data_center_id="us-tx-1",
        attention_profile="portable",
        worker_env=PrebuiltPythonEnvReport(
            label="worker",
            python_path="/opt/reigh-worker-live-test-venv/bin/python",
            cwd="/opt/reigh-livetest-prebuilt/worker",
            python_version="3.11.11",
            import_ok=True,
        ),
        vibecomfy_env=PrebuiltPythonEnvReport(
            label="vibecomfy",
            python_path="/opt/reigh-livetest-prebuilt/vibecomfy/.venv/bin/python",
            cwd="/opt/reigh-livetest-prebuilt/vibecomfy",
            python_version="3.11.11",
            import_ok=True,
        ),
        issues=[
            PrebuiltHealthIssue(
                group="assets",
                code="missing_model_asset",
                message="missing checkpoints/z_image.safetensors",
            )
        ],
        targets_path="/workspace/reigh-livetest-prebuilt/runs/run/targets.json",
        enriched_path="/workspace/reigh-livetest-prebuilt/runs/run/targets.enriched.json",
    )

    monkeypatch.setattr(
        variant_prebuilt,
        "_prepare_context",
        lambda _args: {
            "db": object(),
            "token": "token-1",
            "user_id": "user-1",
            "project_id": "project-1",
            "cases": variant_prebuilt._build_matrix_cases(args),
        },
    )
    monkeypatch.setattr(variant_prebuilt.config, "require_env", lambda name: f"{name.lower()}-value")
    monkeypatch.setattr(variant_prebuilt, "prune_stale_live_test_pods", lambda _api_key: SimpleNamespace(terminated=[]))
    monkeypatch.setattr(variant_prebuilt, "_resolve_volume", lambda _args, _api_key: (args.prebuilt_volume_name, "us-tx-1"))
    monkeypatch.setattr(variant_prebuilt, "_resolve_runpod_gpu_type_id", lambda _api_key, _gpu_type: ("gpu-4090", "RTX 4090"))
    monkeypatch.setattr(variant_prebuilt, "_runs_root", lambda: tmp_path / "runs")
    monkeypatch.setattr(variant_prebuilt, "_timestamp_label", lambda: "20260513T000000Z")
    monkeypatch.setattr(variant_prebuilt, "open_session", lambda _pod_id, _api_key: fake_ssh)
    monkeypatch.setattr(variant_prebuilt, "_ensure_mounted", lambda *_args, **_kwargs: events.append("mounted"))
    monkeypatch.setattr(variant_prebuilt, "_install_prebuilt_system_tools", lambda *_args, **_kwargs: events.append("system_tools"))
    monkeypatch.setattr(variant_prebuilt, "read_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(variant_prebuilt, "extract_bundle_to_container_disk", lambda *_args, **_kwargs: events.append("extract"))
    monkeypatch.setattr(variant_prebuilt, "ensure_git_ref_synced", lambda *_args, **_kwargs: events.append("git_sync"))
    monkeypatch.setattr(variant_prebuilt, "_write_extra_model_paths_yaml", lambda *_args, **_kwargs: events.append("models_bound"))
    monkeypatch.setattr(variant_prebuilt, "_write_remote_json", lambda *_args, **_kwargs: events.append("targets_written"))
    monkeypatch.setattr(
        variant_prebuilt,
        "_enrich_targets_on_consumer",
        lambda *_args, **_kwargs: {"schema_version": 1, "targets": []},
    )
    monkeypatch.setattr(
        variant_prebuilt,
        "run_prebuilt_health_probes",
        lambda *_args, **_kwargs: health_report,
    )
    monkeypatch.setattr(variant_prebuilt, "write_health_report", lambda *_args, **_kwargs: events.append("health_written"))
    monkeypatch.setattr(variant_prebuilt, "fetch_worker_logs", lambda *_args, **_kwargs: "worker logs")
    monkeypatch.setattr(variant_prebuilt, "guarded_terminate", lambda pod_id, *_args, **_kwargs: events.append(f"terminate:{pod_id}"))
    monkeypatch.setattr(variant_prebuilt, "register_worker_record", forbidden("register_worker_record"))
    monkeypatch.setattr(variant_prebuilt, "launch_worker_detached", forbidden("launch_worker_detached"))
    monkeypatch.setattr(variant_prebuilt, "wait_until_ready", forbidden("wait_until_ready"))
    monkeypatch.setattr(variant_prebuilt, "queue_matrix", forbidden("queue_matrix"))

    from runpod_lifecycle import api as runpod_api

    monkeypatch.setattr(
        runpod_api,
        "create_pod",
        lambda **_kwargs: events.append("create_pod") or {"id": "pod-health-fail"},
    )
    monkeypatch.setattr(
        runpod_api,
        "get_network_volumes",
        lambda _api_key: [{"id": "vol-1", "name": args.prebuilt_volume_name}],
    )

    with pytest.raises(RuntimeError) as info:
        variant_prebuilt.run(args)

    message = str(info.value)
    assert "Prebuilt consumer health validation failed before worker registration/launch" in message
    assert "assets/missing_model_asset" in message
    assert events.index("health_written") < events.index("terminate:pod-health-fail")
    assert "create_pod" in events
    assert "targets_written" in events
    assert "health_written" in events
    assert (tmp_path / "runs" / "20260513T000000Z" / "targets.json").exists()
    assert (tmp_path / "runs" / "20260513T000000Z" / "targets.enriched.json").exists()
    assert (tmp_path / "runs" / "20260513T000000Z" / "env.health.json").exists()


def test_install_prebuilt_system_tools_installs_archive_progress_and_video_tools():
    calls = []

    class DummySSH:
        def execute_command(self, command, timeout=600):
            calls.append((command, timeout))
            return 0, "", ""

    _install_prebuilt_system_tools(DummySSH())
    command, timeout = calls[0]
    assert timeout == 600
    assert "apt-get install -y zstd pv ffmpeg" in command


# --------------------------------------------------------------------------- #
# (j) variant_update guard — fake SSH reports manifest present → raises with
# literal `Prebuilt cache present` BEFORE any uv sync.
# --------------------------------------------------------------------------- #


class _ManifestPresentSSH:
    """Fake SSH whose `test -f` for the prebuilt manifest succeeds (exit 0)."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def execute_command(self, command: str, timeout: int = 600):
        self.commands.append(command)
        return 0, "", ""


def test_variant_update_guard_aborts_when_prebuilt_manifest_present():
    ssh = _ManifestPresentSSH()
    with pytest.raises(RuntimeError) as info:
        variant_update._abort_if_prebuilt_cache_present(ssh)
    msg = str(info.value)
    assert "Prebuilt cache present" in msg
    assert "--variant prebuilt" in msg
    assert "rl prebuilt invalidate" in msg
    # The guard only issues the test-f check; no `uv sync` was invoked.
    assert all("uv sync" not in cmd for cmd in ssh.commands)


class _ManifestAbsentSSH:
    """Fake SSH where `test -f` for the prebuilt manifest returns non-zero."""

    def execute_command(self, command: str, timeout: int = 600):
        return 1, "", "no such file"


def test_variant_update_guard_passes_when_no_prebuilt_manifest():
    # Returns None (no raise).
    assert variant_update._abort_if_prebuilt_cache_present(_ManifestAbsentSSH()) is None


# --------------------------------------------------------------------------- #
# (k) terminate_guard regex — three prefixes match canonical pod names.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "reigh-livetest-prebuilt-20260513t120000z",
        "reigh-livetest-builder-20260513t120000z",
        "reigh-live-test-fresh-20260513t120000z",
    ],
)
def test_terminate_guard_regex_matches_each_prefix(name):
    matched = any(pattern.fullmatch(name) for pattern in _LIVE_TEST_POD_NAME_RES)
    assert matched, f"no regex matched {name!r}"


def test_terminate_guard_prefixes_exact_tuple():
    assert LIVE_TEST_POD_PREFIXES == (
        "reigh-live-test-fresh-",
        "reigh-livetest-prebuilt-",
        "reigh-livetest-builder-",
    )


def test_terminate_guard_regex_rejects_non_live_test_names():
    bogus = (
        "reigh-prod-2026-05-13",
        "reigh-live-test-fresh-",  # missing timestamp
        "reigh-livetest-prebuilt-ABCD1234",  # not the right charset
    )
    for name in bogus:
        for pattern in _LIVE_TEST_POD_NAME_RES:
            assert pattern.fullmatch(name) is None, f"unexpected match: {name!r}"


# --------------------------------------------------------------------------- #
# (l) Legacy import re-export check — variant_fresh keeps exposing _phase et al.
# --------------------------------------------------------------------------- #


def test_legacy_imports_from_variant_fresh_still_resolve():
    from scripts.live_test.variant_fresh import (
        _phase,
        _redact_sensitive_text,
        _capture_and_redact_noisy_lifecycle_output,
    )

    assert callable(_phase)
    assert callable(_redact_sensitive_text)
    assert callable(_capture_and_redact_noisy_lifecycle_output)


# --------------------------------------------------------------------------- #
# (a)/(b)/(c)/(d)/(h) Drift-decision behaviour at the variant_prebuilt level.
# We test the decision functions / shell predicates that determine whether a
# given drift triggers `_uv_sync_shell`, `_vibecomfy_install_shell(run_nodes_restore=True)`,
# or just `pip install -e .`. Doing this end-to-end against the full
# `variant_prebuilt.run()` requires an SSH session and a real pod; here we
# verify the underlying shell-rendering primitives.
# --------------------------------------------------------------------------- #


def test_uv_sync_shell_is_what_pyproject_drift_triggers():
    """When pyproject_hash differs, variant_prebuilt issues `_uv_sync_shell` with cuda124."""
    from scripts.live_test.ssh_bootstrap import _uv_sync_shell

    body = _uv_sync_shell(
        "/opt/reigh-livetest-prebuilt/worker",
        env_path="/opt/reigh-worker-live-test-venv",
        extras=("cuda124",),
    )
    assert "uv sync --extra cuda124" in body
    assert "UV_PROJECT_ENVIRONMENT=/opt/reigh-worker-live-test-venv" in body


def test_vibecomfy_install_shell_with_nodes_restore_is_what_lockfile_drift_triggers():
    """When custom_nodes_lock_hash differs, variant_prebuilt issues run_nodes_restore=True."""
    from scripts.live_test.ssh_bootstrap import _vibecomfy_install_shell

    body = _vibecomfy_install_shell(
        "/opt/reigh-livetest-prebuilt/vibecomfy",
        python_path="python3.11",
        attention_profile="portable",
        run_nodes_restore=True,
    )
    assert "vibecomfy.cli nodes restore --lockfile custom_nodes.lock" in body
    assert "uv venv --seed --python python3.11 /opt/reigh-vibecomfy-live-test-venv" in body
    assert (
        "uv pip install --python /opt/reigh-vibecomfy-live-test-venv/bin/python "
        "-e /opt/reigh-livetest-prebuilt/vibecomfy"
    ) in body


def test_vibecomfy_install_shell_without_nodes_restore_for_commit_only_drift():
    """When only vibecomfy_commit drifts (lockfile matches), no destructive nodes restore."""
    from scripts.live_test.ssh_bootstrap import _vibecomfy_install_shell

    body = _vibecomfy_install_shell(
        "/opt/reigh-livetest-prebuilt/vibecomfy",
        python_path="python3.11",
        attention_profile="portable",
        run_nodes_restore=False,
    )
    assert "uv venv --seed --python python3.11 /opt/reigh-vibecomfy-live-test-venv" in body
    assert (
        "uv pip install --python /opt/reigh-vibecomfy-live-test-venv/bin/python "
        "-e /opt/reigh-livetest-prebuilt/vibecomfy"
    ) in body
    assert "vibecomfy.cli nodes restore" not in body


def test_vibecomfy_install_shell_uses_venv_local_uv_for_venv_python():
    from scripts.live_test.ssh_bootstrap import _vibecomfy_install_shell

    body = _vibecomfy_install_shell(
        "/opt/reigh-livetest-prebuilt/vibecomfy",
        python_path="/opt/reigh-livetest-prebuilt/vibecomfy/.venv/bin/python",
        attention_profile="portable",
        run_nodes_restore=True,
    )
    assert (
        "/opt/reigh-livetest-prebuilt/vibecomfy/.venv/bin/uv pip install --python "
        "/opt/reigh-livetest-prebuilt/vibecomfy/.venv/bin/python"
    ) in body


def test_variant_prebuilt_phase_order_is_documented_in_run():
    """Static check that run() invokes the bootstrap phases in the required order.

    Reads the source of variant_prebuilt.run and confirms the documented _phase
    contexts appear in the required sequence (attach → read_manifest →
    hard_fail → extract_venv → extract_vibecomfy → sync_worker → sync_vibecomfy
    → bind_models → target/enrichment evidence → health probes → registration
    → launch). The shared health probe must run AFTER the sync/model-path
    phases and BEFORE worker registration/launch.
    """
    import inspect

    src = inspect.getsource(variant_prebuilt.run)
    required_order = [
        '"create_runpod_pod"',
        '_phase("open_ssh_session"',
        '_phase("attach_prebuilt_volume"',
        '_phase("read_prebuilt_manifest"',
        '_phase("check_hard_fail_drift"',
        '_phase("extract_venv_bundle"',
        '_phase("extract_vibecomfy_bundle"',
        '_phase("sync_worker_ref"',
        '_phase("sync_vibecomfy_ref"',
        '_phase("bind_models_dir"',
        '_phase("write_prebuilt_targets_evidence"',
        '_phase("enrich_prebuilt_targets"',
        '_phase("run_prebuilt_consumer_health_probes"',
        "register_worker_record(",
        '_phase("launch_worker"',
    ]
    last_index = -1
    for marker in required_order:
        idx = src.find(marker)
        assert idx != -1, f"phase marker not found in run(): {marker!r}"
        assert idx > last_index, (
            f"phase {marker!r} appears out of order (expected after previous phase)"
        )
        last_index = idx
    # Explicit health-after-syncs-before-registration assertion.
    health_idx = src.find('_phase("run_prebuilt_consumer_health_probes"')
    worker_sync_idx = src.find('_phase("sync_worker_ref"')
    vibe_sync_idx = src.find('_phase("sync_vibecomfy_ref"')
    register_idx = src.find("register_worker_record(")
    launch_idx = src.find('_phase("launch_worker"')
    assert health_idx > worker_sync_idx
    assert health_idx > vibe_sync_idx
    assert health_idx < register_idx
    assert health_idx < launch_idx
