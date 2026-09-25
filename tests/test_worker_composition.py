from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from source.runtime import supervisor
from source.runtime.worker import preflight
from source.runtime.worker.preflight import PreflightCheck, WorkerPreflightResult
from source.runtime.vibecomfy_profile import VerifiedFacts


def _config(tmp_path: Path) -> supervisor.HostLaunchConfig:
    source = tmp_path / "Astrid"
    host = source / "astrid" / "core" / "execution" / "generic_host.py"
    host.parent.mkdir(parents=True)
    host.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv\n"
        "ready = Path(args[args.index('--ready-file') + 1])\n"
        "(ready.parent / 'argv.json').write_text(json.dumps(args))\n"
        "ready.write_text(json.dumps({'status': 'ready', 'pid': os.getpid()}))\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    (source / "astrid" / "packs").mkdir(parents=True)
    support = tmp_path / "support"
    (support / "credentials").mkdir(parents=True)
    credential = support / "credentials" / "worker.token"
    credential.write_text("sentinel-worker-token", encoding="utf-8")
    credential.chmod(0o600)
    manifest = support / "boot-manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return supervisor.HostLaunchConfig(
        host_python=Path(sys.executable).resolve(), source_checkout=source.resolve(),
        pack_root=(source / "astrid" / "packs").resolve(), runtime_endpoint="http://127.0.0.1:18765",
        credential_file=credential.resolve(), support_root=support.resolve(), runtime_instance_id="runtime-1",
        ready_file=(support / "host-ready.json").resolve(), state_file=(support / "worker-state.json").resolve(),
        boot_manifest_path=manifest.resolve(), boot_manifest_hash="sha256:test",
    )


def _facts_fixture(tmp_path: Path, config: supervisor.HostLaunchConfig, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    source = config.source_checkout
    (source / "uv.lock").write_text("runtime", encoding="utf-8")
    (source / "engine.lock").write_text("engine", encoding="utf-8")
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_bytes = b"model"
    (model_root / "model.bin").write_bytes(model_bytes)
    model_manifest = tmp_path / "models.json"
    model_manifest.write_text(json.dumps({"files": [{"path": "model.bin", "sha256": hashlib.sha256(model_bytes).hexdigest()}]}), encoding="utf-8")
    node_root = tmp_path / "nodes"
    node_root.mkdir()
    node_bytes = b"node"
    (node_root / "node.py").write_bytes(node_bytes)
    node_manifest = tmp_path / "nodes.json"
    node_manifest.write_text(json.dumps({"files": [{"path": "node.py", "sha256": hashlib.sha256(node_bytes).hexdigest()}]}), encoding="utf-8")
    scratch = tmp_path / "scratch"
    cas = tmp_path / "cas"
    output = tmp_path / "output"
    scratch.mkdir()
    cas.mkdir()
    output.mkdir()
    values = {
        "REIGH_INTERPRETER": str(Path(sys.executable).resolve()),
        "REIGH_ENGINE_INTERPRETER": str(Path(sys.executable).resolve()),
        "REIGH_RUNTIME_LOCK_PATH": str(source / "uv.lock"),
        "REIGH_ENGINE_LOCK_PATH": str(source / "engine.lock"),
        "REIGH_MODEL_ROOT": str(model_root),
        "REIGH_MODEL_MANIFEST_PATH": str(model_manifest),
        "REIGH_CUSTOM_NODE_ROOT": str(node_root),
        "REIGH_CUSTOM_NODE_MANIFEST_PATH": str(node_manifest),
        "REIGH_SCRATCH_ROOT": str(scratch),
        "REIGH_CAS_ROOT": str(cas),
        "REIGH_OUTPUT_ROOT": str(output),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


def _discovery(config: supervisor.HostLaunchConfig, *, actor: str = "astrid-pack-host") -> None:
    realm_root = config.support_root.parent / "realm"
    realm_root.mkdir(exist_ok=True)
    record = {
        "version": 1, "endpoint": config.runtime_endpoint, "pid": os.getpid(),
        "process_birth_id": "fixture-birth", "active_realm": "realm-1", "runtime_instance_id": config.runtime_instance_id,
        "realm_root": str(realm_root.resolve()),
        "protocol_version": "workspace.v1", "schema_version": "workspace-schema-v1", "coordinator_epoch": config.runtime_instance_id,
        "credential_file": str(config.support_root / "credentials" / "owner.token"),
        "worker_credential_file": str(config.credential_file), "worker_actor": actor,
        "worker_scopes": ["handshake", "worker:register", "worker:execute", "tasks:read", "objects:read", "objects:write"],
    }
    path = config.support_root / "discovery.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    path.chmod(0o600)


def test_composed_startup_publishes_secret_free_profile_before_one_spawn(tmp_path, monkeypatch):
    config = _config(tmp_path)
    fact_inputs = _facts_fixture(tmp_path, config, monkeypatch)
    _discovery(config)
    monkeypatch.setattr(preflight, "_process_birth_identity", lambda pid: "fixture-birth")
    monkeypatch.setattr(preflight, "_verify_runtime_process", lambda binding: None)
    monkeypatch.setattr(preflight, "_read_runtime_health", lambda binding: {"status": "ok", "protocol": "workspace.v1", "schema_digest": "sha256:" + "1" * 64, "runtime_epoch": 1})
    monkeypatch.setattr(preflight, "_probe_runtime_binding", lambda binding: {"ok": True})
    real_neutral = preflight.run_neutral_worker_preflight
    calls = []

    def _neutral(**kwargs):
        calls.append(kwargs)
        return real_neutral(
            **kwargs,
            probes=preflight.FactProbeOverrides(gpu=lambda: {"uuid": "GPU", "name": "fixture", "driver": "driver", "cuda": "12.4", "vram_bytes": 1024}),
        )

    monkeypatch.setattr(preflight, "run_neutral_worker_preflight", _neutral)
    env = os.environ.copy()
    env.update(fact_inputs)
    env["PATH"] = "/bin"
    result = supervisor.launch_generic_pack_host(config, environ=env)
    assert result == 0
    assert len(calls) == 1
    profile_path = config.support_root / "worker-readiness-profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["status"] == "ready"
    assert profile["runtime"]["credential_reference"] == str(config.credential_file)
    assert "sentinel-worker-token" not in profile_path.read_text(encoding="utf-8")
    child_argv = json.loads((config.support_root / "argv.json").read_text(encoding="utf-8"))
    assert child_argv[child_argv.index("--readiness-profile-path") + 1] == str(profile_path)
    profile_hash = child_argv[child_argv.index("--readiness-profile-hash") + 1]
    assert profile_hash == "sha256:" + hashlib.sha256(profile_path.read_bytes()).hexdigest()
    assert config.state_file.exists()


@pytest.mark.parametrize("mutation", ["missing", "malformed", "swapped", "actor", "credential"])
def test_composed_startup_fails_closed_before_spawn(tmp_path, monkeypatch, mutation):
    config = _config(tmp_path)
    _facts_fixture(tmp_path, config, monkeypatch)
    _discovery(config)
    monkeypatch.setattr(preflight, "_process_birth_identity", lambda pid: "fixture-birth")
    discovery_path = config.support_root / "discovery.json"
    if mutation == "missing":
        discovery_path.unlink()
    elif mutation == "malformed":
        discovery_path.write_text("not-json", encoding="utf-8")
    else:
        record = json.loads(discovery_path.read_text(encoding="utf-8"))
        if mutation == "swapped":
            record["runtime_instance_id"] = "runtime-2"
        elif mutation == "actor":
            record["worker_actor"] = "owner"
        else:
            record["worker_credential_file"] = str(config.support_root / "credentials" / "other.token")
        discovery_path.write_text(json.dumps(record), encoding="utf-8")
    spawned = False

    def _no_spawn(*args, **kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("host spawned before readiness gate")

    monkeypatch.setattr(supervisor.subprocess, "Popen", _no_spawn)
    with pytest.raises(supervisor.LauncherConfigurationError):
        supervisor.launch_generic_pack_host(config, environ={"PATH": "/bin"})
    assert spawned is False
    assert not (config.support_root / "worker-readiness-profile.json").exists()


def test_targeted_route_fails_closed_without_credential_backed_placement_issuer(tmp_path, monkeypatch):
    config = _config(tmp_path)
    spawned = False

    def _no_spawn(*args, **kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("targeted route spawned without placement evidence")

    monkeypatch.setattr(supervisor.subprocess, "Popen", _no_spawn)
    env = {
        "PATH": "/bin",
        "ASTRID_EXECUTION_TARGET_JSON": json.dumps(
            {"kind": "runpod", "pod_id": "pod-1", "provider_account_ref": "account-1"}
        ),
    }
    with pytest.raises(
        supervisor.LauncherConfigurationError,
        match="credential-backed placement issuer",
    ):
        supervisor.launch_generic_pack_host(config, environ=env, enforce_readiness=False)
    assert spawned is False


def test_profile_publication_failure_fails_closed(tmp_path, monkeypatch):
    config = _config(tmp_path)
    fact_inputs = _facts_fixture(tmp_path, config, monkeypatch)
    _discovery(config)
    monkeypatch.setattr(preflight, "_process_birth_identity", lambda pid: "fixture-birth")
    monkeypatch.setattr(preflight, "_verify_runtime_process", lambda binding: None)
    monkeypatch.setattr(preflight, "_read_runtime_health", lambda binding: {"status": "ok", "protocol": "workspace.v1", "schema_digest": "sha256:" + "1" * 64, "runtime_epoch": 1})
    monkeypatch.setattr(preflight, "_probe_runtime_binding", lambda binding: {"ok": True})
    monkeypatch.setattr(
        preflight,
        "run_neutral_worker_preflight",
        lambda **kwargs: WorkerPreflightResult(
            status="passed", checks=[PreflightCheck("fixture", True, "ok")], started_at=1.0, completed_at=2.0,
            verified_facts=VerifiedFacts(
                exact={"interpreter": "python", "runtime_lock": "runtime", "engine_lock": "engine", "model_digest": "model", "custom_node_digest": "nodes", "driver": "driver", "root": "root", "port": 18765},
                minimum={"vram_bytes": 1, "scratch_bytes": 1},
            ),
        ),
    )
    monkeypatch.setattr(supervisor, "_atomic_write_json", lambda path, value: (_ for _ in ()).throw(OSError("publication")))
    env = os.environ.copy()
    env.update(fact_inputs)
    env["PATH"] = "/bin"
    with pytest.raises(supervisor.LauncherConfigurationError, match="publication"):
        supervisor.launch_generic_pack_host(config, environ=env)
    assert not (config.support_root / "worker-readiness-profile.json").exists()


def test_incomplete_neutral_facts_fail_before_spawn(tmp_path, monkeypatch):
    config = _config(tmp_path)
    fact_inputs = _facts_fixture(tmp_path, config, monkeypatch)
    _discovery(config)
    monkeypatch.setattr(preflight, "_process_birth_identity", lambda pid: "fixture-birth")
    monkeypatch.setattr(preflight, "_verify_runtime_process", lambda binding: None)
    monkeypatch.setattr(preflight, "_read_runtime_health", lambda binding: {"status": "ok", "protocol": "workspace.v1", "schema_digest": "sha256:" + "1" * 64, "runtime_epoch": 1})
    monkeypatch.setattr(preflight, "_probe_runtime_binding", lambda binding: {"ok": True})
    monkeypatch.setattr(
        preflight,
        "run_neutral_worker_preflight",
        lambda **kwargs: WorkerPreflightResult(
            status="failed", checks=[PreflightCheck("fact_schema", False, "incomplete")], started_at=1.0, completed_at=2.0,
            verified_facts=VerifiedFacts(exact={}, minimum={}),
        ),
    )
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("spawned")))
    env = os.environ.copy()
    env.update(fact_inputs)
    with pytest.raises(supervisor.LauncherConfigurationError, match="facts"):
        supervisor.launch_generic_pack_host(config, environ=env)
