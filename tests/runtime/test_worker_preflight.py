"""Worker preflight and readiness metadata contracts."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from source.runtime.worker import guardian
from source.runtime.worker.health_labels import write_worker_route_state
from source.runtime.worker.preflight import (
    PREFLIGHT_STATUS_PASSED,
    PreflightCheck,
    RuntimeBinding,
    WorkerPreflightResult,
    FactProbeOverrides,
    _probe_runtime_binding,
    _process_birth_identity,
    _canonical_digest,
    _inspect_model_manifest,
    _model_inventory_digest,
    collect_verified_facts,
    finalize_preflight_result,
    preflight_state_path,
    publish_preflight_metadata,
    run_worker_preflight,
    write_preflight_state,
)
from source.runtime.vibecomfy_profile import VerifiedFacts
from source.runtime.worker.resource_pressure import ResourcePressureResult, write_resource_pressure_state


def _make_worker_repo(tmp_path):
    repo_root = tmp_path / "Reigh-Worker"
    wan2gp = repo_root / "Wan2GP"
    (wan2gp / "models").mkdir(parents=True)
    (wan2gp / "plugins").mkdir()
    (wan2gp / ".git").write_text("gitdir: ../.git/modules/Wan2GP\n", encoding="utf-8")
    (wan2gp / "wgp.py").write_text("# wgp\n", encoding="utf-8")
    (repo_root / "source" / "task_handlers" / "tasks").mkdir(parents=True)
    (repo_root / "source" / "task_handlers" / "tasks" / "dispatch_manifest.py").write_text("# manifest\n", encoding="utf-8")
    (repo_root / "source" / "models" / "lora").mkdir(parents=True)
    (repo_root / "source" / "models" / "lora" / "module_manifest.py").write_text("# manifest\n", encoding="utf-8")

    vibecomfy = tmp_path / "vibecomfy"
    (vibecomfy / "workflow_corpus" / "manifests").mkdir(parents=True)
    (vibecomfy / "template_index.json").write_text("{}", encoding="utf-8")
    (vibecomfy / "workflow_corpus" / "manifests" / "coverage.json").write_text("{}", encoding="utf-8")
    return repo_root, wan2gp


def _configure_verified_facts(tmp_path, monkeypatch, repo_root):
    runtime_lock = repo_root / "uv.lock"
    runtime_lock.write_text("runtime lock\n", encoding="utf-8")
    engine_lock = repo_root / "engine.lock"
    engine_lock.write_text("engine lock\n", encoding="utf-8")

    def _manifest(root_name, file_name, contents):
        root = tmp_path / root_name
        root.mkdir()
        file_path = root / file_name
        file_path.write_bytes(contents)
        digest = hashlib.sha256(contents).hexdigest()
        manifest = tmp_path / f"{root_name}.json"
        manifest.write_text(
            json.dumps({"files": [{"path": file_name, "sha256": f"sha256:{digest}"}]}),
            encoding="utf-8",
        )
        return root, manifest

    model_root, model_manifest = _manifest("models", "model.bin", b"model bytes")
    custom_root, custom_manifest = _manifest("custom-nodes", "node.py", b"node bytes")
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    cas_root = tmp_path / "cas"
    cas_root.mkdir()

    monkeypatch.setenv("REIGH_INTERPRETER", sys.executable)
    monkeypatch.setenv("REIGH_RUNTIME_LOCK_PATH", str(runtime_lock))
    monkeypatch.setenv("REIGH_ENGINE_LOCK_PATH", str(engine_lock))
    monkeypatch.setenv("REIGH_MODEL_ROOT", str(model_root))
    monkeypatch.setenv("REIGH_MODEL_MANIFEST_PATH", str(model_manifest))
    monkeypatch.setenv("REIGH_CUSTOM_NODE_ROOT", str(custom_root))
    monkeypatch.setenv("REIGH_CUSTOM_NODE_MANIFEST_PATH", str(custom_manifest))
    monkeypatch.setenv("REIGH_SCRATCH_ROOT", str(scratch_root))
    monkeypatch.setenv("REIGH_CAS_ROOT", str(cas_root))
    runtime_binding = _runtime_binding(tmp_path)

    return FactProbeOverrides(
        gpu=lambda: {
            "uuid": "GPU-fixture",
            "name": "GPU fixture",
            "driver": "550.1",
            "cuda": "12.4",
            "vram_bytes": 16 * 1024**3,
        },
        runtime=lambda binding: {"ok": True, "detail": f"{binding.endpoint} injected verifier"},
    ), runtime_binding


def _runtime_binding(tmp_path, *, port=18765, **overrides):
    credential_path = tmp_path / "runtime-worker.token"
    credential_path.write_text("worker-token\n", encoding="utf-8")
    credential_path.chmod(0o600)
    values = {
        "endpoint": f"http://127.0.0.1:{port}",
        "port": port,
        "pid": os.getpid(),
        "process_birth_id": "fixture-process",
        "runtime_instance_id": "fixture-runtime",
        "runtime_epoch": 1,
        "schema_digest": "sha256:" + "0" * 64,
        "credential_path": credential_path,
    }
    values.update(overrides)
    return RuntimeBinding(**values)


def test_runtime_binding_verifies_live_listener_and_credentialed_health(tmp_path):
    token = "worker-token"
    digest = "sha256:" + "0" * 64
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            body = json.dumps(
                {
                    "status": "ok",
                    "protocol": "workspace.v1",
                    "schema_digest": digest,
                    "runtime_epoch": 1,
                }
            ).encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    credential_path = tmp_path / "runtime.token"
    credential_path.write_text(token, encoding="utf-8")
    credential_path.chmod(0o600)
    binding = RuntimeBinding(
        endpoint=f"http://127.0.0.1:{server.server_port}",
        port=server.server_port,
        pid=os.getpid(),
        process_birth_id=_process_birth_identity(os.getpid()),
        runtime_instance_id="runtime-live",
        runtime_epoch=1,
        schema_digest=digest,
        credential_path=credential_path,
    )
    try:
        assert _probe_runtime_binding(binding)["ok"] is True
        assert seen == [f"Bearer {token}"]
        parent_birth = _process_birth_identity(os.getppid())
        if parent_birth:
            unrelated = RuntimeBinding(
                endpoint=binding.endpoint,
                port=binding.port,
                pid=os.getppid(),
                process_birth_id=parent_birth,
                runtime_instance_id=binding.runtime_instance_id,
                runtime_epoch=binding.runtime_epoch,
                schema_digest=binding.schema_digest,
                credential_path=binding.credential_path,
            )
            with pytest.raises(OSError, match="owned by the bound process"):
                _probe_runtime_binding(unrelated)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"port": 0, "endpoint": "http://127.0.0.1:0"}, "port"),
        ({"endpoint": "http://127.0.0.1:18766"}, "match"),
        ({"process_birth_id": "stale"}, "birth"),
    ],
)
def test_runtime_binding_rejects_invalid_or_stale_identity(tmp_path, overrides, expected):
    binding = _runtime_binding(tmp_path, **overrides)
    with pytest.raises((OSError, ValueError), match=expected):
        _probe_runtime_binding(binding)


def test_runtime_binding_rejects_unused_port_without_creating_listener(tmp_path):
    binding = _runtime_binding(tmp_path, port=18766, process_birth_id=_process_birth_identity(os.getpid()))
    with pytest.raises(OSError, match="owned by the bound process"):
        _probe_runtime_binding(binding)


@pytest.mark.parametrize("health_override", [{"runtime_epoch": 2}, {"schema_digest": "sha256:" + "1" * 64}])
def test_runtime_binding_rejects_mismatched_health(tmp_path, health_override):
    expected_digest = "sha256:" + "0" * 64
    health = {
        "status": "ok",
        "protocol": "workspace.v1",
        "schema_digest": expected_digest,
        "runtime_epoch": 1,
    }
    health.update(health_override)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps(health).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    binding = _runtime_binding(
        tmp_path,
        port=server.server_port,
        endpoint=f"http://127.0.0.1:{server.server_port}",
        pid=os.getpid(),
        process_birth_id=_process_birth_identity(os.getpid()),
        schema_digest=expected_digest,
    )
    try:
        with pytest.raises(ValueError, match="identity"):
            _probe_runtime_binding(binding)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_missing_runtime_binding_fails_closed_without_fallback(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    called = False

    def _runtime(_binding):
        nonlocal called
        called = True
        return {"ok": True}

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp,
        main_output_dir=tmp_path / "outputs",
        backend="vibecomfy",
        probes=FactProbeOverrides(runtime=_runtime),
    )
    assert result.ready_for_tasks is False
    assert "fact:port" in result.failed_checks
    assert called is False


def _complete_verified_facts():
    return VerifiedFacts(
        exact={
            "interpreter": "python",
            "runtime_lock": "runtime",
            "engine_lock": "engine",
            "model_digest": "model",
            "custom_node_digest": "nodes",
            "driver": "driver",
            "root": "root",
            "port": 8765,
        },
        minimum={"vram_bytes": 1, "scratch_bytes": 1},
    )


def test_readiness_validates_direct_facts_and_mapping_mutation_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("REIGH_PREFLIGHT_STATE_DIR", str(tmp_path))
    facts = _complete_verified_facts()
    result = WorkerPreflightResult(
        status=PREFLIGHT_STATUS_PASSED,
        checks=[],
        started_at=1.0,
        completed_at=2.0,
        verified_facts=facts,
    )
    assert result.ready_for_tasks is True
    facts.exact["port"] = "8765"
    assert result.ready_for_tasks is False
    metadata = result.to_metadata()
    assert metadata["readiness"] == "not_ready"
    assert metadata["verified_facts_complete"] is False
    client = _FakeSupabase()
    publish_preflight_metadata(
        supabase_client=client,
        worker_id="worker-invalid-facts",
        result=result,
        ready_for_tasks=True,
    )
    assert client.updated_payload["metadata"]["ready_for_tasks"] is False

    facts.exact["port"] = 8765
    facts.minimum["vram_bytes"] = -1
    assert result.ready_for_tasks is False
    assert result.to_metadata()["readiness_reason"].startswith("verified_facts_invalid:")


def test_finalize_preserves_verified_facts_while_failed_readiness_stays_closed():
    facts = _complete_verified_facts()
    base = WorkerPreflightResult(
        status=PREFLIGHT_STATUS_PASSED,
        checks=[],
        started_at=1.0,
        completed_at=2.0,
        verified_facts=facts,
    )
    final = finalize_preflight_result(
        base,
        extra_checks=[PreflightCheck("late_check", False, "failed")],
    )
    assert final.verified_facts is facts
    assert final.verified_facts.to_dict() == base.verified_facts.to_dict()
    assert final.ready_for_tasks is False
    assert final.to_metadata()["verified_facts_digest"] == base.to_metadata()["verified_facts_digest"]


def test_worker_preflight_passes_when_required_paths_and_manifests_exist(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    probes, runtime_binding = _configure_verified_facts(tmp_path, monkeypatch, repo_root)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("VIBECOMFY_PATH", str(tmp_path / "vibecomfy"))

    def _fake_find_spec(module_name):
        if module_name in {"torch", "dotenv", "fastapi"}:
            return SimpleNamespace(origin=f"/fake/{module_name}.py")
        return None

    monkeypatch.setattr("source.runtime.worker.preflight.importlib.util.find_spec", _fake_find_spec)

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp,
        main_output_dir=tmp_path / "outputs",
        backend="vibecomfy",
        probes=probes,
        runtime_binding=runtime_binding,
    )

    assert result.status == PREFLIGHT_STATUS_PASSED
    assert result.failed_checks == []
    assert {check.name for check in result.checks} >= {
        "wan2gp_path",
        "wgp_entrypoint",
        "vibecomfy_available",
        "vibecomfy_attention_profile",
        "vibecomfy_template_index",
        "vibecomfy_custom_nodes_manifest",
        "task_dispatch_manifest",
        "lora_module_manifest",
        "wan2gp_models_dir",
        "wan2gp_plugins_dir",
        "main_output_dir",
        "uv_cache_dir",
    }
    assert set(result.verified_facts.exact) == {
        "interpreter",
        "runtime_lock",
        "engine_lock",
        "model_digest",
        "custom_node_digest",
        "driver",
        "root",
        "port",
    }
    assert set(result.verified_facts.minimum) == {"vram_bytes", "scratch_bytes"}
    assert result.readiness == "ready"
    assert result.to_metadata()["readiness_reason"] is None


def test_worker_preflight_fails_closed_when_verified_model_bytes_drift(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    probes, runtime_binding = _configure_verified_facts(tmp_path, monkeypatch, repo_root)
    (tmp_path / "models" / "model.bin").write_bytes(b"changed model bytes")

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp,
        main_output_dir=tmp_path / "outputs",
        backend="vibecomfy",
        probes=probes,
        runtime_binding=runtime_binding,
    )

    assert result.status == "failed"
    assert result.readiness == "not_ready"
    assert "fact:model_digest" in result.failed_checks
    assert result.verified_facts.exact.get("model_digest") is None


def test_verified_model_manifest_produces_canonical_launch_inventory(tmp_path):
    root = tmp_path / "models"
    (root / "nested").mkdir(parents=True)
    files = {"z.bin": b"z", "nested/a.bin": b"alpha"}
    entries = []
    for relative, content in files.items():
        path = root / relative
        path.write_bytes(content)
        entries.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"files": entries}), encoding="utf-8")

    model_digest, _detail, binding = _inspect_model_manifest(root, manifest)

    expected_inventory = [
        {"subdir": "", "name": "z.bin", "size": 1, "sha256": "sha256:" + hashlib.sha256(b"z").hexdigest()},
        {"subdir": "nested", "name": "a.bin", "size": 5, "sha256": "sha256:" + hashlib.sha256(b"alpha").hexdigest()},
    ]
    expected_model_digest = _canonical_digest({
        "files": [
            {"path": "z.bin", "sha256": "sha256:" + hashlib.sha256(b"z").hexdigest()},
            {"path": "nested/a.bin", "sha256": "sha256:" + hashlib.sha256(b"alpha").hexdigest()},
        ]
    })
    assert model_digest == expected_model_digest
    assert binding == {
        "schema_version": 1,
        "path": str(root.resolve()),
        "inventory": expected_inventory,
        "inventory_digest": _model_inventory_digest(expected_inventory),
    }


def test_verified_facts_and_launch_inventory_share_model_inspection(tmp_path, monkeypatch):
    repo_root, _wan2gp = _make_worker_repo(tmp_path)
    probes, runtime_binding = _configure_verified_facts(tmp_path, monkeypatch, repo_root)
    details = {}

    facts, checks = collect_verified_facts(
        repo_root=repo_root,
        main_output_dir=tmp_path / "outputs",
        probes=probes,
        runtime_binding=runtime_binding,
        verified_details=details,
    )

    assert all(check.ok for check in checks)
    assert details["model_root_binding"]["inventory"] == [
        {
            "subdir": "",
            "name": "model.bin",
            "size": len(b"model bytes"),
            "sha256": "sha256:" + hashlib.sha256(b"model bytes").hexdigest(),
        }
    ]
    assert facts.exact["model_digest"] == _canonical_digest({
        "files": [{
            "path": "model.bin",
            "sha256": "sha256:" + hashlib.sha256(b"model bytes").hexdigest(),
        }]
    })


@pytest.mark.parametrize("case", ["duplicate", "escaping", "missing", "unlisted", "changed"])
def test_verified_model_manifest_rejects_unbound_inventory_changes(tmp_path, case):
    root = tmp_path / "models"
    root.mkdir()
    model_file = root / "model.bin"
    model_file.write_bytes(b"verified")
    entry = {"path": "model.bin", "sha256": hashlib.sha256(b"verified").hexdigest()}
    entries = [entry, dict(entry)] if case == "duplicate" else [entry]
    if case == "escaping":
        entries = [{"path": "../outside.bin", "sha256": entry["sha256"]}]
    if case == "missing":
        model_file.unlink()
    if case == "unlisted":
        (root / "extra.bin").write_bytes(b"extra")
    if case == "changed":
        model_file.write_bytes(b"changed")
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"files": entries}), encoding="utf-8")

    digest, detail, binding = _inspect_model_manifest(root, manifest)

    assert digest is None
    assert binding is None
    assert detail


def test_worker_preflight_fails_closed_without_fact_configuration(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    runtime_binding = _runtime_binding(tmp_path)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("VIBECOMFY_PATH", str(tmp_path / "vibecomfy"))
    monkeypatch.setattr(
        "source.runtime.worker.preflight.importlib.util.find_spec",
        lambda name: SimpleNamespace(origin=f"/fake/{name}.py"),
    )

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp,
        main_output_dir=tmp_path / "outputs",
        backend="vibecomfy",
        probes=FactProbeOverrides(
            gpu=lambda: {
                "uuid": "GPU-fixture",
                "name": "GPU fixture",
                "driver": "550.1",
                "cuda": "12.4",
                "vram_bytes": 16 * 1024**3,
            },
            runtime=lambda binding: {"ok": True},
        ),
        runtime_binding=runtime_binding,
    )

    assert result.status == "failed"
    assert result.readiness == "not_ready"
    assert "fact:engine_lock" in result.failed_checks
    assert "fact:model_digest" in result.failed_checks
    assert "model_digest" not in result.verified_facts.exact


def test_worker_preflight_fails_when_wgp_path_is_missing(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("VIBECOMFY_PATH", str(tmp_path / "vibecomfy"))
    monkeypatch.setattr(
        "source.runtime.worker.preflight.importlib.util.find_spec",
        lambda name: None if name == "vibecomfy" else SimpleNamespace(origin="/fake"),
    )

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp / "missing",
        main_output_dir=tmp_path / "outputs",
        backend="wgp",
    )

    assert result.status == "failed"
    assert "wan2gp_path" in result.failed_checks
    assert "wgp_entrypoint" in result.failed_checks


def test_worker_preflight_does_not_require_vibecomfy_for_wgp_backend(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    probes, runtime_binding = _configure_verified_facts(tmp_path, monkeypatch, repo_root)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.delenv("VIBECOMFY_PATH", raising=False)
    monkeypatch.delenv("REIGH_PREFLIGHT_REQUIRE_VIBECOMFY", raising=False)
    monkeypatch.setattr(
        "source.runtime.worker.preflight.importlib.util.find_spec",
        lambda name: None if name == "vibecomfy" else SimpleNamespace(origin="/fake"),
    )

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp,
        main_output_dir=tmp_path / "outputs",
        backend="wgp",
        probes=probes,
        runtime_binding=runtime_binding,
    )

    assert result.status == PREFLIGHT_STATUS_PASSED
    vibecomfy_checks = {check.name: check for check in result.checks if check.name.startswith("vibecomfy")}
    assert vibecomfy_checks["vibecomfy_available"].required is False


def test_worker_preflight_does_not_require_wan2gp_for_vibecomfy_backend(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    probes, runtime_binding = _configure_verified_facts(tmp_path, monkeypatch, repo_root)
    shutil.rmtree(wan2gp)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("VIBECOMFY_PATH", str(tmp_path / "vibecomfy"))
    monkeypatch.setattr(
        "source.runtime.worker.preflight.importlib.util.find_spec",
        lambda name: SimpleNamespace(origin=f"/fake/{name}.py"),
    )

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp,
        main_output_dir=tmp_path / "outputs",
        backend="vibecomfy",
        probes=probes,
        runtime_binding=runtime_binding,
    )

    assert result.status == PREFLIGHT_STATUS_PASSED
    wan2gp_checks = {check.name: check for check in result.checks if check.name.startswith("wan2gp") or check.name == "wgp_entrypoint"}
    assert wan2gp_checks["wan2gp_path"].required is False
    assert wan2gp_checks["wan2gp_submodule_marker"].required is False
    assert wan2gp_checks["wgp_entrypoint"].required is False
    assert wan2gp_checks["wan2gp_models_dir"].required is False
    assert wan2gp_checks["wan2gp_plugins_dir"].required is False


def test_worker_preflight_env_can_force_vibecomfy_for_wgp_backend(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    shutil.rmtree(tmp_path / "vibecomfy")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.delenv("VIBECOMFY_PATH", raising=False)
    monkeypatch.setenv("REIGH_PREFLIGHT_REQUIRE_VIBECOMFY", "1")
    monkeypatch.setattr(
        "source.runtime.worker.preflight.importlib.util.find_spec",
        lambda name: None if name == "vibecomfy" else SimpleNamespace(origin="/fake"),
    )

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp,
        main_output_dir=tmp_path / "outputs",
        backend="wgp",
    )

    assert result.status == "failed"
    assert "vibecomfy_available" in result.failed_checks


def test_worker_preflight_requires_sageattention_for_sage_profile(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("VIBECOMFY_PATH", str(tmp_path / "vibecomfy"))
    monkeypatch.setenv("REIGH_VIBECOMFY_ATTENTION_PROFILE", "sage")

    def _fake_find_spec(module_name):
        if module_name in {"torch", "dotenv", "fastapi", "vibecomfy"}:
            return SimpleNamespace(origin=f"/fake/{module_name}.py")
        return None

    monkeypatch.setattr("source.runtime.worker.preflight.importlib.util.find_spec", _fake_find_spec)

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp,
        main_output_dir=tmp_path / "outputs",
        backend="vibecomfy",
    )

    assert result.status == "failed"
    assert "import:sageattention" in result.failed_checks


def test_worker_preflight_accepts_verified_sageattention_profile(tmp_path, monkeypatch):
    repo_root, wan2gp = _make_worker_repo(tmp_path)
    probes, runtime_binding = _configure_verified_facts(tmp_path, monkeypatch, repo_root)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "uv-cache"))
    monkeypatch.setenv("VIBECOMFY_PATH", str(tmp_path / "vibecomfy"))
    monkeypatch.setenv("REIGH_VIBECOMFY_ATTENTION_PROFILE", "sage")

    monkeypatch.setattr(
        "source.runtime.worker.preflight.importlib.util.find_spec",
        lambda name: SimpleNamespace(origin=f"/fake/{name}.py"),
    )

    result = run_worker_preflight(
        repo_root=repo_root,
        wan2gp_path=wan2gp,
        main_output_dir=tmp_path / "outputs",
        backend="vibecomfy",
        probes=probes,
        runtime_binding=runtime_binding,
    )

    assert result.status == PREFLIGHT_STATUS_PASSED
    assert "import:sageattention" not in result.failed_checks


class _FakeQuery:
    def __init__(self, client):
        self.client = client

    def select(self, *_args, **_kwargs):
        return self

    def update(self, payload):
        self.client.updated_payload = payload
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.client.updated_payload is None:
            return SimpleNamespace(data=[{"metadata": {"existing": "kept"}}])
        return SimpleNamespace(data=[self.client.updated_payload])


class _FakeSupabase:
    def __init__(self):
        self.updated_payload = None

    def table(self, table_name):
        assert table_name == "workers"
        return _FakeQuery(self)


def test_publish_preflight_metadata_merges_existing_metadata_and_ready_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("REIGH_PREFLIGHT_STATE_DIR", str(tmp_path))
    client = _FakeSupabase()
    result = WorkerPreflightResult(
        status="passed",
        checks=[PreflightCheck("wgp_import", True, "ok")],
        started_at=1.0,
        completed_at=2.0,
        verified_facts=VerifiedFacts(
            exact={
                "interpreter": "python",
                "runtime_lock": "runtime",
                "engine_lock": "engine",
                "model_digest": "model",
                "custom_node_digest": "nodes",
                "driver": "driver",
                "root": "root",
                "port": 8765,
            },
            minimum={"vram_bytes": 1, "scratch_bytes": 1},
        ),
    )

    assert publish_preflight_metadata(
        supabase_client=client,
        worker_id="worker-1",
        result=result,
        ready_for_tasks=True,
    )

    metadata = client.updated_payload["metadata"]
    assert metadata["existing"] == "kept"
    assert metadata["preflight_status"] == "passed"
    assert metadata["preflight_phase"] == "passed"
    assert metadata["ready_for_tasks"] is True
    assert metadata["verified_facts_complete"] is True
    assert json.loads(preflight_state_path("worker-1").read_text(encoding="utf-8"))["preflight_status"] == "passed"


def test_guardian_heartbeat_includes_preflight_status_log(tmp_path, monkeypatch):
    monkeypatch.setenv("REIGH_PREFLIGHT_STATE_DIR", str(tmp_path))
    write_preflight_state(
        "worker-1",
        {
            "preflight_status": "failed",
            "preflight_ok": False,
            "preflight_failed_checks": ["wgp_entrypoint"],
        },
    )
    captured = {}

    def _fake_run_subprocess(args, **_kwargs):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout=b'{"success": true}')

    monkeypatch.setattr(guardian, "run_subprocess", _fake_run_subprocess)

    assert guardian.send_heartbeat_with_logs(
        worker_id="worker-1",
        vram_total=1024,
        vram_used=512,
        logs=[],
        config={"db_url": "https://example.test", "api_key": "key"},
    )
    payload = json.loads(captured["args"][captured["args"].index("-d") + 1])
    preflight_logs = [log for log in payload["logs_param"] if log["message"] == "worker_preflight_status"]
    assert preflight_logs[-1]["metadata"]["preflight_status"] == "failed"
    health_logs = [log for log in payload["logs_param"] if log["message"] == "worker_health_labels"]
    assert health_logs[-1]["metadata"]["preflight"]["status"] == "failed"


def test_guardian_heartbeat_exposes_queryable_safe_telemetry_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("REIGH_PREFLIGHT_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("REIGH_WARM_CACHE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("REIGH_RESOURCE_PRESSURE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("REIGH_ROUTE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("REIGH_BACKEND", "vibecomfy")
    monkeypatch.setenv("REIGH_WORKER_PROFILE", "3")
    monkeypatch.setenv("REIGH_SELECTOR_NAMESPACE", "canary")
    monkeypatch.setenv("REIGH_SELECTOR_VERSION", "42")
    monkeypatch.setenv("REIGH_WORKER_RUN_ID", "run-abc")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "must-not-leak")
    write_preflight_state(
        "worker-telemetry",
        {
            "preflight_status": "passed",
            "preflight_ok": True,
            "preflight_failed_checks": [],
        },
    )
    write_worker_route_state(
        "worker-telemetry",
        {
            "task_id": "11111111-1111-1111-1111-111111111111",
            "task_type": "z_image_turbo",
            "params": {
                "route_contract": {
                    "selector_namespace": "canary",
                    "selector_version": 42,
                    "route_key": "z_image_turbo",
                    "selected_backend": "vibecomfy",
                    "selected_profile": "3",
                    "selected_template_id": "image/z_image",
                    "route_run_id": "run-abc",
                    "worker_contract_version": 1,
                }
            },
        },
    )
    write_resource_pressure_state(
        "worker-telemetry",
        ResourcePressureResult(
            status="near_full",
            action="claim_suppressed",
            allow_work=False,
            quota_alert=True,
            required_free_bytes=1024,
            recovered_bytes=0,
            volumes=(),
            cleanup={"lora": {}, "artifacts": {}},
            reason="disk_pressure_unrecoverable",
        ),
    )
    captured = {}

    def _fake_run_subprocess(args, **_kwargs):
        captured["args"] = args
        return SimpleNamespace(returncode=0, stdout=b'{"success": true}')

    monkeypatch.setattr(guardian, "run_subprocess", _fake_run_subprocess)

    assert guardian.send_heartbeat_with_logs(
        worker_id="worker-telemetry",
        vram_total=1024,
        vram_used=512,
        logs=[{"level": "info", "message": "task log", "metadata": {"api_token": "secret"}}],
        config={"db_url": "https://example.test", "api_key": "key"},
    )
    payload = json.loads(captured["args"][captured["args"].index("-d") + 1])
    health_logs = [log for log in payload["logs_param"] if log["message"] == "worker_health_labels"]
    metadata = health_logs[-1]["metadata"]

    assert metadata["backend"] == "vibecomfy"
    assert metadata["profile"] == "3"
    assert metadata["route_key"] == "z_image_turbo"
    assert metadata["template_id"] == "image/z_image"
    assert metadata["run_id"] == "run-abc"
    assert metadata["selector_namespace"] == "canary"
    assert metadata["selector_version"] == "42"
    assert metadata["preflight_status"] == "passed"
    assert metadata["disk_status"] in {"ok", "near_full"}
    assert metadata["resource_pressure_status"] == "near_full"
    assert metadata["quota_alert"] is True
    assert metadata["route"]["current_task_type"] == "z_image_turbo"
    assert "must-not-leak" not in json.dumps(payload)
