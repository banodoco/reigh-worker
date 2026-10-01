from __future__ import annotations

import copy
import json
import os
import signal
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from source.runtime import supervisor
from source.runtime.worker import preflight


def test_private_control_canonical_json_is_utf8_for_registration_and_acks() -> None:
    value = {"actor": "astrid-pack-host", "label": "München/東京"}
    encoded = b'{"actor":"astrid-pack-host","label":"M\xc3\xbcnchen/\xe6\x9d\xb1\xe4\xba\xac"}'
    assert supervisor._canonical_json(value) == encoded
    expected = "sha256:" + __import__("hashlib").sha256(encoded).hexdigest()
    assert supervisor._sha256_json(value) == expected

    runtime_to_worker = supervisor.LocalWorkerPreparerAdapter._ack_payload(
        {"version": supervisor.CONTROL_VERSION, "status": "prepared", "note": "café"}
    )
    assert runtime_to_worker["ack_sha256"] == supervisor._sha256_json(
        {name: item for name, item in runtime_to_worker.items() if name != "ack_sha256"}
    )
    worker_to_host = {
        "version": supervisor.HOST_CONTROL_VERSION,
        "command": "pause_prepare_ack",
        "note": "再開",
    }
    assert supervisor._sha256_json(worker_to_host) == (
        "sha256:" + __import__("hashlib").sha256(supervisor._canonical_json(worker_to_host)).hexdigest()
    )

    runtime = {
        "endpoint": "http://127.0.0.1:47111",
        "protocol": "workspace.v1",
        "schema_digest": "sha256:" + "1" * 64,
        "runtime_epoch": 1,
        "runtime_instance_id": "runtime-a",
        "runtime_session_id": "session-a",
    }
    registered = supervisor._registered_state(
        _registered(runtime, label="München/東京")
    )
    executor_body = registered["registration_bodies"]["/v1/executors"][0]
    raw = supervisor._canonical_json(executor_body)
    assert b"M\xc3\xbcnchen/\xe6\x9d\xb1\xe4\xba\xac" in raw
    assert b"\\u" not in raw
    executor_allowlist = next(
        item
        for item in registered["registration_allowlist"]
        if item["path"] == "/v1/executors"
    )
    assert executor_allowlist["body_sha256"] == [
        "sha256:" + __import__("hashlib").sha256(raw).hexdigest()
    ]


@pytest.mark.parametrize(
    ("expected", "observed"),
    [
        (
            "ps-lstart:Thu Oct  1 09:08:07 2026",
            "ps-lstart:Thu Oct 1 09:08:07 2026",
        ),
        ("proc-start-ticks:123456", "proc-start-ticks:123456"),
    ],
)
def test_process_birth_identity_comparison_accepts_padding_only_or_exact(
    expected, observed
):
    assert supervisor._process_birth_identities_match(expected, observed)


@pytest.mark.parametrize(
    ("expected", "observed"),
    [
        (
            "ps-lstart:Thu Oct  1 09:08:07 2026",
            "ps-lstart:Thu Oct 1 09:08:08 2026",
        ),
        ("ps-lstart:Thu Oct  1 09:08 2026", "ps-lstart:Thu Oct 1 09:08 2026"),
        ("", ""),
        ("proc-start-ticks:123456", "proc-start-ticks:123457"),
        ("proc-start-ticks:123456", "other-format:123456"),
    ],
)
def test_process_birth_identity_comparison_rejects_changed_or_invalid_values(
    expected, observed
):
    assert not supervisor._process_birth_identities_match(expected, observed)


@pytest.mark.parametrize("ack_pid", [6201, True])
def test_host_ack_binding_requires_exact_pid(ack_pid):
    request = {
        "command": "pause_prepare",
        "handoff_id": "handoff-1",
        "nonce_digest": "sha256:" + "1" * 64,
    }
    response = {
        "version": supervisor.HOST_CONTROL_VERSION,
        "command": "pause_prepare_ack",
        "handoff_id": request["handoff_id"],
        "nonce_digest": request["nonce_digest"],
        "status": "paused",
        "host": {
            "pid": ack_pid,
            "birth_id": "ps-lstart:Thu Oct 1 09:08:07 2026",
        },
        "phase": "PAUSED",
    }
    response["ack_sha256"] = supervisor._sha256_json(response)
    handle = SimpleNamespace(
        host=SimpleNamespace(pid=6200),
        host_birth_id="ps-lstart:Thu Oct  1 09:08:07 2026",
    )

    with pytest.raises(supervisor._HandoffAuthorizedFailure, match="binding"):
        supervisor.LocalWorkerPreparerAdapter._verify_host_ack(
            SimpleNamespace(),
            handle,
            request,
            response,
            statuses=frozenset({"paused"}),
        )


def test_cleanup_argv_digest_preserves_argument_boundaries() -> None:
    assert supervisor._argv_digest([b"a b", b"c"]) != supervisor._argv_digest(
        [b"a", b"b c"]
    )


def test_supervisor_module_entrypoint_serves_private_control_channel() -> None:
    runtime, worker = socket.socketpair()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    repository = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "source.runtime.supervisor",
            "--prepared-control-fd",
            str(worker.fileno()),
        ],
        cwd=repository,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        pass_fds=(worker.fileno(),),
    )
    worker.close()
    runtime.settimeout(5)
    try:
        supervisor._send_private_frame(
            runtime,
            {"version": supervisor.CONTROL_VERSION, "command": "entrypoint-probe"},
        )
        response = supervisor._receive_private_frame(runtime)
        assert response == {
            "version": supervisor.CONTROL_VERSION,
            "status": "error",
            "error": "prepared Worker rejected the handoff",
            "error_code": "worker_configuration",
            "error_stage": "entrypoint-probe",
        }
    finally:
        runtime.close()
    assert process.wait(timeout=5) == 78
    assert process.stdout is not None
    assert process.stderr is not None
    assert process.stdout.read() == b""
    assert process.stderr.read() == b""


def test_supervisor_module_entrypoint_rejects_unsupported_arguments() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-m", "source.runtime.supervisor", "--unsupported"],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 78
    assert completed.stdout == ""
    assert completed.stderr == (
        "Worker launcher configuration error: unsupported private arguments\n"
    )


class FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def _config(tmp_path: Path) -> supervisor.HostLaunchConfig:
    source = tmp_path / "Astrid"
    pack = source / "astrid" / "packs"
    pack.mkdir(parents=True)
    support = tmp_path / "support"
    credentials = support / "credentials"
    credentials.mkdir(parents=True)
    credential = credentials / "astrid-pack-host.token"
    credential.write_text("disabled-token", encoding="utf-8")
    credential.chmod(0o600)
    manifest = support / "boot.json"
    manifest.write_text("{}", encoding="utf-8")
    return supervisor.HostLaunchConfig(
        host_python=Path(sys.executable).resolve(),
        source_checkout=source.resolve(),
        pack_root=pack.resolve(),
        runtime_endpoint="http://127.0.0.1:9181",
        credential_file=credential.resolve(),
        support_root=support.resolve(),
        runtime_instance_id="runtime-1",
        ready_file=(support / "ready.json").resolve(),
        state_file=(support / "state.json").resolve(),
        boot_manifest_path=manifest.resolve(),
        boot_manifest_hash="sha256:" + "b" * 64,
    )


def _profile(config: supervisor.HostLaunchConfig):
    return SimpleNamespace(
        profile_id="astrid",
        workspace_uuid="realm-1",
        realm_root=config.support_root.parent / "realm",
        support_root=config.support_root,
        machine_id="machine-1",
        worker_executable=Path(sys.executable).resolve(),
        host_executable=config.host_python,
        engine_executable=Path(sys.executable).resolve(),
        engine_listener_executable=Path(sys.executable).resolve(),
        worker_artifact_digest="sha256:" + "1" * 64,
        host_artifact_digest="sha256:" + "2" * 64,
        engine_artifact_digest="sha256:" + "3" * 64,
        engine_listener_artifact_digest="sha256:" + "4" * 64,
        session_config_digest="sha256:" + "c" * 64,
        profile_revision="profile-1",
        profile_digest="sha256:" + "5" * 64,
        release_digest="sha256:" + "6" * 64,
    )


def _install_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _config(tmp_path)
    profile = _profile(config)
    engine_process = FakeProcess(4300)
    engine = supervisor._OwnedVibeComfySession(
        root=tmp_path / "engine",
        process=engine_process,
        daemon_pid=4300,
        comfy_pid=4301,
        comfy_process_birth_id="birth-4301",
        server_url="http://127.0.0.1:8188",
    )
    engine_report = {
        "pid": 4300,
        "process_birth_id": "birth-4300",
        "comfy_pid": 4301,
        "comfy_process_birth_id": "birth-4301",
        "server_url": "http://127.0.0.1:8188",
        "config_digest": profile.session_config_digest,
    }
    stopped = []
    terminated = []
    host_threads: list[threading.Thread] = []
    next_pid = iter((4200, 4201, 4202))

    monkeypatch.setattr(
        supervisor.LocalWorkerPreparerAdapter,
        "_validate_profile",
        lambda self, value: None,
    )
    monkeypatch.setattr(
        supervisor,
        "_start_owned_vibecomfy_session",
        lambda *_args, **_kwargs: (dict(engine_report), engine),
    )
    monkeypatch.setattr(
        supervisor,
        "_read_owned_vibecomfy_session",
        lambda *_args, **_kwargs: dict(engine_report),
    )
    monkeypatch.setattr(supervisor, "_stop_owned_vibecomfy_session", stopped.append)
    monkeypatch.setattr(preflight, "_process_birth_identity", lambda pid: f"birth-{pid}")
    monkeypatch.setattr(supervisor.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(supervisor.os, "getsid", lambda pid: pid)
    def fake_cleanup_identity(pid):
        return supervisor._CleanupIdentity(
            pid=pid,
            birth_id=preflight._process_birth_identity(pid),
            uid=os.getuid(),
            parent_pid=os.getpid(),
            process_group=pid,
            session_id=pid,
            executable=Path(sys.executable).resolve(),
            artifact_digest="sha256:" + "a" * 64,
            argv_digest=supervisor._argv_digest(
                [b"fake-process", str(pid).encode("ascii")]
            ),
        )
    monkeypatch.setattr(supervisor, "_capture_cleanup_identity", fake_cleanup_identity)
    monkeypatch.setattr(
        supervisor,
        "_signal_owned_group",
        lambda process, signum: terminated.append((process.pid, signum)),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_and_wait",
        lambda process, pgid: (terminated.append((process.pid, pgid)), setattr(process, "returncode", 0)),
    )

    def fake_popen(_argv, **kwargs):
        assert kwargs.pop("custody_role") == "generic_pack_host"
        process = FakeProcess(next(next_pid))
        assert len(kwargs["pass_fds"]) == 2
        assert kwargs["pass_fds"][0] != kwargs["pass_fds"][1]
        assert _argv[_argv.index("--activation-fd") + 1] == str(kwargs["pass_fds"][0])
        assert _argv[_argv.index("--host-control-fd") + 1] == str(kwargs["pass_fds"][1])
        inherited = os.dup(kwargs["pass_fds"][0])

        def host_side() -> None:
            channel = socket.socket(fileno=inherited)
            frame = channel.makefile("rb").readline()
            if not frame:
                channel.close()
                return
            grant = json.loads(frame)
            ack = {
                "version": supervisor.ACTIVATION_ACCEPTED_VERSION,
                "operation_id": grant["operation_id"],
                "channel_id": grant["channel_id"],
                "executor_incarnation": grant["executor_incarnation"],
                "evidence_digest": grant["evidence_digest"],
                "host": grant["host"],
            }
            channel.sendall(json.dumps(ack, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            channel.close()

        thread = threading.Thread(target=host_side, daemon=True)
        thread.start()
        host_threads.append(thread)
        return process

    monkeypatch.setattr(supervisor, "_custodied_popen", fake_popen)
    adapter = supervisor.LocalWorkerPreparerAdapter(config, environ={})
    return adapter, profile, engine_report, stopped, terminated, host_threads


def _grant(handle: supervisor.PreparedHostHandle, config: supervisor.HostLaunchConfig):
    return {
        "version": supervisor.ACTIVATION_VERSION,
        "operation_id": handle.operation_id,
        "channel_id": handle.channel_id,
        "credential_file": str(config.credential_file),
        "executor_incarnation": "incarnation-1",
        "evidence_digest": "sha256:" + "d" * 64,
    }


def _receipt(adapter, handle, grant):
    report = adapter.report(handle)
    return {
        "version": supervisor.RECEIPT_VERSION,
        **{name: {**report["processes"][name]} for name in report["processes"]},
        "session_config_digest": report["session_config_digest"],
        "evidence_digest": grant["evidence_digest"],
        "executor_incarnation": grant["executor_incarnation"],
    }


def test_runtime_process_preparer_round_trips_parked_worker_control(tmp_path, monkeypatch):
    config = _config(tmp_path)
    profile = _profile(config)
    report = {
        "version": supervisor.PREPARATION_VERSION,
        "operation_id": "operation-1",
        "channel_id": "channel-1",
    }
    events = []

    monkeypatch.setattr(preflight, "_process_birth_identity", lambda pid: f"birth-{pid}")

    def fake_popen(_argv, **kwargs):
        assert kwargs.pop("custody_role") == "worker"
        process = FakeProcess(5100)
        inherited = os.dup(kwargs["pass_fds"][0])

        def worker_side() -> None:
            channel = socket.socket(fileno=inherited)
            try:
                while True:
                    request = supervisor._receive_private_frame(channel)
                    command = request["command"]
                    events.append(command)
                    if command == "prepare":
                        response = {
                            "version": supervisor.CONTROL_VERSION,
                            "status": "ok",
                            "report": report,
                        }
                    elif command == "report":
                        response = {
                            "version": supervisor.CONTROL_VERSION,
                            "status": "ok",
                            "report": report,
                        }
                    elif command == "reconnect":
                        response = {
                            "version": supervisor.CONTROL_VERSION,
                            "status": "ok",
                            "reconnected": True,
                        }
                    elif command == "abort":
                        response = {
                            "version": supervisor.CONTROL_VERSION,
                            "status": "ok",
                        }
                        supervisor._send_private_frame(channel, response)
                        return
                    else:
                        response = {
                            "version": supervisor.CONTROL_VERSION,
                            "status": "ok",
                        }
                    supervisor._send_private_frame(channel, response)
            finally:
                channel.close()

        threading.Thread(target=worker_side, daemon=True).start()
        return process

    monkeypatch.setattr(supervisor, "_custodied_popen", fake_popen)
    preparer = supervisor.LocalWorkerProcessPreparer(config, environ={}, timeout_seconds=2)
    handle = preparer.prepare(profile, operation_id="operation-1", channel_id="channel-1")
    assert handle.report_value == report
    assert preparer.report(handle) == report
    preparer.activate(handle, {"executor_incarnation": "incarnation-1"})
    assert preparer.reconnect({"executor_incarnation": "incarnation-1"}) is handle
    preparer.abort(handle)

    assert events == ["prepare", "report", "activate", "reconnect", "abort"]
    assert handle.closed is True


def test_prepare_reports_distinct_engine_identities_and_activates_same_host(tmp_path, monkeypatch):
    adapter, profile, _engine, _stopped, _terminated, threads = _install_fakes(tmp_path, monkeypatch)
    handle = adapter.prepare(profile, operation_id="operation-1", channel_id="channel-1")
    report = adapter.report(handle)

    assert not adapter.config.ready_file.exists()
    assert report["processes"]["host"] == {"pid": 4200, "birth_id": "birth-4200"}
    assert report["processes"]["engine"] == {"pid": 4300, "birth_id": "birth-4300"}
    assert report["processes"]["engine_listener"] == {"pid": 4301, "birth_id": "birth-4301"}
    grant = _grant(handle, adapter.config)
    adapter.activate(handle, grant)
    assert handle.activated is True
    assert adapter.reconnect(_receipt(adapter, handle, grant)) is handle
    for thread in threads:
        thread.join(timeout=2)


@pytest.mark.parametrize("field", ["operation_id", "channel_id", "executor_incarnation", "evidence_digest"])
def test_wrong_or_stale_activation_is_rejected_while_host_remains_parked(tmp_path, monkeypatch, field):
    adapter, profile, _engine, _stopped, _terminated, _threads = _install_fakes(tmp_path, monkeypatch)
    handle = adapter.prepare(profile, operation_id="operation-1", channel_id="channel-1")
    grant = _grant(handle, adapter.config)
    grant[field] = "" if field in {"executor_incarnation", "evidence_digest"} else "stale"
    with pytest.raises(supervisor.LauncherConfigurationError):
        adapter.activate(handle, grant)
    assert handle.activated is False
    assert not adapter.config.ready_file.exists()


def test_identity_change_aborts_without_signalling_replacement(tmp_path, monkeypatch):
    adapter, profile, _engine, _stopped, terminated, _threads = _install_fakes(tmp_path, monkeypatch)
    handle = adapter.prepare(profile, operation_id="operation-1", channel_id="channel-1")
    original = preflight._process_birth_identity
    monkeypatch.setattr(preflight, "_process_birth_identity", lambda pid: "replacement" if pid == handle.host.pid else original(pid))

    with pytest.raises(supervisor.LauncherConfigurationError, match="birth identity changed"):
        adapter.abort(handle)
    assert terminated == []


def test_parked_host_crash_is_reported_and_abort_cleans_engine(tmp_path, monkeypatch):
    adapter, profile, _engine, stopped, _terminated, _threads = _install_fakes(tmp_path, monkeypatch)
    handle = adapter.prepare(profile, operation_id="operation-1", channel_id="channel-1")
    handle.host.returncode = 9
    with pytest.raises(supervisor.LauncherConfigurationError, match="not alive"):
        adapter.report(handle)
    adapter.abort(handle)
    assert stopped
    assert adapter.reconnect({}) is None


def test_reconnect_rejects_stale_incarnation_and_replacement_cleans_old_host(tmp_path, monkeypatch):
    adapter, profile, _engine, stopped, terminated, threads = _install_fakes(tmp_path, monkeypatch)
    first = adapter.prepare(profile, operation_id="operation-1", channel_id="channel-1")
    grant = _grant(first, adapter.config)
    adapter.activate(first, grant)
    stale = _receipt(adapter, first, grant)
    stale["executor_incarnation"] = "stale"
    assert adapter.reconnect(stale) is None

    second = adapter.prepare(profile, operation_id="operation-2", channel_id="channel-2")
    assert first.closed is True
    assert second.host.pid == 4201
    assert terminated == [(4200, signal.SIGTERM)]
    assert stopped
    adapter.abort(second)
    for thread in threads:
        thread.join(timeout=2)


def test_reconnect_accepts_v3_and_rejects_older_receipt(tmp_path, monkeypatch):
    adapter, profile, _engine, _stopped, _terminated, threads = _install_fakes(tmp_path, monkeypatch)
    handle = adapter.prepare(profile, operation_id="operation-1", channel_id="channel-1")
    grant = _grant(handle, adapter.config)
    adapter.activate(handle, grant)
    current = _receipt(adapter, handle, grant)
    assert current["version"] == "runtime.local-worker-receipt/v3"
    assert adapter.reconnect(current) is handle
    stale = dict(current)
    stale["version"] = "runtime.local-worker-receipt/v2"
    assert adapter.reconnect(stale) is None
    adapter.abort(handle)
    for thread in threads:
        thread.join(timeout=2)


def test_inherited_worker_control_dispatches_full_preparer_lifecycle(tmp_path, monkeypatch):
    config = _config(tmp_path)
    profile = _profile(config)
    events = []
    owned_handle = object()
    report = {
        "version": supervisor.PREPARATION_VERSION,
        "operation_id": "operation-1",
        "channel_id": "channel-1",
    }

    class FakeAdapter:
        def __init__(self, received_config, *, environ):
            assert received_config == config
            assert environ is os.environ

        def prepare(self, received_profile, *, operation_id, channel_id):
            assert received_profile.workspace_uuid == profile.workspace_uuid
            events.append(("prepare", operation_id, channel_id))
            return owned_handle

        def report(self, handle):
            assert handle is owned_handle
            events.append(("report",))
            return report

        def activate(self, handle, grant):
            assert handle is owned_handle
            events.append(("activate", grant["executor_incarnation"]))

        def reconnect(self, receipt):
            events.append(("reconnect", receipt["executor_incarnation"]))
            return owned_handle

        def abort(self, handle):
            assert handle is owned_handle
            events.append(("abort",))

    monkeypatch.setattr(supervisor, "LocalWorkerPreparerAdapter", FakeAdapter)
    runtime, worker = socket.socketpair()
    outcome = []
    thread = threading.Thread(
        target=lambda: outcome.append(supervisor._serve_prepared_worker(worker.detach()))
    )
    thread.start()

    supervisor._send_private_frame(
        runtime,
        {
            "version": supervisor.CONTROL_VERSION,
            "command": "prepare",
            "operation_id": "operation-1",
            "channel_id": "channel-1",
            "profile": supervisor._private_profile_payload(profile),
            "config": supervisor._private_config_payload(config),
        },
    )
    assert supervisor._receive_private_frame(runtime)["report"] == report
    supervisor._send_private_frame(
        runtime,
        {"version": supervisor.CONTROL_VERSION, "command": "report"},
    )
    assert supervisor._receive_private_frame(runtime)["status"] == "ok"
    supervisor._send_private_frame(
        runtime,
        {
            "version": supervisor.CONTROL_VERSION,
            "command": "activate",
            "grant": {"executor_incarnation": "incarnation-1"},
        },
    )
    assert supervisor._receive_private_frame(runtime)["status"] == "ok"
    supervisor._send_private_frame(
        runtime,
        {
            "version": supervisor.CONTROL_VERSION,
            "command": "reconnect",
            "receipt": {"executor_incarnation": "incarnation-1"},
        },
    )
    assert supervisor._receive_private_frame(runtime)["reconnected"] is True
    supervisor._send_private_frame(
        runtime,
        {"version": supervisor.CONTROL_VERSION, "command": "abort"},
    )
    assert supervisor._receive_private_frame(runtime)["status"] == "ok"
    thread.join(timeout=2)

    assert outcome == [0]
    assert events == [
        ("prepare", "operation-1", "channel-1"),
        ("report",),
        ("report",),
        ("activate", "incarnation-1"),
        ("reconnect", "incarnation-1"),
        ("abort",),
    ]
    runtime.close()


def _runtime_identity(suffix: str) -> dict[str, object]:
    return {
        "endpoint": "http://127.0.0.1:9181",
        "protocol": "workspace.v1",
        "schema_digest": "sha256:" + "a" * 64,
        "runtime_epoch": 1 if suffix == "old" else 2,
        "runtime_instance_id": f"runtime-{suffix}",
        "runtime_session_id": f"session-{suffix}",
    }


def _generation() -> dict[str, str]:
    return {
        "generation": "generation-1",
        "token_sha256": "sha256:" + "1" * 64,
        "metadata_sha256": "sha256:" + "2" * 64,
        "commit_sha256": "sha256:" + "3" * 64,
    }


def _registered(
    runtime: dict[str, object], *, label: str | None = None
) -> dict[str, object]:
    body = {
        "executor_id": supervisor.GENERIC_HOST_EXECUTOR_ID,
        "runtime_epoch": runtime["runtime_epoch"],
    }
    if label is not None:
        body["label"] = label
    admission_bodies = {
        "/v1/capabilities": [],
        "/v1/executors": [body],
    }
    value = {
        "executor_id": supervisor.GENERIC_HOST_EXECUTOR_ID,
        "source_epoch": "source-1",
        "runtime": dict(runtime),
        "capabilities": [
            {
                "capability_id": "echo",
                "capability_digest": "sha256:" + "4" * 64,
                "source_digest": "5" * 64,
                "dependency_digest": "6" * 64,
                "ready": True,
                "preflight_digest": "sha256:" + "7" * 64,
            }
        ],
        "registration_actor": supervisor.GENERIC_HOST_EXECUTOR_ID,
        "registration_bodies": admission_bodies,
        "registration_allowlist": [
            {
                "method": "POST",
                "path": path,
                "actor": supervisor.GENERIC_HOST_EXECUTOR_ID,
                "body_sha256": sorted(supervisor._sha256_json(item) for item in items),
            }
            for path, items in sorted(admission_bodies.items())
        ],
    }
    return value


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("source_digest", "sha256:" + "5" * 64),
        ("source_digest", "5" * 63),
        ("dependency_digest", "sha256:" + "6" * 64),
        ("dependency_digest", "G" * 64),
    ],
)
def test_registered_state_requires_generic_host_canonical_digest_shape(
    field, invalid
):
    value = _registered(_runtime_identity("old"))
    value["capabilities"][0][field] = invalid
    with pytest.raises(
        supervisor.LauncherConfigurationError,
        match=rf"registered capability {field} is invalid",
    ):
        supervisor._registered_state(value)


def test_registered_state_accepts_generic_host_canonical_digest_shape():
    value = _registered(_runtime_identity("old"))
    normalized = supervisor._registered_state(value)
    assert normalized["capabilities"][0]["source_digest"] == "5" * 64
    assert normalized["capabilities"][0]["dependency_digest"] == "6" * 64


def _handoff_fixture(tmp_path, monkeypatch, statuses):
    config = _config(tmp_path)
    adapter = supervisor.LocalWorkerPreparerAdapter(config, environ={})
    worker_control, host_control = socket.socketpair()
    activation, host_activation = socket.socketpair()
    host_activation.close()
    host = FakeProcess(6200)
    engine = supervisor._OwnedVibeComfySession(
        root=tmp_path / "engine",
        process=FakeProcess(6300),
        daemon_pid=6300,
        comfy_pid=6301,
        comfy_process_birth_id="birth-6301",
        server_url="http://127.0.0.1:8188",
    )
    handle = supervisor.PreparedHostHandle(
        profile=object(),
        operation_id="operation-1",
        channel_id="channel-1",
        host=host,
        host_birth_id="birth-6200",
        host_cleanup_identity=supervisor._CleanupIdentity(
            pid=6200,
            birth_id="birth-6200",
            uid=os.getuid(),
            parent_pid=os.getpid(),
            process_group=6200,
            session_id=6200,
            executable=Path(sys.executable).resolve(),
            artifact_digest="sha256:" + "a" * 64,
            argv_digest=supervisor._argv_digest([b"fake-host", b"6200"]),
        ),
        activation=activation,
        host_control=worker_control,
        engine=engine,
        engine_report={
            "pid": 6300,
            "process_birth_id": "birth-6300",
            "comfy_pid": 6301,
            "comfy_process_birth_id": "birth-6301",
            "server_url": "http://127.0.0.1:8188",
            "config_digest": "sha256:" + "c" * 64,
        },
        activated=True,
        activation_grant={
            "version": supervisor.ACTIVATION_VERSION,
            "operation_id": "operation-1",
            "channel_id": "channel-1",
            "credential_file": str(config.credential_file),
            "executor_incarnation": "incarnation-1",
            "evidence_digest": "sha256:" + "d" * 64,
        },
    )
    adapter._active = handle
    stopped = []
    terminated = []
    monkeypatch.setattr(adapter, "_assert_live", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(preflight, "_process_birth_identity", lambda pid: f"birth-{pid}")
    monkeypatch.setattr(supervisor, "_stop_owned_vibecomfy_session", stopped.append)
    monkeypatch.setattr(supervisor, "_verify_cleanup_identity", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        supervisor,
        "_signal_owned_group",
        lambda process, signum: terminated.append((process.pid, signum)),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_and_wait",
        lambda process, pgid: (terminated.append((process.pid, pgid)), setattr(process, "returncode", 0)),
    )
    old_runtime = _runtime_identity("old")
    new_runtime = _runtime_identity("new")
    unicode_label = "München/東京"
    registered = _registered(old_runtime, label=unicode_label)
    observed = []

    def host_side():
        try:
            for status in statuses:
                request = supervisor._receive_private_frame(host_control)
                observed.append(request)
                if status == "close":
                    return
                payload = {
                    "version": supervisor.HOST_CONTROL_VERSION,
                    "command": request["command"] + "_ack",
                    "handoff_id": request["handoff_id"],
                    "nonce_digest": request["nonce_digest"],
                    "status": status,
                    "host": {"pid": 6200, "birth_id": "birth-6200"},
                    "phase": {
                        "active_work": "ACTIVE",
                        "paused": "PAUSED",
                        "rebind_prepared": "REBIND_PREPARED",
                        "rebind_committed": "REBIND_COMMITTED",
                        "resume_prepared": "RESUME_PREPARED",
                        "resumed": "RESUMED",
                        "adopted": "ADOPTED",
                        "pause_cancelled": "ACTIVE",
                        "aborted": "ABORTED",
                    }[status],
                }
                if request["command"] in {"pause_prepare", "rebind_prepare", "rebind_commit"}:
                    payload["activation"] = {
                        **handle.activation_grant,
                        "host": {"pid": 6200, "birth_id": "birth-6200"},
                    }
                    payload["registered_state"] = _registered(
                        new_runtime if request["command"] in {"rebind_prepare", "rebind_commit"} else old_runtime,
                        label=unicode_label,
                    )
                if status == "active_work":
                    payload["inflight_claim_iterations"] = 1
                if request["command"] in {"rebind_prepare", "rebind_commit"}:
                    payload["credential_generation"] = _generation()
                if request["command"] == "rebind_commit":
                    payload["registration"] = {
                        "runtime_registration": {"registered": True},
                        "withdrawn_capabilities": [],
                    }
                payload["ack_sha256"] = supervisor._sha256_json(payload)
                supervisor._send_private_frame(host_control, payload)
        finally:
            host_control.close()

    thread = threading.Thread(target=host_side, daemon=True)
    thread.start()
    nonce = "9" * 64
    common = {
        "version": supervisor.CONTROL_VERSION,
        "handoff_id": "handoff-1",
        "nonce_digest": supervisor._nonce_digest(nonce),
        "deadline_monotonic": time.monotonic() + 10,
        "deadline_unix_ms": int(time.time() * 1000) + 10_000,
    }
    common["sealed_record_digest"] = _initial_sealed_record(common)[
        "sealed_record_digest"
    ]
    return (
        adapter, handle, thread, observed, stopped, terminated, common,
        nonce, old_runtime, new_runtime, registered,
    )


def _prepare_request(common, old_runtime):
    return {
        **common,
        "command": "handoff_prepare",
        "old_owner": {"pid": 4100, "birth_id": "birth-4100"},
        "sealed_record": _initial_sealed_record(common),
        "old_runtime": old_runtime,
        "receipt_evidence_digest": "sha256:" + "d" * 64,
        "credential_generation": _generation(),
    }


def _adopt_request(common, nonce, old_runtime, new_runtime, registered, credential_file):
    seal = _seal_request(common, registered)
    return {
        **common,
        "command": "handoff_adopt",
        "nonce": nonce,
        "request_id": "handoff-1",
        "old_owner": {"pid": 4100, "birth_id": "birth-4100"},
        "new_owner": {"pid": 5100, "birth_id": "birth-5100"},
        "export_sealed_digest": seal["export_sealed_digest"],
        "export_record_digest": seal["export_record_digest"],
        "adopter_record_digest": "sha256:" + "f" * 64,
        "old_runtime": old_runtime,
        "new_runtime": new_runtime,
        "endpoint": new_runtime["endpoint"],
        "credential_file": str(credential_file),
        "credential_generation": _generation(),
        "receipt_evidence_digest": "sha256:" + "d" * 64,
        "executor_incarnation": "incarnation-1",
        "registered_state": registered,
    }


def _initial_sealed_record(common, old_runtime=None):
    if old_runtime is None:
        old_runtime = _runtime_identity("old")
    value = {
        "version": supervisor.HANDOFF_RECORD_VERSION,
        "state": "OWNED",
        "handoff_id": common["handoff_id"],
        "realm_id": "realm-1",
        "deadline_monotonic": common["deadline_monotonic"],
        "deadline_unix_ms": common["deadline_unix_ms"],
        "nonce_digest": common["nonce_digest"],
        "sealed_record_digest": None,
        "old_owner": {
            "pid": 4100,
            "birth_id": "birth-4100",
            "runtime_instance_id": old_runtime["runtime_instance_id"],
            "runtime": dict(old_runtime),
        },
        "export": None,
        "export_sealed_digest": None,
        "adopter": None,
    }
    value["sealed_record_digest"] = supervisor._sha256_json(
        {
            key: item
            for key, item in value.items()
            if key != "sealed_record_digest"
        }
    )
    value["record_digest"] = supervisor._sha256_json(value)
    return value


def _reseal_initial_record(value):
    value.pop("record_digest", None)
    value["sealed_record_digest"] = supervisor._sha256_json(
        {
            key: item
            for key, item in value.items()
            if key != "sealed_record_digest"
        }
    )
    value["record_digest"] = supervisor._sha256_json(value)
    return value


def test_sealed_handoff_record_binds_full_owner_runtime_and_digests():
    old_runtime = _runtime_identity("old")
    common = {
        "handoff_id": "handoff-owner-binding",
        "nonce_digest": "sha256:" + "9" * 64,
        "deadline_monotonic": 1234.5,
        "deadline_unix_ms": 1_900_000_000_000,
    }
    record = _initial_sealed_record(common, old_runtime)
    common["sealed_record_digest"] = record["sealed_record_digest"]
    validation = {
        "handoff_id": common["handoff_id"],
        "nonce_digest": common["nonce_digest"],
        "sealed_record_digest": common["sealed_record_digest"],
        "old_owner": {"pid": 4100, "birth_id": "birth-4100"},
        "old_runtime": old_runtime,
        "deadline_monotonic": common["deadline_monotonic"],
        "deadline_unix_ms": common["deadline_unix_ms"],
    }

    assert supervisor._sealed_handoff_record(record, **validation) == record

    identity_mutations = {
        "owner pid": lambda value: value["old_owner"].update(pid=4101),
        "owner birth": lambda value: value["old_owner"].update(
            birth_id="birth-4101"
        ),
        "Runtime identity": lambda value: value["old_owner"]["runtime"].update(
            endpoint="http://127.0.0.1:9999"
        ),
        "Runtime instance": lambda value: value["old_owner"].update(
            runtime_instance_id="runtime-other"
        ),
    }
    for label, mutate in identity_mutations.items():
        changed = copy.deepcopy(record)
        mutate(changed)
        _reseal_initial_record(changed)
        changed_validation = {
            **validation,
            "sealed_record_digest": changed["sealed_record_digest"],
        }
        with pytest.raises(
            supervisor._HandoffRejected,
            match="sealed handoff owner does not match owner A",
        ):
            supervisor._sealed_handoff_record(changed, **changed_validation)

    changed_nonce = copy.deepcopy(record)
    changed_nonce["nonce_digest"] = "sha256:" + "8" * 64
    _reseal_initial_record(changed_nonce)
    with pytest.raises(supervisor._HandoffRejected, match="sealed handoff record"):
        supervisor._sealed_handoff_record(changed_nonce, **validation)

    changed_seal = copy.deepcopy(record)
    changed_seal["sealed_record_digest"] = "sha256:" + "7" * 64
    changed_seal["record_digest"] = supervisor._sha256_json(
        {key: item for key, item in changed_seal.items() if key != "record_digest"}
    )
    with pytest.raises(supervisor._HandoffRejected, match="sealed handoff record"):
        supervisor._sealed_handoff_record(changed_seal, **validation)

    changed_record = copy.deepcopy(record)
    changed_record["record_digest"] = "sha256:" + "6" * 64
    with pytest.raises(supervisor._HandoffRejected, match="record digest"):
        supervisor._sealed_handoff_record(changed_record, **validation)


def test_handoff_diagnostic_codes_are_bounded_and_secret_free():
    assert supervisor._bounded_handoff_error_code(
        supervisor._HandoffRejected(
            "private Worker sealed handoff owner does not match owner A"
        )
    ) == "sealed_owner_mismatch"
    assert supervisor._bounded_handoff_error_code(
        supervisor._HandoffAuthorizedFailure(
            "GenericPackHost old Runtime state changed"
        )
    ) == "host_runtime_changed"
    assert supervisor._bounded_handoff_error_code(
        supervisor._HandoffRejected("unrecognized detail: secret-material")
    ) == "handoff_rejected"


def _handoff_export(registered):
    identity = {
        "evidence_digest": "sha256:" + "d" * 64,
        "worker": {"pid": os.getpid(), "birth_id": f"birth-{os.getpid()}"},
        "host": {"pid": 6200, "birth_id": "birth-6200"},
        "engine": {"pid": 6300, "birth_id": "birth-6300"},
        "engine_listener": {"pid": 6301, "birth_id": "birth-6301"},
    }
    return {
        "receipt": {
            "version": supervisor.RECEIPT_VERSION,
            **identity,
            "executor_incarnation": "incarnation-1",
        },
        "identity": identity,
        "credential_generation": _generation(),
        "registered_state": registered,
    }


def _seal_request(common, registered):
    export = _handoff_export(registered)
    export_sealed_digest = supervisor._sha256_json(
        {
            "version": supervisor.HANDOFF_EXPORT_SEAL_VERSION,
            "sealed_record_digest": common["sealed_record_digest"],
            "nonce_digest": common["nonce_digest"],
            "export": export,
        }
    )
    bound_record = {
        key: item
        for key, item in _initial_sealed_record(common).items()
        if key != "record_digest"
    }
    bound_record["export"] = export
    bound_record["export_sealed_digest"] = export_sealed_digest
    return {
        "version": supervisor.CONTROL_VERSION,
        "command": "handoff_seal",
        "handoff_id": common["handoff_id"],
        "nonce": "9" * 64,
        "nonce_digest": common["nonce_digest"],
        "sealed_record_digest": common["sealed_record_digest"],
        "export_sealed_digest": export_sealed_digest,
        "export_record_digest": supervisor._sha256_json(bound_record),
        "export": export,
        "old_owner": {"pid": 4100, "birth_id": "birth-4100"},
    }


def test_runtime_verifies_unicode_worker_ack_over_private_channel(tmp_path, monkeypatch):
    config = _config(tmp_path)
    preparer = supervisor.LocalWorkerProcessPreparer(config, environ={})
    runtime_channel, worker_channel = socket.socketpair()
    worker = FakeProcess(7100)
    handle = supervisor.PreparedWorkerHandle(
        worker=worker,
        worker_birth_id="birth-7100",
        control=runtime_channel,
        report_value={},
        activated=True,
    )
    preparer._active = handle
    monkeypatch.setattr(preparer, "_birth", lambda pid: f"birth-{pid}")
    request = {
        "version": supervisor.CONTROL_VERSION,
        "command": "handoff_finalize",
        "handoff_id": "handoff-unicode",
    }

    def worker_side():
        received = supervisor._receive_private_frame(worker_channel)
        assert received == request
        response = supervisor.LocalWorkerPreparerAdapter._ack_payload(
            {
                "version": supervisor.CONTROL_VERSION,
                "command": "handoff_finalize_ack",
                "handoff_id": "handoff-unicode",
                "status": "finalized",
                "host_ack": {"registration_label": "München/東京"},
            }
        )
        supervisor._send_private_frame(worker_channel, response)
        worker_channel.close()

    thread = threading.Thread(target=worker_side)
    thread.start()
    response = preparer.handoff(handle, request)
    thread.join(timeout=2)
    runtime_channel.close()
    assert response["host_ack"]["registration_label"] == "München/東京"


def test_handoff_active_work_refusal_is_nonmutating(tmp_path, monkeypatch):
    fixture = _handoff_fixture(tmp_path, monkeypatch, ["active_work"])
    adapter, handle, thread, observed, *_rest = fixture
    common, _nonce, old_runtime, _new_runtime, _registered_value = fixture[6:]
    request = _prepare_request(common, old_runtime)
    assert "nonce" not in request
    response, should_exit = adapter.handoff_command(handle, request)
    thread.join(timeout=2)
    assert should_exit is False
    assert response["status"] == "active_work"
    assert response["worker_phase"] == "owned"
    assert "nonce" not in response
    assert adapter._handoff is None
    assert observed[0]["command"] == "pause_prepare"
    assert "nonce" not in observed[0]
    assert response["ack_sha256"] == supervisor._sha256_json(
        {key: value for key, value in response.items() if key != "ack_sha256"}
    )


def test_handoff_state_transitions_nonce_replay_lost_final_ack_and_hashes(
    tmp_path, monkeypatch
):
    fixture = _handoff_fixture(
        tmp_path,
        monkeypatch,
        [
            "paused", "rebind_prepared", "rebind_committed", "resume_prepared",
            "resumed", "adopted",
        ],
    )
    (
        adapter, handle, thread, observed, _stopped, _terminated, common,
        nonce, old_runtime, new_runtime, registered,
    ) = fixture
    tampered_record = _prepare_request(common, old_runtime)
    tampered_record["sealed_record"]["realm_id"] = "tampered-realm"
    with pytest.raises(supervisor._HandoffRejected, match="sealed handoff digest"):
        adapter.handoff_command(handle, tampered_record)
    premature_authority = {
        **_prepare_request(common, old_runtime),
        "nonce": nonce,
    }
    with pytest.raises(supervisor._HandoffRejected, match="invalid shape"):
        adapter.handoff_command(handle, premature_authority)
    raw_record = _prepare_request(common, old_runtime)
    raw_record["sealed_record"]["nested"] = {"nonce": nonce}
    with pytest.raises(supervisor._HandoffRejected, match="sealed handoff record"):
        adapter.handoff_command(handle, raw_record)
    assert adapter._handoff is None
    assert observed == []
    response, _ = adapter.handoff_command(handle, _prepare_request(common, old_runtime))
    assert response["worker_phase"] == "paused"
    assert "nonce" not in response
    assert "nonce" not in vars(adapter._handoff)
    assert adapter.handoff_command(handle, _prepare_request(common, old_runtime))[0] == response
    seal = _seal_request(common, registered)
    wrong_receipt = json.loads(json.dumps(seal))
    wrong_receipt["export"]["receipt"]["executor_incarnation"] = "wrong"
    with pytest.raises(supervisor._HandoffRejected, match="receipt"):
        adapter.handoff_command(handle, wrong_receipt)
    wrong_graph = json.loads(json.dumps(seal))
    wrong_graph["export"]["identity"]["host"]["pid"] = 6201
    wrong_graph["export"]["receipt"]["host"]["pid"] = 6201
    with pytest.raises(supervisor._HandoffRejected, match="graph"):
        adapter.handoff_command(handle, wrong_graph)
    tampered_seal = {**seal, "export_sealed_digest": "sha256:" + "0" * 64}
    with pytest.raises(supervisor._HandoffRejected, match="export seal"):
        adapter.handoff_command(handle, tampered_seal)
    assert adapter._handoff is not None
    assert adapter._handoff.phase == "paused"
    assert set(adapter._handoff.acknowledgements or {}) == {"handoff_prepare"}
    assert "nonce" not in vars(adapter._handoff)
    sealed, _ = adapter.handoff_command(handle, seal)
    assert sealed["status"] == "sealed"
    assert sealed["worker_phase"] == "export_sealed"
    assert "nonce" not in sealed
    assert "nonce" not in adapter._handoff.acknowledgements["handoff_seal"][1]
    sealed_state = copy.deepcopy(vars(adapter._handoff))
    with pytest.raises(supervisor._HandoffRejected, match="already consumed"):
        adapter.handoff_command(handle, seal)
    altered_seal = {**seal, "nonce": "8" * 64}
    with pytest.raises(supervisor._HandoffRejected, match="already consumed"):
        adapter.handoff_command(handle, altered_seal)
    assert vars(adapter._handoff) == sealed_state
    wrong_export_digest = _adopt_request(
        common, nonce, old_runtime, new_runtime, registered, adapter.config.credential_file
    )
    wrong_export_digest["export_record_digest"] = "sha256:" + "0" * 64
    with pytest.raises(supervisor._HandoffRejected, match="evidence"):
        adapter.handoff_command(handle, wrong_export_digest)
    invalid_adopter = _adopt_request(
        common, nonce, old_runtime, new_runtime, registered, adapter.config.credential_file
    )
    invalid_adopter["adopter_record_digest"] = "invalid"
    with pytest.raises(supervisor._HandoffRejected, match="evidence"):
        adapter.handoff_command(handle, invalid_adopter)
    assert adapter._handoff is not None
    assert adapter._handoff.phase == "export_sealed"
    assert len(observed) == 1
    wrong = _adopt_request(
        common, "wrong", old_runtime, new_runtime, registered, adapter.config.credential_file
    )
    with pytest.raises(supervisor._HandoffRejected, match="nonce"):
        adapter.handoff_command(handle, wrong)
    assert adapter._handoff is not None and adapter._handoff.phase == "export_sealed"
    wrong_owner = _adopt_request(
        common, nonce, old_runtime, new_runtime, registered, adapter.config.credential_file
    )
    wrong_owner["old_owner"] = {"pid": 4101, "birth_id": "birth-4101"}
    with pytest.raises(supervisor._HandoffRejected, match="evidence"):
        adapter.handoff_command(handle, wrong_owner)
    same_owner = _adopt_request(
        common, nonce, old_runtime, new_runtime, registered, adapter.config.credential_file
    )
    same_owner["new_owner"] = same_owner["old_owner"]
    with pytest.raises(supervisor._HandoffRejected, match="evidence"):
        adapter.handoff_command(handle, same_owner)
    assert adapter._handoff is not None and adapter._handoff.phase == "export_sealed"
    adopt = _adopt_request(
        common, nonce, old_runtime, new_runtime, registered, adapter.config.credential_file
    )
    response, _ = adapter.handoff_command(handle, adopt)
    assert response["worker_phase"] == "adopt_prepared"
    prospective = response["host_ack"]["registered_state"]
    assert prospective["runtime"] == new_runtime
    replayed, _ = adapter.handoff_command(handle, adopt)
    assert replayed == response
    changed_replay = {**adopt, "request_id": "request-2"}
    with pytest.raises(supervisor._HandoffRejected, match="replay changed"):
        adapter.handoff_command(handle, changed_replay)
    with pytest.raises(supervisor._HandoffRejected, match="phase"):
        adapter.handoff_command(handle, _prepare_request(common, old_runtime))
    commit_request = {
        **common,
        "command": "handoff_commit",
        "new_owner": {"pid": 5100, "birth_id": "birth-5100"},
        "new_runtime": new_runtime,
        "credential_generation": _generation(),
        "registered_state": prospective,
    }
    response, _ = adapter.handoff_command(handle, commit_request)
    assert response["worker_phase"] == "rebind_committed"
    response, _ = adapter.handoff_command(
        handle,
        {
            **common,
            "command": "resume_prepare",
            "new_owner": {"pid": 5100, "birth_id": "birth-5100"},
            "new_runtime": new_runtime,
        },
    )
    assert response["worker_phase"] == "resume_armed"
    response, _ = adapter.handoff_command(
        handle,
        {
            **common,
            "command": "resume_commit",
            "new_owner": {"pid": 5100, "birth_id": "birth-5100"},
            "new_runtime": new_runtime,
        },
    )
    assert response["worker_phase"] == "resumed"
    resumed_request = {
        **common,
        "command": "resume_commit",
        "new_owner": {"pid": 5100, "birth_id": "birth-5100"},
        "new_runtime": new_runtime,
    }
    assert adapter.handoff_command(handle, resumed_request)[0] == response
    finalize_request = {
        **common,
        "command": "handoff_finalize",
        "new_owner": {"pid": 5100, "birth_id": "birth-5100"},
    }
    finalized, _ = adapter.handoff_command(handle, finalize_request)
    assert finalized["status"] == "finalized"
    assert finalized["worker_phase"] == "finalized"
    assert adapter._handoff is None
    finalized_state = (
        copy.deepcopy(adapter._runtime_owner_identity),
        copy.deepcopy(adapter._finalized_handoff_acks),
        copy.deepcopy(observed),
    )
    replayed_finalize, should_exit = adapter.handoff_command(
        handle, finalize_request
    )
    assert should_exit is False
    assert replayed_finalize == finalized
    for altered_finalize in (
        {
            **finalize_request,
            "new_owner": {"pid": 5101, "birth_id": "birth-5101"},
        },
        {**finalize_request, "nonce_digest": "sha256:" + "0" * 64},
    ):
        with pytest.raises(supervisor._HandoffRejected, match="finalized"):
            adapter.handoff_command(handle, altered_finalize)
    assert (
        adapter._runtime_owner_identity,
        adapter._finalized_handoff_acks,
        observed,
    ) == finalized_state
    for replay in (adopt, commit_request, resumed_request):
        with pytest.raises(supervisor._HandoffRejected, match="finalized"):
            adapter.handoff_command(handle, replay)
    stale_prepare = _prepare_request(
        {
            **common,
            "handoff_id": "handoff-2",
            "sealed_record_digest": "sha256:" + "a" * 64,
        },
        new_runtime,
    )
    stale_prepare["old_owner"] = {"pid": 4100, "birth_id": "birth-4100"}
    with pytest.raises(supervisor._HandoffRejected, match="stale"):
        adapter.handoff_command(handle, stale_prepare)
    thread.join(timeout=2)
    assert [item["command"] for item in observed] == [
        "pause_prepare", "rebind_prepare", "rebind_commit", "resume_prepare",
        "resume_commit", "handoff_finalize",
    ]
    assert observed[0]["old_owner"] == {"pid": 4100, "birth_id": "birth-4100"}
    assert observed[1]["old_owner"] == observed[0]["old_owner"]
    assert observed[1]["new_owner"] == {"pid": 5100, "birth_id": "birth-5100"}
    assert all(
        item["new_owner"] == observed[1]["new_owner"] for item in observed[2:]
    )
    for item in observed:
        assert "nonce" not in item


def test_finalized_handoff_ack_cache_is_bounded_to_eight(tmp_path):
    adapter = supervisor.LocalWorkerPreparerAdapter(_config(tmp_path), environ={})
    for index in range(9):
        request = {
            "version": supervisor.CONTROL_VERSION,
            "command": "handoff_finalize",
            "handoff_id": f"handoff-{index}",
            "nonce_digest": "sha256:" + f"{index:x}" * 64,
            "sealed_record_digest": "sha256:" + "a" * 64,
            "deadline_monotonic": 100.0,
            "deadline_unix_ms": 100_000,
            "new_owner": {"pid": 5100 + index, "birth_id": f"birth-{index}"},
        }
        adapter._remember_finalized_handoff_ack(
            request,
            {
                "version": supervisor.CONTROL_VERSION,
                "command": "handoff_finalize_ack",
                "handoff_id": request["handoff_id"],
                "status": "finalized",
            },
        )
    assert [item.handoff_id for item in adapter._finalized_handoff_acks] == [
        f"handoff-{index}" for index in range(1, 9)
    ]


def test_bound_host_control_eof_requires_full_cleanup(tmp_path, monkeypatch):
    fixture = _handoff_fixture(tmp_path, monkeypatch, ["paused", "close"])
    (
        adapter, handle, thread, _observed, stopped, terminated, common,
        nonce, old_runtime, new_runtime, registered,
    ) = fixture
    adapter.handoff_command(handle, _prepare_request(common, old_runtime))
    adapter.handoff_command(handle, _seal_request(common, registered))
    adopt = _adopt_request(
        common, nonce, old_runtime, new_runtime, registered, adapter.config.credential_file
    )
    with pytest.raises(supervisor._HandoffAuthorizedFailure, match="control channel"):
        adapter.handoff_command(handle, adopt)
    adapter.abort(handle)
    thread.join(timeout=2)
    assert stopped
    assert terminated == [(6200, signal.SIGTERM)]
    assert handle.closed is True


def test_export_sealed_rollback_forwards_old_owner(tmp_path, monkeypatch):
    fixture = _handoff_fixture(tmp_path, monkeypatch, ["paused", "pause_cancelled"])
    adapter, handle, thread, observed, *_rest = fixture
    common, _nonce, old_runtime, _new_runtime, registered = fixture[6:]
    adapter.handoff_command(handle, _prepare_request(common, old_runtime))
    adapter.handoff_command(handle, _seal_request(common, registered))
    response, should_exit = adapter.handoff_command(
        handle,
        {
            **common,
            "command": "handoff_abort",
            "reason_code": "coordinator_cancelled",
            "old_owner": {"pid": 4100, "birth_id": "birth-4100"},
        },
    )
    thread.join(timeout=2)
    assert should_exit is False
    assert response["worker_phase"] == "owned"
    assert adapter._handoff is None
    assert observed[1]["command"] == "pause_cancel"
    assert observed[1]["old_owner"] == {"pid": 4100, "birth_id": "birth-4100"}


def test_prepared_handoff_deadline_requires_full_cleanup(tmp_path, monkeypatch):
    fixture = _handoff_fixture(tmp_path, monkeypatch, ["paused"])
    (
        adapter, handle, thread, _observed, stopped, terminated, common,
        nonce, old_runtime, new_runtime, registered,
    ) = fixture
    adapter.handoff_command(handle, _prepare_request(common, old_runtime))
    adapter.handoff_command(handle, _seal_request(common, registered))
    assert adapter._handoff is not None
    expired = time.monotonic()
    adapter._handoff.deadline_monotonic = expired
    common["deadline_monotonic"] = expired
    adopt = _adopt_request(
        common, nonce, old_runtime, new_runtime, registered, adapter.config.credential_file
    )
    with pytest.raises(supervisor._HandoffAuthorizedFailure, match="deadline expired"):
        adapter.handoff_command(handle, adopt)
    adapter.abort(handle)
    thread.join(timeout=2)
    assert stopped
    assert terminated == [(6200, signal.SIGTERM)]


def test_host_ack_hash_mismatch_is_authorized_failure(tmp_path, monkeypatch):
    fixture = _handoff_fixture(tmp_path, monkeypatch, [])
    adapter, handle, thread, _observed, *_rest = fixture
    common, _nonce, old_runtime, _new_runtime, registered = fixture[6:]
    request = {
        "version": supervisor.HOST_CONTROL_VERSION,
        "command": "pause_prepare",
        "handoff_id": common["handoff_id"],
        "nonce_digest": common["nonce_digest"],
        "deadline_monotonic": common["deadline_monotonic"],
        "deadline_unix_ms": common["deadline_unix_ms"],
        "old_runtime": old_runtime,
    }
    response = {
        "version": supervisor.HOST_CONTROL_VERSION,
        "command": "pause_prepare_ack",
        "handoff_id": common["handoff_id"],
        "nonce_digest": common["nonce_digest"],
        "status": "paused",
        "host": {"pid": 6200, "birth_id": "birth-6200"},
        "phase": "PAUSED",
        "activation": adapter._expected_activation(handle),
        "registered_state": registered,
        "ack_sha256": "sha256:" + "0" * 64,
    }
    with pytest.raises(supervisor._HandoffAuthorizedFailure, match="digest"):
        adapter._verify_host_ack(
            handle,
            request,
            response,
            statuses=frozenset({"paused"}),
            snapshot=True,
        )
    thread.join(timeout=2)


def test_persistent_host_control_eof_keeps_normal_worker_cleanup(tmp_path, monkeypatch):
    config = _config(tmp_path)
    profile = _profile(config)
    runtime, worker = socket.socketpair()
    worker_host, host = socket.socketpair()
    events = []
    owned = SimpleNamespace(host_control=worker_host)

    class FakeAdapter:
        _handoff = None

        def __init__(self, received_config, *, environ):
            assert received_config == config

        def prepare(self, *_args, **_kwargs):
            return owned

        def report(self, received):
            assert received is owned
            return {
                "version": supervisor.PREPARATION_VERSION,
                "operation_id": "operation-1",
                "channel_id": "channel-1",
            }

        def abort(self, received):
            assert received is owned
            events.append("abort")
            worker_host.close()

    monkeypatch.setattr(supervisor, "LocalWorkerPreparerAdapter", FakeAdapter)
    outcome = []
    thread = threading.Thread(
        target=lambda: outcome.append(supervisor._serve_prepared_worker(worker.detach()))
    )
    thread.start()
    supervisor._send_private_frame(
        runtime,
        {
            "version": supervisor.CONTROL_VERSION,
            "command": "prepare",
            "operation_id": "operation-1",
            "channel_id": "channel-1",
            "profile": supervisor._private_profile_payload(profile),
            "config": supervisor._private_config_payload(config),
        },
    )
    assert supervisor._receive_private_frame(runtime)["status"] == "ok"
    host.close()
    error = supervisor._receive_private_frame(runtime)
    assert error["status"] == "error"
    assert error["error"] == "prepared Worker rejected the handoff"
    assert error["error_code"] == "host_control_closed"
    assert error["error_stage"] == "control"
    thread.join(timeout=2)
    runtime.close()
    assert outcome == [78]
    assert events == ["abort"]


def test_persistent_host_control_refusal_precedes_blocked_cleanup(tmp_path, monkeypatch):
    config = _config(tmp_path)
    profile = _profile(config)
    runtime, worker = socket.socketpair()
    worker_host, host = socket.socketpair()
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    owned = SimpleNamespace(host_control=worker_host)

    class FakeAdapter:
        _handoff = None

        def __init__(self, received_config, *, environ):
            assert received_config == config

        def prepare(self, *_args, **_kwargs):
            return owned

        def report(self, received):
            assert received is owned
            return {
                "version": supervisor.PREPARATION_VERSION,
                "operation_id": "operation-1",
                "channel_id": "channel-1",
            }

        def abort(self, received):
            assert received is owned
            cleanup_started.set()
            assert allow_cleanup.wait(timeout=2)
            worker_host.close()

    monkeypatch.setattr(supervisor, "LocalWorkerPreparerAdapter", FakeAdapter)
    outcome = []
    thread = threading.Thread(
        target=lambda: outcome.append(supervisor._serve_prepared_worker(worker.detach()))
    )
    thread.start()
    supervisor._send_private_frame(
        runtime,
        {
            "version": supervisor.CONTROL_VERSION,
            "command": "prepare",
            "operation_id": "operation-1",
            "channel_id": "channel-1",
            "profile": supervisor._private_profile_payload(profile),
            "config": supervisor._private_config_payload(config),
        },
    )
    assert supervisor._receive_private_frame(runtime)["status"] == "ok"
    host.close()
    error = supervisor._receive_private_frame(runtime)
    assert error == {
        "version": supervisor.CONTROL_VERSION,
        "status": "error",
        "error": "prepared Worker rejected the handoff",
        "error_code": "host_control_closed",
        "error_stage": "control",
    }
    assert cleanup_started.wait(timeout=1)
    assert thread.is_alive()
    allow_cleanup.set()
    thread.join(timeout=2)
    runtime.close()
    assert outcome == [78]
