from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace

import pytest

from source.runtime import supervisor
from source.runtime.worker import preflight


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
    monkeypatch.setattr(
        supervisor,
        "_terminate_and_wait",
        lambda process, pgid: (terminated.append((process.pid, pgid)), setattr(process, "returncode", 0)),
    )

    def fake_popen(_argv, **kwargs):
        process = FakeProcess(next(next_pid))
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

    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)
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

    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)
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

    with pytest.raises(supervisor.LauncherConfigurationError, match="replacement"):
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
    assert terminated == [(4200, 4200)]
    assert stopped
    adapter.abort(second)
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
