from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import source.runtime.supervisor as supervisor
from source.runtime.supervisor import (
    LauncherConfigurationError,
    _OwnedVibeComfySession,
    _read_owned_vibecomfy_custody,
    _read_owned_vibecomfy_session,
    _stop_owned_vibecomfy_session,
)


def test_host_spawn_failure_cleans_owned_session(tmp_path, monkeypatch):
    config = supervisor.HostLaunchConfig(
        host_python=Path(sys.executable),
        source_checkout=tmp_path,
        pack_root=tmp_path,
        runtime_endpoint="http://127.0.0.1:9000",
        credential_file=tmp_path / "credential",
        support_root=tmp_path,
        runtime_instance_id="test-runtime",
        ready_file=tmp_path / "ready.json",
        state_file=tmp_path / "state.json",
        boot_manifest_path=tmp_path / "boot.json",
        boot_manifest_hash="sha256:" + "a" * 64,
    )
    owned = object()
    stopped = []
    profile = tmp_path / "worker-readiness-profile.json"
    profile.write_text("{}")
    monkeypatch.setattr(supervisor, "_start_owned_vibecomfy_session", lambda *_: ({}, owned))
    monkeypatch.setattr(supervisor, "_prepare_worker_readiness", lambda *a, **kw: (profile, "hash"))
    monkeypatch.setattr(supervisor, "_stop_owned_vibecomfy_session", stopped.append)

    def fail_spawn(*args, **kwargs):
        raise OSError("host spawn failed")

    monkeypatch.setattr(supervisor.subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="host spawn failed"):
        supervisor.launch_generic_pack_host(config, environ={}, enforce_readiness=True)
    assert stopped == [owned]
    assert not profile.exists()


def _registry(tmp_path: Path) -> Path:
    root = tmp_path / "session"
    root.mkdir()
    from source.runtime.worker.preflight import _process_birth_identity
    birth_id = _process_birth_identity(os.getpid()) or "birth-a"
    (root / "pid").write_text(str(os.getpid()), encoding="utf-8")
    (root / "comfy_pid").write_text(str(os.getpid()), encoding="utf-8")
    (root / "comfy_process_start_identity").write_text(birth_id, encoding="utf-8")
    (root / "url").write_text("http://127.0.0.1:8188", encoding="utf-8")
    (root / "config.json").write_text('{"base_directory":"/models"}\n', encoding="utf-8")
    (root / "source_revision").write_text("vibe-revision", encoding="utf-8")
    (root / "source_content_digest").write_text("sha256:" + "b" * 64, encoding="utf-8")
    (root / "daemon.log").write_text("ready\n", encoding="utf-8")
    (root / "launch.json").write_text(
        json.dumps(
            {
                "launch_token": "token-a",
                "pid": os.getpid(),
                "process_start_identity": birth_id,
                "comfy_pid": os.getpid(),
                "comfy_process_start_identity": birth_id,
                "url": "http://127.0.0.1:8188",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_missing_optional_session_does_not_advertise_checkout_binding() -> None:
    # No target directory means the Worker does not start an optional checkout
    # session, so readiness publishes no Vibe extension.
    assert supervisor._start_owned_vibecomfy_session(
        SimpleNamespace(), {}
    ) == (None, None)


def test_session_binding_publishes_registry_identity_and_config_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _registry(tmp_path)
    original_run = supervisor.subprocess.run

    def fake_run(command, *args, **kwargs):
        if command and command[0] == "lsof":
            return SimpleNamespace(returncode=0, stdout="python 1 :8188 (LISTEN)\n")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        fake_run,
    )
    monkeypatch.setattr(
        supervisor,
        "_process_parent_pid",
        lambda _pid: os.getpid(),
    )
    payload = _read_owned_vibecomfy_session(root, verify_parent=True)
    assert payload["pid"] == os.getpid()
    assert payload["launch_token"] == "token-a"
    expected = "sha256:" + hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    assert payload["config_digest"] == expected


def test_session_binding_rejects_unparented_comfy_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _registry(tmp_path)
    original_run = supervisor.subprocess.run
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda command, *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="python 1 :8188 (LISTEN)\n",
        )
        if command and command[0] == "lsof"
        else original_run(command, *args, **kwargs),
    )
    monkeypatch.setattr(supervisor, "_process_parent_pid", lambda _pid: 999999)
    with pytest.raises(LauncherConfigurationError, match="unreadable or stale"):
        _read_owned_vibecomfy_session(root, verify_parent=True)


def test_session_binding_fails_closed_on_listener_observation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _registry(tmp_path)
    original_run = supervisor.subprocess.run
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda command, *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="lsof: permission denied",
        )
        if command and command[0] == "lsof"
        else original_run(command, *args, **kwargs),
    )
    with pytest.raises(LauncherConfigurationError, match="unreadable or stale"):
        _read_owned_vibecomfy_session(root, verify_parent=False)


def test_session_binding_requires_source_content_attestation(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    (root / "source_content_digest").unlink()
    with pytest.raises(LauncherConfigurationError, match="incomplete"):
        _read_owned_vibecomfy_session(root, verify_parent=False)


def test_startup_custody_recovery_never_adopts_a_later_port_binder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _registry(tmp_path)
    custody = _read_owned_vibecomfy_custody(
        root,
        expected_daemon_pid=os.getpid(),
        expected_server_url="http://127.0.0.1:8188",
    )
    assert custody is not None

    class DeadProcess:
        pid = os.getpid() + 1000

        @staticmethod
        def poll() -> int:
            return 1

    monkeypatch.setattr(supervisor, "_owned_listener_pid", lambda _port: 99999)
    monotonic_values = iter((100.0, 200.0))
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        supervisor.os,
        "kill",
        lambda *_args, **_kwargs: pytest.fail("unrelated listener must not be signalled"),
    )
    with pytest.raises(LauncherConfigurationError, match="remained"):
        _stop_owned_vibecomfy_session(
            _OwnedVibeComfySession(
                root=root,
                process=DeadProcess(),
                daemon_pid=DeadProcess.pid,
                server_url="http://127.0.0.1:8188",
            )
        )


def test_startup_custody_recovery_rejects_unknown_non_listening_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _registry(tmp_path)

    class DeadProcess:
        pid = os.getpid() + 1001

        @staticmethod
        def poll() -> int:
            return 1

    monkeypatch.setattr(supervisor, "_owned_listener_pid", lambda _port: None)
    monotonic_values = iter((100.0, 200.0))
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(monotonic_values))
    with pytest.raises(LauncherConfigurationError, match="child absence"):
        _stop_owned_vibecomfy_session(
            _OwnedVibeComfySession(
                root=root,
                process=DeadProcess(),
                daemon_pid=DeadProcess.pid,
                server_url="http://127.0.0.1:8188",
            )
        )


def test_partial_or_mismatched_session_registry_fails_closed(tmp_path: Path) -> None:
    root = _registry(tmp_path)
    (root / "launch.json").write_text(
        json.dumps(
            {
                "launch_token": "wrong",
                "pid": os.getpid(),
                "process_start_identity": (root / "comfy_process_start_identity").read_text().strip(),
                "comfy_pid": os.getpid(),
                "comfy_process_start_identity": (root / "comfy_process_start_identity").read_text().strip(),
                "url": "http://127.0.0.1:8189",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LauncherConfigurationError, match="unreadable or stale"):
        _read_owned_vibecomfy_session(root, verify_parent=False)
