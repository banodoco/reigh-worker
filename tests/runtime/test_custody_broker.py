from __future__ import annotations

import ast
import inspect
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from source.runtime import custody_broker
from source.runtime import supervisor


@pytest.fixture(autouse=True)
def _clear_unresolved_launches():
    supervisor._UNRESOLVED_CUSTODY.clear()
    yield
    supervisor._UNRESOLVED_CUSTODY.clear()


def _exercise_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, authority_journal: Path | None = None,
):
    pid = 43123
    identity = {"pid": pid, "birth_id": "birth-43123", "uid": os.getuid()}
    tokens = iter(
        (
            {"pid": pid, "uid": os.getuid(), "pidversion": 7, "words": [1] * 8, "sha256": "sha256:" + "1" * 64},
            {"pid": pid, "uid": os.getuid(), "pidversion": 8, "words": [2] * 8, "sha256": "sha256:" + "2" * 64},
        )
    )
    monkeypatch.setattr(custody_broker.sys, "platform", "darwin")
    monkeypatch.setattr(custody_broker, "_token_details", lambda _connection: next(tokens))
    broker = custody_broker.RoleBoundCustodyBroker(
        role="generic_pack_host",
        identity_provider=lambda observed_pid: identity if observed_pid == pid else None,
        ledger_root=tmp_path / "ledger",
        authority_journal=authority_journal,
    )
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(broker.socket_path))
    frame = {
        "version": custody_broker.PROTOCOL_VERSION,
        "command": "register_pre_exec",
        "run_id": broker.run_id,
        "role": broker.role,
        "pid": pid,
        "ppid": os.getpid(),
        "argv_digest": "sha256:" + "a" * 64,
    }
    custody_broker._send_frame(connection, frame)
    ack = custody_broker._read_frame(connection)
    broker.wait_until_sealed()
    connection.close()
    return broker, identity, ack


def test_registration_is_kernel_authenticated_durable_before_ack_and_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker, identity, ack = _exercise_registration(tmp_path, monkeypatch)
    assert ack["status"] == "registered"
    assert ack["pid"] == identity["pid"]
    assert ack["ledger_state_digest"].startswith("sha256:")
    events = [json.loads(line) for line in broker.journal_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "registration_pre_exec",
        "registration_ack_checkpoint",
        "registration_post_exec",
        "admission_sealed",
    ]
    assert all(
        event["predecessor_digest"] == (None if index == 0 else events[index - 1]["event_digest"])
        for index, event in enumerate(events)
    )
    ledger = json.loads(broker.ledger_path.read_text())
    assert ledger["state"] == "sealed"
    assert ledger["registration"]["audit_token_pidversion"] == 8


def test_post_exec_authority_is_escrowed_before_final_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    escrow_root = tmp_path / "escrow"
    escrow_root.mkdir(mode=0o700)
    journal = escrow_root / "authorities.jsonl"
    broker, identity, _ack = _exercise_registration(
        tmp_path, monkeypatch, authority_journal=journal,
    )
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["pid"] == identity["pid"]
    assert records[0]["role"] == broker.role
    assert records[0]["identity"] == identity
    assert records[0]["audit_token_words"] == [2] * 8
    assert records[0]["binding"]["state"] == "post-exec-authority-validated"
    assert journal.stat().st_mode & 0o777 == 0o600


def test_cleanup_routes_only_through_registered_audit_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker, identity, _ack = _exercise_registration(tmp_path, monkeypatch)
    observed = []
    monkeypatch.setattr(
        custody_broker,
        "_signal_token",
        lambda words, signum: observed.append((list(words), signum)),
    )
    broker.signal(signal.SIGTERM, expected_pid=identity["pid"])
    assert observed == [([2] * 8, signal.SIGTERM)]


def test_cleanup_fails_closed_for_changed_identity_or_unsealed_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker, identity, _ack = _exercise_registration(tmp_path, monkeypatch)
    broker.identity_provider = lambda _pid: {**identity, "birth_id": "replacement"}
    monkeypatch.setattr(
        custody_broker,
        "_signal_token",
        lambda *_args: pytest.fail("changed identity was signalled"),
    )
    with pytest.raises(custody_broker.CustodyError, match="absent or changed"):
        broker.signal(signal.SIGKILL, expected_pid=identity["pid"])
    broker.state = "accepting"
    with pytest.raises(custody_broker.CustodyError, match="not sealed"):
        broker.signal(signal.SIGTERM, expected_pid=identity["pid"])


def test_registration_rejects_kernel_peer_token_for_another_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(custody_broker.sys, "platform", "darwin")
    monkeypatch.setattr(
        custody_broker,
        "_token_details",
        lambda _connection: {
            "pid": 99999,
            "uid": os.getuid(),
            "pidversion": 1,
            "words": [1] * 8,
            "sha256": "sha256:" + "1" * 64,
        },
    )
    broker = custody_broker.RoleBoundCustodyBroker(
        role="worker",
        identity_provider=lambda pid: {"pid": pid, "birth_id": "birth", "uid": os.getuid()},
        ledger_root=tmp_path / "ledger",
        timeout=0.2,
    )
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(0.5)
    connection.connect(str(broker.socket_path))
    custody_broker._send_frame(connection, {
        "version": custody_broker.PROTOCOL_VERSION,
        "command": "register_pre_exec",
        "run_id": broker.run_id,
        "role": broker.role,
        "pid": 43123,
        "ppid": os.getpid(),
        "argv_digest": "sha256:" + "a" * 64,
    })
    with pytest.raises((custody_broker.CustodyError, TimeoutError, OSError)):
        custody_broker._read_frame(connection)
    with pytest.raises(custody_broker.CustodyError, match="kernel identity differs"):
        broker.wait_until_sealed()
    connection.close()


def _failed_after_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
):
    pid = 43123
    identity = {"pid": pid, "birth_id": "birth-43123", "uid": os.getuid()}
    pre = {
        "pid": pid, "uid": os.getuid(), "pidversion": 7,
        "words": [1] * 8, "sha256": "sha256:" + "1" * 64,
    }
    post = {
        "pid": pid, "uid": os.getuid(), "pidversion": 8,
        "words": [2] * 8, "sha256": "sha256:" + "2" * 64,
    }
    tokens = iter((pre, post))
    monkeypatch.setattr(custody_broker.sys, "platform", "darwin")
    if failure == "post_exec_refresh":
        refresh_calls = iter((pre, custody_broker.CustodyError("injected post-exec token refresh failure")))

        def failed_refresh(_connection):
            value = next(refresh_calls)
            if isinstance(value, BaseException):
                raise value
            return value

        monkeypatch.setattr(custody_broker, "_token_details", failed_refresh)
    else:
        monkeypatch.setattr(custody_broker, "_token_details", lambda _connection: next(tokens))
    broker = custody_broker.RoleBoundCustodyBroker(
        role="worker",
        identity_provider=lambda observed_pid: identity if observed_pid == pid else None,
        ledger_root=tmp_path / failure,
        timeout=0.03,
    )
    original_persist = broker._persist

    def injected_persist(event_name: str) -> None:
        if event_name == failure:
            raise custody_broker.CustodyError(f"injected {failure} failure")
        original_persist(event_name)

    if failure != "post_exec_refresh":
        monkeypatch.setattr(broker, "_persist", injected_persist)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(broker.socket_path))
    custody_broker._send_frame(connection, {
        "version": custody_broker.PROTOCOL_VERSION,
        "command": "register_pre_exec",
        "run_id": broker.run_id,
        "role": broker.role,
        "pid": pid,
        "ppid": os.getpid(),
        "argv_digest": "sha256:" + "a" * 64,
    })
    assert custody_broker._read_frame(connection)["status"] == "registered"
    with pytest.raises(custody_broker.CustodyError, match="registration failed"):
        broker.wait_until_sealed()
    connection.close()
    return broker, identity


@pytest.mark.parametrize(
    ("failure", "cleanup_available"),
    [
        ("post_exec_refresh", False),
        ("registration_post_exec", True),
        ("admission_sealed", True),
    ],
)
def test_post_ack_failure_injections_preserve_truthful_cleanup_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    cleanup_available: bool,
) -> None:
    broker, identity = _failed_after_ack(tmp_path, monkeypatch, failure)
    observed: list[tuple[list[int], int]] = []
    monkeypatch.setattr(
        custody_broker,
        "_signal_token",
        lambda words, signum: observed.append((list(words), signum)),
    )
    if cleanup_available:
        broker.signal_failed_admission(signal.SIGTERM, expected_pid=identity["pid"])
        assert observed == [([2] * 8, signal.SIGTERM)]
    else:
        with pytest.raises(custody_broker.CustodyError, match="no validated"):
            broker.signal_failed_admission(signal.SIGTERM, expected_pid=identity["pid"])
        assert observed == []


class _AdmissionProcess:
    def __init__(self, pid: int = 47001):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            if timeout == 0:
                raise subprocess.TimeoutExpired("custody", timeout)
            self.returncode = -signal.SIGTERM
        return self.returncode


class _AdmissionBroker:
    def __init__(self, *, cleanup_available: bool, **_kwargs):
        self.cleanup_available = cleanup_available
        self.signals: list[int] = []

    def child_environment(self, _argv, *, start_new_session):
        return {"D76_CUSTODY_FIXTURE": "1" if start_new_session else "0"}

    def wait_until_sealed(self):
        raise custody_broker.CustodyError("injected post-ACK admission failure")

    def signal_failed_admission(self, signum, *, expected_pid):
        assert expected_pid == 47001
        if not self.cleanup_available:
            raise custody_broker.CustodyError("no validated post-exec authority")
        self.signals.append(signum)


@pytest.mark.parametrize("cleanup_available", [False, True])
def test_custodied_popen_retains_handle_and_latches_only_unresolved_admission(
    monkeypatch: pytest.MonkeyPatch,
    cleanup_available: bool,
) -> None:
    brokers = []

    def make_broker(**kwargs):
        broker = _AdmissionBroker(cleanup_available=cleanup_available, **kwargs)
        brokers.append(broker)
        return broker

    process = _AdmissionProcess()
    monkeypatch.setattr(supervisor, "RoleBoundCustodyBroker", make_broker)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_args, **_kwargs: process)
    owner = supervisor._CustodyLaunchOwner("worker")
    with pytest.raises(supervisor._CustodyAdmissionFailure) as caught:
        supervisor._custodied_popen(
            [sys.executable, "-c", "pass"],
            custody_role="worker",
            custody_owner=owner,
            start_new_session=True,
        )
    assert caught.value.owner is owner
    assert owner.process is process
    assert owner.broker is brokers[0]
    if cleanup_available:
        assert owner.state == "failed_reaped"
        assert process.returncode == -signal.SIGTERM
        assert brokers[0].signals == [signal.SIGTERM]
        assert not supervisor._UNRESOLVED_CUSTODY
    else:
        assert owner.state == "unresolved"
        assert process.returncode is None
        assert list(supervisor._UNRESOLVED_CUSTODY.values()) == [owner]
        with pytest.raises(supervisor.LauncherConfigurationError, match="refuses follow-on"):
            supervisor._custodied_popen(
                [sys.executable, "-c", "pass"], custody_role="worker"
            )


def test_all_four_product_callers_publish_owner_before_admission_wait() -> None:
    tree = ast.parse(inspect.getsource(supervisor))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_custodied_popen"
    ]
    assert len(calls) == 4
    assert all(
        {keyword.arg for keyword in call.keywords} >= {"custody_role", "custody_owner"}
        for call in calls
    )


def test_custodied_popen_success_releases_latch_and_returns_same_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SealedBroker(_AdmissionBroker):
        def wait_until_sealed(self):
            return None

    broker = SealedBroker(cleanup_available=True)
    process = _AdmissionProcess()
    monkeypatch.setattr(supervisor, "RoleBoundCustodyBroker", lambda **_kwargs: broker)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_args, **_kwargs: process)
    owner = supervisor._CustodyLaunchOwner("worker")
    returned = supervisor._custodied_popen(
        [sys.executable, "-c", "pass"],
        custody_role="worker",
        custody_owner=owner,
    )
    assert returned is process
    assert owner.process is process
    assert owner.broker is broker
    assert owner.state == "sealed"
    assert not supervisor._UNRESOLVED_CUSTODY


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin audit-token API")
def test_real_worker_launch_registers_seals_and_signals_with_kernel_audit_token() -> None:
    process = supervisor._custodied_popen(
        ["/bin/sleep", "30"],
        custody_role="worker",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    broker = process._reigh_custody_broker
    assert broker.state == "sealed"
    assert broker.registration["audit_token_pidversion"] != broker.registration["pre_exec_pidversion"]
    supervisor._terminate_and_wait(process, process.pid)
    assert process.returncode == -signal.SIGTERM


def test_resolve_executable_preserves_lexical_venv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.write_bytes(b"#!/bin/sh\nexit 0\n")
    target.chmod(0o755)
    lexical = tmp_path / "venv-python"
    lexical.symlink_to(target.name)

    assert custody_broker._resolve_executable(str(lexical)) == str(lexical)
    assert lexical.resolve(strict=True) == target
    broker = object.__new__(custody_broker.RoleBoundCustodyBroker)
    broker.socket_path = tmp_path / "custody.sock"
    broker.run_id = "sha256:" + "1" * 64
    broker.role = "engine_daemon"
    environment = broker.child_environment(
        [str(lexical), "-I", "-m", "vibecomfy.commands.session"],
        start_new_session=True,
    )
    normalized = json.loads(
        custody_broker.base64.b64decode(
            environment["ASTRID_CUSTODY_TARGET_B64"], validate=True
        )
    )
    assert normalized[0] == str(lexical)
    assert Path(normalized[0]).resolve(strict=True) == target


def test_resolve_executable_rejects_retarget_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical = tmp_path / "venv-python"
    first = tmp_path / "python-a"
    second = tmp_path / "python-b"
    for target in (first, second):
        target.write_bytes(b"#!/bin/sh\nexit 0\n")
        target.chmod(0o755)
    lexical.symlink_to(first.name)
    original = custody_broker._executable_identity
    calls = 0

    def retarget(candidate: Path):
        nonlocal calls
        calls += 1
        observed = original(candidate)
        if calls == 1:
            lexical.unlink()
            lexical.symlink_to(second.name)
        return observed

    monkeypatch.setattr(custody_broker, "_executable_identity", retarget)
    with pytest.raises(custody_broker.CustodyError, match="changed during validation"):
        custody_broker._resolve_executable(str(lexical))


def test_resolve_executable_rejects_lexical_replacement_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical = tmp_path / "venv-python"
    lexical.write_bytes(b"#!/bin/sh\nexit 0\n")
    lexical.chmod(0o755)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"#!/bin/sh\nexit 0\n")
    replacement.chmod(0o755)
    original = custody_broker._executable_identity
    calls = 0

    def replace(candidate: Path):
        nonlocal calls
        calls += 1
        observed = original(candidate)
        if calls == 1:
            os.replace(replacement, lexical)
        return observed

    monkeypatch.setattr(custody_broker, "_executable_identity", replace)
    with pytest.raises(custody_broker.CustodyError, match="changed during validation"):
        custody_broker._resolve_executable(str(lexical))


def test_resolve_executable_rejects_missing_and_nonexecutable(tmp_path: Path) -> None:
    with pytest.raises(custody_broker.CustodyError, match="unavailable"):
        custody_broker._resolve_executable(str(tmp_path / "missing"))
    candidate = tmp_path / "python"
    candidate.write_bytes(b"#!/bin/sh\nexit 0\n")
    candidate.chmod(0o600)
    with pytest.raises(custody_broker.CustodyError, match="not executable"):
        custody_broker._resolve_executable(str(candidate))
