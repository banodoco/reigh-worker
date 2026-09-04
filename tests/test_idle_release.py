from __future__ import annotations

import ast
import inspect
import signal

import pytest

from source.runtime import supervisor
from source.runtime import worker_protocol
from source.runtime.worker import idle_release, server


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_tracker(*, idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False, clock=None):
    cfg = idle_release.IdleReleaseConfig(
        idle_minutes=idle_minutes,
        grace_seconds=grace_seconds,
        is_service_mode=is_service_mode,
    )
    return idle_release.IdleReleaseTracker(cfg, clock=clock or _FakeClock()), clock


def test_tracker_disabled_when_idle_minutes_zero() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(3600)
    tracker.record_empty_poll()
    clock.advance(3600)
    assert tracker.should_release() is False


def test_tracker_blocked_in_service_mode() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=True),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(120)
    tracker.record_empty_poll()
    clock.advance(120)
    assert tracker.should_release() is False


def test_tracker_blocked_when_not_onboarded() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.record_empty_poll()
    clock.advance(3600)
    assert tracker.should_release() is False


def test_tracker_blocked_during_onboarding_grace() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(30)  # still within grace
    tracker.record_empty_poll()
    clock.advance(120)  # past idle window but onboarded only 150s ago — actually past grace
    # Re-test with shorter advance to stay inside grace:
    clock2 = _FakeClock()
    tracker2 = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=300.0, is_service_mode=False),
        clock=clock2,
    )
    tracker2.mark_onboarded()
    clock2.advance(30)
    tracker2.record_empty_poll()
    clock2.advance(120)
    assert tracker2.should_release() is False  # 150s onboarded, < 300s grace


def test_tracker_blocked_when_no_empty_poll_recorded() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(3600)
    assert tracker.should_release() is False


def test_tracker_fires_when_idle_window_elapsed() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(120)  # past grace
    tracker.record_empty_poll()
    clock.advance(60)  # exactly at idle window
    assert tracker.should_release() is True


def test_tracker_does_not_fire_before_window() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=60.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    clock.advance(120)
    tracker.record_empty_poll()
    clock.advance(30)  # half the idle window
    assert tracker.should_release() is False


def test_record_empty_poll_first_wins() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=0.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    tracker.record_empty_poll()
    first = tracker.last_successful_empty_poll_at
    clock.advance(30)
    tracker.record_empty_poll()  # should be a no-op
    assert tracker.last_successful_empty_poll_at == first


def test_record_claim_resets_idle_window() -> None:
    clock = _FakeClock()
    tracker = idle_release.IdleReleaseTracker(
        idle_release.IdleReleaseConfig(idle_minutes=1.0, grace_seconds=0.0, is_service_mode=False),
        clock=clock,
    )
    tracker.mark_onboarded()
    tracker.record_empty_poll()
    clock.advance(120)
    assert tracker.should_release() is True
    tracker.record_claim()
    assert tracker.should_release() is False
    assert tracker.last_successful_empty_poll_at is None


def test_worker_sigterm_handler_raises_keyboardinterrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = {}
    real_signal = signal.signal

    class _StopInstall(Exception):
        pass

    def _capture(sig, handler):
        if sig == signal.SIGTERM:
            installed["handler"] = handler
            raise _StopInstall
        return real_signal(sig, handler)

    monkeypatch.setattr(server, "bootstrap_runtime_environment", lambda: None)
    monkeypatch.setattr(server.signal, "signal", _capture)

    with pytest.raises(_StopInstall):
        server.main()

    previous = real_signal(signal.SIGTERM, installed["handler"])
    try:
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
    finally:
        real_signal(signal.SIGTERM, previous)


def test_idle_release_exit_code_constant() -> None:
    assert worker_protocol.IDLE_RELEASE_EXIT_CODE == 75
    assert supervisor.IDLE_RELEASE_EXIT_CODE is worker_protocol.IDLE_RELEASE_EXIT_CODE


def _has_top_level_idle_assign(source_text: str) -> bool:
    tree = ast.parse(source_text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "IDLE_RELEASE_EXIT_CODE":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                        return True
    return False



def test_no_literal_idle_release_exit_code_in_server_or_supervisor() -> None:
    server_src = inspect.getsource(server)
    sup_src = inspect.getsource(supervisor)
    assert not _has_top_level_idle_assign(server_src), "server.py must import IDLE_RELEASE_EXIT_CODE, not redefine it"
    assert not _has_top_level_idle_assign(sup_src), "supervisor.py must import IDLE_RELEASE_EXIT_CODE, not redefine it"
