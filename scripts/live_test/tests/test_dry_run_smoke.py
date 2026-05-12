"""Dry-run smoke tests (T16).

Cover the dry-run code paths for ``--variant prebuilt`` and ``--variant auto``
without requiring RunPod credentials or a real volume. These tests exercise
the CLI from ``main()`` end-to-end via in-process invocation rather than a
subprocess, so the harness is fully reproducible in CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_test import main as live_test_main
from scripts.live_test import variant_prebuilt


def _patch_dry_run_env(monkeypatch, *, allow_select=False):
    """Force the early dry-run branch by unsetting REIGH_LIVE_TEST_TOKEN.

    Also avoids any accidental real API calls by short-circuiting helpers that
    would otherwise reach RunPod.
    """
    monkeypatch.delenv("REIGH_LIVE_TEST_TOKEN", raising=False)
    # Override get_env so the early dry-run branch is taken (no token).
    monkeypatch.setattr(
        "scripts.live_test.variant_prebuilt.config.get_env",
        lambda name, default=None: None if name == "REIGH_LIVE_TEST_TOKEN" else default,
    )
    if not allow_select:
        monkeypatch.setattr(
            "scripts.live_test._shared.select_network_volume",
            lambda api_key, *, name_prefix, data_center_filter=None: None,
        )


def test_variant_prebuilt_dry_run_prints_contract_paths(monkeypatch, capsys):
    """`--variant prebuilt --dry-run` completes without credentials and shows
    contract paths (NOT the fresh defaults)."""
    _patch_dry_run_env(monkeypatch)
    rc = live_test_main.main(
        [
            "--variant",
            "prebuilt",
            "--dry-run",
            "--backend",
            "vibecomfy",
            "--case",
            "z_image_turbo",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    # Prebuilt runtime paths visible in the plan, not the fresh defaults.
    assert "/opt/reigh-livetest-prebuilt/worker" in captured
    assert "/opt/reigh-worker-live-test-venv" in captured
    assert "/opt/reigh-livetest-prebuilt/vibecomfy" in captured
    # Fresh-only path must NOT appear.
    assert "/workspace/Reigh-Worker-LiveTest" not in captured


def test_variant_auto_dry_run_falls_back_to_fresh_when_no_volume(monkeypatch, capsys):
    """auto + no volume → prebuilt_unavailable JSON log + dispatch to fresh."""
    # Force the auto preflight to return None (no volume found).
    monkeypatch.setattr(
        "scripts.live_test._shared.select_network_volume",
        lambda api_key, *, name_prefix, data_center_filter=None: None,
    )
    # Ensure the fresh path's early dry-run branch fires (no token).
    monkeypatch.delenv("REIGH_LIVE_TEST_TOKEN", raising=False)
    monkeypatch.setattr(
        "scripts.live_test.variant_fresh.config.get_env",
        lambda name, default=None: None if name == "REIGH_LIVE_TEST_TOKEN" else default,
    )
    # The CLI's auto handler reads RUNPOD_API_KEY from config.get_env; force a value
    # so the preflight runs select_network_volume rather than short-circuiting
    # on missing-api-key.
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(
        "scripts.live_test.main.config.get_env",
        lambda name, default=None: "test-key" if name == "RUNPOD_API_KEY" else None if name == "REIGH_LIVE_TEST_TOKEN" else default,
    )
    rc = live_test_main.main(
        [
            "--variant",
            "auto",
            "--dry-run",
            "--backend",
            "vibecomfy",
            "--case",
            "z_image_turbo",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    # The auto preflight emits a structured prebuilt_unavailable log.
    assert "prebuilt_unavailable" in captured
    # And the fresh variant's dry-run plan must have been printed (not prebuilt).
    assert "Variant: fresh" in captured


def test_variant_auto_dry_run_picks_prebuilt_when_volume_reported(monkeypatch, capsys):
    """auto + matching volume → prebuilt_available JSON log + dispatch to prebuilt."""
    monkeypatch.setattr(
        "scripts.live_test._shared.select_network_volume",
        lambda api_key, *, name_prefix, data_center_filter=None: (
            "vol-123",
            "reigh-livetest-prebuilt-portable-eu-no-1",
            "eu-no-1",
        ),
    )
    monkeypatch.delenv("REIGH_LIVE_TEST_TOKEN", raising=False)
    monkeypatch.setattr(
        "scripts.live_test.variant_prebuilt.config.get_env",
        lambda name, default=None: None if name == "REIGH_LIVE_TEST_TOKEN" else default,
    )
    monkeypatch.setattr(
        "scripts.live_test.main.config.get_env",
        lambda name, default=None: "test-key" if name == "RUNPOD_API_KEY" else None if name == "REIGH_LIVE_TEST_TOKEN" else default,
    )
    rc = live_test_main.main(
        [
            "--variant",
            "auto",
            "--dry-run",
            "--backend",
            "vibecomfy",
            "--case",
            "z_image_turbo",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "prebuilt_available" in captured
    assert "Variant: prebuilt" in captured
    # And the prebuilt contract paths are surfaced in the dry-run plan.
    assert "/opt/reigh-livetest-prebuilt/worker" in captured


def test_rl_prebuilt_subcommands_have_help_text():
    """All four prebuilt subcommands expose --help."""
    from runpod_lifecycle import cli as rl_cli

    parser = rl_cli.build_parser()
    # Verify the prebuilt group + four subcommands are registered.
    actions = {a.dest: a for a in parser._actions}
    assert "cmd" in actions
    # Inspect subparser choices via the prebuilt subparser action.
    sub_action = None
    for action in parser._subparsers._actions if parser._subparsers else []:
        if getattr(action, "choices", None) and "prebuilt" in action.choices:
            sub_action = action.choices["prebuilt"]
            break
    assert sub_action is not None, "prebuilt subparser not registered"
    # Grab its own sub-subparser choices.
    for action in sub_action._actions:
        if getattr(action, "choices", None):
            assert set(action.choices.keys()) == {"build", "inspect", "invalidate", "list"}
            break
    else:
        pytest.fail("prebuilt sub-subparser not found")


def test_rl_prebuilt_build_requires_volume_name_and_data_center():
    """`rl prebuilt build` without --volume-name or --data-center exits with error."""
    from runpod_lifecycle import cli as rl_cli

    parser = rl_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["prebuilt", "build"])
    # Specify volume but not data center → still errors.
    with pytest.raises(SystemExit):
        parser.parse_args(["prebuilt", "build", "--volume-name", "v"])
    # Specify data center but not volume → still errors.
    with pytest.raises(SystemExit):
        parser.parse_args(["prebuilt", "build", "--data-center", "EU-NO-1"])
    # Both → parses successfully.
    args = parser.parse_args(["prebuilt", "build", "--volume-name", "v", "--data-center", "EU-NO-1"])
    assert args.volume_name == "v"
    assert args.data_center == "EU-NO-1"
    assert args.python_version == "3.10"  # default
