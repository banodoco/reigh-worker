"""Regression tests for the ssh_bootstrap bundle-primitives refactor (T3).

These golden-string comparisons lock in that ``run_install`` and
``clone_and_install_vibecomfy`` emit the same shell commands they did before
``_uv_sync_shell`` / ``_vibecomfy_install_shell`` were extracted, so the
prebuilt variant can compose those primitives without quietly changing the
behaviour of the existing fresh path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_test.ssh_bootstrap import (
    _uv_sync_shell,
    _vibecomfy_install_shell,
    clone_and_install_vibecomfy,
    ensure_git_ref_synced,
    run_install,
)


class _CapturingSSH:
    def __init__(self) -> None:
        self.commands: list[tuple[str, int]] = []

    def execute_command(self, command: str, timeout: int = 600) -> tuple[int, str, str]:
        self.commands.append((command, timeout))
        return 0, "", ""


def _capture(callable_, *args, **kwargs) -> str:
    ssh = _CapturingSSH()
    callable_(ssh, *args, **kwargs)
    assert ssh.commands, "callable did not invoke ssh.execute_command"
    return ssh.commands[0][0]


# --------------------------------------------------------------------------- #
# Golden-string snapshots of the pre-refactor commands. These were captured
# from the pre-refactor codebase and must continue to match byte-for-byte.
# --------------------------------------------------------------------------- #

_EXPECTED_RUN_INSTALL = (
    "bash -lc 'set -euo pipefail\n"
    "apt-get update\n"
    "apt-get install -y python3.10-venv python3.10-dev build-essential ffmpeg git curl wget\n"
    "if ! command -v uv >/dev/null 2>&1; then\n"
    "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
    '  export PATH="$HOME/.local/bin:$PATH"\n'
    "fi\n"
    "cd /workspace/Reigh-Worker-LiveTest\n"
    'export PATH="$HOME/.local/bin:$PATH"\n'
    "export UV_PROJECT_ENVIRONMENT=/opt/reigh-worker-live-test-venv\n"
    "export UV_LINK_MODE=copy\n"
    "for attempt in 1 2 3; do\n"
    "  if uv sync --extra cuda124; then\n"
    "    break\n"
    "  fi\n"
    '  echo "uv sync attempt $attempt failed; cleaning partial venv and retrying"\n'
    '  rm -rf .venv "$UV_PROJECT_ENVIRONMENT"\n'
    "  sleep 5\n"
    "  if [ $attempt -eq 3 ]; then exit 1; fi\n"
    "done\n"
    "'"
)


def test_run_install_emits_byte_identical_shell():
    command = _capture(run_install, "/workspace/Reigh-Worker-LiveTest")
    assert command == _EXPECTED_RUN_INSTALL


def test_clone_and_install_vibecomfy_emits_byte_identical_shell_portable():
    command = _capture(
        clone_and_install_vibecomfy,
        repo_url="https://example.com/v.git",
        branch="main",
        workdir="/workspace/vibecomfy",
        python_path="python3.11",
        attention_profile="portable",
    )
    # Lock in every shell line in order — any drift fails the test.
    expected_lines = [
        "bash -lc 'set -euo pipefail",
        "export VIBECOMFY_ATTENTION_PROFILE=portable",
        "mkdir -p /workspace",
        "rm -rf /workspace/vibecomfy",
        "git clone --branch main --single-branch https://example.com/v.git /workspace/vibecomfy",
        "git -C /workspace/vibecomfy fetch origin main",
        "git -C /workspace/vibecomfy checkout main",
        "git -C /workspace/vibecomfy reset --hard FETCH_HEAD",
        "git -C /workspace/vibecomfy clean -ffd -e .venv/",
        'echo "VibeComfy checkout: $(git -C /workspace/vibecomfy rev-parse --short HEAD)"',
        "uv venv --seed --python python3.11 /workspace/vibecomfy/.venv",
        "uv pip install --python /workspace/vibecomfy/.venv/bin/python -e /workspace/vibecomfy",
        # uv pip install of comfyui + comfy-script
        "cd /workspace/vibecomfy",
        "test -f custom_nodes.lock",
        "/workspace/vibecomfy/.venv/bin/python -m vibecomfy.cli nodes restore --lockfile custom_nodes.lock",
        "test -f /workspace/vibecomfy/template_index.json",
        "test -f /workspace/vibecomfy/workflow_corpus/manifests/coverage.json",
    ]
    for line in expected_lines:
        assert line in command, f"missing expected shell line {line!r} in:\n{command}"
    # Critical embedded literal — the comfyui pin URL must survive intact.
    assert (
        "'comfyui@git+https://github.com/peteromallet/ComfyUI.git@fix/latentupscale-model-mmap-residency'"
        in command
    )
    assert "'comfy-script[default]'" in command


def test_clone_and_install_vibecomfy_adds_sageattention_for_sage_profile():
    command = _capture(
        clone_and_install_vibecomfy,
        repo_url="https://example.com/v.git",
        branch="main",
        workdir="/workspace/vibecomfy",
        python_path="python3.11",
        attention_profile="sage",
    )
    assert "git clone --depth 1 https://github.com/thu-ml/SageAttention.git /tmp/sageattention" in command
    assert "uv pip install --python /workspace/vibecomfy/.venv/bin/python --no-build-isolation /tmp/sageattention" in command
    assert "VIBECOMFY_ATTENTION_PROFILE=sage" in command


# --------------------------------------------------------------------------- #
# _uv_sync_shell helper
# --------------------------------------------------------------------------- #


def test_uv_sync_shell_rejects_empty_extras():
    with pytest.raises(ValueError):
        _uv_sync_shell("/workspace/x", env_path="/opt/y", extras=())


def test_uv_sync_shell_default_extras_is_cuda124():
    import inspect

    sig = inspect.signature(_uv_sync_shell)
    assert sig.parameters["extras"].default == ("cuda124",)
    body = _uv_sync_shell("/workspace/x", env_path="/opt/y")
    assert "uv sync --extra cuda124" in body
    assert " --locked " not in body  # default is with_locked=False


def test_uv_sync_shell_with_locked_emits_locked_flag():
    body = _uv_sync_shell("/workspace/x", env_path="/opt/y", with_locked=True)
    assert "uv sync --locked --extra cuda124" in body


def test_uv_sync_shell_multiple_extras_renders_each():
    body = _uv_sync_shell("/workspace/x", env_path="/opt/y", extras=("cuda124", "test"))
    assert "uv sync --extra cuda124 --extra test" in body


# --------------------------------------------------------------------------- #
# ensure_git_ref_synced — no uv sync should appear in either path.
# --------------------------------------------------------------------------- #


def test_ensure_git_ref_synced_warm_path_does_not_invoke_uv_sync():
    ssh = _CapturingSSH()
    ensure_git_ref_synced(
        ssh,
        workdir="/opt/reigh-livetest-prebuilt/worker",
        repo_url="https://github.com/banodoco/Reigh-Worker.git",
        ref="main",
    )
    assert len(ssh.commands) == 1
    command = ssh.commands[0][0]
    # Expected warm-path shell sequence.
    assert "git -C /opt/reigh-livetest-prebuilt/worker fetch origin main" in command
    assert "git -C /opt/reigh-livetest-prebuilt/worker checkout main" in command
    assert "git -C /opt/reigh-livetest-prebuilt/worker reset --hard FETCH_HEAD" in command
    assert "git -C /opt/reigh-livetest-prebuilt/worker clean -ffd -e .venv/" in command
    # No uv sync, no clone.
    assert "uv sync" not in command
    assert "git clone" not in command


def test_ensure_git_ref_synced_force_clone_path_emits_full_clone_without_uv_sync():
    ssh = _CapturingSSH()
    ensure_git_ref_synced(
        ssh,
        workdir="/opt/reigh-livetest-prebuilt/worker",
        repo_url="https://github.com/banodoco/Reigh-Worker.git",
        ref="feature/x",
        force_clone=True,
    )
    assert len(ssh.commands) == 1
    command = ssh.commands[0][0]
    assert "mkdir -p /opt/reigh-livetest-prebuilt" in command
    assert "rm -rf /opt/reigh-livetest-prebuilt/worker" in command
    assert "git clone https://github.com/banodoco/Reigh-Worker.git /opt/reigh-livetest-prebuilt/worker" in command
    assert "git -C /opt/reigh-livetest-prebuilt/worker fetch origin feature/x" in command
    assert "git -C /opt/reigh-livetest-prebuilt/worker reset --hard FETCH_HEAD" in command
    # No uv sync in the git-only primitive.
    assert "uv sync" not in command
