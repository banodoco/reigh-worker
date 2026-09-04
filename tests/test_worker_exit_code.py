from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from source.runtime import supervisor


ENTRYPOINT = Path(__file__).parents[1] / "source" / "runtime" / "entrypoints" / "worker.py"


def test_worker_entrypoint_propagates_failed_preflight_exit_code(monkeypatch):
    class FailedPreflightServer:
        @staticmethod
        def main():
            return 1

    monkeypatch.setattr("importlib.import_module", lambda _name: FailedPreflightServer)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")

    assert exc_info.value.code == 1


def test_supervisor_restarts_disposable_failed_worker(tmp_path, monkeypatch):
    state_file = tmp_path / "attempts"
    child_script = tmp_path / "child.py"
    child_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "state = Path(sys.argv[1])\n"
        "attempt = int(state.read_text()) if state.exists() else 0\n"
        "state.write_text(str(attempt + 1))\n"
        "raise SystemExit(1 if attempt == 0 else 0)\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        supervisor,
        "_build_child_cmd",
        lambda _argv_tail: [sys.executable, str(child_script), str(state_file)],
    )
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)

    assert supervisor.main([]) == 0
    assert state_file.read_text(encoding="utf-8") == "2"
