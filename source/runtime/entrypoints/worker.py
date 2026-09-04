"""Runtime worker entrypoint."""

from __future__ import annotations

from importlib import import_module


def _server_module():
    return import_module("source.runtime.worker.server")


def main():
    return _server_module().main()
if __name__ == "__main__":
    raise SystemExit(main() or 0)
