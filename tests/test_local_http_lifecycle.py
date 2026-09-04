from __future__ import annotations

from pathlib import Path

from source.runtime.worker import local_http, server


def test_supported_worker_keeps_local_http_materialization_boundary() -> None:
    server_source = server.__file__
    assert server_source is not None
    source_text = Path(server_source).read_text(encoding="utf-8")
    assert "start_local_http_server" in source_text
    assert callable(local_http.start_local_http_server)
    assert not hasattr(server, "_is_local_worker_mode")
