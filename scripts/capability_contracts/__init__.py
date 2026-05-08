"""Worker-local capability contract reporting package.

This package is deliberately stdlib-only and import-safe for report-time use.
It may import worker routing contracts, but it must not import live-test modules
or configuration.
"""

from __future__ import annotations

__all__ = [
    "loaders",
    "models",
    "report",
]
