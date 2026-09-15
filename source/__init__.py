"""Neutral Worker package boundary.

Importing ``source`` must not execute optional model, media, LoRA, heartbeat,
or database bootstrap code. Supported Runtime startup imports the neutral
supervisor first; legacy modules must import their own optional dependencies at
the point where they are explicitly used.
"""

from __future__ import annotations
