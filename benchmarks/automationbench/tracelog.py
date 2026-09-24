"""Per-task trace routing for concurrent AutomationBench runs.

Identical mechanism to benchmarks.itbench.tracelog: Stirrup's `AgentLogger`
writes to a module-global Rich Console; we rebind it once to a proxy that
dispatches to a per-task Console selected via a ContextVar, so each concurrent
(task, repeat) unit's trace lands in its own file.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import stirrup.utils.logging as slog
from rich.console import Console

# Console used when no unit is bound (e.g. import-time / orchestrator-level work).
_fallback: Console = Console()
_current: ContextVar[Console] = ContextVar("automationbench_trace_console", default=_fallback)


class _ConsoleProxy:
    """Forwards all attribute access to the ContextVar-selected Console."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 — passthrough proxy
        return getattr(_current.get(), name)


_installed = False


def install() -> None:
    """Rebind Stirrup's global console to the per-task proxy (idempotent)."""
    global _installed
    if not _installed:
        slog.console = _ConsoleProxy()
        _installed = True


@contextlib.contextmanager
def trace_to(path: Path, width: int = 100):
    """Bind a fresh file-backed Console for the current task, writing to `path`.

    The file is line-buffered so `tail -f` streams the trace live. Plain text,
    no ANSI color/markup — these are log files, not a TTY.
    """
    install()
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8", buffering=1)  # line-buffered for tail -f
    console = Console(
        file=f, width=width, force_terminal=False, no_color=True,
        highlight=False, soft_wrap=False,
    )
    token = _current.set(console)
    try:
        yield console
    finally:
        _current.reset(token)
        f.flush()
        f.close()
