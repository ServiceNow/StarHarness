"""Compact turn+score progress stream for concurrent AutomationBench runs.

Same design as benchmarks.itbench.progress: a background thread prints a
one-line status every `interval` seconds; each unit prints a one-line summary
on completion. Verbose per-turn agent traces go to per-task `trace.log` files
(see `tracelog`); this is the only thing on stdout.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Unit:
    start: float
    turn: int = 0
    max_turns: int = 0


@dataclass
class ProgressReporter:
    total: int
    max_turns: int
    interval: float = 30.0
    _clock: "callable" = time.monotonic

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _running: dict = field(default_factory=dict, init=False)   # key -> _Unit
    _done: int = field(default=0, init=False)
    _strict_pass: int = field(default=0, init=False)
    _score_sum: float = field(default=0.0, init=False)
    _thread: "threading.Thread | None" = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------ updates
    def register(self, key: str) -> None:
        with self._lock:
            self._running[key] = _Unit(start=self._clock(), max_turns=self.max_turns)

    def update_turn(self, key: str, turn: int) -> None:
        with self._lock:
            u = self._running.get(key)
            if u is not None:
                u.turn = turn

    def complete(self, key: str, *, score: float, strict: bool,
                 obj_passed: int, obj_total: int, guardrails_broken: int) -> None:
        with self._lock:
            u = self._running.pop(key, None)
            self._done += 1
            self._score_sum += score
            if strict:
                self._strict_pass += 1
            elapsed = self._clock() - u.start if u else 0.0
            turn = u.turn if u else 0
            done, total = self._done, self.total
            avg = self._score_sum / self._done
        print(
            f"  ✓ [{done}/{total}] {key}  aa={score:.3f}  strict={'PASS' if strict else 'fail'}  "
            f"obj={obj_passed}/{obj_total} guard_broken={guardrails_broken}  ({turn}t, {elapsed:.0f}s)"
            f"  | running avg={avg:.3f}",
            flush=True,
        )

    # ------------------------------------------------------------------ monitor
    def _status_line(self) -> str:
        with self._lock:
            now = self._clock()
            running = [
                (k, u.turn, u.max_turns, now - u.start)
                for k, u in self._running.items()
            ]
            done, total = self._done, self.total
            avg = (self._score_sum / done) if done else 0.0
            strict = self._strict_pass
        running.sort(key=lambda r: -r[3])  # longest-running first
        shown = ", ".join(f"{k}(t{t}/{mt},{el:.0f}s)" for k, t, mt, el in running[:10])
        more = f" +{len(running) - 10} more" if len(running) > 10 else ""
        return (
            f"[status] {done}/{total} done | avg_aa={avg:.3f} (N={done}) | "
            f"strict_pass={strict} | {len(running)} running: {shown}{more}"
        )

    def _monitor(self) -> None:
        while not self._stop.wait(timeout=self.interval):
            with self._lock:
                if not self._running:
                    continue
            print(self._status_line(), flush=True)
