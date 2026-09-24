"""Proposer wrapper: drive `omp` (oh-my-pi) non-interactively and parse its session.

Analog of the evolving-harness reference's `claude_wrapper.py`, but for omp. omp has no
stream-json-to-a-clean-object mode (`--mode json` emits verbose per-delta state), so we run
omp in normal text mode with an isolated `--session-dir` and parse the compact session
`.jsonl` it writes (one event per line) for tool calls, token usage, and the final text.

Usage as a library:
    from omp_wrapper import run
    result = run(task_file="…/task.md", cwd=".", log_dir="…/proposer_logs/iter1",
                 model="openai/gpt-5.4", prior_path="prompts/proposer_prior.md",
                 timeout_s=2400)

Usage as a CLI smoke test:
    python omp_wrapper.py --task-text "List the files in itbench/ then stop." \
        --model openai/gpt-5.4 --timeout 120
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Locating the omp binary
# --------------------------------------------------------------------------------------

def resolve_omp_bin() -> str:
    """Find the omp executable. omp is installed via mise and may not be on PATH."""
    env = os.environ.get("OMP_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("omp")
    if found:
        return found
    # mise shim / install locations
    candidates = [
        Path.home() / ".local/share/mise/shims/omp",
        Path.home() / ".local/share/mise/installs/github-can1357-oh-my-pi/latest/omp",
        Path.home() / ".local/bin/omp",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # last resort: try `mise which omp`
    try:
        out = subprocess.run(["mise", "which", "omp"], capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "omp"  # will fail loudly if truly absent


# --------------------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------------------

@dataclass
class SessionResult:
    returncode: int
    timed_out: bool
    duration_s: float
    text: str                       # final assistant text
    tool_calls: list[dict] = field(default_factory=list)  # [{"name","summary"}]
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    model: str = ""
    session_path: str | None = None
    log_path: str | None = None
    stdout_tail: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# --------------------------------------------------------------------------------------
# Session parsing
# --------------------------------------------------------------------------------------

def find_latest_session(session_dir: Path) -> Path | None:
    """Return the most-recently-modified *.jsonl under session_dir (recursively)."""
    files = list(session_dir.rglob("*.jsonl"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _content_blocks(msg: dict) -> list[dict]:
    content = msg.get("content")
    return content if isinstance(content, list) else []


def parse_session(session_path: Path) -> dict:
    """Parse an omp session .jsonl into {text, tool_calls, usage, model}.

    Session events are one JSON object per line. We care about `message` events whose
    `message.role == "assistant"`: their content blocks carry text and `toolCall`s, and the
    final assistant event carries cumulative `usage`.
    """
    final_text = ""
    tool_calls: list[dict] = []
    input_tokens = output_tokens = total_tokens = 0
    cost = 0.0
    model = ""

    try:
        lines = session_path.read_text(errors="replace").splitlines()
    except OSError:
        return {"text": "", "tool_calls": [], "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "cost": 0.0, "model": ""}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "model_change" and ev.get("model"):
            model = ev["model"]
        if ev.get("type") != "message":
            continue
        msg = ev.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        if msg.get("model"):
            model = msg["model"]
        # tool calls + latest text
        text_parts = []
        for block in _content_blocks(msg):
            btype = block.get("type")
            if btype == "toolCall":
                args = block.get("arguments") or {}
                # `i` is omp's short intent label for the call; fall back to a compact dump
                summary = args.get("i") or args.get("summary") or ""
                if not summary:
                    compact = {k: v for k, v in args.items() if k not in ("i", "summary")}
                    summary = json.dumps(compact)[:160]
                tool_calls.append({"name": block.get("name", "?"), "summary": str(summary)[:200]})
            elif btype == "text" and block.get("text"):
                text_parts.append(block["text"])
        if text_parts:
            final_text = "\n".join(text_parts)
        # usage (assistant events carry cumulative usage in the event or message)
        usage = None
        ev_evt = ev.get("assistantMessageEvent") or {}
        for src in (msg.get("usage"), ev_evt.get("usage")):
            if isinstance(src, dict):
                usage = src
        if usage:
            input_tokens = usage.get("input", input_tokens) or input_tokens
            output_tokens = usage.get("output", output_tokens) or output_tokens
            total_tokens = usage.get("totalTokens", total_tokens) or total_tokens
            c = usage.get("cost")
            if isinstance(c, dict):
                cost = c.get("total", cost) or cost

    return {"text": final_text, "tool_calls": tool_calls, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "total_tokens": total_tokens, "cost": cost,
            "model": model}


# --------------------------------------------------------------------------------------
# Running omp
# --------------------------------------------------------------------------------------

def run(
    *,
    task_file: str | Path | None = None,
    task_text: str | None = None,
    cwd: str | Path = ".",
    log_dir: str | Path,
    model: str = "openai/gpt-5.4",
    prior_path: str | Path | None = None,
    timeout_s: int = 2400,
    thinking: str = "medium",
    extra_flags: list[str] | None = None,
) -> SessionResult:
    """Invoke omp non-interactively on a task and parse the resulting session.

    Exactly one of `task_file` / `task_text` must be provided. The task is passed to omp as an
    `@file` message (robust to very large prompts). `prior_path`, if given, is appended to the
    system prompt. All omp session files land under `log_dir/session/` for isolated parsing.
    """
    if (task_file is None) == (task_text is None):
        raise ValueError("provide exactly one of task_file or task_text")

    cwd = Path(cwd).resolve()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    session_dir = log_dir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Materialize the task to a file so we can use omp's @file message syntax (avoids ARG_MAX).
    tmp_task: Path | None = None
    if task_text is not None:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".md", dir=log_dir, delete=False)
        tmp.write(task_text)
        tmp.close()
        tmp_task = Path(tmp.name)
        task_path = tmp_task
    else:
        task_path = Path(task_file).resolve()

    omp = resolve_omp_bin()
    cmd = [
        omp, "-p",
        "--approval-mode", "yolo",
        "--model", model,
        "--thinking", thinking,
        "--no-title",                       # avoid title-gen using a mis-resolving smol role
        "--no-skills",                      # disable global skill discovery (e.g. ~/.claude/skills/stop-slop)
        "--session-dir", str(session_dir),
        "--max-time", str(timeout_s),       # omp's own wall-clock cap
        "--cwd", str(cwd),
    ]
    prior_text = ""
    if prior_path:
        prior_text = Path(prior_path).read_text()
        cmd += ["--append-system-prompt", prior_text]
    if extra_flags:
        cmd += extra_flags
    cmd += [f"@{task_path}"]

    # --- Persist the full INPUT side (nothing truncated): exact command, the prior, and
    # the exact task prompt sent. The prior value is elided from the displayed command only
    # because it is stored verbatim in prior.md alongside. ---
    display_cmd, skip = [], False
    for c in cmd:
        if skip:
            display_cmd.append("<prior.md>")
            skip = False
            continue
        display_cmd.append(c)
        if c == "--append-system-prompt":
            skip = True
    (log_dir / "request.json").write_text(json.dumps({
        "command": display_cmd, "model": model, "thinking": thinking,
        "timeout_s": timeout_s, "cwd": str(cwd),
        "prior_file": "prior.md" if prior_path else None,
        "task_file": "task_sent.md",
    }, indent=2))
    if prior_text:
        (log_dir / "prior.md").write_text(prior_text)
    try:
        (log_dir / "task_sent.md").write_text(Path(task_path).read_text(errors="replace"))
    except OSError:
        pass

    log_path = log_dir / "stdout.log"
    start = time.monotonic()
    timed_out = False
    # Backstop timeout a bit beyond omp's own --max-time.
    hard_timeout = timeout_s + 120
    with open(log_path, "w") as logf:
        logf.write(f"$ {' '.join(display_cmd)}\n\n")
        logf.flush()
        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT,
                timeout=hard_timeout, text=True,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
    duration = time.monotonic() - start

    if tmp_task is not None:
        tmp_task.unlink(missing_ok=True)

    session_path = find_latest_session(session_dir)
    parsed = parse_session(session_path) if session_path else {
        "text": "", "tool_calls": [], "input_tokens": 0, "output_tokens": 0,
        "total_tokens": 0, "cost": 0.0, "model": model}

    tail = ""
    try:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
    except OSError:
        pass

    return SessionResult(
        returncode=returncode,
        timed_out=timed_out,
        duration_s=round(duration, 1),
        text=parsed["text"],
        tool_calls=parsed["tool_calls"],
        tool_call_count=len(parsed["tool_calls"]),
        input_tokens=parsed["input_tokens"],
        output_tokens=parsed["output_tokens"],
        total_tokens=parsed["total_tokens"],
        cost=parsed["cost"],
        model=parsed["model"] or model,
        session_path=str(session_path) if session_path else None,
        log_path=str(log_path),
        stdout_tail=tail,
    )


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the omp proposer wrapper.")
    ap.add_argument("--task-text", help="Inline task prompt.")
    ap.add_argument("--task-file", help="Path to a task prompt file.")
    ap.add_argument("--model", default="openai/gpt-5.4")
    ap.add_argument("--prior", default=None, help="Path to a system-prompt prior file.")
    ap.add_argument("--log-dir", default="evolving_runs/_wrapper_smoke")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    res = run(
        task_file=args.task_file,
        task_text=args.task_text or (None if args.task_file else "Say the single word: pong"),
        log_dir=args.log_dir,
        model=args.model,
        prior_path=args.prior,
        timeout_s=args.timeout,
    )
    print(res.to_json())


if __name__ == "__main__":
    _cli()
