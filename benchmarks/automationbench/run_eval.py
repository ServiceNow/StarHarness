"""Orchestrate an AutomationBench-AA run: load tasks -> per task x repeats -> grade -> aggregate.

Usage (smoke test, one finance task):
    python -m benchmarks.automationbench.run_eval --tasks 4001

Full finance baseline (AutomationBench-AA setting: 1 run, 50-turn cap):
    python -m benchmarks.automationbench.run_eval --domain finance
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

# A buggy assertion must fail that assertion, not crash the run (upstream
# non-strict mode; strict mode is their default for development).
os.environ.setdefault("AUTOMATIONBENCH_STRICT_ASSERTIONS", "0")

from . import data
from .config import RunConfig
from .grader import grade
from .progress import ProgressReporter

HERE = Path(__file__).resolve().parents[2]
# Override with RUNS_ROOT to land runs elsewhere (e.g. a persistent mount).
RUNS_DIR = Path(os.environ.get("RUNS_ROOT") or (HERE / "runs"))
SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _harness_metadata(harness: str) -> dict:
    """Hook for recording harness provenance alongside scores."""
    return {}


async def eval_repeat(
    cfg: RunConfig,
    task: data.Task,
    run_dir: Path,
    rep: int,
    unit_index: int,
    sem: asyncio.Semaphore,
    reporter: ProgressReporter,
) -> dict:
    """Run + grade one (task, repeat) unit. Returns the score payload.

    A unit failure is recorded as score 0.0 (AA: infrastructure errors score 0)
    rather than propagated, so one bad unit doesn't abort the batch.
    """
    tag = f"{task.task_id}/r{rep}"
    rep_dir = run_dir / task.task_id / f"repeat_{rep}"
    async with sem:
        reporter.register(tag)
        try:
            trace_path = rep_dir / "trace.log"
            # Pin this unit to one endpoint for all its turns (affinity -> prefix-cache
            # reuse); units are spread round-robin across cfg.base_urls by unit_index.
            endpoint = cfg.endpoint_for(unit_index)
            from .agent_setup import _ProgressLogger
            from .agent_setup import run_task as run_stirrup_task

            run_coro = run_stirrup_task(
                cfg, task.for_agent(), trace_path, _ProgressLogger(reporter, tag), endpoint
            )
            if cfg.unit_timeout and cfg.unit_timeout > 0:
                try:
                    world = await asyncio.wait_for(run_coro, timeout=cfg.unit_timeout)
                except asyncio.TimeoutError:
                    raise RuntimeError(f"unit timed out after {cfg.unit_timeout:.0f}s") from None
            else:
                world = await run_coro
            result = grade(world, task.initial_state, task.assertions)
            payload = (result.to_dict()
                       | {"endpoint": endpoint, "harness": cfg.harness}
                       | _harness_metadata(cfg.harness))
            rep_dir.mkdir(parents=True, exist_ok=True)
            (rep_dir / "score.json").write_text(json.dumps(payload, indent=2))
            reporter.complete(
                tag, score=result.aa_score, strict=result.task_completed_correctly,
                obj_passed=result.objectives_passed, obj_total=result.objectives_total,
                guardrails_broken=result.guardrails_broken,
            )
            return payload
        except Exception as e:  # noqa: BLE001 — isolate a unit failure, don't kill the batch
            note = f"error: {type(e).__name__}: {e}"
            payload = {
                "aa_score": 0.0, "partial_credit": 0.0, "task_completed_correctly": False,
                "objectives_total": 0, "objectives_passed": 0,
                "guardrails_total": 0, "guardrails_broken": 0,
                "notes": note, "assertions": [],
                "endpoint": cfg.endpoint_for(unit_index), "harness": cfg.harness,
                **_harness_metadata(cfg.harness),
            }
            try:
                rep_dir.mkdir(parents=True, exist_ok=True)
                (rep_dir / "score.json").write_text(json.dumps(payload, indent=2))
            except Exception:  # noqa: BLE001 — best-effort failure record
                pass
            print(f"  ✗ {tag} FAILED -> {note}", flush=True)
            reporter.complete(tag, score=0.0, strict=False, obj_passed=0,
                              obj_total=0, guardrails_broken=0)
            return payload


async def eval_task(
    cfg: RunConfig, task: data.Task, task_index: int, run_dir: Path,
    sem: asyncio.Semaphore, reporter: ProgressReporter,
) -> dict:
    """Run all repeats of a task concurrently (bounded by `sem`) and aggregate."""
    repeat_payloads = await asyncio.gather(
        *[eval_repeat(cfg, task, run_dir, rep, task_index * cfg.repeats + rep, sem, reporter)
          for rep in range(cfg.repeats)]
    )
    aa_scores = [p["aa_score"] for p in repeat_payloads]
    return {
        "task_id": task.task_id,
        "name": task.name,
        "repeat_scores": aa_scores,
        "mean": sum(aa_scores) / len(aa_scores) if aa_scores else 0.0,
        "partial_credit": (sum(p["partial_credit"] for p in repeat_payloads)
                           / len(repeat_payloads) if repeat_payloads else 0.0),
        "task_completed_correctly": all(p["task_completed_correctly"] for p in repeat_payloads),
        "guardrails_broken": max((p["guardrails_broken"] for p in repeat_payloads), default=0),
    }


async def main_async(args) -> None:
    # Line-buffer stdout so orchestrator-level progress streams to a redirected log live.
    sys.stdout.reconfigure(line_buffering=True)
    cfg = RunConfig()
    cfg.harness = args.harness
    if args.repeats:
        cfg.repeats = args.repeats
    if args.max_turns is not None:
        cfg.max_turns = args.max_turns
    if args.max_concurrency:
        cfg.max_concurrency = args.max_concurrency
    if args.domain:
        cfg.domain = args.domain
    if args.unit_timeout is not None:
        cfg.unit_timeout = args.unit_timeout
    if cfg.max_turns <= 0:
        raise SystemExit("--max-turns must be positive")
    if cfg.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if cfg.max_concurrency <= 0:
        raise SystemExit("--max-concurrency must be positive")
    if cfg.unit_timeout < 0:
        raise SystemExit("--unit-timeout must not be negative")

    cfg.register_zero_cost_model()

    if len(cfg.base_urls) > 1:
        print(f"model={cfg.litellm_model} base_urls ({len(cfg.base_urls)}, load-balanced per unit):")
        for u in cfg.base_urls:
            print(f"  - {u}")
    else:
        print(f"model={cfg.litellm_model} base_url={cfg.base_url}")
    print(f"harness={cfg.harness} domain={cfg.domain} repeats={cfg.repeats} max_turns={cfg.max_turns}")
    print(f"max_concurrency={cfg.max_concurrency} unit_timeout={cfg.unit_timeout or 'off'}")

    tasks = data.load_tasks(cfg.domain)
    task_ids = None
    tasks_raw = args.tasks or os.environ.get("TASKS", "")
    if tasks_raw.strip():
        task_ids = [t.strip() for t in tasks_raw.split(",") if t.strip()]
    tasks = data.filter_tasks(tasks, task_ids=task_ids, limit=args.limit)
    if not tasks:
        raise SystemExit("No tasks matched. Check --tasks / --domain / --limit.")

    run_name = args.name or "smoke"
    if not SAFE_RUN_NAME.fullmatch(run_name):
        raise SystemExit(f"invalid --name: {run_name!r}")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    n_units = len(tasks) * cfg.repeats
    print(f"running {len(tasks)} task(s) x {cfg.repeats} repeat(s) "
          f"= {n_units} unit(s), up to {cfg.max_concurrency} concurrent", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    sem = asyncio.Semaphore(cfg.max_concurrency)
    reporter = ProgressReporter(total=n_units, max_turns=cfg.max_turns,
                                interval=float(os.environ.get("PROGRESS_INTERVAL", "30")))
    reporter.start()
    try:
        results = list(await asyncio.gather(
            *[eval_task(cfg, t, ti, run_dir, sem, reporter)
              for ti, t in enumerate(tasks)]))
    finally:
        reporter.stop()

    overall = sum(r["mean"] for r in results) / len(results)
    mean_partial = sum(r["partial_credit"] for r in results) / len(results)
    strict_rate = sum(1 for r in results if r["task_completed_correctly"]) / len(results)
    summary = {
        "model": cfg.model, "harness": cfg.harness, "domain": cfg.domain, "repeats": cfg.repeats,
        "max_turns": cfg.max_turns, "n_tasks": len(results),
        "overall_score": overall,
        "mean_partial_credit": mean_partial,
        "strict_pass_rate": strict_rate,
        "per_task": results,
    } | _harness_metadata(cfg.harness)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== OVERALL: aa={overall:.3f} partial={mean_partial:.3f} "
          f"strict={strict_rate:.3f} over {len(results)} task(s) ===")
    print(f"Summary written to {run_dir / 'summary.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description="AutomationBench-AA harness evaluation")
    p.add_argument("--harness", choices=["stirrup"],
                   default=os.environ.get("AGENT_HARNESS", "stirrup"))
    p.add_argument("--domain", help="AutomationBench domain (default: finance)")
    p.add_argument("--tasks",
                   help="Comma-separated task ids (example_ids, e.g. 4001,4017) — "
                        "scopes the run to exactly these, in order. Env fallback: TASKS.")
    p.add_argument("--repeats", type=int, help="Repeats per task (AA uses 1)")
    p.add_argument("--max-turns", type=int, help="Turn cap per task (AA uses 50)")
    p.add_argument("--max-concurrency", type=int,
                   help="Max (task, repeat) units run concurrently (default 8)")
    p.add_argument("--unit-timeout", type=float,
                   help="Per-(task,repeat) wall-clock cap in seconds; exceeding it scores "
                        "the unit 0.0 with a note (default 0 = no timeout; env UNIT_TIMEOUT).")
    p.add_argument("--limit", type=int, help="Cap number of tasks")
    p.add_argument("--name", help="Run name (subdir under runs/)")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
