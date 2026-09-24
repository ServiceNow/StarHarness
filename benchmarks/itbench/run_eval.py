"""Orchestrate an ITBench-AA run: download -> per task x repeats -> grade -> aggregate.

Usage (smoke test with one public scenario):
    python -m benchmarks.itbench.run_eval --scenario Scenario-1 --repeats 1 --max-turns 12

Full public split (all 40 tasks, 3 repeats, 100 turns — the AA setting):
    python -m benchmarks.itbench.run_eval --split public
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import data
from .agent_setup import _ProgressLogger, run_task
from .config import RunConfig
from .grader import grade
from .progress import ProgressReporter

REPO = Path(__file__).resolve().parents[2]
# Override with RUNS_ROOT to store runs on a persistent mount.
RUNS_DIR = Path(os.environ.get("RUNS_ROOT") or (REPO / "runs"))
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


async def eval_repeat(
    cfg: RunConfig,
    task: data.Task,
    run_dir: Path,
    rep: int,
    unit_index: int,
    judge,
    sem: asyncio.Semaphore,
    reporter: ProgressReporter,
) -> float:
    """Run + grade one (task, repeat) unit. Returns the precision-at-full-recall.

    A unit is fully isolated (own rep_dir/sandbox/output), so units run concurrently
    bounded by `sem`. Failures are recorded as score 0.0 (with an error note in
    score.json) rather than propagated, so one bad unit doesn't abort the batch.
    """
    tag = f"{task.scenario_id}/r{rep}"
    rep_dir = run_dir / task.scenario_id / f"repeat_{rep}"
    sandbox = rep_dir / "sandbox"
    async with sem:
        reporter.register(tag)
        try:
            # Offload the snapshot copy: it's a blocking shutil.copytree (~seconds on
            # the mount). Run inline on the event loop it serializes all concurrent
            # units' staging and stalls in-flight LLM I/O — so no requests reach the
            # endpoint until staging drains. to_thread lets the copies run in parallel.
            await asyncio.to_thread(data.stage_sandbox, task, sandbox)
            trace_path = rep_dir / "trace.log"
            # Pin this unit to one endpoint for all its turns (affinity → prefix-cache
            # reuse); units are spread round-robin across cfg.base_urls by unit_index.
            endpoint = cfg.endpoint_for(unit_index)
            run_coro = run_task(
                cfg, sandbox, rep_dir / "output", trace_path,
                _ProgressLogger(reporter, tag), endpoint,
            )
            # Per-unit wall-clock cap: a non-converging unit can otherwise run to the
            # 100-turn cap (~hours). On timeout the coro is cancelled (Stirrup's async
            # context managers clean up the exec env) and surfaced as a clear note.
            if cfg.unit_timeout and cfg.unit_timeout > 0:
                try:
                    output = await asyncio.wait_for(run_coro, timeout=cfg.unit_timeout)
                except asyncio.TimeoutError:
                    raise RuntimeError(f"unit timed out after {cfg.unit_timeout:.0f}s") from None
            else:
                output = await run_coro
            # grade() calls the synchronous LLM judge; offload so it doesn't block the
            # event loop (which would serialize every other concurrent unit).
            score = await asyncio.to_thread(grade, task.ground_truth_yaml, output, judge)
            payload = {
                "predictions": score.predictions, "score": score.precision_at_full_recall,
                "full_recall": score.full_recall, "tp": score.tp, "fp": score.fp,
                "matched_root_cause_sets": score.matched_root_cause_sets,
                "total_root_cause_sets": score.total_root_cause_sets, "notes": score.notes,
                "endpoint": endpoint, "harness": cfg.harness,
            }
            rep_dir.mkdir(parents=True, exist_ok=True)
            (rep_dir / "score.json").write_text(json.dumps(payload, indent=2))
            reporter.complete(
                tag, score=score.precision_at_full_recall, full_recall=score.full_recall,
                tp=score.tp, fp=score.fp, total_rc=score.total_root_cause_sets,
                preds=len(score.predictions),
            )
            return score.precision_at_full_recall
        except Exception as e:  # noqa: BLE001 — isolate a unit failure, don't kill the batch
            note = f"error: {type(e).__name__}: {e}"
            try:
                rep_dir.mkdir(parents=True, exist_ok=True)
                (rep_dir / "score.json").write_text(
                    json.dumps({"predictions": [], "score": 0.0, "full_recall": False,
                                "tp": 0, "fp": 0, "matched_root_cause_sets": 0,
                                "total_root_cause_sets": 0, "notes": note,
                                "harness": cfg.harness},
                               indent=2)
                )
            except Exception:  # noqa: BLE001 — best-effort failure record
                pass
            print(f"  ✗ {tag} FAILED -> {note}", flush=True)
            reporter.complete(tag, score=0.0, full_recall=False, tp=0, fp=0,
                              total_rc=0, preds=0)
            return 0.0
        finally:
            # Drop the staged sandbox (a full snapshot copy, tens-hundreds of MB) on
            # every path — success, failure, or timeout — so 100s of units (and dev
            # loops that time units out) don't bloat the data store. Keep trace.log +
            # score.json. Set KEEP_SANDBOX=1 to retain for debugging.
            if os.environ.get("KEEP_SANDBOX", "0") != "1":
                # Offload too: a sync rmtree here fires on every unit completion and
                # would stall the loop (freezing other units' I/O) each time.
                await asyncio.to_thread(shutil.rmtree, sandbox, ignore_errors=True)


async def eval_task(
    cfg: RunConfig, task: data.Task, task_index: int, run_dir: Path, judge,
    sem: asyncio.Semaphore, reporter: ProgressReporter,
) -> dict:
    """Run all repeats of a task concurrently (bounded by `sem`) and aggregate.

    `task_index` is the task's position in the run; combined with `rep` it forms a
    stable global unit index used to round-robin units across cfg.base_urls.
    """
    repeat_scores = await asyncio.gather(
        *[eval_repeat(cfg, task, run_dir, rep, task_index * cfg.repeats + rep,
                      judge, sem, reporter)
          for rep in range(cfg.repeats)]
    )
    avg = sum(repeat_scores) / len(repeat_scores) if repeat_scores else 0.0
    return {"id_aa": task.id_aa, "scenario_id": task.scenario_id,
            "split": task.source_split, "repeat_scores": list(repeat_scores), "mean": avg}


async def main_async(args) -> None:
    # Line-buffer stdout so orchestrator-level progress streams to a redirected
    # log (`> run.log`) live; per-task agent traces stream via their own files.
    sys.stdout.reconfigure(line_buffering=True)
    cfg = RunConfig()
    if args.repeats:
        cfg.repeats = args.repeats
    if args.max_turns:
        cfg.max_turns = args.max_turns
    if args.split:
        cfg.split = args.split
    if args.max_concurrency:
        cfg.max_concurrency = args.max_concurrency
    if args.unit_timeout is not None:
        cfg.unit_timeout = args.unit_timeout
    if args.harness:
        cfg.harness = args.harness
    if cfg.harness != "stirrup":
        raise SystemExit("ITBench adapter supports only the vendored Stirrup harness")

    # Self-hosted endpoints may not have a LiteLLM price-map entry. Do not register public
    # OpenAI models as zero-cost because that would make the run's cost reporting incorrect.
    if all("api.openai.com" not in url for url in cfg.base_urls):
        cfg.register_zero_cost_model()

    # All blocking work (sandbox copy, Stirrup's exec-dir upload, the LLM judge,
    # cleanup rmtree) runs via asyncio.to_thread on the loop's default executor,
    # whose default size is only cpu_count+4 (~12 here). With up to max_concurrency
    # units offloading at once that would re-serialize the copies, so widen the pool
    # to cover all in-flight units (plus headroom for judge + cleanup threads).
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=max(32, cfg.max_concurrency * 2)))

    judge = cfg.build_judge()
    judge_desc = (
        f"{cfg.judge_model} (effort={cfg.judge_reasoning_effort}, key={'set' if cfg.judge_api_key else 'MISSING'})"
        if judge is not None
        else "deterministic regex matcher"
    )
    if len(cfg.base_urls) > 1:
        print(f"model={cfg.litellm_model} base_urls ({len(cfg.base_urls)}, load-balanced per unit):")
        for u in cfg.base_urls:
            print(f"  - {u}")
    else:
        print(f"model={cfg.litellm_model} base_url={cfg.base_url}")
    print(f"harness={cfg.harness} split={cfg.split} repeats={cfg.repeats} "
          f"max_turns={cfg.max_turns}")
    print(f"judge={judge_desc}")
    print(f"max_concurrency={cfg.max_concurrency} "
          f"unit_timeout={cfg.unit_timeout or 'off'}")

    # Explicit scenario selection (single --scenario or comma-list --scenarios).
    # When set, it scopes both the HF download AND the task list, in the requested
    # order — this is how the dev-subset loop pins its 10 scenarios in one command.
    scenario_filter: list[str] | None = None
    # CLI --scenario / --scenarios take precedence; SCENARIOS is the environment
    # fallback. Empty or unset means the whole requested split.
    scenarios_raw = args.scenarios or os.environ.get("SCENARIOS", "")
    if args.scenario:
        scenario_filter = [args.scenario]
    elif scenarios_raw.strip():
        scenario_filter = [s.strip() for s in scenarios_raw.split(",") if s.strip()]

    # One-time download: called once here, before any fan-out. With ITBENCH_DATA
    # on a persistent mount, the snapshot is reused across job re-runs too. A
    # scenario_filter scopes allow_patterns so a smoke run does not pull the full dataset.
    print(f"downloading data (root={cfg.data_root or 'HF cache'}, "
          f"scenarios={scenario_filter or cfg.split})...", flush=True)
    sre_root = data.download(cfg.data_root or None, scenarios=scenario_filter)
    tasks = data.load_tasks(sre_root, split=cfg.split if not scenario_filter else "all")
    if scenario_filter:
        order = {s: i for i, s in enumerate(scenario_filter)}
        tasks = [t for t in tasks if t.scenario_id in order]
        tasks.sort(key=lambda t: order[t.scenario_id])
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("No tasks matched. Check --scenario / --scenarios / --split / download.")

    run_dir = RUNS_DIR / (args.name or "smoke")
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
            *[eval_task(cfg, t, ti, run_dir, judge, sem, reporter)
              for ti, t in enumerate(tasks)]))
    finally:
        reporter.stop()

    overall = sum(r["mean"] for r in results) / len(results)
    summary = {"model": cfg.model, "harness": cfg.harness, "split": cfg.split,
               "repeats": cfg.repeats,
               "max_turns": cfg.max_turns, "n_tasks": len(results),
               "judge": (cfg.judge_model if judge is not None else "deterministic"),
               "judge_reasoning_effort": (cfg.judge_reasoning_effort if judge is not None else None),
               "overall_score": overall, "per_task": results}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== OVERALL: {overall:.3f} over {len(results)} task(s) ===")
    print(f"Summary written to {run_dir / 'summary.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description="ITBench-AA harness evaluation")
    p.add_argument("--harness", choices=["stirrup"],
                   default=os.environ.get("AGENT_HARNESS", "stirrup"),
                   help="Agent harness under test (default: stirrup)")
    p.add_argument("--scenario", help="Single scenario id (e.g. Scenario-1) — fast smoke fetch")
    p.add_argument("--scenarios",
                   help="Comma-separated scenario ids (e.g. Scenario-6,Scenario-8,...) — "
                        "scopes the download + task list to exactly these, in order. "
                        "Use for the dev-subset loop. Ignored if --scenario is set.")
    p.add_argument("--unit-timeout", type=float,
                   help="Per-(task,repeat) wall-clock cap in seconds; exceeding it scores "
                        "the unit 0.0 with a note (default 0 = no timeout; env UNIT_TIMEOUT).")
    p.add_argument("--split", choices=["public", "private", "all"], help="Task split")
    p.add_argument("--repeats", type=int, help="Repeats per task (AA uses 3)")
    p.add_argument("--max-turns", type=int, help="Turn cap per task (AA uses 100)")
    p.add_argument("--max-concurrency", type=int,
                   help="Max (task, repeat) units run concurrently (default 4)")
    p.add_argument("--limit", type=int, help="Cap number of tasks")
    p.add_argument("--name", help="Run name (subdir under runs/)")
    args = p.parse_args()
    run_name = args.name or "smoke"
    if not _RUN_NAME.fullmatch(run_name):
        p.error("--name must contain only letters, digits, dot, underscore, or hyphen")
    for label in ("repeats", "max_turns", "max_concurrency"):
        value = getattr(args, label)
        if value is not None and value < 1:
            p.error(f"--{label.replace('_', '-')} must be positive")
    if args.unit_timeout is not None and args.unit_timeout < 0:
        p.error("--unit-timeout must be nonnegative")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
