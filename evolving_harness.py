"""Evolving-harness search loop: omp proposes edits to the Stirrup harness; a benchmark scores them.

The *proposer* is `omp` (via omp_wrapper) and the *evaluator* is a benchmark adapter
(`python -m benchmarks.<name>.run_eval`). See the benchmark's `domain_spec.md` for the
full experiment definition.

Phases:
  Phase 0  Baseline eval on the dev set  -> records selection/search frontier means.
  Phase i  Propose -> capture(git diff) -> guardrail -> validate -> smoke -> benchmark -> select.
  Final    Evaluate the winning frontier on the held-out set (if applicable).

Candidate isolation is git-based and scoped to the adapter's `editable_dirs`. Tracked edits outside
that boundary are rejected and restored. Keep = commit (frontier advances); discard/regress/crash
= `git checkout` + scoped `git clean` back to the frontier commit.

Stray files the proposer writes OUTSIDE the editable dirs are detected by diffing the untracked
set against a snapshot taken at iteration start (only genuinely NEW files count), and are removed
only if they are regular files not on PROTECTED_FILES — arbitrary files are never auto-deleted.

Run:
  set -a; . ./.env; set +a
  .venv/bin/python evolving_harness.py --benchmark automationbench --run-name demo --iterations 1
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import omp_wrapper
from benchmarks.base import AdapterValidationError, BenchmarkAdapter

# --------------------------------------------------------------------------------------
# Fixed configuration
# --------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
GROUND_TRUTH_HINTS = ("ground_truth",)                 # any changed path containing this -> reject

PROPOSER_MODEL_DEFAULT = "openai/gpt-5.4"
PYTHON = str(REPO / ".venv/bin/python")
RUNS_DIR = Path(os.environ.get("RUNS_ROOT") or (REPO / "runs"))
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TRACE_ANALYSIS_MODES = ("inline", "subagents")

# Git root may be a parent dir; strip the repo-relative prefix from all git path output.
_GIT_PREFIX = subprocess.run(
    ["git", "rev-parse", "--show-prefix"], cwd=str(REPO), capture_output=True, text=True,
).stdout.strip()


def _gp(path: str) -> str:
    """Strip the git prefix so paths are relative to REPO (matching editable_dirs)."""
    if _GIT_PREFIX and path.startswith(_GIT_PREFIX):
        return path[len(_GIT_PREFIX):]
    return path


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------

# Set once in main() when the run dir is known, so logs persist in the repo.
_RUN_LOG_FH = None
_EVAL_LOG_DIR = None


def log(msg: str) -> None:
    line = f"[evolve] {msg}"
    print(line, flush=True)
    if _RUN_LOG_FH is not None:
        _RUN_LOG_FH.write(line + "\n")
        _RUN_LOG_FH.flush()


def tail(path: Path | None, n_chars: int = 800) -> str:
    """Last n_chars of a log file (for failure diagnostics)."""
    if path is None:
        return "(no log)"
    try:
        return path.read_text(errors="replace")[-n_chars:]
    except OSError:
        return "(log unavailable)"


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True, check=check)


def eval_env(adapter: BenchmarkAdapter) -> dict:
    env = os.environ.copy()
    env.update(adapter.agent_env)
    venv_bin = str(REPO / ".venv" / "bin")
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(REPO / ".venv")
    return env


def load_adapter(name: str) -> BenchmarkAdapter:
    """Dynamically load a benchmark adapter by name."""
    if not name.isidentifier() or name.lower() != name:
        raise AdapterValidationError(
            f"benchmark must be a lowercase Python identifier: {name!r}"
        )
    mod = importlib.import_module(f"benchmarks.{name}.benchmark")
    return mod.Adapter()


def trace_analysis_prompt(mode: str, scenarios: list[str], max_wave: int) -> str:
    """Return optional proposer instructions for per-trace subagent analysis."""
    if mode == "inline":
        return ""
    if mode != "subagents":
        raise ValueError(f"unknown trace analysis mode: {mode}")
    if max_wave < 1:
        raise ValueError("trace subagent wave size must be positive")

    scenario_list = ", ".join(scenarios)
    return f"""## Trace analysis mode: per-trace subagents

Analyze the search evidence with one isolated OMP `task` subagent per scenario before editing.
Spawn {len(scenarios)} workers for these scenarios: {scenario_list}. Dispatch them in waves of at
most {max_wave} workers.

- Give each worker the evidence for one scenario and no evidence from another scenario.
- Ask each worker to report the first causal agent error, recurring failure candidates, supporting
  trace evidence, and a general harness intervention. Workers must not edit files or spawn workers.
- Wait for every worker. Check that each scenario produced a report.
- As the parent proposer, aggregate the reports across scenarios. Choose one general intervention,
  make the candidate edit, and write `pending_eval.json`.

This mode spends one subagent call per search trace. Keep all selection and held-out evidence hidden.
"""


# --------------------------------------------------------------------------------------
# Git baseline + candidate isolation (scoped to adapter.editable_dirs)
# --------------------------------------------------------------------------------------

def ensure_gitignore() -> None:
    """Keep pycache/venv/evolving_runs out of scoped baseline commits so diffs stay clean."""
    gi = REPO / ".gitignore"
    lines = gi.read_text().splitlines() if gi.exists() else []
    needed = ["__pycache__/", "*.pyc", ".venv/", "evolving_runs/", ".DS_Store"]
    missing = [n for n in needed if n not in lines]
    if missing:
        gi.write_text("\n".join(lines + missing) + "\n")


def head_sha() -> str | None:
    r = git("rev-parse", "--verify", "HEAD", check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def commit_baseline(adapter: BenchmarkAdapter, message: str) -> str:
    """Commit the current state of editable_dirs as a (new) baseline. Returns the commit sha."""
    dirs = adapter.editable_dirs
    git("add", "--", *dirs)
    staged = git("diff", "--cached", "--quiet", "--", *dirs, check=False)
    if staged.returncode != 0:  # something staged
        git("-c", "user.name=evolving-harness", "-c", "user.email=meta@harness.local",
            "commit", "-m", message, "--", *dirs)
    return head_sha() or ""


def capture_diff(adapter: BenchmarkAdapter, dest: Path) -> tuple[list[str], str]:
    """Stage editable_dirs changes, write the unified diff to `dest`, return (changed_files, diff)."""
    dirs = adapter.editable_dirs
    git("add", "-A", "--", *dirs)
    diff = git("diff", "--cached", "--", *dirs, check=False).stdout
    files = [_gp(f) for f in git("diff", "--cached", "--name-only", "--", *dirs,
                            check=False).stdout.splitlines() if f]
    git("reset", "--quiet", "--", *dirs, check=False)  # unstage; leave working tree as-is
    dest.write_text(diff)
    return files, diff


def snapshot_untracked() -> set[str]:
    """Set of currently-untracked paths (git status '?? ' entries), for stray detection."""
    r = git("status", "--porcelain", check=False)
    out = set()
    for line in r.stdout.splitlines():
        if line.startswith("?? "):
            out.add(_gp(line[3:].strip().strip('"')))
    return out


def tracked_changes_outside(adapter: BenchmarkAdapter) -> list[str]:
    """Tracked paths changed outside the adapter's declared edit boundary."""
    paths: set[str] = set()
    for args in (("diff", "--name-only"), ("diff", "--cached", "--name-only")):
        result = git(*args, check=False)
        paths.update(_gp(path) for path in result.stdout.splitlines() if path)
    return sorted(
        path for path in paths
        if not any(path == root or path.startswith(root + "/") for root in adapter.editable_dirs)
    )


def restore_outside_tracked_changes(paths: list[str]) -> None:
    """Restore proposer changes outside edit scope; the run starts from a clean tree."""
    if not paths:
        return
    git("reset", "--quiet", "--", *paths, check=False)
    git("checkout", "--", *paths, check=False)


def new_strays(adapter: BenchmarkAdapter, pre_untracked: set[str]) -> list[str]:
    """Untracked paths OUTSIDE editable_dirs that appeared since `pre_untracked` was taken."""
    strays = []
    for p in snapshot_untracked() - pre_untracked:
        if p == "pending_eval.json":
            continue
        if any(p == d or p.startswith(d + "/") for d in adapter.editable_dirs):
            continue
        strays.append(p)
    return sorted(strays)


def remove_strays(adapter: BenchmarkAdapter, strays: list[str]) -> None:
    """Delete flagged NEW stray files only — regular files, never protected ones, never dirs."""
    for p in strays:
        if p in adapter.protected_files:
            log(f"  refusing to delete protected stray {p}; leaving for manual review")
            continue
        fp = REPO / p
        if fp.is_file():
            fp.unlink(missing_ok=True)


def revert_to_baseline(adapter: BenchmarkAdapter, strays: list[str] | None = None) -> None:
    """Restore editable_dirs to the last committed baseline and drop new untracked files."""
    dirs = adapter.editable_dirs
    git("checkout", "--", *dirs, check=False)
    git("clean", "-fdq", "--", *dirs, check=False)
    if strays:
        remove_strays(adapter, strays)
    (REPO / "pending_eval.json").unlink(missing_ok=True)


# --------------------------------------------------------------------------------------
# Guardrails + validation
# --------------------------------------------------------------------------------------

def guardrail_violation(
    adapter: BenchmarkAdapter,
    changed_files: list[str],
    diff: str,
) -> str | None:
    for f in changed_files:
        if f in adapter.out_of_scope:
            return f"edited out-of-scope file {f}"
        if any(h in f for h in GROUND_TRUTH_HINTS):
            return f"touched a ground-truth path {f}"
        if not any(f == d or f.startswith(d + "/") for d in adapter.editable_dirs):
            return f"edited outside {adapter.editable_dirs}: {f}"
    return adapter.candidate_violation(changed_files, diff)


def _discover_benchmarks() -> list[str]:
    """Find all benchmark names by scanning benchmarks/*/benchmark.py for an Adapter class."""
    bench_dir = REPO / "benchmarks"
    names = []
    for child in sorted(bench_dir.iterdir()):
        if child.is_dir() and (child / "benchmark.py").exists():
            names.append(child.name)
    return names


def import_ok(adapter: BenchmarkAdapter, tag: str = "") -> bool:
    """Check that the active benchmark AND all other benchmarks still import.

    If the candidate touched only benchmark-specific files (under benchmarks/<active>/),
    other benchmarks are unaffected and the check is trivially fast. If it touched
    shared harness files, this catches breakages before we waste eval time.
    """
    names = _discover_benchmarks()
    failures = []
    for name in names:
        r = subprocess.run(
            [PYTHON, "-c", f"import benchmarks.{name}.run_eval"],
            cwd=str(REPO), env=eval_env(adapter), capture_output=True, text=True,
        )
        if r.returncode != 0:
            out = (r.stdout or "") + (r.stderr or "")
            failures.append(f"benchmarks.{name}: {out[-500:]}")
    if _EVAL_LOG_DIR is not None:
        suffix = f"_{tag}" if tag else ""
        (_EVAL_LOG_DIR / f"import_check{suffix}.log").write_text(
            f"checked: {', '.join(names)}\nfailures: {len(failures)}\n"
            + "\n".join(failures)
        )
    if failures:
        log(f"  import check failed:\n{chr(10).join(failures)[:1000]}")
        return False
    return True


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

def run_eval(adapter: BenchmarkAdapter, name: str, scenarios: list[str], repeats: int,
             max_turns: int, concurrency: int, unit_timeout: float | None = None) -> tuple[int, Path]:
    """Run the benchmark eval, streaming output to evolving_runs/<run>/eval_logs/<name>.log."""
    log_path = (_EVAL_LOG_DIR or RUNS_DIR) / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rc = adapter.run_eval(
        name=name, scenarios=scenarios, repeats=repeats, max_turns=max_turns,
        concurrency=concurrency, unit_timeout=unit_timeout, repo=REPO, python=PYTHON,
        eval_env=eval_env(adapter), log_path=log_path,
    )
    return rc, log_path


def read_summary(
    adapter: BenchmarkAdapter,
    name: str,
    expected_scenarios: list[str],
) -> tuple[float, dict[str, float], float] | None:
    return adapter.read_summary(name, RUNS_DIR, expected_scenarios)


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------

def decide(mean: float, per: dict[str, float], vrate: float, frontier: dict, eps: float = 1e-9) -> tuple[bool, str]:
    if mean > frontier["mean"] + eps:
        return True, f"improved {frontier['mean']:.3f} -> {mean:.3f}"
    if mean <= frontier["mean"] + eps and mean >= frontier["mean"] - eps:
        # Task rate is tied — use verifier pass rate as tiebreaker
        fv = frontier.get("verifier_rate", 0.0)
        if vrate > fv + eps:
            return True, f"task rate tied ({mean:.3f}), verifier rate improved {fv:.3f} -> {vrate:.3f}"
        return False, f"no improvement (task {mean:.3f} tied, verifier {vrate:.3f} <= {fv:.3f})"
    return False, f"no mean improvement ({mean:.3f} < {frontier['mean']:.3f})"


def _score_delta_table(reference: dict[str, float], candidate: dict[str, float],
                       baseline: dict[str, float] | None = None) -> str:
    rows = []
    baseline = baseline or {}
    all_scenarios = sorted(set(reference) | set(candidate) | set(baseline))
    for sid in all_scenarios:
        ref = reference.get(sid, 0.0)
        cand = candidate.get(sid, 0.0)
        base = baseline.get(sid, 0.0)
        delta = cand - ref
        tag = "WIN" if delta > 1e-9 else ("REGRESS" if delta < -1e-9 else "SAME")
        base_tag = "below-baseline" if cand + 1e-9 < base else ""
        rows.append(f"- {tag} {sid}: frontier={ref:.3f} candidate={cand:.3f} "
                    f"delta={delta:+.3f} baseline={base:.3f} {base_tag}".rstrip())
    return "\n".join(rows)


def _parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def promotion_review(adapter: BenchmarkAdapter, *, iter_log_dir: Path, model: str,
                     timeout_s: int, candidate_record: dict, frontier: dict,
                     patch_path: Path, candidate_run: str, test_flip_result: dict | None) -> tuple[bool, str, dict]:
    """Make a deterministic frontier decision from selection task and verifier scores."""
    baseline_per = frontier.get("baseline_per_scenario") or frontier.get("per_scenario", {})
    candidate_per = candidate_record.get("per_scenario") or {}
    frontier_per = frontier.get("per_scenario", {})
    below_baseline = [
        sid for sid, base in baseline_per.items()
        if candidate_per.get(sid, 0.0) + 1e-9 < base
    ]
    key_wins = [
        sid for sid, val in candidate_per.items()
        if val > frontier_per.get(sid, 0.0) + 1e-9
    ]
    key_regressions = [
        sid for sid, val in candidate_per.items()
        if val + 1e-9 < frontier_per.get(sid, 0.0)
    ]
    mean = float(candidate_record.get("mean") or 0.0)
    frontier_mean = float(frontier.get("mean") or 0.0)
    vrate = float(candidate_record.get("verifier_rate") or 0.0)
    frontier_vrate = float(frontier.get("verifier_rate") or 0.0)
    if mean > frontier_mean + 1e-9:
        promote = True
        reason = f"selection accepted: task mean improved {frontier_mean:.3f} -> {mean:.3f}"
    elif abs(mean - frontier_mean) <= 1e-9 and vrate > frontier_vrate + 1e-9:
        promote = True
        reason = (
            f"selection accepted: task mean tied at {mean:.3f}, "
            f"verifier rate improved {frontier_vrate:.3f} -> {vrate:.3f}"
        )
    else:
        promote = False
        reason = (
            f"selection rejected: task mean {mean:.3f} vs frontier {frontier_mean:.3f}, "
            f"verifier rate {vrate:.3f} vs frontier {frontier_vrate:.3f}"
        )
    payload = {
        "promote": promote,
        "reason": reason,
        "decision_source": "deterministic_task_and_verifier_scores",
        "candidate_run": candidate_run,
        "patch": str(patch_path),
        "test_flip": test_flip_result or {},
        "selection_mean": mean,
        "frontier_selection_mean": frontier_mean,
        "selection_verifier_rate": vrate,
        "frontier_selection_verifier_rate": frontier_vrate,
        "below_baseline": below_baseline,
        "key_wins": key_wins,
        "key_regressions": key_regressions,
        "delta_table": _score_delta_table(frontier_per, candidate_per, baseline_per),
    }
    return promote, reason, payload


def proceed_gate(adapter: BenchmarkAdapter, *, iter_log_dir: Path, model: str,
                  timeout_s: int, candidate_name: str, hypothesis: str | None,
                  test_flip_result: dict | None, frontier: dict,
                  patch_path: Path) -> tuple[bool, str, dict]:
    """Ask omp whether to spend a full benchmark eval after the test-flip probe."""
    n_scenarios = len(frontier.get("per_scenario", {}))
    task = f"""You are the proceed gate for a {adapter.name} harness-evolution run.

A candidate harness change has been proposed and a single-scenario test-flip probe has been run.
Your job is to decide whether it is worth spending a full {n_scenarios}-scenario
developer benchmark evaluation on this candidate, or whether the evidence is weak enough to
discard early and save compute.

Do not edit files. Read the patch and flip result if useful.

Candidate:
- name: {candidate_name}
- hypothesis: {hypothesis}
- patch: `{patch_path}`

Test-flip result:
{json.dumps(test_flip_result or {}, indent=2)}

Current frontier:
- mean: {frontier.get("mean", 0.0):.3f}
- verifier_rate: {frontier.get("verifier_rate", 0.0):.3f}

Consider:
- Did the test-flip show any signal (score change, verifier-rate change, partial progress)?
- Is the hypothesis plausible given the flip result?
- Could the change still help other scenarios even if the flip didn't validate?
- Is the flip scenario representative or an outlier?

Return ONLY a JSON object:
{{
  "proceed": true or false,
  "reason": "short evidence-based reason",
  "confidence": "high" | "medium" | "low"
}}
"""
    log_dir = iter_log_dir / "proceed_gate"
    res = omp_wrapper.run(
        task_text=task, cwd=REPO, log_dir=log_dir, model=model,
        prior_path=None, timeout_s=timeout_s,
    )
    (log_dir / "session_result.json").write_text(res.to_json())
    payload = _parse_json_object(res.text)
    proceed = bool(payload.get("proceed", True))
    reason = str(payload.get("reason") or "proceed gate returned no reason")
    return proceed, reason, payload


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True,
                    help="Benchmark adapter to evolve (for example, automationbench).")
    ap.add_argument("--run-name", required=True, help="Isolates outputs under evolving_runs/<name>/.")
    ap.add_argument("--scenarios", default=None,
                    help="Comma-separated search scenario override (default: the adapter's search set).")
    ap.add_argument("--selection-scenarios", default=None,
                    help="Comma-separated selection scenario override (default: the adapter's selection set).")
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--start-iter", type=int, default=1,
                    help="First iteration number (use with --skip-baseline to CONTINUE a prior "
                         "run in the same evolving_runs/<name>/ without clobbering earlier iterN artifacts).")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--proposer-model", default=PROPOSER_MODEL_DEFAULT)
    ap.add_argument("--propose-timeout", type=int, default=2400)
    ap.add_argument("--trace-chars", type=int, default=0,
                    help="Per-scenario trace cap fed to the proposer (0 = unlimited, default).")
    ap.add_argument("--trace-analysis", choices=TRACE_ANALYSIS_MODES, default="inline",
                    help="Analyze traces in the proposer context (inline) or ask the proposer to "
                         "spawn one isolated subagent per search trace (subagents).")
    ap.add_argument("--trace-subagent-wave", type=int, default=8,
                    help="Maximum concurrent trace-analysis workers in subagents mode (default: 8).")
    ap.add_argument("--smoke-max-turns", type=int, default=20)
    ap.add_argument("--test-flip", action="store_true", default=True,
                    help="Run the proposer's test_scenario before full benchmark as an evidence "
                         "probe. Promotion review sees the result; a failed probe is not an "
                         "automatic rejection.")
    ap.add_argument("--no-test-flip", dest="test_flip", action="store_false")
    ap.add_argument("--fresh", action="store_true", help="Reset the working tree + clear evolving_runs/<name>.")
    ap.add_argument("--skip-baseline", action="store_true", help="Reuse existing frontier.json.")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--held-out", dest="held_out", action="store_true", default=True)
    ap.add_argument("--no-held-out", dest="held_out", action="store_false")
    args = ap.parse_args()

    if not SAFE_NAME.fullmatch(args.run_name):
        print(f"FATAL: invalid run name: {args.run_name!r}", file=sys.stderr)
        return 2
    if args.trace_subagent_wave < 1:
        print("FATAL: --trace-subagent-wave must be positive", file=sys.stderr)
        return 2

    try:
        adapter = load_adapter(args.benchmark)
        adapter.validate(REPO)
    except (AdapterValidationError, ImportError, AttributeError, TypeError) as exc:
        print(f"FATAL: invalid benchmark adapter: {exc}", file=sys.stderr)
        return 2
    log(f"loaded benchmark adapter: {adapter.name}")

    dev_scenarios = adapter.dev_scenarios
    selection_scenarios = adapter.selection_scenarios()
    if args.scenarios:
        dev_scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        log(f"scenario override: {dev_scenarios}")
    if args.selection_scenarios:
        selection_scenarios = [s.strip() for s in args.selection_scenarios.split(",") if s.strip()]
        log(f"selection scenario override: {selection_scenarios}")
    try:
        adapter.validate_scenario_sets(
            dev_scenarios,
            selection_scenarios,
            adapter.held_out_scenarios(),
        )
    except AdapterValidationError as exc:
        print(f"FATAL: invalid scenario split: {exc}", file=sys.stderr)
        return 2

    run_root = REPO / "evolving_runs" / args.run_name
    if args.fresh and run_root.exists():
        shutil.rmtree(run_root)
    (run_root / "candidates").mkdir(parents=True, exist_ok=True)
    (run_root / "proposer_logs").mkdir(parents=True, exist_ok=True)
    frontier_path = run_root / "frontier.json"
    summary_log = run_root / "evolution_summary.jsonl"

    global _RUN_LOG_FH, _EVAL_LOG_DIR
    _EVAL_LOG_DIR = run_root / "eval_logs"
    _EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _RUN_LOG_FH = open(run_root / "run.log", "a")
    log(f"=== evolving-harness run '{args.run_name}' start | benchmark={args.benchmark} args={vars(args)} ===")

    ensure_gitignore()
    if args.fresh:
        revert_to_baseline(adapter)

    # --- Establish/advance the git baseline (scoped to editable_dirs) ---
    if head_sha() is None:
        log("FATAL: StarHarness needs an initial Git commit before evolution can run")
        return 2
    outside = tracked_changes_outside(adapter)
    if outside:
        log(f"FATAL: tracked changes exist outside adapter edit scope: {outside}")
        return 2
    base_commit = commit_baseline(adapter, f"StarHarness baseline ({adapter.name})")
    log(f"baseline commit: {base_commit[:10]}")

    trace_chars = args.trace_chars if args.trace_chars > 0 else 10_000_000

    # --- Phase 0: baseline eval ---
    if args.skip_baseline and frontier_path.exists():
        frontier = json.loads(frontier_path.read_text())
        log(f"reusing frontier: selection mean={frontier['mean']:.3f}")
    else:
        search_base_run = f"evolving_{args.run_name}_base_search"
        selection_base_run = f"evolving_{args.run_name}_base_selection"
        log(f"Phase 0a: search baseline/traces -> runs/{search_base_run} "
            f"({len(dev_scenarios)} scenarios x {args.repeats})")
        _, elog = run_eval(adapter, search_base_run, dev_scenarios, args.repeats, args.max_turns, args.concurrency)
        search_parsed = read_summary(adapter, search_base_run, dev_scenarios)
        if search_parsed is None:
            log(f"FATAL: search baseline produced no summary. log tail:\n{tail(elog, 2000)}")
            return 1
        search_mean, search_per, search_vrate = search_parsed
        log(f"Phase 0b: selection baseline/gate -> runs/{selection_base_run} "
            f"({len(selection_scenarios)} scenarios x {args.repeats})")
        _, elog = run_eval(adapter, selection_base_run, selection_scenarios, args.repeats, args.max_turns, args.concurrency)
        parsed = read_summary(adapter, selection_base_run, selection_scenarios)
        if parsed is None:
            log(f"FATAL: selection baseline produced no summary. log tail:\n{tail(elog, 2000)}")
            return 1
        mean, per, vrate = parsed
        frontier = {
            "mean": mean,
            "per_scenario": per,
            "baseline_per_scenario": per,
            "verifier_rate": vrate,
            "commit": base_commit,
            "traces_run": search_base_run,
            "search_run": search_base_run,
            "search_mean": search_mean,
            "search_per_scenario": search_per,
            "baseline_search_per_scenario": search_per,
            "search_verifier_rate": search_vrate,
            "search_scenarios": dev_scenarios,
            "selection_run": selection_base_run,
            "selection_scenarios": selection_scenarios,
            "hypotheses": [],
        }
        frontier_path.write_text(json.dumps(frontier, indent=2))
        log(f"search baseline mean={search_mean:.3f} verifier_rate={search_vrate:.3f}")
        log(f"selection baseline mean={mean:.3f} verifier_rate={vrate:.3f}  per-scenario={json.dumps(per)}")

    # --- Iterations ---
    last_iter = args.start_iter + args.iterations - 1
    for i in range(args.start_iter, last_iter + 1):
        t0 = time.monotonic()
        iter_log_dir = run_root / "proposer_logs" / f"iter{i}"
        iter_log_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== Iteration {i} (of {args.start_iter}..{last_iter}) "
            f"(selection frontier mean={frontier['mean']:.3f}, "
            f"search mean={frontier.get('search_mean', frontier['mean']):.3f}) ===")

        pre_untracked = snapshot_untracked()

        # 1) Propose
        traces = adapter.gather_traces(frontier.get("search_run") or frontier["traces_run"],
                                       dev_scenarios, RUNS_DIR, trace_chars)

        # Read discarded hypotheses from the summary log so the proposer can learn from failures
        discarded: list[dict] = []
        if summary_log.exists():
            for line in summary_log.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    if rec.get("decision") != "kept":
                        reason = rec.get("reason", "")
                        if rec.get("decision") == "discarded":
                            reason = "selection gate did not improve task/verifier score enough to promote"
                        discarded.append({
                            "name": rec.get("name", "?"),
                            "hypothesis": rec.get("hypothesis", "?"),
                            "delta": rec.get("delta"),
                            "decision": rec["decision"],
                            "reason": reason,
                            "selection_mean": rec.get("mean"),
                            "selection_verifier_rate": rec.get("verifier_rate"),
                            "search_per_scenario": rec.get("search_per_scenario"),
                            "test_flip": rec.get("test_flip"),
                            "proceed_gate": rec.get("proceed_gate"),
                            "traces_run": rec.get("search_run"),
                        })
                except (json.JSONDecodeError, KeyError):
                    pass
        frontier["discarded"] = discarded

        task = adapter.render_task(i, args.iterations, frontier, traces)

        analysis_prompt = trace_analysis_prompt(
            args.trace_analysis,
            dev_scenarios,
            args.trace_subagent_wave,
        )
        if analysis_prompt:
            task = f"{analysis_prompt}\n\n---\n\n{task}"

        # Prepend the proposer prior to the task instead of sending as system prompt
        prior_path = adapter.prompts_dir / "proposer_prior.md"
        if prior_path.exists():
            prior_text = prior_path.read_text()
            task = f"{prior_text}\n\n---\n\n{task}"

        task_file = iter_log_dir / "task.md"
        task_file.write_text(task)
        (REPO / "pending_eval.json").unlink(missing_ok=True)
        log("  proposing (omp)…")
        res = omp_wrapper.run(
            task_file=task_file, cwd=REPO, log_dir=iter_log_dir,
            model=args.proposer_model, prior_path=None,
            timeout_s=args.propose_timeout,
        )
        (iter_log_dir / "session_result.json").write_text(res.to_json())
        log(f"  proposer: {res.tool_call_count} tool calls, {res.total_tokens} tok, "
            f"${res.cost:.3f}, {res.duration_s}s" + (" [TIMED OUT]" if res.timed_out else ""))

        outside_tracked = tracked_changes_outside(adapter)
        if outside_tracked:
            restore_outside_tracked_changes(outside_tracked)

        record = {"iter": i, "name": None, "hypothesis": None, "mean": None, "delta": None,
                  "per_scenario": None, "decision": None, "proposer_cost": res.cost,
                  "session_log": res.log_path, "test_flip": None}
        strays = new_strays(adapter, pre_untracked)

        def finish(decision: str, extra: str = "") -> None:
            record["decision"] = decision
            record["reason"] = extra
            with open(summary_log, "a") as f:
                f.write(json.dumps(record) + "\n")
            log(f"  -> {decision}. {extra} ({time.monotonic()-t0:.0f}s)")

        if outside_tracked:
            finish("rejected-guardrail", f"edited tracked paths outside scope: {outside_tracked}")
            revert_to_baseline(adapter, new_strays(adapter, pre_untracked))
            continue

        # 2) Capture candidate diff (scoped to editable dirs)
        patch_path = run_root / "candidates" / f"iter{i}.patch"
        changed, diff = capture_diff(adapter, patch_path)
        pend_path = REPO / "pending_eval.json"
        pend = json.loads(pend_path.read_text()) if pend_path.exists() else {}
        record["name"] = pend.get("name") or f"iter{i}"
        record["hypothesis"] = pend.get("hypothesis")
        if not changed:
            finish("no-op", f"proposer made no edits to {adapter.editable_dirs}")
            revert_to_baseline(adapter, strays)
            continue
        log(f"  candidate '{record['name']}' touches: {changed}")

        # 3) Guardrail check
        # Hard violation: edited an out-of-scope file -> reject
        # Soft violation: created stray temp files outside editable_dirs -> clean and proceed
        hard_violation = guardrail_violation(adapter, changed, diff)
        if hard_violation:
            finish("rejected-guardrail", hard_violation)
            revert_to_baseline(adapter, strays)
            continue

        if strays:
            # Auto-clean stray temp files and proceed with the candidate
            log(f"  cleaning stray files outside editable_dirs: {strays}")
            remove_strays(adapter, strays)
            strays = []

        # 4) Validate import (active benchmark + cross-benchmark safety)
        if not import_ok(adapter, f"iter{i}"):
            finish("invalid-import", "import check failed (see import_check log)")
            revert_to_baseline(adapter, strays)
            continue

        # 5) Smoke
        if not args.skip_smoke and dev_scenarios:
            smoke_run = f"evolving_{args.run_name}_iter{i}_smoke"
            _, elog = run_eval(adapter, smoke_run, [dev_scenarios[0]], 1, args.smoke_max_turns, 1)
            if read_summary(adapter, smoke_run, [dev_scenarios[0]]) is None:
                finish("crash-smoke", f"smoke crashed. log tail:\n{tail(elog)}")
                revert_to_baseline(adapter, strays)
                continue

        # 5b) Test probe: run the proposer's chosen test_scenario, but treat it as evidence.
        test_flip_result = None
        if args.test_flip and dev_scenarios:
            test_scenario = pend.get("test_scenario", "")
            if not test_scenario:
                log("  no test_scenario in pending_eval.json — skipping test-flip (proceeding to full benchmark)")
            else:
                per = frontier.get("search_per_scenario") or frontier.get("per_scenario", {})
                frontier_val = per.get(test_scenario, None)
                if frontier_val is None:
                    log(f"  test_scenario '{test_scenario}' not in dev set — skipping test-flip")
                else:
                    flip_run = f"evolving_{args.run_name}_iter{i}_flip"
                    log(f"  test-flip on '{test_scenario}' (frontier={frontier_val:.3f}) -> runs/{flip_run}")
                    run_eval(adapter, flip_run, [test_scenario], 1, args.max_turns, 1)
                    flip_parsed = read_summary(adapter, flip_run, [test_scenario])
                    if flip_parsed is None:
                        finish("crash-flip", f"test-flip crashed. log: runs/{flip_run}")
                        revert_to_baseline(adapter, strays)
                        continue
                    flip_mean, flip_per, flip_vrate = flip_parsed
                    flip_val = flip_per.get(test_scenario, 0.0)
                    validated = flip_val >= 1.0 and frontier_val < 1.0
                    regressed = frontier_val >= 1.0 - 1e-9 and flip_val + 1e-9 < frontier_val
                    test_flip_result = {
                        "scenario": test_scenario,
                        "frontier": frontier_val,
                        "flipped": flip_val,
                        "validated": validated,
                        "regressed": regressed,
                        "verifier_rate": flip_vrate,
                    }
                    record["test_flip"] = test_flip_result

                    if validated:
                        log(f"  test-flip: validated ({frontier_val:.3f} -> {flip_val:.3f})")
                    elif regressed:
                        log(f"  test-flip: regressed ({frontier_val:.3f} -> {flip_val:.3f})")
                    else:
                        log(f"  test-flip: did not validate ({frontier_val:.3f} -> {flip_val:.3f})")

        # 5c) Proceed gate — ask OMP whether to spend the full benchmark eval.
        if test_flip_result is not None:
            proceed, proceed_reason, proceed_payload = proceed_gate(
                adapter,
                iter_log_dir=iter_log_dir,
                model=args.proposer_model,
                timeout_s=min(args.propose_timeout, 600),
                candidate_name=record["name"],
                hypothesis=record["hypothesis"],
                test_flip_result=test_flip_result,
                frontier=frontier,
                patch_path=patch_path,
            )
            record["proceed_gate"] = proceed_payload
            if not proceed:
                log(f"  proceed gate: ABORT — {proceed_reason}")
                finish("aborted-proceed-gate", proceed_reason)
                revert_to_baseline(adapter, strays)
                continue
            log(f"  proceed gate: PROCEED — {proceed_reason}")

        # 6) Benchmark on the proposer-hidden selection set
        bench_run = f"evolving_{args.run_name}_iter{i}"
        log(f"  selection benchmarking -> runs/{bench_run}")
        run_eval(adapter, bench_run, selection_scenarios, args.repeats, args.max_turns, args.concurrency)
        parsed = read_summary(adapter, bench_run, selection_scenarios)
        if parsed is None:
            finish("crash-bench", f"benchmark produced no summary. check eval_logs/{bench_run}.log")
            revert_to_baseline(adapter, strays)
            continue
        mean, per, vrate = parsed
        record["mean"] = mean
        record["per_scenario"] = per
        record["verifier_rate"] = vrate
        record["delta"] = round(mean - frontier["mean"], 4)

        # 7) Select deterministically from selection task score + verifier score.
        pre_review_diff = git("diff", "--", *adapter.editable_dirs, check=False).stdout
        keep, reason, review_payload = promotion_review(
            adapter,
            iter_log_dir=iter_log_dir,
            model=args.proposer_model,
            timeout_s=min(args.propose_timeout, 900),
            candidate_record=record,
            frontier=frontier,
            patch_path=patch_path,
            candidate_run=bench_run,
            test_flip_result=test_flip_result,
        )
        post_review_diff = git("diff", "--", *adapter.editable_dirs, check=False).stdout
        if post_review_diff != pre_review_diff:
            keep = False
            reason = "promotion reviewer unexpectedly edited the candidate; rejecting to preserve evaluated state"
            review_payload = {**review_payload, "review_mutated_candidate": True}
        record["promotion_review"] = review_payload
        if keep:
            search_run = f"evolving_{args.run_name}_iter{i}_search"
            log(f"  refreshing proposer-visible search traces -> runs/{search_run}")
            _, elog = run_eval(adapter, search_run, dev_scenarios, args.repeats, args.max_turns, args.concurrency)
            search_parsed = read_summary(adapter, search_run, dev_scenarios)
            if search_parsed is None:
                finish("crash-search-refresh", f"accepted candidate could not refresh search traces. log tail:\n{tail(elog)}")
                revert_to_baseline(adapter, strays)
                continue
            search_mean, search_per, search_vrate = search_parsed
            record["search_mean"] = search_mean
            record["search_per_scenario"] = search_per
            record["search_verifier_rate"] = search_vrate
            record["search_run"] = search_run
            new_commit = commit_baseline(adapter, f"iter{i}: {record['name']} ({record['hypothesis'] or ''})"[:200])
            frontier.update({
                "mean": mean, "per_scenario": per, "verifier_rate": vrate,
                "commit": new_commit,
                "traces_run": search_run,
                "search_run": search_run,
                "search_mean": search_mean,
                "search_per_scenario": search_per,
                "search_verifier_rate": search_vrate,
                "selection_run": bench_run,
                "hypotheses": frontier["hypotheses"] + [{
                    "name": record["name"], "hypothesis": record["hypothesis"],
                    "delta": record["delta"],
                    "search_delta": round(search_mean - frontier.get("search_mean", 0.0), 4)}],
            })
            frontier_path.write_text(json.dumps(frontier, indent=2))
            finish("kept", reason)
        else:
            revert_to_baseline(adapter, strays)
            finish("discarded", reason)

    # --- Final: held-out eval on the winning frontier ---
    if args.held_out:
        ho = adapter.held_out_scenarios()
        if ho:
            ho_run = f"evolving_{args.run_name}_heldout"
            log(f"Final: held-out eval on {len(ho)} scenarios -> runs/{ho_run}")
            run_eval(adapter, ho_run, ho, args.repeats, args.max_turns, args.concurrency)
            parsed = read_summary(adapter, ho_run, ho)
            if parsed:
                mean, per, vrate = parsed
                (run_root / "held_out_result.json").write_text(json.dumps(
                    {"held_out_mean": mean, "per_scenario": per, "verifier_rate": vrate, "scenarios": ho}, indent=2))
                log(f"held-out mean={mean:.3f} verifier_rate={vrate:.3f}")

    log(f"Done. Frontier mean={frontier['mean']:.3f}. "
        f"Kept {len(frontier['hypotheses'])} change(s). See {summary_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
