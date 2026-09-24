"""Build search, selection, and holdout splits from baseline task traces with OMP."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import omp_wrapper

DEFAULT_MODEL = "openai/gpt-5.4"
SPLIT_FILENAMES = {
    "search": "search_scenarios.json",
    "selection": "selection_scenarios.json",
    "holdout": "holdout_scenarios.json",
}


class StratificationError(ValueError):
    """Raised when the manifest, OMP response, or requested split is invalid."""


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load reproducible baseline task records and validate their descriptors."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StratificationError(f"cannot read manifest: {path}") from exc

    raw_tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(raw_tasks, list):
        raise StratificationError("manifest must contain a tasks list")

    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict):
            raise StratificationError(f"task {index} must be an object")
        task_id = raw.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip() or task_id != task_id.strip():
            raise StratificationError(f"task {index} has an invalid task_id")
        if task_id in seen:
            raise StratificationError(f"duplicate task_id: {task_id}")
        seen.add(task_id)

        reproducible = raw.get("reproducible", True)
        if not isinstance(reproducible, bool):
            raise StratificationError(f"{task_id}: reproducible must be boolean")
        if not reproducible:
            continue

        score = _unit_interval(raw.get("baseline_score"), f"{task_id}: baseline_score")
        verifier = _unit_interval(
            raw.get("verifier_pass_rate"), f"{task_id}: verifier_pass_rate"
        )
        trace_path = raw.get("trace_path")
        if not isinstance(trace_path, str) or not trace_path.strip():
            raise StratificationError(f"{task_id}: trace_path must name a baseline trace")
        resolved_trace = (path.parent / trace_path).resolve()
        if not resolved_trace.is_file():
            raise StratificationError(f"{task_id}: trace does not exist: {resolved_trace}")

        tasks.append({
            "task_id": task_id,
            "baseline_score": score,
            "verifier_pass_rate": verifier,
            "trace_path": str(resolved_trace),
        })

    if len(tasks) < 3:
        raise StratificationError("at least three reproducible tasks are required")
    return tasks


def _unit_interval(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StratificationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise StratificationError(f"{label} must be finite and between 0 and 1")
    return result


def split_limits(
    task_count: int,
    max_evolution_fraction: float,
    min_holdout: int,
    search_size: int | None,
    selection_size: int | None,
) -> tuple[int, int | None, int | None]:
    """Validate size controls and return the maximum evolution-pool size."""
    if not 0.0 < max_evolution_fraction < 1.0:
        raise StratificationError("max evolution fraction must be between 0 and 1")
    if min_holdout < 1 or min_holdout > task_count - 2:
        raise StratificationError("min holdout must leave at least two evolution tasks")
    if (search_size is None) != (selection_size is None):
        raise StratificationError("set both search size and selection size, or neither")

    max_evolution = min(
        task_count - min_holdout,
        max(2, math.floor(task_count * max_evolution_fraction)),
    )
    if search_size is not None and selection_size is not None:
        if search_size < 1 or selection_size < 1:
            raise StratificationError("search and selection sizes must be positive")
        if search_size + selection_size > max_evolution:
            raise StratificationError(
                "requested search and selection sizes exceed the evolution-pool limit"
            )
    return max_evolution, search_size, selection_size


def build_prompt(
    tasks: list[dict[str, Any]],
    *,
    max_wave: int,
    max_evolution: int,
    min_holdout: int,
    search_size: int | None,
    selection_size: int | None,
) -> str:
    """Render the paper-aligned OMP stratification prompt."""
    if max_wave < 1:
        raise StratificationError("max wave must be positive")
    size_rule = (
        f"Use exactly {search_size} search tasks and {selection_size} selection tasks."
        if search_size is not None and selection_size is not None
        else (
            "Choose the smallest non-empty search and selection sets that preserve the baseline "
            "failure-mode, score, and verifier-pass distributions."
        )
    )
    records = json.dumps(tasks, indent=2)
    return f"""# Stratify benchmark tasks

Create the task partition before harness evolution. Follow this procedure:

1. Spawn one isolated OMP `task` subagent for each of the {len(tasks)} baseline task records below.
   Use waves of at most {max_wave}. Give each worker one record and its trace path. Workers must
   inspect one trace, assign a concise failure-mode label, cite the causal evidence in their report,
   and avoid file edits or nested subagents.
2. Wait for all workers and verify one report per task ID. Merge labels that describe the same
   causal failure pattern.
3. Form an evolution pool from proposer-visible search tasks and proposer-hidden selection tasks.
   Match both sets on failure-mode coverage, baseline score, and verifier pass rate. Put each
   failure mode in both sets when at least two tasks share that mode. Put a singleton mode in one
   evolution set. These coverage rules define the minimum pool. {size_rule}
4. Keep the evolution pool at or below {max_evolution} tasks. Reserve at least {min_holdout} tasks
   for holdout. Holdout receives every reproducible task outside the evolution pool.

Use baseline descriptors only for this pre-evolution partition. During evolution, the proposer
must receive search traces and outcomes only. Selection contents and outcomes stay hidden, and
holdout tasks must not affect proposal or acceptance.

Return only one JSON object with this shape:

{{
  "task_strata": {{"task-id": "failure_mode"}},
  "task_evidence": {{"task-id": "brief causal trace evidence"}},
  "search": ["task-id"],
  "selection": ["task-id"],
  "holdout": ["task-id"],
  "rationale": "brief distribution and size rationale"
}}

All task IDs must appear once across the three splits. `task_strata` and `task_evidence` must each
contain every task ID.

Baseline task records:

```json
{records}
```
"""


def parse_response(text: str) -> dict[str, Any]:
    """Extract one JSON object from the final OMP response."""
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise StratificationError("OMP did not return a JSON object") from None
        try:
            payload = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError as exc:
            raise StratificationError("OMP returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise StratificationError("OMP response must be a JSON object")
    return payload


def validate_plan(
    payload: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    max_evolution: int,
    min_holdout: int,
    search_size: int | None,
    selection_size: int | None,
) -> dict[str, Any]:
    """Reject incomplete, overlapping, oversized, or invented partitions."""
    known = {task["task_id"] for task in tasks}
    split_sets: dict[str, set[str]] = {}
    for name in SPLIT_FILENAMES:
        values = payload.get(name)
        if not isinstance(values, list) or not values:
            raise StratificationError(f"OMP response needs a non-empty {name} list")
        if any(not isinstance(value, str) or value not in known for value in values):
            raise StratificationError(f"{name} contains an unknown task ID")
        if len(values) != len(set(values)):
            raise StratificationError(f"{name} contains duplicate task IDs")
        split_sets[name] = set(values)

    if split_sets["search"] & split_sets["selection"]:
        raise StratificationError("search and selection overlap")
    if split_sets["search"] & split_sets["holdout"]:
        raise StratificationError("search and holdout overlap")
    if split_sets["selection"] & split_sets["holdout"]:
        raise StratificationError("selection and holdout overlap")
    covered = set().union(*split_sets.values())
    if covered != known:
        raise StratificationError("splits must cover every reproducible task exactly once")
    if len(split_sets["search"] | split_sets["selection"]) > max_evolution:
        raise StratificationError("OMP response exceeds the evolution-pool limit")
    if len(split_sets["holdout"]) < min_holdout:
        raise StratificationError("OMP response has too few holdout tasks")
    if search_size is not None and len(split_sets["search"]) != search_size:
        raise StratificationError("OMP response has the wrong search size")
    if selection_size is not None and len(split_sets["selection"]) != selection_size:
        raise StratificationError("OMP response has the wrong selection size")

    strata = payload.get("task_strata")
    if not isinstance(strata, dict) or set(strata) != known:
        raise StratificationError("task_strata must map every reproducible task ID")
    if any(not isinstance(value, str) or not value.strip() for value in strata.values()):
        raise StratificationError("task_strata contains an invalid failure-mode label")
    evidence = payload.get("task_evidence")
    if not isinstance(evidence, dict) or set(evidence) != known:
        raise StratificationError("task_evidence must map every reproducible task ID")
    if any(not isinstance(value, str) or not value.strip() for value in evidence.values()):
        raise StratificationError("task_evidence contains an invalid evidence summary")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise StratificationError("OMP response needs a rationale")

    tasks_by_mode: dict[str, set[str]] = {}
    for task_id, mode in strata.items():
        tasks_by_mode.setdefault(mode.strip(), set()).add(task_id)
    evolution = split_sets["search"] | split_sets["selection"]
    minimum_evolution = 0
    for mode, members in tasks_by_mode.items():
        if not members & evolution:
            raise StratificationError(f"evolution pool omits failure mode: {mode}")
        if len(members) >= 2:
            minimum_evolution += 2
            if not (members & split_sets["search"]) or not (members & split_sets["selection"]):
                raise StratificationError(
                    f"search and selection must both represent failure mode: {mode}"
                )
        else:
            minimum_evolution += 1
    if search_size is None and len(evolution) != minimum_evolution:
        raise StratificationError(
            "automatic split is not the smallest pool that represents each failure mode"
        )
    return payload


def build_report(payload: dict[str, Any], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute reviewable split statistics from trusted baseline descriptors."""
    by_id = {task["task_id"]: task for task in tasks}
    report: dict[str, Any] = {"rationale": payload["rationale"], "splits": {}}
    for name in SPLIT_FILENAMES:
        ids = payload[name]
        modes = Counter(payload["task_strata"][task_id] for task_id in ids)
        scores = [by_id[task_id]["baseline_score"] for task_id in ids]
        verifier_rates = [by_id[task_id]["verifier_pass_rate"] for task_id in ids]
        report["splits"][name] = {
            "n_tasks": len(ids),
            "mean_baseline_score": sum(scores) / len(scores),
            "baseline_score_range": [min(scores), max(scores)],
            "mean_verifier_pass_rate": sum(verifier_rates) / len(verifier_rates),
            "verifier_pass_rate_range": [min(verifier_rates), max(verifier_rates)],
            "failure_modes": dict(sorted(modes.items())),
        }
    return report


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    report: dict[str, Any],
    *,
    force: bool,
) -> None:
    """Write adapter-compatible split files plus the analysis and report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_outputs_available(output_dir, force=force)
    for name, filename in SPLIT_FILENAMES.items():
        (output_dir / filename).write_text(json.dumps({"scenarios": payload[name]}, indent=2) + "\n")
    (output_dir / "stratification.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "stratification_report.json").write_text(json.dumps(report, indent=2) + "\n")


def ensure_outputs_available(output_dir: Path, *, force: bool) -> None:
    """Fail before model invocation when a run would overwrite split results."""
    targets = [output_dir / filename for filename in SPLIT_FILENAMES.values()]
    targets += [output_dir / "stratification.json", output_dir / "stratification_report.json"]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise StratificationError(
            "refusing to overwrite output files without --force: "
            + ", ".join(path.name for path in existing)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("stratification_runs/latest")
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", default="medium")
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--max-wave", type=int, default=8)
    parser.add_argument("--search-size", type=int)
    parser.add_argument("--selection-size", type=int)
    parser.add_argument("--max-evolution-fraction", type=float, default=0.5)
    parser.add_argument("--min-holdout", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        manifest_path = args.manifest.resolve()
        tasks = load_manifest(manifest_path)
        max_evolution, search_size, selection_size = split_limits(
            len(tasks),
            args.max_evolution_fraction,
            args.min_holdout,
            args.search_size,
            args.selection_size,
        )
        if args.max_wave < 1 or args.timeout < 1:
            raise StratificationError("max wave and timeout must be positive")

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        ensure_outputs_available(output_dir, force=args.force)
        prompt = build_prompt(
            tasks,
            max_wave=args.max_wave,
            max_evolution=max_evolution,
            min_holdout=args.min_holdout,
            search_size=search_size,
            selection_size=selection_size,
        )
        result = omp_wrapper.run(
            task_text=prompt,
            cwd=output_dir,
            log_dir=output_dir / "omp_analysis",
            model=args.model,
            thinking=args.thinking,
            timeout_s=args.timeout,
        )
        if result.returncode != 0 or result.timed_out:
            raise StratificationError(
                f"OMP failed with exit code {result.returncode}; see {result.log_path}"
            )
        payload = validate_plan(
            parse_response(result.text),
            tasks,
            max_evolution=max_evolution,
            min_holdout=args.min_holdout,
            search_size=search_size,
            selection_size=selection_size,
        )
        report = build_report(payload, tasks)
        write_outputs(output_dir, payload, report, force=args.force)
    except StratificationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote stratified splits to {output_dir}")
    print(json.dumps(report["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
