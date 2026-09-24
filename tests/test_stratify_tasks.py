import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from stratify_tasks import (
    StratificationError,
    build_prompt,
    build_report,
    load_manifest,
    main,
    split_limits,
    validate_plan,
)


def _manifest(tmp_path: Path, count: int = 6) -> Path:
    tasks = []
    for index in range(count):
        trace = tmp_path / f"task-{index}.log"
        trace.write_text(f"trace {index}")
        tasks.append({
            "task_id": f"task-{index}",
            "reproducible": True,
            "baseline_score": index / count,
            "verifier_pass_rate": (count - index) / count,
            "trace_path": trace.name,
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"tasks": tasks}))
    return path


def _plan() -> dict:
    return {
        "task_strata": {
            "task-0": "tool_use",
            "task-1": "tool_use",
            "task-2": "tool_use",
            "task-3": "tool_use",
            "task-4": "tool_use",
            "task-5": "tool_use",
        },
        "task_evidence": {
            f"task-{index}": f"Trace {index} shows a tool-use error." for index in range(6)
        },
        "search": ["task-0", "task-2"],
        "selection": ["task-1"],
        "holdout": ["task-3", "task-4", "task-5"],
        "rationale": "The evolution pool covers two failure modes and score ranges.",
    }


def test_manifest_requires_traces_and_filters_nonreproducible(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["tasks"][5]["reproducible"] = False
    payload["tasks"][5]["trace_path"] = "missing.log"
    path.write_text(json.dumps(payload))

    tasks = load_manifest(path)

    assert len(tasks) == 5
    assert all(Path(task["trace_path"]).is_absolute() for task in tasks)


def test_prompt_requires_one_isolated_worker_per_trace(tmp_path: Path) -> None:
    tasks = load_manifest(_manifest(tmp_path))
    prompt = build_prompt(
        tasks,
        max_wave=3,
        max_evolution=3,
        min_holdout=3,
        search_size=None,
        selection_size=None,
    )

    assert "one isolated OMP `task` subagent for each of the 6" in prompt
    assert "waves of at most 3" in prompt
    assert "smallest non-empty search and selection sets" in prompt
    assert "baseline failure-mode, score, and verifier-pass distributions" in prompt
    assert "Selection contents and outcomes stay hidden" in prompt


def test_plan_validation_and_report_use_trusted_scores(tmp_path: Path) -> None:
    tasks = load_manifest(_manifest(tmp_path))
    plan = validate_plan(
        _plan(),
        tasks,
        max_evolution=3,
        min_holdout=3,
        search_size=2,
        selection_size=1,
    )
    report = build_report(plan, tasks)

    assert report["splits"]["search"]["n_tasks"] == 2
    assert report["splits"]["selection"]["failure_modes"] == {"tool_use": 1}
    assert report["splits"]["holdout"]["n_tasks"] == 3


def test_plan_validation_rejects_overlap(tmp_path: Path) -> None:
    tasks = load_manifest(_manifest(tmp_path))
    plan = _plan()
    plan["selection"] = ["task-0"]

    with pytest.raises(StratificationError, match="search and selection overlap"):
        validate_plan(
            plan,
            tasks,
            max_evolution=3,
            min_holdout=3,
            search_size=None,
            selection_size=None,
        )


def test_automatic_plan_must_use_smallest_failure_mode_cover(tmp_path: Path) -> None:
    tasks = load_manifest(_manifest(tmp_path))

    with pytest.raises(StratificationError, match="not the smallest pool"):
        validate_plan(
            _plan(),
            tasks,
            max_evolution=3,
            min_holdout=3,
            search_size=None,
            selection_size=None,
        )


def test_automatic_plan_accepts_minimum_failure_mode_cover(tmp_path: Path) -> None:
    tasks = load_manifest(_manifest(tmp_path))
    plan = _plan()
    plan["search"] = ["task-0"]
    plan["selection"] = ["task-1"]
    plan["holdout"] = ["task-2", "task-3", "task-4", "task-5"]

    assert validate_plan(
        plan,
        tasks,
        max_evolution=3,
        min_holdout=3,
        search_size=None,
        selection_size=None,
    ) == plan


def test_split_limits_require_space_for_search_and_selection() -> None:
    with pytest.raises(StratificationError, match="leave at least two"):
        split_limits(6, 0.5, 5, None, None)


def test_cli_runs_omp_and_writes_adapter_split_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "splits"
    result = SimpleNamespace(
        returncode=0,
        timed_out=False,
        text=json.dumps(_plan()),
        log_path=str(output / "omp_analysis" / "stdout.log"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "stratify_tasks.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--search-size",
            "2",
            "--selection-size",
            "1",
        ],
    )

    with patch("stratify_tasks.omp_wrapper.run", return_value=result) as run:
        assert main() == 0

    assert run.call_args.kwargs["cwd"] == output.resolve()
    assert json.loads((output / "search_scenarios.json").read_text()) == {
        "scenarios": ["task-0", "task-2"]
    }
    assert (output / "stratification_report.json").is_file()
