import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.base import BenchmarkAdapter
from benchmarks.itbench.benchmark import Adapter
from benchmarks.itbench.data import Task, stage_sandbox

REPO = Path(__file__).resolve().parents[1]


def test_itbench_adapter_contract_and_public_partition() -> None:
    adapter = Adapter()
    search = adapter.dev_scenarios
    selection = adapter.selection_scenarios()
    holdout = adapter.held_out_scenarios()

    assert isinstance(adapter, BenchmarkAdapter)
    assert adapter.name == "itbench"
    prior_development_set = {
        "Scenario-102",
        "Scenario-105",
        "Scenario-11",
        "Scenario-19",
        "Scenario-20",
        "Scenario-23",
        "Scenario-33",
        "Scenario-6",
        "Scenario-8",
        "Scenario-91",
    }
    baseline_scores = {
        "Scenario-6": 0.0,
        "Scenario-8": 0.0,
        "Scenario-11": 0.0,
        "Scenario-19": 1.0,
        "Scenario-20": 1.0,
        "Scenario-23": 1.0,
        "Scenario-33": 0.5,
        "Scenario-91": 0.5,
        "Scenario-102": 0.0,
        "Scenario-105": 1.0,
    }

    assert [len(search), len(selection), len(holdout)] == [5, 5, 30]
    assert set(search + selection) == prior_development_set
    assert set(holdout).isdisjoint(prior_development_set)
    assert sorted(baseline_scores[item] for item in search) == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert sorted(baseline_scores[item] for item in selection) == [0.0, 0.0, 0.5, 1.0, 1.0]
    assert len(set(search + selection + holdout)) == 40
    adapter.validate(REPO)


def test_itbench_scoring_and_split_files_are_protected() -> None:
    adapter = Adapter()
    required = {
        "benchmarks/itbench/benchmark.py",
        "benchmarks/itbench/config.py",
        "benchmarks/itbench/data.py",
        "benchmarks/itbench/grader.py",
        "benchmarks/itbench/judge.py",
        "benchmarks/itbench/run_eval.py",
        "benchmarks/itbench/search_scenarios.json",
        "benchmarks/itbench/selection_scenarios.json",
        "benchmarks/itbench/holdout_scenarios.json",
    }

    assert required <= adapter.out_of_scope
    assert "vendor/stirrup" in adapter.editable_dirs


def test_itbench_agent_environment_respects_endpoint_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("AGENT_BASE_URLS", raising=False)
    env = Adapter().agent_env
    assert env["AGENT_BASE_URL"] == "http://localhost:8000/v1"
    assert env["AGENT_HARNESS"] == "stirrup"

    monkeypatch.setenv("AGENT_BASE_URLS", "http://one/v1,http://two/v1")
    env = Adapter().agent_env
    assert env["AGENT_BASE_URLS"] == "http://one/v1,http://two/v1"
    assert "AGENT_BASE_URL" not in env


def test_itbench_run_eval_pins_stirrup_and_scenarios(tmp_path: Path) -> None:
    result = SimpleNamespace(returncode=0)
    with patch("benchmarks.itbench.benchmark.subprocess.run", return_value=result) as run:
        rc = Adapter().run_eval(
            name="test-run",
            scenarios=["Scenario-12", "Scenario-13"],
            repeats=1,
            max_turns=5,
            concurrency=1,
            unit_timeout=30,
            repo=tmp_path,
            python="python",
            eval_env={"AGENT_HARNESS": "other"},
            log_path=tmp_path / "eval.log",
        )

    command = run.call_args.args[0]
    assert rc == 0
    assert command[command.index("--harness") + 1] == "stirrup"
    assert command[command.index("--scenarios") + 1] == "Scenario-12,Scenario-13"
    assert run.call_args.kwargs["env"]["AGENT_HARNESS"] == "stirrup"


def _write_summary(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    payload = {
        "harness": "stirrup",
        "n_tasks": 2,
        "overall_score": 0.5,
        "per_task": [
            {
                "scenario_id": "Scenario-12",
                "mean": 0.25,
                "repeat_scores": [0.0, 0.5],
            },
            {
                "scenario_id": "Scenario-13",
                "mean": 0.75,
                "repeat_scores": [0.5, 1.0],
            },
        ],
    }
    path = run_dir / "summary.json"
    path.write_text(json.dumps(payload))
    return path


def test_itbench_read_summary_requires_exact_valid_results(tmp_path: Path) -> None:
    path = _write_summary(tmp_path, "valid")
    adapter = Adapter()

    assert adapter.read_summary(
        "valid", tmp_path, ["Scenario-12", "Scenario-13"]
    ) == (0.5, {"Scenario-12": 0.25, "Scenario-13": 0.75}, 0.0)
    assert adapter.read_summary("valid", tmp_path, ["Scenario-12"]) is None

    payload = json.loads(path.read_text())
    payload["per_task"][0]["mean"] = 0.4
    path.write_text(json.dumps(payload))
    assert adapter.read_summary("valid", tmp_path) is None


def test_itbench_stage_sandbox_removes_ground_truth(tmp_path: Path) -> None:
    snapshot = tmp_path / "source"
    snapshot.mkdir()
    (snapshot / "ground_truth.yaml").write_text("answer: secret")
    (snapshot / "alerts.json").write_text("[]")
    task = Task("row", "Scenario-12", "public", "answer: secret", snapshot)

    destination = stage_sandbox(task, tmp_path / "sandbox")

    assert not (destination / "ground_truth.yaml").exists()
    assert (destination / "alerts.json").read_text() == "[]"


def test_itbench_gather_traces_caps_search_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    sre = data_root / "sre"
    sre.mkdir(parents=True)
    (sre / "data.jsonl").write_text(json.dumps({
        "scenario_id": "Scenario-12",
        "ground_truth_yaml": "root: deployment-a",
    }))
    repeat = tmp_path / "runs" / "run" / "Scenario-12" / "repeat_0"
    repeat.mkdir(parents=True)
    (repeat / "score.json").write_text('{"score": 0.5}')
    (repeat / "trace.log").write_text("0123456789")
    monkeypatch.setenv("ITBENCH_DATA", str(data_root))

    evidence = Adapter().gather_traces(
        "run", ["Scenario-12"], tmp_path / "runs", trace_chars=4
    )

    assert "root: deployment-a" in evidence
    assert "6789" in evidence
    assert "012345" not in evidence


@pytest.mark.parametrize(
    "added",
    [
        "+value = ground_truth\n",
        "+tasks = load_tasks(path)\n",
        '+if name == "Scenario-12": pass\n',
        '+Path("runs/x/score.json").read_text()\n',
    ],
)
def test_itbench_candidate_guard_rejects_answer_access(added: str) -> None:
    assert Adapter().candidate_violation(["benchmarks/itbench/prompts.py"], added)


def test_itbench_render_task_uses_runtime_search_override() -> None:
    frontier = {
        "search_mean": 0.25,
        "search_per_scenario": {"Scenario-6": 0.25},
        "baseline_search_per_scenario": {"Scenario-6": 0.0},
        "search_scenarios": ["Scenario-6"],
        "hypotheses": [],
        "discarded": [],
    }

    rendered = Adapter().render_task(1, 2, frontier, "TRACE")

    assert "Scenario-6: 0.250 (baseline 0.000)" in rendered
    assert "Scenario-11" not in rendered


def test_itbench_candidate_guard_allows_generic_harness_edit() -> None:
    diff = '+system_prompt = "Check evidence before answering."\n'
    assert Adapter().candidate_violation(["benchmarks/itbench/prompts.py"], diff) is None


def test_itbench_readme_mini_run_uses_the_development_pool() -> None:
    readme = (REPO / "README.md").read_text()
    adapter = Adapter()

    assert "--scenarios Scenario-6" in readme
    assert "--selection-scenarios Scenario-11" in readme
    assert "Scenario-6" in adapter.dev_scenarios
    assert "Scenario-11" in adapter.selection_scenarios()
    assert {"Scenario-6", "Scenario-11"}.isdisjoint(adapter.held_out_scenarios())
