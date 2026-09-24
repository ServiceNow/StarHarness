import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.automationbench.benchmark import Adapter
from benchmarks.automationbench.data import Task
from benchmarks.base import AdapterValidationError, BenchmarkAdapter
from evolving_harness import guardrail_violation, trace_analysis_prompt

REPO = Path(__file__).resolve().parents[1]


def test_finance_adapter_contract() -> None:
    adapter = Adapter()

    assert isinstance(adapter, BenchmarkAdapter)
    assert adapter.name == "automationbench"
    assert adapter.dev_scenarios
    assert adapter.selection_scenarios()
    assert adapter.held_out_scenarios()
    assert set(adapter.dev_scenarios).isdisjoint(adapter.selection_scenarios())
    assert set(adapter.dev_scenarios).isdisjoint(adapter.held_out_scenarios())
    assert set(adapter.selection_scenarios()).isdisjoint(adapter.held_out_scenarios())
    assert Path(adapter.domain_spec_path).is_file()
    assert Path(adapter.surfaces_path).is_file()
    adapter.validate(REPO)


def test_protected_scoring_files_are_not_editable() -> None:
    adapter = Adapter()

    required = {
        "benchmarks/automationbench/benchmark.py",
        "benchmarks/automationbench/config.py",
        "benchmarks/automationbench/data.py",
        "benchmarks/automationbench/grader.py",
        "benchmarks/automationbench/run_eval.py",
        "benchmarks/automationbench/proposer_prior.md",
        "benchmarks/automationbench/propose_task.md",
        "benchmarks/automationbench/search_scenarios.json",
        "benchmarks/automationbench/selection_scenarios.json",
        "benchmarks/automationbench/holdout_scenarios.json",
    }
    assert required <= adapter.out_of_scope
    assert "vendor/automationbench" not in adapter.editable_dirs


def test_split_validation_rejects_leakage() -> None:
    adapter = Adapter()

    with pytest.raises(AdapterValidationError, match="search and held-out"):
        adapter.validate_scenario_sets(["4002"], ["4005"], adapter.held_out_scenarios())


def test_agent_environment_respects_endpoint_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("AGENT_BASE_URLS", raising=False)
    assert Adapter().agent_env["AGENT_BASE_URL"] == "http://localhost:8000/v1"

    monkeypatch.setenv("AGENT_BASE_URLS", "http://one/v1,http://two/v1")
    env = Adapter().agent_env
    assert env["AGENT_BASE_URLS"] == "http://one/v1,http://two/v1"
    assert "AGENT_BASE_URL" not in env


def test_render_task_uses_search_scores_and_runtime_override() -> None:
    frontier = {
        "mean": 0.9,
        "per_scenario": {"4005": 0.9},
        "baseline_per_scenario": {"4005": 0.8},
        "search_mean": 0.25,
        "search_per_scenario": {"4001": 0.25},
        "baseline_search_per_scenario": {"4001": 0.1},
        "search_scenarios": ["4001"],
        "hypotheses": [],
        "discarded": [],
    }

    rendered = Adapter().render_task(1, 2, frontier, "TRACE")

    assert "search-1" in rendered
    assert "Mean score over proposer-visible search-1: 0.250" in rendered
    assert "4001: 0.250 (baseline 0.100)" in rendered
    assert "4005" not in rendered


def _write_summary(root: Path, name: str, task_ids: list[str]) -> None:
    tasks = [
        {"task_id": task_id, "mean": score, "guardrails_broken": broken}
        for task_id, score, broken in zip(task_ids, [0.25, 0.75], [0, 1], strict=True)
    ]
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "n_tasks": len(tasks),
        "overall_score": sum(task["mean"] for task in tasks) / len(tasks),
        "per_task": tasks,
    }))


def test_read_summary_requires_complete_expected_set(tmp_path: Path) -> None:
    _write_summary(tmp_path, "valid", ["4001", "4005"])
    adapter = Adapter()

    assert adapter.read_summary("valid", tmp_path, ["4001", "4005"]) == (
        0.5,
        {"4001": 0.25, "4005": 0.75},
        0.5,
    )
    assert adapter.read_summary("valid", tmp_path, ["4001", "4007"]) is None


def test_read_summary_rejects_score_mismatch(tmp_path: Path) -> None:
    _write_summary(tmp_path, "bad", ["4001", "4005"])
    path = tmp_path / "bad" / "summary.json"
    payload = json.loads(path.read_text())
    payload["overall_score"] = 1.0
    path.write_text(json.dumps(payload))

    assert Adapter().read_summary("bad", tmp_path, ["4001", "4005"]) is None


def test_run_eval_pins_finance_and_stirrup(tmp_path: Path) -> None:
    result = SimpleNamespace(returncode=0)
    with patch("benchmarks.automationbench.benchmark.subprocess.run", return_value=result) as run:
        rc = Adapter().run_eval(
            name="test-run",
            scenarios=["4001"],
            repeats=1,
            max_turns=5,
            concurrency=1,
            unit_timeout=30,
            repo=tmp_path,
            python="python",
            eval_env={"AB_DOMAIN": "other", "AGENT_HARNESS": "other"},
            log_path=tmp_path / "eval.log",
        )

    command = run.call_args.args[0]
    assert rc == 0
    assert command[command.index("--domain") + 1] == "finance"
    assert command[command.index("--harness") + 1] == "stirrup"
    assert command[command.index("--tasks") + 1] == "4001"


def test_gather_traces_caps_search_evidence(tmp_path: Path) -> None:
    repeat = tmp_path / "run" / "4001" / "repeat_0"
    repeat.mkdir(parents=True)
    (repeat / "score.json").write_text('{"aa_score": 0.5}')
    (repeat / "trace.log").write_text("0123456789")
    task = SimpleNamespace(task_id="4001", name="finance.test", assertions=[{"type": "x"}])

    with patch("benchmarks.automationbench.data.load_tasks", return_value=[task]):
        evidence = Adapter().gather_traces("run", ["4001"], tmp_path, trace_chars=4)

    assert "finance.test" in evidence
    assert '"type": "x"' in evidence
    assert "6789" in evidence
    assert "012345" not in evidence


def test_agent_task_hides_grading_data_and_copies_state() -> None:
    task = Task(
        task_id="4001",
        name="finance.test",
        system_prompt="system",
        user_prompt="user",
        initial_state={"meta": {"current_time": "2026-01-01T00:00:00Z"}, "gmail": {}},
        assertions=[{"type": "gmail_message_sent"}],
        zapier_tools=[],
    )

    agent_task = task.for_agent()
    agent_task.initial_state["gmail"]["changed"] = True

    assert not hasattr(agent_task, "assertions")
    assert not hasattr(agent_task, "raw_info")
    assert not hasattr(agent_task, "task_id")
    assert not hasattr(agent_task, "name")
    assert task.initial_state["gmail"] == {}
    assert "gmail" in agent_task.allowed_services


@pytest.mark.parametrize(
    "added",
    [
        "+value = task.assertions\n",
        "+tasks = load_tasks(\"finance\")\n",
        "+if task.task_id == \"4001\": pass\n",
        "+Path(\"runs/x/score.json\").read_text()\n",
    ],
)
def test_candidate_guard_rejects_answer_access(added: str) -> None:
    assert Adapter().candidate_violation(["benchmarks/automationbench/agent_setup.py"], added)


def test_candidate_guard_allows_generic_harness_edit() -> None:
    diff = '+system_prompt = "Check tool results before writing."\n'
    assert Adapter().candidate_violation(
        ["benchmarks/automationbench/agent_setup.py"], diff
    ) is None


def test_evolution_loop_calls_adapter_candidate_guard() -> None:
    path = "benchmarks/automationbench/agent_setup.py"
    diff = '+value = task.assertions\n'
    assert guardrail_violation(Adapter(), [path], diff)


def test_readme_mini_run_keeps_held_out_tasks_hidden() -> None:
    readme = (REPO / "README.md").read_text()
    adapter = Adapter()

    assert "--scenarios 4001" in readme
    assert "--selection-scenarios 4005" in readme
    assert "4001" in adapter.dev_scenarios
    assert "4005" in adapter.selection_scenarios()
    assert {"4001", "4005"}.isdisjoint(adapter.held_out_scenarios())


def test_inline_trace_analysis_adds_no_prompt() -> None:
    assert trace_analysis_prompt("inline", ["4001"], 8) == ""


def test_subagent_trace_analysis_is_isolated_and_bounded() -> None:
    prompt = trace_analysis_prompt("subagents", ["4001", "4003"], 2)

    assert "one isolated OMP `task` subagent per scenario" in prompt
    assert "Spawn 2 workers" in prompt
    assert "4001, 4003" in prompt
    assert "waves of at\nmost 2 workers" in prompt
    assert "Workers must not edit files or spawn workers" in prompt
    assert "aggregate the reports across scenarios" in prompt
    assert "selection and held-out evidence hidden" in prompt


def test_subagent_trace_analysis_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="wave size must be positive"):
        trace_analysis_prompt("subagents", ["4001"], 0)
