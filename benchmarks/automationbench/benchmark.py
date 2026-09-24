"""AutomationBenchAdapter — BenchmarkAdapter implementation for AutomationBench-AA (finance)."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.base import BenchmarkAdapter

_BENCH_DIR = Path(__file__).resolve().parent
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _read_scenario_file(filename: str) -> list[str]:
    path = _BENCH_DIR / filename
    return list(json.loads(path.read_text())["scenarios"])


class Adapter(BenchmarkAdapter):
    """AutomationBench (finance domain) benchmark adapter."""

    @property
    def name(self) -> str:
        return "automationbench"

    @property
    def editable_dirs(self) -> list[str]:
        return ["benchmarks/automationbench", "vendor/stirrup"]

    @property
    def out_of_scope(self) -> set[str]:
        return {
            "benchmarks/automationbench/benchmark.py",
            "benchmarks/automationbench/config.py",
            "benchmarks/automationbench/grader.py",
            "benchmarks/automationbench/data.py",
            "benchmarks/automationbench/domain_spec.md",
            "benchmarks/automationbench/progress.py",
            "benchmarks/automationbench/propose_task.md",
            "benchmarks/automationbench/proposer_prior.md",
            "benchmarks/automationbench/run_eval.py",
            "benchmarks/automationbench/search_scenarios.json",
            "benchmarks/automationbench/selection_scenarios.json",
            "benchmarks/automationbench/holdout_scenarios.json",
            "benchmarks/automationbench/surfaces.md",
            "benchmarks/automationbench/tracelog.py",
        }

    @property
    def protected_files(self) -> set[str]:
        return {
            "evolving_harness.py", "omp_wrapper.py",
            "benchmarks/automationbench/domain_spec.md",
            "benchmarks/automationbench/surfaces.md",
            "benchmarks/automationbench/search_scenarios.json",
            "benchmarks/automationbench/selection_scenarios.json",
            "benchmarks/automationbench/holdout_scenarios.json",
            "README.md", "pyproject.toml", "uv.lock",
            ".gitignore", ".env", ".env.example",
        }

    @property
    def agent_env(self) -> dict[str, str]:
        env = {
            "AGENT_LLM": os.environ.get("AGENT_LLM", "gpt-5.4"),
            "AGENT_REASONING_EFFORT": os.environ.get("AGENT_REASONING_EFFORT", "medium"),
            "AGENT_HARNESS": "stirrup",
            "AB_DOMAIN": "finance",
        }
        if urls := os.environ.get("AGENT_BASE_URLS", "").strip():
            env["AGENT_BASE_URLS"] = urls
        else:
            env["AGENT_BASE_URL"] = os.environ.get(
                "AGENT_BASE_URL", "https://api.openai.com/v1"
            )
        return env

    @property
    def dev_scenarios(self) -> list[str]:
        """Search split (proposer-visible) from search_scenarios.json."""
        return _read_scenario_file("search_scenarios.json")

    def selection_scenarios(self) -> list[str]:
        """Selection split (proposer-hidden) from selection_scenarios.json."""
        return _read_scenario_file("selection_scenarios.json")

    def held_out_scenarios(self) -> list[str]:
        """Held-out generalization set from holdout_scenarios.json."""
        return _read_scenario_file("holdout_scenarios.json")

    @property
    def prompts_dir(self) -> Path:
        return _BENCH_DIR

    @property
    def surfaces_path(self) -> Path:
        return _BENCH_DIR / "surfaces.md"

    @property
    def domain_spec_path(self) -> Path:
        return _BENCH_DIR / "domain_spec.md"

    def run_eval(
        self,
        name: str,
        scenarios: list[str],
        repeats: int,
        max_turns: int,
        concurrency: int,
        unit_timeout: float | None,
        repo: Path,
        python: str,
        eval_env: dict[str, str],
        log_path: Path,
    ) -> int:
        if not _RUN_NAME.fullmatch(name):
            raise ValueError(f"invalid run name: {name!r}")
        if not scenarios:
            raise ValueError("AutomationBench evaluations require explicit scenario IDs")
        if repeats < 1 or max_turns < 1 or concurrency < 1:
            raise ValueError("repeats, max_turns, and concurrency must be positive")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            python, "-m", "benchmarks.automationbench.run_eval",
            "--harness", "stirrup",
            "--domain", "finance",
            "--repeats", str(repeats),
            "--max-turns", str(max_turns),
            "--max-concurrency", str(concurrency),
            "--name", name,
        ]
        if scenarios:
            cmd += ["--tasks", ",".join(scenarios)]
        if unit_timeout:
            cmd += ["--unit-timeout", str(unit_timeout)]
        with open(log_path, "w") as f:
            f.write(f"$ {shlex.join(cmd)}\n\n")
            f.flush()
            proc = subprocess.run(
                cmd, cwd=str(repo), env=eval_env, text=True,
                stdout=f, stderr=subprocess.STDOUT, check=False,
            )
        return proc.returncode

    def read_summary(
        self,
        name: str,
        runs_dir: Path,
        expected_scenarios: list[str] | None = None,
    ) -> tuple[float, dict[str, float], float] | None:
        path = runs_dir / name / "summary.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

        tasks = data.get("per_task") if isinstance(data, dict) else None
        if not isinstance(tasks, list) or not tasks:
            return None
        per: dict[str, float] = {}
        intact = 0
        for task in tasks:
            if not isinstance(task, dict):
                return None
            task_id = task.get("task_id")
            mean = task.get("mean")
            broken = task.get("guardrails_broken")
            if (
                not isinstance(task_id, str)
                or task_id in per
                or not isinstance(mean, (int, float))
                or isinstance(mean, bool)
                or not math.isfinite(mean)
                or not 0.0 <= mean <= 1.0
                or not isinstance(broken, (int, float))
                or isinstance(broken, bool)
                or not math.isfinite(broken)
                or broken < 0
            ):
                return None
            per[task_id] = float(mean)
            intact += broken == 0

        if data.get("n_tasks") != len(tasks):
            return None
        if expected_scenarios is not None and (
            len(tasks) != len(expected_scenarios) or set(per) != set(expected_scenarios)
        ):
            return None

        overall = data.get("overall_score")
        calculated = sum(per.values()) / len(per)
        if (
            not isinstance(overall, (int, float))
            or isinstance(overall, bool)
            or not math.isfinite(overall)
            or not 0.0 <= overall <= 1.0
            or not math.isclose(float(overall), calculated, abs_tol=1e-9)
        ):
            return None

        # Guardrail-intact rate as the tiebreaker metric: fraction of tasks with
        # no guardrail broken (AA zeroes a task on any guardrail violation).
        return float(overall), per, intact / len(tasks)

    def import_check(self, repo: Path, python: str, eval_env: dict[str, str]) -> bool:
        r = subprocess.run(
            [python, "-c", "import benchmarks.automationbench.run_eval"],
            cwd=str(repo), env=eval_env, capture_output=True, text=True,
            check=False,
        )
        return r.returncode == 0

    def candidate_violation(self, changed_files: list[str], diff: str) -> str | None:
        """Block task-answer access and per-task branches in editable harness code."""
        added = "\n".join(
            line[1:] for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        lowered = added.casefold()
        forbidden = {
            "task.assertions",
            "task.raw_info",
            "task.task_id",
            "task.name",
            "load_tasks(",
            "get_domain_dataset",
            "automationbench.domains",
            "automationbench.rubric",
            "benchmarks.automationbench.grader",
            "harness_world",
            "summary.json",
            "score.json",
        }
        if match := next((value for value in sorted(forbidden) if value in lowered), None):
            return f"candidate added forbidden benchmark-answer access: {match}"

        known_ids = self.dev_scenarios + self.selection_scenarios() + self.held_out_scenarios()
        task_id_pattern = rf"[\"'](?:{'|'.join(map(re.escape, known_ids))})[\"']"
        if re.search(task_id_pattern, added):
            return "candidate hardcoded an AutomationBench task ID"
        return None

    def gather_traces(
        self,
        run_name: str,
        scenarios: list[str],
        runs_dir: Path,
        trace_chars: int,
    ) -> str:
        from benchmarks.automationbench import data as abdata

        tasks_by_id = {t.task_id: t for t in abdata.load_tasks("finance")}
        sections = []
        for sc in scenarios:
            task = tasks_by_id.get(sc)
            assertions_txt = (
                json.dumps(task.assertions, indent=2) if task
                else "(task not found in dataset)"
            )
            rep = runs_dir / run_name / sc / "repeat_0"
            score_txt = "(score.json missing)"
            trace_txt = "(trace.log missing)"
            sp = rep / "score.json"
            tp = rep / "trace.log"
            if sp.exists():
                score_txt = sp.read_text()
            if tp.exists():
                t = tp.read_text(errors="replace")
                if trace_chars and len(t) > trace_chars:
                    t = f"…[truncated to last {trace_chars} chars]…\n" + t[-trace_chars:]
                trace_txt = t
            sections.append(
                f"### {sc} ({task.name if task else 'unknown'})\n\n"
                f"**assertions (what the final state SHOULD satisfy — for diagnosis only, "
                f"never leak to the agent)**\n```json\n{assertions_txt}\n```\n\n"
                f"**score.json (per-assertion grading)**\n```json\n{score_txt}\n```\n\n"
                f"**trace.log (what the agent actually did)**\n```\n{trace_txt}\n```\n"
            )
        return "\n".join(sections)

    def render_task(
        self,
        iteration: int,
        n_iterations: int,
        frontier: dict[str, Any],
        traces: str,
    ) -> str:
        tmpl = (self.prompts_dir / "propose_task.md").read_text()
        scenarios = frontier.get("search_scenarios") or self.dev_scenarios
        search_per = frontier.get("search_per_scenario") or {}
        baseline_search_per = frontier.get("baseline_search_per_scenario") or search_per
        per = "\n".join(
            f"  - {scenario}: {search_per.get(scenario, 0.0):.3f} "
            f"(baseline {baseline_search_per.get(scenario, 0.0):.3f})"
            for scenario in scenarios
        )
        if frontier["hypotheses"]:
            kept = "\n".join(
                f"- **KEPT** ({h['name']}, +{h['delta']:.3f}): {h['hypothesis']}"
                for h in frontier["hypotheses"]
            )
        else:
            kept = "- (nothing kept yet — the frontier is the unmodified baseline)"
        discarded = frontier.get("discarded", [])
        if discarded:
            disc = "\n".join(
                f"- **{d['decision'].upper()}** ({d['name']}): {d['hypothesis']}"
                for d in discarded
            )
        else:
            disc = "- (nothing discarded yet)"
        history = f"### Kept (on the frontier)\n{kept}\n\n### Tried & discarded\n{disc}"
        return tmpl.format(
            iteration=iteration,
            n_iterations=n_iterations,
            n_scenarios=len(scenarios),
            frontier_score=f"{frontier.get('search_mean', 0.0):.3f}",
            frontier_per_scenario=per,
            kept_hypotheses=history,
            scenario_traces_section=traces,
            surfaces_path=str(self.surfaces_path),
            domain_spec_path=str(self.domain_spec_path),
            pending_eval_path="pending_eval.json",
        )
