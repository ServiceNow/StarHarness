"""BenchmarkAdapter implementation for the ITBench SRE benchmark."""

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
    return list(json.loads((_BENCH_DIR / filename).read_text())["scenarios"])


class Adapter(BenchmarkAdapter):
    """ITBench SRE adapter with search, selection, and holdout isolation."""

    @property
    def name(self) -> str:
        return "itbench"

    @property
    def editable_dirs(self) -> list[str]:
        return ["benchmarks/itbench", "vendor/stirrup"]

    @property
    def out_of_scope(self) -> set[str]:
        return {
            "benchmarks/itbench/benchmark.py",
            "benchmarks/itbench/config.py",
            "benchmarks/itbench/data.py",
            "benchmarks/itbench/domain_spec.md",
            "benchmarks/itbench/grader.py",
            "benchmarks/itbench/judge.py",
            "benchmarks/itbench/progress.py",
            "benchmarks/itbench/propose_task.md",
            "benchmarks/itbench/proposer_prior.md",
            "benchmarks/itbench/run_eval.py",
            "benchmarks/itbench/search_scenarios.json",
            "benchmarks/itbench/selection_scenarios.json",
            "benchmarks/itbench/holdout_scenarios.json",
            "benchmarks/itbench/surfaces.md",
            "benchmarks/itbench/tracelog.py",
        }

    @property
    def protected_files(self) -> set[str]:
        return {
            "evolving_harness.py", "omp_wrapper.py", "stratify_tasks.py",
            "benchmarks/itbench/domain_spec.md", "benchmarks/itbench/surfaces.md",
            "benchmarks/itbench/search_scenarios.json",
            "benchmarks/itbench/selection_scenarios.json",
            "benchmarks/itbench/holdout_scenarios.json",
            "README.md", "pyproject.toml", "uv.lock", ".gitignore", ".env", ".env.example",
        }

    @property
    def agent_env(self) -> dict[str, str]:
        env = {
            "AGENT_LLM": os.environ.get("AGENT_LLM", "gpt-5.4"),
            "AGENT_REASONING_EFFORT": os.environ.get("AGENT_REASONING_EFFORT", "medium"),
            "AGENT_HARNESS": "stirrup",
            "ITBENCH_DATA": os.environ.get("ITBENCH_DATA", "./data"),
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
        return _read_scenario_file("search_scenarios.json")

    def selection_scenarios(self) -> list[str]:
        return _read_scenario_file("selection_scenarios.json")

    def held_out_scenarios(self) -> list[str]:
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
            raise ValueError("ITBench evaluations require explicit scenario IDs")
        if repeats < 1 or max_turns < 1 or concurrency < 1:
            raise ValueError("repeats, max_turns, and concurrency must be positive")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            python, "-m", "benchmarks.itbench.run_eval",
            "--harness", "stirrup",
            "--scenarios", ",".join(scenarios),
            "--repeats", str(repeats),
            "--max-turns", str(max_turns),
            "--max-concurrency", str(concurrency),
            "--name", name,
        ]
        if unit_timeout:
            cmd += ["--unit-timeout", str(unit_timeout)]
        with open(log_path, "w") as stream:
            stream.write(f"$ {shlex.join(cmd)}\n\n")
            stream.flush()
            result = subprocess.run(
                cmd,
                cwd=str(repo),
                env={**eval_env, "AGENT_HARNESS": "stirrup"},
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return result.returncode

    def read_summary(
        self,
        name: str,
        runs_dir: Path,
        expected_scenarios: list[str] | None = None,
    ) -> tuple[float, dict[str, float], float] | None:
        try:
            data = json.loads((runs_dir / name / "summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None
        tasks = data.get("per_task") if isinstance(data, dict) else None
        if not isinstance(tasks, list) or not tasks or data.get("harness") != "stirrup":
            return None

        per: dict[str, float] = {}
        for task in tasks:
            if not isinstance(task, dict):
                return None
            scenario_id = task.get("scenario_id")
            mean = task.get("mean")
            repeats = task.get("repeat_scores")
            if (
                not isinstance(scenario_id, str)
                or scenario_id in per
                or not isinstance(mean, (int, float))
                or isinstance(mean, bool)
                or not math.isfinite(mean)
                or not 0.0 <= mean <= 1.0
                or not isinstance(repeats, list)
                or not repeats
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or not 0.0 <= value <= 1.0
                    for value in repeats
                )
                or not math.isclose(float(mean), sum(repeats) / len(repeats), abs_tol=1e-9)
            ):
                return None
            per[scenario_id] = float(mean)

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
        return float(overall), per, 0.0

    def import_check(self, repo: Path, python: str, eval_env: dict[str, str]) -> bool:
        result = subprocess.run(
            [python, "-c", "import benchmarks.itbench.run_eval"],
            cwd=str(repo), env=eval_env, capture_output=True, text=True, check=False,
        )
        return result.returncode == 0

    def candidate_violation(self, changed_files: list[str], diff: str) -> str | None:
        added = "\n".join(
            line[1:] for line in diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        lowered = added.casefold()
        forbidden = {
            "ground_truth", "scenario_id", "summary.json", "score.json",
            "benchmarks.itbench.data", "benchmarks.itbench.grader", "benchmarks.itbench.judge",
            "load_tasks(", "download(",
        }
        if match := next((value for value in sorted(forbidden) if value in lowered), None):
            return f"candidate added forbidden ITBench answer access: {match}"
        known_ids = self.dev_scenarios + self.selection_scenarios() + self.held_out_scenarios()
        if re.search(rf"[\"'](?:{'|'.join(map(re.escape, known_ids))})[\"']", added):
            return "candidate hardcoded an ITBench scenario ID"
        return None

    def gather_traces(
        self,
        run_name: str,
        scenarios: list[str],
        runs_dir: Path,
        trace_chars: int,
    ) -> str:
        data_root = Path(os.environ.get("ITBENCH_DATA") or "./data")
        data_jsonl = data_root / "sre" / "data.jsonl"
        ground_truth: dict[str, str] = {}
        if data_jsonl.is_file():
            for line in data_jsonl.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if scenario_id := record.get("scenario_id"):
                    ground_truth[scenario_id] = record.get("ground_truth_yaml", "")

        sections = []
        for scenario in scenarios:
            repeat_dir = runs_dir / run_name / scenario / "repeat_0"
            score_path = repeat_dir / "score.json"
            trace_path = repeat_dir / "trace.log"
            score = score_path.read_text() if score_path.is_file() else "(score.json missing)"
            trace = trace_path.read_text(errors="replace") if trace_path.is_file() else "(trace.log missing)"
            if trace_chars and len(trace) > trace_chars:
                trace = f"…[truncated to last {trace_chars} chars]…\n" + trace[-trace_chars:]
            sections.append(
                f"### {scenario}\n\n"
                "**ground truth (proposer diagnosis only; never expose it to the agent)**\n"
                f"```yaml\n{ground_truth.get(scenario, '(ground truth unavailable)')}\n```\n\n"
                f"**score.json**\n```json\n{score}\n```\n\n"
                f"**trace.log**\n```\n{trace}\n```\n"
            )
        return "\n".join(sections)

    def render_task(
        self,
        iteration: int,
        n_iterations: int,
        frontier: dict[str, Any],
        traces: str,
    ) -> str:
        template = (self.prompts_dir / "propose_task.md").read_text()
        scenarios = frontier.get("search_scenarios") or self.dev_scenarios
        search_per = frontier.get("search_per_scenario") or {}
        baseline_per = frontier.get("baseline_search_per_scenario") or search_per
        scores = "\n".join(
            f"  - {scenario}: {search_per.get(scenario, 0.0):.3f} "
            f"(baseline {baseline_per.get(scenario, 0.0):.3f})"
            for scenario in scenarios
        )
        kept = frontier.get("hypotheses") or []
        discarded = frontier.get("discarded") or []
        history = "\n".join(
            [f"- KEPT {item['name']}: {item['hypothesis']}" for item in kept]
            + [f"- {item['decision'].upper()} {item['name']}: {item['hypothesis']}" for item in discarded]
        ) or "- No prior attempts."
        return template.format(
            iteration=iteration,
            n_iterations=n_iterations,
            n_scenarios=len(scenarios),
            frontier_score=f"{frontier.get('search_mean', 0.0):.3f}",
            frontier_per_scenario=scores,
            kept_hypotheses=history,
            scenario_traces_section=traces,
            surfaces_path=str(self.surfaces_path),
            domain_spec_path=str(self.domain_spec_path),
            pending_eval_path="pending_eval.json",
        )
