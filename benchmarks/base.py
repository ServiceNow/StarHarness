"""BenchmarkAdapter protocol — the interface the evolving harness calls for each benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class AdapterValidationError(ValueError):
    """Raised when an adapter violates the StarHarness contract."""


class BenchmarkAdapter(ABC):
    """Plug-in interface: the evolving harness calls these methods to run a benchmark.

    Each benchmark lives under ``benchmarks/<name>/`` and provides a
    ``benchmark.py`` module that exports an ``Adapter`` class implementing this ABC.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique Python package name under ``benchmarks/``."""

    # ------------------------------------------------------------------
    # Editable surface configuration
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def editable_dirs(self) -> list[str]:
        """Repository-relative directories the proposer may edit."""

    @property
    @abstractmethod
    def out_of_scope(self) -> set[str]:
        """Files within editable_dirs that must never be edited (e.g. grader.py, judge.py)."""

    @property
    @abstractmethod
    def protected_files(self) -> set[str]:
        """Repo-root files never auto-deleted by stray cleanup."""

    # ------------------------------------------------------------------
    # Agent / eval environment
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def agent_env(self) -> dict[str, str]:
        """Env vars for the agent-under-test (AGENT_LLM, AGENT_BASE_URL, etc.)."""

    @property
    @abstractmethod
    def dev_scenarios(self) -> list[str]:
        """Scenario IDs for the dev/search set."""

    @abstractmethod
    def selection_scenarios(self) -> list[str]:
        """Scenario IDs for proposer-hidden candidate scoring."""

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @abstractmethod
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
        """Run the benchmark eval. Return exit code. Write output to log_path."""

    @abstractmethod
    def read_summary(
        self,
        name: str,
        runs_dir: Path,
        expected_scenarios: list[str] | None = None,
    ) -> tuple[float, dict[str, float], float] | None:
        """Parse the benchmark's summary output.

        Return (overall_mean, {scenario_id: mean}, verifier_pass_rate).
        verifier_pass_rate is the fraction of individual verifier checks that passed
        (used as a tiebreaker when task-level mean is tied). Return ``None`` when
        output is missing, malformed, or incomplete for ``expected_scenarios``.
        """

    @abstractmethod
    def import_check(self, repo: Path, python: str, eval_env: dict[str, str]) -> bool:
        """Verify the benchmark is importable after a candidate edit."""

    @abstractmethod
    def gather_traces(
        self,
        run_name: str,
        scenarios: list[str],
        runs_dir: Path,
        trace_chars: int,
    ) -> str:
        """Assemble per-scenario evidence (ground truth + score + trace) for the proposer."""

    @abstractmethod
    def held_out_scenarios(self) -> list[str]:
        """Scenarios for the held-out generalization test."""

    # ------------------------------------------------------------------
    # Proposer prompt paths
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def prompts_dir(self) -> Path:
        """Directory containing propose_task.md, proposer_prior.md for this benchmark."""

    @property
    @abstractmethod
    def surfaces_path(self) -> Path:
        """Path to the edit catalog (surfaces.md) for this benchmark."""

    @property
    @abstractmethod
    def domain_spec_path(self) -> Path:
        """Path to the domain spec for this benchmark."""

    @abstractmethod
    def render_task(
        self,
        iteration: int,
        n_iterations: int,
        frontier: dict[str, Any],
        traces: str,
    ) -> str:
        """Render the proposer task prompt for one iteration."""

    def validate_scenario_sets(
        self,
        search: list[str],
        selection: list[str],
        held_out: list[str],
    ) -> None:
        """Reject empty, duplicate, or overlapping benchmark partitions."""
        sets: dict[str, set[str]] = {}
        for label, scenarios in (
            ("search", search),
            ("selection", selection),
            ("held-out", held_out),
        ):
            if not scenarios:
                raise AdapterValidationError(f"{label} scenarios must not be empty")
            if any(
                not isinstance(item, str) or not item.strip() or item != item.strip()
                for item in scenarios
            ):
                raise AdapterValidationError(
                    f"{label} scenarios must contain trimmed, non-empty strings"
                )
            if len(scenarios) != len(set(scenarios)):
                raise AdapterValidationError(f"{label} scenarios contain duplicates")
            sets[label] = set(scenarios)

        for left, right in (("search", "selection"), ("search", "held-out"),
                            ("selection", "held-out")):
            overlap = sets[left] & sets[right]
            if overlap:
                raise AdapterValidationError(
                    f"{left} and {right} scenarios overlap: {sorted(overlap)}"
                )

    def validate(self, repo: Path) -> None:
        """Validate paths, prompts, environment values, and task partitions."""
        repo = repo.resolve()
        if not self.name.isidentifier() or self.name.lower() != self.name:
            raise AdapterValidationError(
                f"adapter name must be a lowercase Python identifier: {self.name!r}"
            )

        def check_relative(label: str, paths: list[str] | set[str]) -> None:
            for value in paths:
                path = Path(value)
                if path.is_absolute() or ".." in path.parts or value in {"", "."}:
                    raise AdapterValidationError(
                        f"{label} path must stay inside the repository: {value!r}"
                    )

        if not self.editable_dirs:
            raise AdapterValidationError("editable_dirs must not be empty")
        if len(self.editable_dirs) != len(set(self.editable_dirs)):
            raise AdapterValidationError("editable_dirs contains duplicates")
        check_relative("editable", self.editable_dirs)
        check_relative("out-of-scope", self.out_of_scope)
        check_relative("protected", self.protected_files)

        for value in self.editable_dirs:
            path = (repo / value).resolve()
            try:
                path.relative_to(repo)
            except ValueError as exc:
                raise AdapterValidationError(
                    f"editable directory resolves outside repository: {value}"
                ) from exc
            if not path.is_dir():
                raise AdapterValidationError(f"editable directory does not exist: {value}")
        for value in self.out_of_scope:
            if not any(value == root or value.startswith(root + "/") for root in self.editable_dirs):
                raise AdapterValidationError(f"out-of-scope path is not editable: {value}")
            path = (repo / value).resolve()
            try:
                path.relative_to(repo)
            except ValueError as exc:
                raise AdapterValidationError(
                    f"out-of-scope file resolves outside repository: {value}"
                ) from exc
            if not path.is_file():
                raise AdapterValidationError(f"out-of-scope file does not exist: {value}")

        module = repo / "benchmarks" / self.name / "benchmark.py"
        if not module.is_file():
            raise AdapterValidationError(f"adapter module does not exist: {module}")

        required_files = {
            self.prompts_dir / "proposer_prior.md",
            self.prompts_dir / "propose_task.md",
            self.surfaces_path,
            self.domain_spec_path,
        }
        for path in required_files:
            resolved = path.resolve()
            try:
                resolved.relative_to(repo)
            except ValueError as exc:
                raise AdapterValidationError(
                    f"required adapter file resolves outside repository: {path}"
                ) from exc
            if not resolved.is_file():
                raise AdapterValidationError(f"required adapter file does not exist: {path}")

        if any(
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\0" in key
            or not isinstance(value, str)
            or "\0" in value
            for key, value in self.agent_env.items()
        ):
            raise AdapterValidationError("agent_env must map strings to strings")

        self.validate_scenario_sets(
            self.dev_scenarios,
            self.selection_scenarios(),
            self.held_out_scenarios(),
        )

    def candidate_violation(self, changed_files: list[str], diff: str) -> str | None:
        """Return an adapter-specific rejection reason for a candidate diff."""
        return None
