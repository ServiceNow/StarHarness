"""Task loading for the AutomationBench-AA replication.

Tasks come from the vendored AutomationBench package's in-repo domain datasets
(e.g. automationbench.domains.finance.get_finance_dataset) — no download needed.
Each row: example_id, task (name), prompt (chat messages: system + user trigger),
info (JSON string with zapier_tools, initial_state, assertions).
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from automationbench.domains import get_domain_dataset
from automationbench.schema.world import WorldState


def strip_none_values(obj: Any) -> Any:
    """Recursively strip None values from nested dicts and lists.

    Verbatim from automationbench.runner — HF Dataset normalization adds all
    possible keys with None for missing values, which breaks Pydantic
    default_factory when the row is re-validated as a WorldState.
    """
    if isinstance(obj, dict):
        return {k: strip_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [strip_none_values(item) for item in obj if item is not None]
    else:
        return obj


@dataclass
class AgentTask:
    """Task fields exposed to editable agent code."""

    system_prompt: str
    user_prompt: str
    initial_state: dict
    allowed_services: list[str]


@dataclass
class Task:
    """Trusted AutomationBench task with private grading fields."""

    task_id: str               # str(example_id), e.g. "4001"
    name: str                  # e.g. "finance.invoice_email_extract"
    system_prompt: str         # embedded system message, used verbatim
    user_prompt: str           # the trigger message
    initial_state: dict        # WorldState dict (None-stripped)
    assertions: list[dict]     # scored assertions (objectives + guardrails)
    zapier_tools: list[str] = field(default_factory=list)

    def for_agent(self) -> AgentTask:
        """Copy only fields that editable harness code may inspect."""
        return AgentTask(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            initial_state=deepcopy(self.initial_state),
            allowed_services=_allowed_services(
                self.initial_state,
                self.assertions,
                self.zapier_tools,
            ),
        )


_SERVICE_FIELDS = sorted(
    (str(field) for field in WorldState.model_fields if field != "meta"),
    key=len,
    reverse=True,
)


def _service_for_name(name: str) -> str | None:
    for service_field in _SERVICE_FIELDS:
        if name == service_field or name.startswith(service_field + "_"):
            return service_field
    return None


def _allowed_services(
    initial_state: dict,
    assertions: list[dict],
    zapier_tools: list[str],
) -> list[str]:
    allowed = {
        key for key in initial_state if key != "meta" and key in WorldState.model_fields
    }
    for assertion in assertions:
        if service := _service_for_name(str(assertion.get("type", ""))):
            allowed.add(service)
    for tool_name in zapier_tools:
        if service := _service_for_name(tool_name):
            allowed.add(service)
    return sorted(allowed)


@lru_cache(maxsize=None)
def load_tasks(domain: str = "finance") -> list[Task]:
    """Load all tasks for a domain, in dataset order (cached — datasets are static)."""
    ds = get_domain_dataset(domain)
    tasks: list[Task] = []
    for row in ds:
        info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
        messages = row["prompt"]
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        initial_state = strip_none_values(info.get("initial_state", {}))
        assertions = [strip_none_values(a) for a in info.get("assertions", [])]
        tasks.append(Task(
            task_id=str(row["example_id"]),
            name=row["task"],
            system_prompt=system,
            user_prompt=user,
            initial_state=initial_state,
            assertions=assertions,
            zapier_tools=info.get("zapier_tools", []),
        ))
    return tasks


def filter_tasks(tasks: list[Task], task_ids: list[str] | None = None,
                 limit: int | None = None) -> list[Task]:
    """Filter to explicit task ids (preserving the requested order), then cap at limit."""
    if task_ids:
        order = {t: i for i, t in enumerate(task_ids)}
        tasks = [t for t in tasks if t.task_id in order or t.name in order]
        tasks.sort(key=lambda t: order.get(t.task_id, order.get(t.name, len(order))))
    if limit:
        tasks = tasks[:limit]
    return tasks
