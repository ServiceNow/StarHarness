"""Wire a Stirrup Agent for one AutomationBench-AA task: the API toolset over a WorldState.

AutomationBench-AA gives the model three structured tools — `api_search`
(discover endpoints), `api_fetch` (call simulated REST endpoints against the
world), and `base64_encode` (Gmail body helper) — discovering and calling the
REST endpoints it needs. Stirrup's Agent IS the multi-turn loop (upstream's
verifiers loop is not used); each (task, repeat) unit gets a fresh WorldState
built from the task's initial_state, mutated in-process by api_fetch.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from automationbench.schema.world import WorldState
from automationbench.tools.api import api_fetch, api_search, base64_encode
from pydantic import BaseModel
from stirrup import Agent
from stirrup.clients.litellm_client import LiteLLMClient
from stirrup.core.models import Tool, ToolResult
from stirrup.utils.logging import AgentLogger

from . import tracelog
from .config import RunConfig
from .data import AgentTask
from .progress import ProgressReporter

# --------------------------------------------------------------------------------------
# API toolset as Stirrup Tools
# --------------------------------------------------------------------------------------


class ApiSearchParams(BaseModel):
    query: str
    top_k: int = 5


class ApiFetchParams(BaseModel):
    method: str
    url: str
    params: str | None = None
    body: str | None = None


class Base64EncodeParams(BaseModel):
    text: str


def build_api_tools(world: WorldState) -> list[Tool]:
    """Build the AutomationBench API toolset as Stirrup Tools bound to `world`.

    Descriptions are the upstream functions' docstrings verbatim — that is the
    surface the verifiers env exposes to the model. Executor errors are returned
    as tool content (success=True) so the agent sees the failure and can recover,
    matching how a real API responds with an error body.
    """

    def _search(p: ApiSearchParams) -> ToolResult:
        try:
            return ToolResult(content=api_search(query=p.query, top_k=p.top_k))
        except Exception as e:  # noqa: BLE001 — surface as a tool error, don't kill the run
            return ToolResult(content=f"error: {type(e).__name__}: {e}")

    def _fetch(p: ApiFetchParams) -> ToolResult:
        try:
            return ToolResult(content=api_fetch(
                world, method=p.method, url=p.url, params=p.params, body=p.body,
            ))
        except Exception as e:  # noqa: BLE001
            return ToolResult(content=f"error: {type(e).__name__}: {e}")

    def _b64(p: Base64EncodeParams) -> ToolResult:
        try:
            return ToolResult(content=base64_encode(text=p.text))
        except Exception as e:  # noqa: BLE001
            return ToolResult(content=f"error: {type(e).__name__}: {e}")

    return [
        Tool(name="api_search", description=inspect.getdoc(api_search) or "",
             parameters=ApiSearchParams, executor=_search),
        Tool(name="api_fetch", description=inspect.getdoc(api_fetch) or "",
             parameters=ApiFetchParams, executor=_fetch),
        Tool(name="base64_encode", description=inspect.getdoc(base64_encode) or "",
             parameters=Base64EncodeParams, executor=_b64),
    ]


def build_world(task: AgentTask) -> WorldState:
    """Fresh WorldState for one unit, with upstream service gating applied."""
    world = WorldState(**task.initial_state)
    world.meta.allowed_services = task.allowed_services
    return world


# --------------------------------------------------------------------------------------
# Agent construction + run
# --------------------------------------------------------------------------------------


class _ProgressLogger(AgentLogger):
    """AgentLogger that also reports each step's turn to a ProgressReporter."""

    def __init__(self, reporter: ProgressReporter, key: str) -> None:
        super().__init__(show_spinner=False)
        self._reporter = reporter
        self._key = key

    def on_step(self, step: int, tool_calls: int = 0, input_tokens: int = 0,
                output_tokens: int = 0) -> None:
        super().on_step(step, tool_calls, input_tokens, output_tokens)
        self._reporter.update_turn(self._key, step)


def build_agent(cfg: RunConfig, task: AgentTask, world: WorldState,
                logger: AgentLogger, base_url: str | None = None) -> Agent:
    client = LiteLLMClient(
        model=cfg.litellm_model,
        api_key=cfg.api_key,
        reasoning_effort=cfg.reasoning_effort,
        kwargs=cfg.litellm_kwargs(base_url),
    )
    return Agent(
        client=client,
        name="automation_agent",
        system_prompt=task.system_prompt,  # the task's embedded system prompt, verbatim
        max_turns=cfg.max_turns,
        tools=build_api_tools(world),  # finish tool is added by Agent automatically
        logger=logger,
    )


async def run_task(cfg: RunConfig, task: AgentTask, trace_path: Path,
                   logger: AgentLogger, base_url: str | None = None) -> WorldState:
    """Run one repeat. Returns the final WorldState for grading.

    The full per-turn agent trace for THIS unit is written to `trace_path` (one
    file per (task, repeat)), so concurrent units never interleave.
    """
    world = build_world(task)
    with tracelog.trace_to(trace_path):
        agent = build_agent(cfg, task, world, logger, base_url)
        async with agent.session() as session:
            await session.run(task.user_prompt)
    return world
