# Stirrup Edit Catalog — AutomationBench (Finance)

Every location in the harness and Stirrup source that can be modified to change agent behavior.
All paths are relative to the repository root. Before editing any file, save a backup:
`cp <file> <file>.bak`. To revert: `cp <file>.bak <file>`. To keep: delete the `.bak`.

> Symbols and defaults verified against source.
>
> The proposer may edit `benchmarks/automationbench/agent_setup.py`, create new harness modules
> under `benchmarks/automationbench/`, and edit `vendor/stirrup/`. The adapter rejects changes to
> the evaluation, scoring, task-loading, split, trace, and proposer-policy files listed below.
>
> **Cross-benchmark safety:** `vendor/stirrup/` is shared across all benchmarks. After each candidate
> edit, the harness checks that **all** benchmarks still import cleanly — a `vendor/stirrup/` change
> that breaks another benchmark is automatically rejected. Benchmark-specific changes (under
> `benchmarks/automationbench/`) are inherently isolated and always safe.
>
> **Out of scope (hard):** `benchmark.py`, `config.py`, `data.py`, `grader.py`, `run_eval.py`,
> `progress.py`, `tracelog.py`, all scenario JSON files, proposer templates, experiment specs,
> and anything under `vendor/automationbench/`. The adapter also rejects added code that reads
> assertions, scores, task IDs, or upstream task loaders.

---

## Change families

Two families of change exist — analyze the evidence and decide which addresses the failure mode
you found. Do not default to either one.

**Family A — Prompt surfaces** (change *what the agent is told / how much it sees*):
- `benchmarks/automationbench/agent_setup.py` — tool descriptions, system-prompt handling, Agent options
- `vendor/stirrup/src/stirrup/constants.py` — `CONTEXT_SUMMARIZATION_CUTOFF`, `TURNS_REMAINING_WARNING_THRESHOLD`
- `vendor/stirrup/src/stirrup/prompts/*.txt` — base system prompt, summarizer prompts

**Family B — Structural / architectural levers** (change *what the agent can do or how it works*):
- add **custom tools** (`stirrup.core.models.Tool`: name, description, pydantic params, executor)
- add **sub-agents** via `Agent.to_tool()` (agent.py:1523) — e.g. a dedicated specialist the main
  agent delegates to
- add **skills** (`skills_dir=` on `agent.session()`) — reusable SKILL.md playbooks injected into
  the system prompt (`vendor/stirrup/src/stirrup/skills/skills.py`)
- replace/extend the **finish tool** (`finish_tool=` on `Agent`, `vendor/stirrup/src/stirrup/tools/finish.py`)
- add **tool providers** (`ToolProvider` protocol) or wrap existing executors (retry, formatting,
  validation, result post-processing)
- modify **Stirrup agent-loop internals** (`core/agent.py`): turn handling, tool-call execution,
  context summarization, overflow recovery, successive-assistant blocking, turns-remaining warnings
- modify the **client layer** (`clients/litellm_client.py`): request construction, reasoning
  effort, message formatting, error/retry behavior

---

## `benchmarks/automationbench/agent_setup.py`

### build_api_tools() — the API toolset wiring

```python
# current
return [
    Tool(name="api_search", description=inspect.getdoc(api_search) or "",
         parameters=ApiSearchParams, executor=_search),
    Tool(name="api_fetch", description=inspect.getdoc(api_fetch) or "",
         parameters=ApiFetchParams, executor=_fetch),
    Tool(name="base64_encode", description=inspect.getdoc(base64_encode) or "",
         parameters=Base64EncodeParams, executor=_b64),
]
```

Everything about how the tools are presented and executed is editable: descriptions, parameter
schemas (add fields, defaults, validation), executor wrappers (pre/post-processing, retries,
mutation echoes, truncation), and the tool list itself (add or remove tools). The executors
currently catch all exceptions into tool content.

### build_world() — environment setup

```python
# current
world = WorldState(**task.initial_state)
world.meta.allowed_services = task.allowed_services
```

The protected task loader computes `allowed_services` before it creates the assertion-free
`AgentTask` passed to editable code. The proposer can change how tools expose this world but cannot
read grading assertions through the task object.

### build_agent() — agent construction

```python
# current
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
```

The task's embedded system prompt is dataset content (currently used verbatim — how it is framed
or combined with harness-level prompting is editable). Other Agent options available
(`core/agent.py:337-345`): `context_summarization_cutoff`, `turns_remaining_warning_threshold`,
`run_sync_in_thread`, `text_only_tool_responses`, `block_successive_assistant_messages`,
`recover_from_context_overflow`, `finish_tool` (custom finish tool or list),
`passthrough_system_prompt`.

### run_task() — session call

```python
# current
world = build_world(task)
with tracelog.trace_to(trace_path):
    agent = build_agent(cfg, task, world, logger, base_url)
    async with agent.session() as session:
        await session.run(task.user_prompt)
```

`agent.session()` accepts `skills_dir=`, `output_dir=`, `input_files=`, `resume=`,
`cache_on_interrupt=`. The user message content and any framing around it are editable. A
sub-agent could also be constructed here and exposed via `Agent.to_tool()`.

---

## Stirrup core (`vendor/stirrup/src/stirrup/`)

- **`core/agent.py`** — the agent loop (`class Agent`, line 245): sequential tool-call execution
  within a turn, finish-tool normalization and multiple-finish-call rejection,
  `block_successive_assistant_messages` (injects a continue message when the model responds
  without tool calls), context-overflow recovery (drops one completed turn and retries),
  context summarization at `context_summarization_cutoff`, turns-remaining warnings,
  `to_tool()` (line 1523) to expose an agent as a parent's tool, `_PARENT_DEPTH` for sub-agent
  nesting.
- **`core/models.py`** — `Tool`, `ToolResult`, `ToolProvider`, message types. The shape of every
  tool the agent sees.
- **`clients/litellm_client.py`** — request construction: reasoning_effort, message formatting,
  streaming, error/retry behavior.
- **`prompts/*.txt`** — `base_system_prompt.txt`, `message_summarizer.txt`,
  `message_summarizer_bridge.txt`.
- **`constants.py`** — `CONTEXT_SUMMARIZATION_CUTOFF`, `TURNS_REMAINING_WARNING_THRESHOLD`, etc.
- **`skills/skills.py`** — `load_skills_metadata(skills_dir)`: scans SKILL.md files, injects
  metadata into the system prompt.
- **`tools/finish.py`** — `SIMPLE_FINISH_TOOL` / finish contract (what "done" means, final answer
  shape). Replaceable via `finish_tool=`.
- **`tools/mcp.py`** — `MCPToolProvider` if external MCP tools are ever wanted.
- **`utils/logging.py`** — `AgentLogger` (turn display, spinner, trace output).
