"""Wire a Stirrup Agent for one ITBench-AA task: run_shell + finish over a sandbox.

ITBench-AA gives the model exactly two tools: a single `run_shell` to inspect the
snapshot, and `finish` to submit. Stirrup's LocalCodeExecToolProvider provides a
shell tool (default name "code_exec"); we subclass it only to rename the tool to
`run_shell` so the surface matches the methodology. The snapshot is uploaded into
the exec environment as the agent's working directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from stirrup import Agent
from stirrup.clients.litellm_client import LiteLLMClient
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.utils.logging import AgentLogger

from . import tracelog
from .config import RunConfig
from .progress import ProgressReporter
from .prompts import OUTPUT_FILENAME, SYSTEM_PROMPT, build_task_prompt

# AA gives the agent a single shell tool named `run_shell`; both backends below
# rename Stirrup's default `code_exec` tool to match that surface.
RUN_SHELL_DESCRIPTION = (
    "Execute a shell command in the execution environment. "
    "Returns exit code, stdout, and stderr as XML."
)

class _ProgressLogger(AgentLogger):
    """AgentLogger that also reports each step's turn to a ProgressReporter.

    Keeps all of AgentLogger's per-task file logging; only adds a turn update so
    the compact stdout status line can show live progress of in-flight units.
    """

    def __init__(self, reporter: ProgressReporter, key: str) -> None:
        super().__init__(show_spinner=False)
        self._reporter = reporter
        self._key = key

    def on_step(self, step: int, tool_calls: int = 0, input_tokens: int = 0,
                output_tokens: int = 0) -> None:
        super().on_step(step, tool_calls, input_tokens, output_tokens)
        self._reporter.update_turn(self._key, step)


class RunShellProvider(LocalCodeExecToolProvider):
    """LocalCodeExecToolProvider whose tool is named `run_shell` (AA surface)."""

    async def __aenter__(self):  # type: ignore[override]
        import tempfile

        if self._temp_base_dir:
            self._temp_base_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir = Path(tempfile.mkdtemp(prefix="itbench_sandbox_", dir=self._temp_base_dir))
        return self.get_code_exec_tool(name="run_shell", description=RUN_SHELL_DESCRIPTION)


def build_shell_provider(cfg: RunConfig, exec_tmp_base: Path):
    """Build the code-exec ToolProvider for one unit, per `cfg.backend`.

    Returns the provider plus the (data_location, output_path) the AA prompt
    should advertise for that backend. The provider's tool is named `run_shell`.

    - "local":   a host temp dir under `exec_tmp_base` (the default).
    - "boxlite": a self-hosted BoxLite microVM with AA's `/home/user` mount and
                 the AA packages installed at box start. Each unit gets its own
                 microVM (full isolation), torn down on context exit.
    """
    if cfg.backend == "boxlite":
        # Imported lazily so the `local` path needs no boxlite install.
        from stirrup.tools.code_backends.boxlite import BoxliteCodeExecToolProvider

        class RunShellBoxliteProvider(BoxliteCodeExecToolProvider):
            """BoxliteCodeExecToolProvider whose tool is named `run_shell`."""

            async def __aenter__(self):  # type: ignore[override]
                await super().__aenter__()  # create microVM + mkdir /home/user + setup
                return self.get_code_exec_tool(name="run_shell", description=RUN_SHELL_DESCRIPTION)

        provider = RunShellBoxliteProvider(
            url=cfg.boxlite_url or None,
            image=cfg.boxlite_image,
            working_dir="/home/user",
            disk_size_gb=cfg.boxlite_disk_gb,
            cpus=cfg.boxlite_cpus,
            memory_mib=cfg.boxlite_memory_mib,
            setup_commands=cfg.boxlite_setup_commands(),
            shell_timeout=cfg.shell_timeout,
        )
        return provider, "/home/user", "/home/user/agent_output.json"

    provider = RunShellProvider(shell_timeout=cfg.shell_timeout, temp_base_dir=exec_tmp_base)
    return provider, None, None  # None -> build_task_prompt uses its local-backend defaults


def build_agent(cfg: RunConfig, shell: RunShellProvider, logger: AgentLogger,
                base_url: str | None = None) -> Agent:
    client = LiteLLMClient(
        model=cfg.litellm_model,
        api_key=cfg.api_key,
        reasoning_effort=cfg.reasoning_effort,
        kwargs=cfg.litellm_kwargs(base_url),
    )
    return Agent(
        client=client,
        name="sre_agent",
        system_prompt=SYSTEM_PROMPT,
        max_turns=cfg.max_turns,
        tools=[shell],  # finish tool is added by Agent automatically
        logger=logger,
    )


async def run_task(cfg: RunConfig, sandbox: Path, output_dir: Path, trace_path: Path,
                   logger: AgentLogger, base_url: str | None = None) -> dict:
    """Run one repeat. Returns the parsed agent_output.json (or {} if none produced).

    The full per-turn agent trace for THIS unit is written to `trace_path` (one
    file per (task, repeat)), so concurrent units never interleave. See
    `tracelog` for how the per-task routing works. `logger` carries the
    progress hook (a `_ProgressLogger`).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage the exec env's working dir under the per-unit rep_dir (on the shared
    # run mount), NOT ephemeral /tmp. At high concurrency, N concurrent
    # ~hundreds-of-MB scenario copies would overflow the 16GiB local-disk cap and
    # crash the job. output_dir is rep_dir/output, so its parent is rep_dir on the
    # mount. The provider rmtree's this per unit on exit, so it doesn't accumulate.
    exec_tmp_base = output_dir.parent / "exec_tmp"
    shell, data_location, output_path = build_shell_provider(cfg, exec_tmp_base)
    # AA's verbatim prompt; the agent discovers files via ls. For boxlite the
    # data/output live at AA's `/home/user`; for local, the prompt defaults apply.
    if data_location is not None:
        task_prompt = build_task_prompt(data_location=data_location, output_path=output_path)
    else:
        task_prompt = build_task_prompt()

    # Upload the staged snapshot (top-level files + dirs) into the exec env.
    input_files = [str(p) for p in sandbox.iterdir()]

    # Route this unit's trace to its own file. show_spinner=False: the Rich Live
    # spinner is TTY-only and would just spew control codes into a log file.
    with tracelog.trace_to(trace_path):
        agent = build_agent(cfg, shell, logger, base_url)
        async with agent.session(output_dir=output_dir, input_files=input_files) as session:
            await session.run(task_prompt)

    out = output_dir / OUTPUT_FILENAME
    if not out.exists():
        return {}
    try:
        return json.loads(out.read_text())
    except json.JSONDecodeError:
        return {}
