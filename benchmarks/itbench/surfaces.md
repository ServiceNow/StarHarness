# ITBench harness edit catalog

The proposer may edit ITBench agent behavior in these locations:

- `benchmarks/itbench/agent_setup.py`: agent construction, tools, skills, context settings, and
  sandbox provider wiring.
- `benchmarks/itbench/prompts.py`: system prompt, task workflow, output contract, and examples.
- `benchmarks/itbench/extra_tools.py` (create if needed): snapshot-analysis tools exposed to the
  agent.
- `benchmarks/itbench/skills/` (create if needed): reusable SRE investigation playbooks.
- `vendor/stirrup/src/stirrup/`: shared agent loop, clients, tool providers, context handling, and
  finish behavior.

The candidate may add modules under `benchmarks/itbench/` when `agent_setup.py` imports them. The
candidate must keep evaluator and policy files unchanged.

## Protected evaluator surface

The adapter rejects changes to:

- `benchmark.py`, `config.py`, `data.py`, `grader.py`, `judge.py`, and `run_eval.py`;
- progress and trace routing;
- search, selection, and holdout scenario files;
- proposer prompts and experiment documentation.

Editable code receives a staged incident snapshot without `ground_truth.yaml`. It must not read
run summaries, score files, scenario IDs, task loaders, graders, or judge code.

## Prompt and agent controls

`prompts.py` controls the SRE role, investigation phases, Kubernetes entity format, causal
irreducibility rules, and `agent_output.json` contract. Keep the output compatible with
`grader.py`: a top-level `contributing_factors` list whose entries contain `name`, `reasoning`, and
`evidence`.

`agent_setup.py` builds the Stirrup `Agent`. The baseline agent has only the `run_shell` tool
and the automatic finish tool, with no skills loaded. Candidate changes may adjust context thresholds,
turn warnings, tool composition, finish behavior, skills, and subagents. The evaluator supplies
the model, turn cap, endpoint, and sandbox. Candidate code must not replace those fixed inputs.

## Structural controls

Use `extra_tools.py` for deterministic snapshot inspection such as alert, event, object-history,
or topology summaries. A tool should expose evidence from the staged snapshot and avoid embedding
scenario-specific conclusions.

Place skills under `benchmarks/itbench/skills/<name>/SKILL.md` and pass `skills_dir=` to
`agent.session()` in `agent_setup.py`. Write procedures that apply across
incidents, such as testing upstream control surfaces or separating causes from downstream
symptoms.

Subagents may divide evidence collection or challenge a root-cause hypothesis. Keep the parent
responsible for the final `agent_output.json` so the grader receives one diagnosis.

Changes under `vendor/stirrup` affect both included benchmark adapters. StarHarness imports all
benchmark evaluators after shared edits and rejects a candidate that breaks either adapter.
