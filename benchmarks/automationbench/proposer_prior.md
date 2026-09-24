# Proposer prior — AutomationBench harness evolution

## The setup (read these before editing)

- **`surfaces.md`** — the catalog of every file/symbol you may edit and what it controls.
  **Read it first.** It groups surfaces into two families — prompt/config edits and
  **structural levers** (custom tools, sub-agents, skills, tool providers, Stirrup internals).
  Analyze the evidence and decide which family addresses the failure mode you found.
- **`domain_spec.md`** — the full experiment definition (metric, eval set, rules). Skim it.

## Purpose

You are running a harness-evolution experiment. The goal is to improve the harness so the fixed
automation agent performs better on this AutomationBench (finance) benchmark. Wherever the evidence
points, enhance the model's effective performance through harness evolution: better prompting,
better structure (tools, sub-agents, skills), better loop behavior.

## Fixed

The benchmark, task data, assertion grading, simulated app environment (WorldState), and
agent-under-test model are not part of the intervention. Treat them as the experimental environment.

## Intervention

Change the harness: prompts, configuration, tools, skills, sub-agents, tool providers, result
formatting, or Stirrup internals. Use `surfaces.md` to find where to edit.

Each iteration is one coherent change. You see the full history of prior attempts — kept and
discarded. Reason about what to try: identify the failure mode with the most headroom, check whether
it has been attacked before, and decide whether to refine a prior approach or pivot to a new
surface. Don't repeat discarded attempts without a meaningfully different hypothesis.

## Evidence

Use per-assertion `score.json` grading, conversation `trace.log` files, prior patches, proposer
sessions, eval logs, and promotion notes. Assertion details are for diagnosis, not for hardcoding
task-specific answers — and must never leak into what the agent-under-test sees.

Use only proposer-visible search evidence while making edits. The selection split is reserved for
accept/reject scoring; do not inspect its scenario file or result traces while proposing.

## Boundaries

- Edit only under `benchmarks/automationbench/` or `vendor/stirrup/`. Never edit anything under
  `vendor/automationbench/` (the vendored upstream benchmark).
- Under `benchmarks/automationbench/`, edit `agent_setup.py` or add harness modules. Do not edit
  adapter, evaluator, config, task data, grading, trace, split, prompt-policy, or specification
  files.
- Do not hardcode task IDs, expected answers, assertion contents, or per-task branches.
- Do not load benchmark tasks, scores, summaries, or assertions from editable code. The protected
  evaluator passes an assertion-free task view to the harness.
- Keep `python -c "import benchmarks.automationbench.run_eval"` importable.

## Output

Make the edit in place, then write `pending_eval.json` at repo root:

```json
{
  "name": "short-slug",
  "hypothesis": "Harness change, expected behavior change, and why it should improve the benchmark.",
  "changed_files": ["relative/path.py"],
  "test_scenario": "scenario id you expect to improve, if there is one"
}
```

`changed_files` must match the actual diff. `test_scenario` is used for a quick check when present.
