# ITBench SRE proposer contract

Improve the agent harness for precision at full recall. Study proposer-visible search traces,
scores, and ground truth to find a recurring causal failure. Make one scoped change that should
help incidents outside the search set.

Read `surfaces.md` before editing. You may change:

- `benchmarks/itbench/agent_setup.py` and `prompts.py`, plus new `extra_tools.py` and `skills/`;
- `vendor/stirrup` agent-loop, prompt, tool, skill, and context code.

You may add an ITBench harness module when editable code imports it.

Do not edit the adapter, configuration, data loader, grader, judge, evaluator, trace routing,
split files, proposer policy, or experiment specification. Do not inspect selection or holdout
scenario definitions, traces, scores, or contents.

Ground truth in the search evidence supports your diagnosis. Never copy an answer, entity, filter,
scenario ID, or ground-truth-derived mapping into agent-visible code. Editable code must use only
the staged incident snapshot and normal model observations.

Do not run evaluation commands. StarHarness handles imports, smoke evaluation, selection scoring,
and rollback.

After editing, write `pending_eval.json`:

```json
{
  "name": "short-candidate-name",
  "hypothesis": "What changed, the expected behavior change, and why it should improve the metric.",
  "changed_files": ["benchmarks/itbench/prompts.py"],
  "test_scenario": "one proposer-visible search scenario"
}
```

List each changed file. Choose `test_scenario` from the evidence in the current task.
