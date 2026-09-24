# Iteration {iteration} of {n_iterations}: propose one ITBench harness change

## Search frontier

- Mean over {n_scenarios} proposer-visible search tasks: {frontier_score}
- Per-scenario search scores:
{frontier_per_scenario}

## Prior attempts

{kept_hypotheses}

## Search evidence

Each section contains trusted ground truth for proposer diagnosis, the current score, and the
agent trace. Keep this information out of agent-visible code. Selection and holdout evidence must
remain hidden.

{scenario_traces_section}

## Candidate task

Read `{surfaces_path}` and `{domain_spec_path}`. Compare failures across search tasks, identify one
recurring causal error, and choose a general harness intervention. Consider prompt changes and
structural changes such as tools, skills, context policy, or subagents.

Make one coherent edit. Do not run the evaluator. Write `{pending_eval_path}` with the candidate
name, hypothesis, changed files, and one search `test_scenario` that should respond to the change.
