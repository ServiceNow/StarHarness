# Iteration {iteration} of {n_iterations} — propose a harness intervention

## Purpose

Improve the harness for this fixed benchmark. The next patch should be a testable intervention:
what harness behavior changes, and why should that improve score?

## Current frontier

- Mean score over proposer-visible search-{n_scenarios}: {frontier_score}
- Per-scenario search scores:
{frontier_per_scenario}

## Prior attempts

{kept_hypotheses}

## Evidence

Each search task's `score.json` contains the per-assertion grading (objectives vs guardrails,
passed/failed, assertion params) and each `trace.log` contains the full conversation: every
api_search/api_fetch call and result. Read whatever is useful from the search split only.
Do not inspect selection-split result JSONs or selection scenario definitions while proposing.

Useful artifacts:
- `evolving_runs/<this-run>/candidates/iter*.patch`
- `evolving_runs/<this-run>/evolution_summary.jsonl`
- prior `runs/<run_name>/<task_id>/repeat_*/score.json` and `trace.log`

{scenario_traces_section}

## Task

Read `{surfaces_path}` and skim `{domain_spec_path}`. Then decide what to try this iteration.

Analyze before you edit. Work out from the evidence *how the agent is actually going wrong* —
aggregate across tasks to find the recurring failure pattern (e.g. a class of unsafe writes, a
class of missed lookups), not the quirks of any single task. Then decide whether and how the
harness can enhance the model's performance on that failure *in general*: the edit must help tasks
you have never seen, not just the ones in the evidence. Do not change something blindly: a
candidate without a failure-mode diagnosis behind it is a wasted iteration, and a fix that only
patches a specific task will not survive the selection split.

You have full visibility into prior attempts below — what was kept, what was discarded, and why. Use
that history to reason about strategy: Which failure mode has the most headroom? Has it been
attacked before? If so, what was tried and why did it fail — should you refine the approach or move
to a different surface? If a surface has been exhausted, pivot.

Do not run any eval, smoke test, or benchmark command yourself. The harness will
benchmark your candidate automatically after you finish. Your job is to analyze the
evidence, make the edit, and write `{pending_eval_path}` — nothing more.

Make the edit in place, then write `{pending_eval_path}` using the contract from the prior.
