# ITBench SRE adapter specification

## Task and metric

Each ITBench-AA SRE task gives the agent an offline Kubernetes incident snapshot. The agent writes
`agent_output.json` with the minimal independent entities that caused the incident. The evaluator
scores precision at full recall: a response receives credit after it covers each root-cause group,
then false-positive entities reduce precision.

StarHarness keeps the agent model, dataset, grader, judge, and task partition fixed. Candidates may
change prompts, tools, skills, agent composition, and the shared Stirrup harness.

## Three-way public split

The adapter partitions the 40 public SRE scenarios into:

- 5 proposer-visible search tasks in `search_scenarios.json`;
- 5 proposer-hidden selection tasks in `selection_scenarios.json`;
- 30 sealed holdout tasks in `holdout_scenarios.json`.

The search and selection files divide the 10-scenario development set from the prior experiment.
The holdout file contains the other 30 public scenarios. The three files remain disjoint and cover
the public 40-task set.

The checked-in 5/5 allocation uses the prior baseline run. Each set contains two zero-score tasks,
one 0.5 task, and two 1.0 tasks. Trace labels place network-partition, invalid-image,
overprediction, and mislocalized-root-cause cases in both sets.

Use `stratify_tasks.py` with a full baseline manifest when a new study needs failure-mode-matched
splits. Replace all three checked-in files together after reviewing its report.

## Evaluation boundary

The evaluator downloads only the requested scenario snapshots from
`ArtificialAnalysis/ITBench-AA`. `data.py` removes ground-truth files before staging a snapshot for
editable agent code. The trusted evaluator retains ground truth for grading and proposer-visible
search diagnosis.

Candidate acceptance uses the hidden selection mean. ITBench has no independent verifier-rate
tiebreaker, so tied task means do not advance the frontier. StarHarness opens holdout tasks after
the final iteration.

## Editable and protected code

The proposer may edit `agent_setup.py`, `prompts.py`, and `vendor/stirrup`, and may add
`extra_tools.py` and benchmark skills. The adapter protects its model configuration, dataset loader, grader, judge,
evaluator, traces, split files, and proposer policy.

The diff guard rejects added code that reads ground truth, score artifacts, evaluator modules, or
scenario IDs. The runtime import check and one-task smoke evaluation run before hidden selection
scoring.

## Runtime

- Agent model: `AGENT_LLM`, default `gpt-5.4`.
- Agent endpoint: `AGENT_BASE_URL` or `AGENT_BASE_URLS`.
- Judge: set `JUDGE_MODEL` for model-assisted normalization; the default is deterministic
  matching.
- Data cache: `ITBENCH_DATA`, default `./data`.
- Default task cap: 100 turns in ITBench, with the evolution CLI supplying its configured cap.
- Default repeats: one during evolution; use more repeats for final measurement.

ITBench snapshots can occupy tens of gigabytes. A scenario override limits the Hugging Face
download to the requested tasks.
