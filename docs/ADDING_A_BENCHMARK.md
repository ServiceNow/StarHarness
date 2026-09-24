# Adding a benchmark adapter

An adapter translates a benchmark into three things StarHarness needs: a safe edit boundary, an
evaluation command with machine-readable results, and failure evidence for the proposer.

Use `benchmarks/automationbench` for a simulated SaaS benchmark and `benchmarks/itbench` for an
offline incident-snapshot benchmark.

## 1. Define the evaluation contract

Before writing code, document:

- the unit of evaluation and primary metric;
- which model and settings remain fixed;
- the allowed harness surfaces;
- the grader and ground-truth files that must never change;
- disjoint search, selection, and held-out task IDs;
- repeat count, timeouts, concurrency, and external services;
- the exact candidate acceptance rule.

Put this in `benchmarks/<name>/domain_spec.md`. Put the editable code catalog in `surfaces.md`.

## 2. Create the package

```text
benchmarks/<name>/
├── __init__.py
├── benchmark.py
├── run_eval.py
├── domain_spec.md
├── surfaces.md
├── proposer_prior.md
├── propose_task.md
├── search_scenarios.json
├── selection_scenarios.json
└── holdout_scenarios.json
```

The three split files should use this shape:

```json
{"scenarios": ["task-001", "task-002"]}
```

You can generate a compact partition from a full reproducible baseline with
`stratify_tasks.py`. Supply one manifest record per task with its baseline score, verifier pass
rate, and trace path. OMP assigns one isolated worker to each trace, groups causal failure modes,
and matches those modes and both numeric descriptors across search and selection. The helper
reserves all remaining tasks for holdout and writes the three split files above.

Run stratification before evolution. Do not expose `stratification.json`, selection traces, or
holdout traces to the evolution proposer.

## 3. Implement `BenchmarkAdapter`

In `benchmark.py`, export a class named `Adapter` extending
`benchmarks.base.BenchmarkAdapter`.

```python
class Adapter(BenchmarkAdapter):
    @property
    def name(self) -> str:
        return "mybenchmark"

    @property
    def editable_dirs(self) -> list[str]:
        return ["benchmarks/mybenchmark", "vendor/my_harness"]

    @property
    def out_of_scope(self) -> set[str]:
        return {
            "benchmarks/mybenchmark/grader.py",
            "benchmarks/mybenchmark/data.py",
            "benchmarks/mybenchmark/search_scenarios.json",
            "benchmarks/mybenchmark/selection_scenarios.json",
            "benchmarks/mybenchmark/holdout_scenarios.json",
        }

    @property
    def protected_files(self) -> set[str]:
        return {"README.md", "pyproject.toml", "uv.lock", ".env"}
```

Implement every abstract member in `benchmarks/base.py`:

- `name`: lowercase Python package name under `benchmarks/`.
- `editable_dirs`: repository-relative directories that may enter a candidate patch.
- `out_of_scope`: existing files inside editable directories that control evaluation or evidence.
- `protected_files`: root files that stray-file cleanup must preserve.
- `agent_env`: model endpoint variables inherited by evaluation subprocesses.
- `dev_scenarios`, `selection_scenarios`, `held_out_scenarios`: explicit task IDs.
- `run_eval`: launch the evaluator, redirect output to `log_path`, and return its exit code.
- `read_summary`: validate exact `expected_scenarios`, then return
  `(mean, per_scenario, verifier_pass_rate)` or `None`.
- `import_check`: cheaply prove the adapter still imports after a candidate edit.
- `gather_traces`: assemble search-only failure evidence without exposing selection artifacts.
- `render_task`: fill the proposal prompt from runtime search scenarios and search scores.
- `prompts_dir`, `surfaces_path`, `domain_spec_path`: prompt/spec locations.
- `candidate_violation`: optionally reject answer access, task-ID branches, or other
  benchmark-specific leakage in added diff lines.

`verifier_pass_rate` is a deterministic tiebreaker. Return `0.0` if the benchmark has no useful
secondary verifier metric.

## 4. Provide a stable evaluator CLI

The adapter's `run_eval` should invoke a module that accepts task IDs, repeat count, turn or step
limit, concurrency, run name, and optional unit timeout. It should always write a summary under:

```text
runs/<run-name>/summary.json
```

Unit failures should become explicit zero-score records rather than aborting the entire batch.
Keep task execution isolated so concurrent units cannot share mutable state.

Reject summaries with missing or duplicate tasks, non-finite scores, an incorrect aggregate, or a
task set that differs from `expected_scenarios`. Otherwise a partial evaluation can enter the
frontier.

## 5. Protect the benchmark

Put graders, task loaders, split files, and ground truth in `out_of_scope`. Keep upstream benchmark
code outside `editable_dirs`. `gather_traces` may expose search-set answers to the proposer for
diagnosis, but the runtime agent must receive only the benchmark's normal observation surface.

Do not let task IDs or expected answers become candidate features. StarHarness also rejects paths
containing `ground_truth`, but adapters should declare all protected files explicitly.

Pass an agent-facing task object to editable code. Keep assertions, expected answers, raw benchmark
rows, and grader handles in protected evaluator code. If the harness needs derived metadata such as
allowed services, compute it before constructing the agent-facing object.

## 6. Write proposer inputs

- `proposer_prior.md` defines the metric, allowed edits, forbidden edits, and output contract.
- `propose_task.md` receives the iteration number, frontier scores, accepted and rejected
  hypotheses, and search traces.
- `surfaces.md` describes both prompt/configuration surfaces and structural changes such as tools,
  skills, providers, subagents, context management, and finish logic.

Keep selection and held-out traces out of proposer prompts.

StarHarness can ask the proposer to analyze each search trace in an isolated subagent with
`--trace-analysis subagents`. Make each scenario section from `gather_traces` self-contained so the
parent can give one section to one worker. The parent aggregates the reports and owns the candidate
edit; workers must not edit files.

Call `adapter.validate(repo)` before evaluation. StarHarness uses it to check repository-relative
paths, required prompt files, environment values, unique task IDs, and disjoint partitions. Test
CLI split overrides too; an override must not draw from the held-out set.

## 7. Adapter acceptance checklist

- A clean checkout passes `adapter.validate(repo)`.
- Search, selection, and held-out IDs contain no duplicates or overlap.
- A one-task evaluator run writes one matching task record and a correct aggregate.
- `read_summary` rejects missing tasks, duplicate tasks, invalid scores, and aggregate mismatches.
- Editable code cannot access grading fields, scores, or upstream task loaders.
- Candidate diffs cannot change the adapter, evaluator, grader, task loader, or split files.
- `render_task` uses search metrics when CLI scenario overrides are active.
- Endpoint variables reach the agent process without embedding credentials in the adapter.
- Import checks and a one-task smoke run catch invalid candidate code before selection scoring.
- The final evaluation reads held-out IDs only after the search loop ends.

## 8. Validate

At minimum, run:

```bash
uv run python -c "from benchmarks.mybenchmark.benchmark import Adapter; print(Adapter().name)"
uv run python -m benchmarks.mybenchmark.run_eval --tasks task-001 --name smoke
uv run pytest
```

Then exercise one evolution iteration with one search and one selection task before launching the
full split. Review the candidate patch, frontier JSON, selection summary, and held-out separation.

## Included adapter mappings

Both included adapters implement the full contract:

| Concern | Finance | ITBench SRE |
|---|---|---|
| Adapter | `benchmarks/automationbench/benchmark.py` | `benchmarks/itbench/benchmark.py` |
| Evaluator | `benchmarks/automationbench/run_eval.py` | `benchmarks/itbench/run_eval.py` |
| Agent wiring | `benchmarks/automationbench/agent_setup.py` | `benchmarks/itbench/agent_setup.py` |
| Task loading | `benchmarks/automationbench/data.py` | `benchmarks/itbench/data.py` |
| Grading | Programmatic assertions | Precision at full recall |
| Search evidence | Assertions and traces | Ground truth and traces |
| Editable catalog | `benchmarks/automationbench/surfaces.md` | `benchmarks/itbench/surfaces.md` |
| Environment | `vendor/automationbench` | Offline snapshot per scenario |
| Agent harness | `vendor/stirrup` | `vendor/stirrup` |
