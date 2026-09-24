# Architecture

StarHarness separates search policy from benchmark semantics.

```text
full reproducible baseline -> OMP task stratification -> search / selection / holdout
                                                        |
                                                        v
search traces -> proposer -> candidate Git diff -> guardrails/import/smoke
                                                    |
                                                    v
held-out <- final frontier <- keep/revert <- selection evaluator
```

`evolving_harness.py` owns frontier state, Git isolation, proposer invocation, and acceptance.
It dynamically imports `benchmarks.<name>.benchmark.Adapter`. The adapter owns task splits,
environment variables, evaluator invocation, result parsing, trace assembly, prompt rendering,
and the set of paths a candidate may change.

The proposer supports two trace-analysis modes. `inline` gives all search evidence to the parent
proposer. `subagents` tells the parent to assign one search scenario to each isolated worker in
bounded waves, check that each scenario has a report, and combine the reports before making one
candidate edit. The workers analyze evidence without editing files. The parent retains ownership
of the candidate diff and `pending_eval.json`.

Evaluation artifacts are written under `runs/`; search metadata, patches, and proposer sessions
are written under `evolving_runs/`. Both are ignored by Git.

`stratify_tasks.py` runs before evolution. It sends each baseline task trace to one isolated OMP
worker, groups the reported failure modes, and selects the smallest evolution pool that represents
each stratum in search and selection. It also matches baseline score and verifier-pass
distributions. The helper assigns all remaining reproducible tasks to holdout and writes a report
for human review. Git ignores the report directory because it contains information about hidden
splits.

The Finance example treats `benchmarks/automationbench` and `vendor/stirrup` as editable surfaces.
Its grader, task loader, and split files are explicitly out of scope. The upstream AutomationBench
environment in `vendor/automationbench` is neither editable nor passed to Git cleanup.
