# Domain Spec: Evolving Stirrup on AutomationBench (Finance) via an omp Evolving-Harness

## Domain summary

We run an automated search over **task-specific harnesses**: the code around a fixed base model
(`gpt-5.4`) that decides how an automation agent interacts with simulated SaaS apps (Gmail, Google
Sheets, Slack, accounting tools) to complete finance workflows. A **proposer** agent (**omp**, model
`gpt-5.4`) iteratively edits the **stirrup** framework + **automationbench** wrapper, and each
candidate is benchmarked on a fixed dev set. We keep only candidates that improve the mean score.

- **Base model (fixed):** `gpt-5.4` with `reasoning_effort=medium` — the agent under test. We do **not** change it.
- **What changes:** the surfaces cataloged in [`surfaces.md`](./surfaces.md).
- **Proposer:** omp (`omp -p --yolo`), model `gpt-5.4`, via [`omp_wrapper.py`](../../omp_wrapper.py).
- **Evaluator:** `python -m benchmarks.automationbench.run_eval` (unchanged).

---

## 1. Problem framing

- **Unit of evaluation:** one AutomationBench task run (agent discovers and calls simulated REST
  endpoints via the API toolset — `api_search` / `api_fetch` / `base64_encode` — to complete a
  finance workflow; graded by programmatic assertions on the final environment state).
- **Fixed:** agent-under-test model `gpt-5.4` (medium reasoning); task data (`data.py` + vendored
  `vendor/automationbench/` upstream), grading (`grader.py`), WorldState + API tools + assertion rubric
  (vendored upstream, read-only).
- **Changes (search space):** all surfaces in `surfaces.md` — **including `vendor/stirrup/src`
  core** (agent logic, tool handling, prompt templates, constants) and the `benchmarks/automationbench/*` wrapper.
- **Proposer model:** `gpt-5.4`.
- **Optimization budget:** fixed **N = 5** iterations per run (override with `--iterations`).

### Scoring (AutomationBench-AA methodology)

- Each assertion is classified against the initial state: **objectives** fail initially and must be
  made true; **guardrails** pass initially and must not be broken.
- **Headline `aa_score`:** a task scores 0 if ANY guardrail is broken, else
  objectives_passed / objectives_total. Infrastructure errors and timeouts score 0. No LLM judge.
- Also recorded per task: native `partial_credit` (broken guardrails count as failures but don't
  zero the task) and `task_completed_correctly` (strict all-or-nothing, Zapier's official metric).

---

## 2. Harness definition

- **Candidate = a git diff** against the clean baseline tree. The proposer edits files in place;
  the loop captures `git diff` as the candidate patch, then restores baseline.
- **Compliance / validity — a candidate must:**
  1. leave the repo importable — `python -c "import benchmarks.automationbench.run_eval"` succeeds;
  2. pass a **smoke test** — one short `run_eval` (1 task, low `--max-turns`) with no crash;
  3. produce parseable `score.json` files with `aa_score` + per-assertion breakdown.
- **Out-of-scope (hard guardrails — enforced in the proposer prior *and* a post-edit check):**
  - never edit adapter, evaluator, model config, task data, grader, traces, splits, or proposer policy;
  - never edit anything under `vendor/automationbench/` (vendored upstream);
  - never hardcode per-task predictions or branch on task IDs;
  - never leak assertion contents into the agent-under-test's view.
  The evaluator passes an assertion-free `AgentTask` to editable code. The adapter rejects diffs
  that add score access, upstream task loading, task IDs, or assertion access.

---

## 3. Evaluation plan

- **Search set (search-30):** 30 finance tasks from the evolution pool. The proposer may read these
  traces and uses them to propose edits. See `search_scenarios.json`.
- **Selection set (selection-20):** the complementary 20 tasks from the same 50-task evolution pool.
  The proposer must not read these traces or scenario definitions. Candidate accept/reject is based
  on this split. See `selection_scenarios.json`.
- **Held-out set (holdout-50):** the 50 finance tasks outside the evolution pool; final-only
  generalization check. See `holdout_scenarios.json`.
- **Split construction:** the 50-task evolution pool was sampled from the 100-task public finance
  baseline (`runs/ab_finance_base_gpt55`, gpt-5.5 medium) to match the overall aa mean and
  failure-mode distribution, then split 30/20 minimizing aa-mean + failure-mode divergence
  (seed=42). All three splits sit at aa ≈ 0.596.
- **Repeats:** **1** per task (AA methodology).
- **Turn cap:** **50** (AA methodology).
- **Candidate score:** mean `aa_score` across the 20 selection tasks
  (from `runs/<name>/summary.json`, field `overall_score`), with guardrail-intact rate as the
  deterministic tiebreaker.
- **Concurrency:** default `--concurrency 8`; tunable at launch.
- **Services:** WorldState and SaaS tools run in-process. The agent and proposer call the configured
  model endpoint.
- **Per-candidate runtime:** model latency, task length, endpoint capacity, and concurrency determine
  runtime.
- **Leakage checks:** the loop validates disjoint partitions, exact summary coverage, protected
  paths, forbidden added code, and search-only proposer traces.

### Selection rule (frontier)

Accept a candidate onto the frontier **iff**:
1. mean selection-20 score **strictly improves** over the frontier, or the task mean ties and the
   selection guardrail-intact rate improves.

Otherwise revert. The OMP proposer may still decide whether a single patch-test result is worth a
full selection eval, but OMP no longer decides promotion. Per-task wins/regressions are recorded as
diagnostic evidence, not as independent vetoes.

---

## 4. Baselines

**Note:** the baseline scores below were measured with **gpt-5.5** medium reasoning; the agent
under test for this evolution run is **gpt-5.4** medium.

Full finance baseline: `runs/ab_finance_base_gpt55` (gpt-5.5 medium, 100 tasks, 1 repeat, 50-turn
cap, 0 infra errors):

- **Full benchmark baseline:** aa **0.596**, native partial_credit **0.704**, strict pass **0.310**,
  guardrail-intact rate **0.79**.
- **Search-30 baseline:** aa **0.596**.
- **Selection-20 baseline:** aa **0.596**.
- **Holdout-50 baseline:** aa **0.595**.
- **Phase 0 baseline:** the frontier is initialized from both a search-30 baseline eval for proposer
  traces and a selection-20 baseline eval for accept/reject scoring.
- **Reusable helpers (do not reimplement):** `benchmarks/automationbench/run_eval.py` (CLI, score/summary JSONs),
  `benchmarks/automationbench/data.py` (task loading), `benchmarks/automationbench/grader.py` (assertion grading),
  `benchmarks/automationbench/agent_setup.py` (agent + API toolset construction).

---

## 5. Search plan (loop)

Driver: `evolving_harness.py`. Phases:

- **Phase 0 — Baseline:** run `run_eval` on search-30 to collect proposer-visible frontier traces,
  then run `run_eval` on selection-20 to record the baseline score into `evolving_runs/<run>/frontier.json`.
  (`--skip-baseline` to reuse an existing frontier.)
- **Phase 1..N — Iterate** (N = 5 default):
  1. **Propose:** `omp_wrapper.run()` with the proposer prior + a per-iteration task that includes
     the search frontier score, all kept hypotheses, and per search task the **assertions +
     per-assertion grading + full trace** (no truncation by default). Proposer edits surfaces in
     place and writes `pending_eval.json`.
  2. **Capture:** snapshot proposer edits as a git diff → candidate patch.
  3. **Guardrail check:** reject if the diff touches out-of-scope paths (§2) or writes strays.
  4. **Validate:** import check.
  5. **Smoke:** one short single-task `run_eval`; revert + skip on crash.
  6. **Benchmark:** selection-20 `run_eval --name evolving_<run>_iter<i>`; parse mean + per-task.
  7. **Frontier update:** apply the deterministic selection rule (§3); keep (commit) or revert.
  8. **Search refresh:** if kept, run the new frontier on search-30 and store those traces for the
     next proposal step.
  9. **Log:** append one line to `evolving_runs/<run>/evolution_summary.jsonl`.
- **Final:** evaluate the winning frontier on the **held-out** set; write `held_out_result.json`.

**Stopping:** fixed N iterations.

---

## 6. Experience & logging (online + offline)

- **Offline experience the proposer may consult:** `vendor/stirrup/docs/`, `surfaces.md`,
  `proposer_prior.md`, and prior traces under `runs/`.
- **Per-run online storage** under `evolving_runs/<run>/`:
  - `frontier.json` — best selection mean, selection per-task baselines, current winning commit,
    plus the current search trace run.
  - `evolution_summary.jsonl` — one line/iteration: `{iter, name, hypothesis, mean, delta,
    per_scenario, decision, session_log}`.
  - `candidates/iter<i>.patch` — the candidate diff.
  - `proposer_logs/iter<i>/` — omp stdout + parsed session + task.md.
  - `held_out_result.json` — final winner's held-out score.
- **CLI (`evolving_harness.py`):** `--benchmark automationbench` (required), `--iterations` (5),
  `--repeats` (1), `--run-name`, `--concurrency` (8), `--proposer-model` (`openai/gpt-5.4`),
  `--propose-timeout`, `--fresh`, `--skip-baseline`, `--skip-smoke`, `--trace-chars` (0 = unlimited),
  `--trace-analysis` (`inline` or `subagents`), `--trace-subagent-wave` (8), `--no-held-out`.

---

## Decisions log

| Field | Decision |
|---|---|
| Proposer model | `gpt-5.4` |
| Agent-under-test (fixed) | `gpt-5.4` with `reasoning_effort=medium` (baselines shown were measured on `gpt-5.5`) |
| Candidate isolation | git-based (diff apply/revert), scoped to `benchmarks/automationbench/` + `vendor/stirrup/` |
| Edit scope | all surfaces incl. `vendor/stirrup/src` core |
| Search set | search-30 from the evolution pool; proposer-visible |
| Selection set | selection-20, complementary to search-30; proposer-hidden accept/reject gate |
| Held-out set | holdout-50; final-only generalization check |
| Repeats | 1 |
| Turn cap | 50 (AA methodology) |
| Budget/stop | fixed N=5 iterations |
| Acceptance | deterministic selection mean improvement, or tied mean with better guardrail-intact rate |
| Proposer context/iter | assertions + per-assertion grading + full trace for search-30 only |
| Concurrency | default 30, set at launch |
| Guardrails | standard set, enforced in prior + post-edit diff check |

## Open questions / unknowns

- **Rate limits** on the agent-under-test model may bound usable concurrency; 30 was stable for the
  (gpt-5.5) baseline.
- **Proposer context size** for 30 search tasks (assertions + grading + trace each) is moderate;
  omp manages its own context/compaction. Set a positive `--trace-chars` only if you hit context limits.
- **Subagent trace analysis** adds one proposer worker per search task. A 30-task search split adds
  30 worker calls. Use `--trace-subagent-wave` to cap concurrent workers.
- **AA comparability:** we run the public 100-task finance split; AA's published numbers use a
  private 657-task held-out split — methodology is identical, absolute numbers are not comparable.
