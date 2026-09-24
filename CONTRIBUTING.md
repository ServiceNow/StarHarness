# Contributing

Contributions are welcome, especially new benchmark adapters, safer candidate-isolation checks,
and reproducibility improvements.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

Use a feature branch and keep each pull request focused. Do not commit `.env`, API keys, run
traces, model outputs, benchmark caches, or proprietary datasets.

## Adapter contributions

Follow [`docs/ADDING_A_BENCHMARK.md`](docs/ADDING_A_BENCHMARK.md). A new adapter should include:

- a deterministic or clearly documented grader;
- disjoint search, selection, and held-out splits;
- hard guardrails protecting graders, task data, and ground truth;
- a one-task smoke command that reviewers can run;
- licensing and download instructions for external benchmark assets;
- tests for summary parsing and adapter discovery.

State expected API costs and external services in the pull request. Never include benchmark
answers in prompts visible to the agent under test.
