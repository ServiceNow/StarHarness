# Third-party software

StarHarness includes these unmodified or adapted MIT-licensed projects:

- `vendor/automationbench`: AutomationBench, copyright 2026 Zapier, Inc.
  Its full license and scope notice are in `vendor/automationbench/LICENSE`.
- `vendor/stirrup`: Stirrup, copyright 2025 Artificial Analysis.
  Its full license is in `vendor/stirrup/LICENSE`.

The benchmark adapter under `benchmarks/automationbench` integrates these projects but is
part of StarHarness. No API keys, benchmark outputs, model traces, or private datasets are
included.

The `benchmarks/itbench` adapter downloads the public
[ITBench-AA SRE dataset](https://huggingface.co/datasets/ArtificialAnalysis/ITBench-AA) at runtime.
The dataset is licensed under CC BY 4.0 and is not included in this repository. ITBench-AA is
Artificial Analysis' release of public scenarios from IBM's ITBench benchmark.

`benchmarks/itbench/prompts.py` and `benchmarks/itbench/judge.py` reproduce Artificial Analysis'
generation and grading prompt text verbatim (see the module docstrings), for faithful replication
of the ITBench-AA methodology. This prompt text is Artificial Analysis' own authorship, credited
here as no license is stated for it upstream; IBM is the original author of the underlying ITBench
benchmark and scenarios that Artificial Analysis' evaluation is built on.
