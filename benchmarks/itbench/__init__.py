"""ITBench-AA replication harness built on top of Stirrup.

Replicates Artificial Analysis' ITBench-AA SRE evaluation
(https://artificialanalysis.ai/methodology/intelligence-benchmarking#itbench-aa)
for the PUBLIC task split only (the 19 private held-out tasks are not shared).

Pipeline:  data.py (download + stage sandbox)
        ->  agent_setup.py (Stirrup Agent w/ run_shell + finish)
        ->  the model writes /sandbox/agent_output.json
        ->  grader.py (precision at full recall)
        ->  run_eval.py (orchestrate + aggregate)
"""
