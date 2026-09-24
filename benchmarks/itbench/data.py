"""Dataset access for ITBench-AA (ArtificialAnalysis/ITBench-AA, `sre` config).

Each row of sre/data.jsonl describes one Kubernetes incident task:
    id_aa, scenario_id, category, source_split (public|private),
    scenario_root (/home/user), ground_truth_yaml

The actual offline snapshot lives in sre/<scenario_id>/ alongside a
ground_truth.yaml. We download the repo (or a single scenario) from the Hub and
stage a sandbox copy of each scenario WITHOUT ground_truth.yaml so the agent
can't read the answer.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "ArtificialAnalysis/ITBench-AA"
CONFIG = "sre"
# Files an agent must never see (they contain the labels).
_SECRET_FILES = {"ground_truth.yaml", "ground_truth.yml"}


@dataclass
class Task:
    id_aa: str
    scenario_id: str
    source_split: str          # public | private
    ground_truth_yaml: str     # raw YAML string with fault/groups/aliases/...
    snapshot_dir: Path         # downloaded sre/<scenario_id>/ (read-only original)


def download(
    data_root: str | None = None,
    scenario: str | None = None,
    scenarios: list[str] | None = None,
) -> Path:
    """Download the sre split (or a scenario subset) from the Hub; return its local root.

    Pass `scenario` (one id) or `scenarios` (a list of ids) to scope the fetch to
    just those `sre/<id>/*` trees plus data.jsonl, avoiding the full dataset
    split (snapshots are 100s of MB each). With neither, the whole `sre/` tree is
    pulled. `scenario` takes precedence over `scenarios` if both are given.
    """
    from huggingface_hub import snapshot_download

    sel = [scenario] if scenario else (list(scenarios) if scenarios else None)
    allow = [f"{CONFIG}/data.jsonl"]
    if sel:
        allow.extend(f"{CONFIG}/{s}/*" for s in sel)
    else:
        allow.append(f"{CONFIG}/*")
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=allow,
        local_dir=data_root or None,
    )
    return Path(local) / CONFIG


def load_tasks(sre_root: Path, split: str = "public") -> list[Task]:
    """Parse data.jsonl and return tasks for the requested split (public|private|all)."""
    rows = [
        json.loads(line)
        for line in (sre_root / "data.jsonl").read_text().splitlines()
        if line.strip()
    ]
    tasks: list[Task] = []
    for r in rows:
        if split != "all" and r.get("source_split") != split:
            continue
        snap = sre_root / r["scenario_id"]
        if not snap.is_dir():
            continue  # scenario not downloaded (e.g. single-scenario smoke fetch)
        tasks.append(
            Task(
                id_aa=r["id_aa"],
                scenario_id=r["scenario_id"],
                source_split=r["source_split"],
                ground_truth_yaml=r["ground_truth_yaml"],
                snapshot_dir=snap,
            )
        )
    return tasks


def stage_sandbox(task: Task, dest: Path) -> Path:
    """Copy the snapshot into `dest`, dropping ground-truth files. Returns `dest`.

    This is what the agent sees as its working directory (AA mounts it under
    /home/user; with Stirrup's local backend it's the exec env temp dir).
    """
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        task.snapshot_dir,
        dest,
        ignore=lambda _d, names: [n for n in names if n in _SECRET_FILES],
    )
    return dest


def snapshot_listing(sandbox: Path) -> str:
    """A short `ls -la`-style listing of the sandbox top level for the prompt."""
    lines = []
    for p in sorted(sandbox.iterdir()):
        kind = "dir " if p.is_dir() else "file"
        size = "" if p.is_dir() else f" ({p.stat().st_size:,} bytes)"
        lines.append(f"  {kind} {p.name}{size}")
    return "\n".join(lines)
