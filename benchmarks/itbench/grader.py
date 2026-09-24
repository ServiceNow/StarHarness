"""Scoring for ITBench-AA: precision at full recall.

AA's grader uses an LLM judge (GPT-5.5 medium) to normalize submitted entities to
ground-truth canonical entities and alias groups. We reproduce that with the
`itbench.judge.LLMJudge` (default GPT-5.4) — passed in as `judge`. When no judge
is configured, we fall back to approximating that normalization DETERMINISTICALLY
using the structured ground truth each scenario ships, which already encodes the
canonical matchers:

  groups:   id, kind, namespace, filter: [<regex>, ...], root_cause: true?
  aliases:  [[group_id, group_id, ...], ...]   # interchangeable entities

Matching a predicted `namespace/Kind/name`:
  - namespace must equal the group's namespace (case-insensitive)
  - kind must equal the group's kind, OR satisfy "Workload Kind Equivalence"
    (Deployment <-> Pod and other workload kinds, when ns + name correspond)
  - name must match one of the group's `filter` regexes

Score (per repeat):
  - Collapse groups into scoring groups via alias sets.
  - Root-cause scoring groups = those containing a `root_cause: true` group.
  - FULL RECALL: every root-cause scoring group must be hit, else score = 0.0.
  - Otherwise precision = TP / (TP + FP), where a prediction matching a
    root-cause group is a TP and any unmatched / non-root-cause prediction is FP.

The LLM judge is the default (JUDGE_MODEL=gpt-5.4); set JUDGE_MODEL="" to use the
deterministic path with no second endpoint.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

import yaml

# Workload kinds AA treats as interchangeable (Deployment <-> Pod, etc.).
_WORKLOAD_KINDS = {"deployment", "pod", "replicaset", "statefulset", "daemonset", "rollout"}


def _compile_filter(f: str) -> re.Pattern:
    """Compile a ground-truth group `filter`.

    Most GT filters are regexes (e.g. `load-generator-.*`), but some are authored
    as shell-style globs (e.g. Scenario-38's `*.*`) which are not valid regex and
    make `re.compile` raise `re.error: nothing to repeat`. Fall back to interpreting
    such a filter as a glob so a single malformed entry can't abort the whole grade.
    (In judge=llm mode these compiled filters are unused anyway; this only affects the
    deterministic matcher, where glob semantics match the obvious authorial intent.)
    """
    try:
        return re.compile(f)
    except re.error:
        return re.compile(fnmatch.translate(f))


@dataclass
class Group:
    id: str
    kind: str
    namespace: str
    filters: list[re.Pattern]
    root_cause: bool


def _parse_groups(gt: dict) -> list[Group]:
    groups = []
    for g in gt.get("groups", []) or []:
        filt = g.get("filter") or []
        if isinstance(filt, str):
            filt = [filt]
        groups.append(
            Group(
                id=g["id"],
                kind=str(g.get("kind", "")),
                namespace=str(g.get("namespace", "")),
                filters=[_compile_filter(f) for f in filt],
                root_cause=bool(g.get("root_cause", False)),
            )
        )
    return groups


def _scoring_sets(groups: list[Group], aliases: list[list[str]]) -> list[set[str]]:
    """Union group ids that share an alias list into scoring sets."""
    parent = {g.id: g.id for g in groups}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a in parent and b in parent:
            parent[find(a)] = find(b)

    for alias_list in aliases or []:
        for other in alias_list[1:]:
            union(alias_list[0], other)
    sets: dict[str, set[str]] = {}
    for g in groups:
        sets.setdefault(find(g.id), set()).add(g.id)
    return list(sets.values())


def _kinds_match(pred_kind: str, gt_kind: str) -> bool:
    a, b = pred_kind.lower(), gt_kind.lower()
    if a == b:
        return True
    return a in _WORKLOAD_KINDS and b in _WORKLOAD_KINDS  # Workload Kind Equivalence


def _match_group(name: str, groups: list[Group]) -> Group | None:
    """Match a predicted `namespace/Kind/name` to a ground-truth group."""
    parts = name.strip().strip("/").split("/")
    if len(parts) != 3:
        return None
    ns, kind, leaf = parts
    for g in groups:
        if g.namespace and ns.lower() != g.namespace.lower():
            continue
        if not _kinds_match(kind, g.kind):
            continue
        if any(p.search(leaf) for p in g.filters):
            return g
    return None


@dataclass
class Score:
    precision_at_full_recall: float
    full_recall: bool
    tp: int
    fp: int
    matched_root_cause_sets: int
    total_root_cause_sets: int
    predictions: list[str]
    notes: str = ""


def grade(ground_truth_yaml: str, agent_output: dict, judge=None) -> Score:
    """Score one repeat.

    Entity normalization (mapping each prediction to a ground-truth group) is done
    by `judge` (an `itbench.judge.LLMJudge`) when provided — this is AA's faithful
    path. With `judge=None`, fall back to the deterministic `filter`-regex matcher.
    """
    gt = yaml.safe_load(ground_truth_yaml) or {}
    payload = gt.get("spec") if isinstance(gt.get("spec"), dict) else gt  # AA: GT.spec when present
    groups = _parse_groups(payload)
    by_id = {g.id: g for g in groups}
    sets = _scoring_sets(groups, payload.get("aliases", []))
    rc_sets = [s for s in sets if any(by_id[i].root_cause for i in s)]

    pred_factors = [
        cf
        for cf in (agent_output.get("contributing_factors") or [])
        if (cf.get("name") or "").strip()
    ]
    preds = [(cf.get("name") or "").strip() for cf in pred_factors]

    note = ""
    if judge is not None:
        try:
            # AA's judge sees the full GT and the full generated response.
            matched_ids = judge.normalize(gt, {"contributing_factors": pred_factors})
            note = "judge=llm"
        except Exception as e:  # noqa: BLE001 — surface, don't crash a whole run
            matched_ids = [g.id if (g := _match_group(n, groups)) else None for n in preds]
            note = f"judge error: {type(e).__name__}: {e}; fell back to deterministic matcher"
    else:
        matched_ids = [g.id if (g := _match_group(n, groups)) else None for n in preds]
        note = "judge=deterministic"

    tp = fp = 0
    covered: set[int] = set()
    for gid in matched_ids:
        if gid is None:
            fp += 1
            continue
        # which scoring set does this group belong to, and is it a root-cause set?
        hit_rc = False
        for s in sets:
            if gid in s:
                if any(by_id[i].root_cause for i in s):
                    hit_rc = True
                    covered.add(idx_of(sets, s))
                break
        if hit_rc:
            tp += 1
        else:
            fp += 1  # matched a non-root-cause entity = false positive

    matched_rc = sum(1 for s in rc_sets if idx_of(sets, s) in covered)
    full_recall = matched_rc == len(rc_sets) and len(rc_sets) > 0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    score = precision if full_recall else 0.0
    if not preds:
        note = "no predictions produced"
    return Score(
        precision_at_full_recall=score,
        full_recall=full_recall,
        tp=tp,
        fp=fp,
        matched_root_cause_sets=matched_rc,
        total_root_cause_sets=len(rc_sets),
        predictions=preds,
        notes=note,
    )


def idx_of(sets: list[set[str]], target: set[str]) -> int:
    for i, s in enumerate(sets):
        if s is target:
            return i
    return -1
