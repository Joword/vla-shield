"""Replay dataset/scenarios/traces.jsonl through ShieldEvaluator."""

from __future__ import annotations

import json
from pathlib import Path

from shield.api.evaluator import EvalInput, ShieldEvaluator

REPO = Path(__file__).resolve().parents[3]
TRACES = REPO / "dataset" / "scenarios" / "traces.jsonl"


def _load() -> list[dict]:
    rows = []
    with TRACES.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_traces_replay_matches_labels() -> None:
    """Each trace's last step matches `label`; risk_tags ⊆ ontology_ids."""
    ev = ShieldEvaluator()
    rows = _load()
    assert len(rows) == 4
    mismatches = []
    for trace in rows:
        actions = list(trace.get("actions") or [])
        joints = list(trace.get("joints") or [])
        assert actions, trace["trace_id"]
        last = None
        prev: list[float] = []
        for i, action in enumerate(actions):
            q = joints[i] if i < len(joints) else (joints[-1] if joints else [0.0] * len(action))
            last = ev.evaluate(
                EvalInput(
                    robot_id=str(trace.get("robot_platform", "trace")),
                    action=[float(x) for x in action],
                    t_ns=i,
                    sequence_id=i + 1,
                    current_joints=[float(x) for x in q],
                    obstacles=trace.get("obstacles"),
                    prev_velocity=prev,
                )
            )
            prev = [float(x) for x in action]
        assert last is not None
        got = last["decision"]
        exp = str(trace["label"]).upper()
        oids = list(last.get("ontology_ids") or [])
        tags = [str(t) for t in (trace.get("risk_tags") or [])]
        missing = [t for t in tags if t not in oids]
        if got != exp or missing:
            mismatches.append(
                {
                    "id": trace["trace_id"],
                    "expected": exp,
                    "got": got,
                    "oids": oids,
                    "missing_tags": missing,
                }
            )
    assert not mismatches, mismatches
