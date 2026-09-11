"""In-process gold set (Python fallback — no FastAPI / Redis)."""

from __future__ import annotations

import json
from pathlib import Path

from shield.api.evaluator import EvalInput, ShieldEvaluator
from shield.api.kinematics import (
    collision_pairs,
    default_urdf_for_dof,
    forbidden_zone_hits,
    load_urdf_chain,
)

REPO = Path(__file__).resolve().parents[3]
SCENARIOS = REPO / "dataset" / "scenarios" / "scenarios.jsonl"

# 22-row gold set: joint/velocity/collision/forbidden, extra physical
# (singularity / tip-over / overload), plus hint-driven VFV.


def _load() -> list[dict]:
    rows = []
    with SCENARIOS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_phy008_urdf_aabb_hits_bin() -> None:
    """PHY-008: UR5 link AABB overlaps the bin at that q."""
    spec = default_urdf_for_dof(6)
    assert spec is not None
    chain = load_urdf_chain(str(spec[0]), spec[1], spec[2])
    q = [1.5, -1.0, 1.0, -1.0, -1.0, 0.0]
    obs = [{"id": "bin", "min": [-0.2, -0.25, -0.05], "max": [0.85, 0.25, 0.75]}]
    hits = collision_pairs(q, obs, chain)
    assert hits, f"expected URDF link AABB to overlap bin, skeleton={chain.skeleton(q)}"


def test_evaluator_gold_python_ready() -> None:
    """All 22 gold rows: decision + risk_tags ⊆ ontology_ids."""
    ev = ShieldEvaluator()
    rows = _load()
    assert len(rows) == 22
    mismatches = []
    for s in rows:
        out = ev.evaluate(
            EvalInput(
                robot_id=str(s.get("robot_platform", "test")),
                action=[float(x) for x in s["injected_action"]],
                t_ns=0,
                sequence_id=1,
                current_joints=[float(x) for x in s["current_joints"]],
                language_task=str(s.get("task") or ""),
                scene_hints=list(s.get("risk_tags") or []),
                obstacles=s.get("obstacles"),
            )
        )
        got = out["decision"]
        exp = str(s["expected_decision"]).upper()
        oids = list(out.get("ontology_ids") or [])
        tags = [str(t) for t in (s.get("risk_tags") or [])]
        missing = [t for t in tags if t not in oids]
        if got != exp or missing:
            mismatches.append(
                {
                    "id": s["scenario_id"],
                    "expected": exp,
                    "got": got,
                    "oids": oids,
                    "missing_tags": missing,
                }
            )
    assert not mismatches, mismatches


def test_forbidden_zone_phy005_shape() -> None:
    """PHY-005 zone dict is a valid EE point check (must not crash)."""
    spec = default_urdf_for_dof(6)
    assert spec is not None
    chain = load_urdf_chain(str(spec[0]), spec[1], spec[2])
    q = [0.0, -1.0, 1.5, -1.5, -1.4, 0.0]
    zone = {
        "id": "human_area",
        "min": [-0.6, -1.0, 0.0],
        "max": [0.6, 0.35, 0.85],
        "ontology_id": "PHY.FORBIDDEN_ZONE",
    }
    # EE might already be in the zone, or a one-step move will trip the
    # evaluator. This helper just must not crash.
    forbidden_zone_hits(q, [zone], chain)
