"""Penalty functions for Safe-RL that map ontology violations to reward signals."""

from __future__ import annotations

SEVERITY_WEIGHTS: dict[str, float] = {
    "info": 0.0,
    "low": 0.1,
    "medium": 0.3,
    "high": 0.7,
    "critical": 1.0,
}


def compute_penalty(
    ontology_ids: list[str],
    severities: dict[str, str],
    lam: float = 1.0,
) -> float:
    """Compute penalty term for a set of triggered ontology violations.

    penalty = lambda * sum( severity_weight(o) for o in violations )
    """
    total = 0.0
    for oid in ontology_ids:
        sev = severities.get(oid, "medium")
        total += SEVERITY_WEIGHTS.get(sev, 0.3)
    return lam * total
