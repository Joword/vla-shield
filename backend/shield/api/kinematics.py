"""Lightweight serial-arm FK for the digital-twin monitor.

This is **not** a URDF solver.  It maps an N-DoF joint vector to a chain of
Cartesian waypoints so the Three.js scene can draw a skeleton and a shadow
polyline without shipping a mesh.  Coordinates are **robot frame, Z-up,
metres**.  The monitor converts to Three.js Y-up at render time.
"""

from __future__ import annotations

import math

# UR5-ish link lengths (base lift is applied separately).  Extra DoF
# beyond 6 reuse 8 cm distal links.
_BASE_LINKS = (0.12, 0.35, 0.30, 0.12, 0.10, 0.08)
_BASE_LIFT = 0.12

# Default table-side obstacle, matching the old SceneView placeholder box.
DEFAULT_ZONES: list[dict] = [
    {
        "min": [0.55, -0.15, 0.0],
        "max": [0.95, 0.25, 0.90],
        "label": "obstacle",
        "ontology_id": "PHY.FORBIDDEN_ZONE",
    }
]


def link_lengths_for_dof(dof: int) -> list[float]:
    if dof <= 0:
        return []
    if dof <= len(_BASE_LINKS):
        return list(_BASE_LINKS[:dof])
    return list(_BASE_LINKS) + [0.08] * (dof - len(_BASE_LINKS))


def fk_skeleton(joints: list[float]) -> list[list[float]]:
    """Return waypoints ``[origin, base, j1, …, jN]`` in robot XYZ (Z-up)."""
    dof = len(joints)
    if dof == 0:
        return [[0.0, 0.0, 0.0], [0.0, 0.0, _BASE_LIFT]]
    lengths = link_lengths_for_dof(dof)
    points: list[list[float]] = [[0.0, 0.0, 0.0], [0.0, 0.0, _BASE_LIFT]]
    x, y, z = 0.0, 0.0, _BASE_LIFT
    yaw = 0.0
    pitch = 0.0
    for i, (q, length) in enumerate(zip(joints, lengths)):
        if i == 0:
            yaw += q
        elif i == 1:
            pitch += q
        elif i % 2 == 0:
            yaw += 0.5 * q
        else:
            pitch += 0.5 * q
        cos_p = math.cos(pitch)
        x += length * cos_p * math.cos(yaw)
        y += length * cos_p * math.sin(yaw)
        z += length * math.sin(pitch)
        z = max(z, 0.02)
        points.append([x, y, z])
    return points


def ee_of(joints: list[float]) -> list[float]:
    return fk_skeleton(joints)[-1]


def shadow_polyline(joint_samples: list[list[float]]) -> list[list[float]]:
    """End-effector XYZ for each joint-space shadow sample."""
    return [ee_of(sample) for sample in joint_samples if sample]


def zones_for(ontology_ids: list[str]) -> list[dict]:
    """Static demo obstacle plus extra AABBs when semantic rules fire."""
    zones = [dict(z) for z in DEFAULT_ZONES]
    triggered = set(ontology_ids)
    if "SEM.HEAT_SOURCE" in triggered:
        zones.append(
            {
                "min": [0.20, 0.30, 0.20],
                "max": [0.50, 0.60, 0.70],
                "label": "heat_source",
                "ontology_id": "SEM.HEAT_SOURCE",
            }
        )
    if "SEM.FORBIDDEN_REGION" in triggered:
        zones.append(
            {
                "min": [-0.20, 0.40, 0.00],
                "max": [0.20, 0.80, 0.50],
                "label": "forbidden_region",
                "ontology_id": "SEM.FORBIDDEN_REGION",
            }
        )
    if "SEM.HUMAN_PROXIMITY" in triggered:
        zones.append(
            {
                "min": [-0.35, -0.55, 0.00],
                "max": [0.35, -0.15, 1.20],
                "label": "human_perimeter",
                "ontology_id": "SEM.HUMAN_PROXIMITY",
            }
        )
    return zones
