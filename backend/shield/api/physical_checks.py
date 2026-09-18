"""Python copies of shield-physics::checks.

So the FastAPI fallback can still fire PHY.SINGULARITY / PHY.TIPOVER /
PHY.OVERLOAD (gold PHY-004 / 006 / 007) without the Rust ext. Numbers come
from dataset/ontology/phy_calibration.json — same file Rust include_str!s.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

_CAL_PATH = Path(__file__).resolve().parents[3] / "dataset" / "ontology" / "phy_calibration.json"


@lru_cache(maxsize=4)
def load_phy_calibration(path: str | None = None) -> dict[str, Any]:
    """Shared coefficients. `path` overrides the shipped JSON."""
    p = Path(path) if path else _CAL_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("phy_calibration.json must be an object")
    return data


def phy_calibration() -> dict[str, Any]:
    """Cached shipped calibration."""
    return load_phy_calibration()


def clamp_joint_velocity(
    action: list[float],
    velocity_max: list[float],
    *,
    prev_velocity: list[float] | None = None,
    acceleration_max: list[float] | None = None,
    dt: float = 0.0,
) -> list[float]:
    """Vel cap, then |Δv| ≤ a_max·dt when prev_velocity length matches DoF."""
    n = len(action)
    prev = list(prev_velocity or [])
    use_accel = len(prev) == n and dt > 0.0 and acceleration_max is not None
    out: list[float] = []
    for i, raw in enumerate(action):
        vmax = float(velocity_max[i]) if i < len(velocity_max) else float("inf")
        vel = max(-vmax, min(vmax, float(raw)))
        if use_accel:
            a_max = float(acceleration_max[i]) if i < len(acceleration_max) else float("inf")
            if math.isfinite(a_max):
                da = a_max * dt
                p = float(prev[i])
                vel = max(p - da, min(p + da, vel))
        out.append(vel)
    return out


def _ee_xyz(chain: Any, q: list[float]) -> list[float] | None:
    skeleton = getattr(chain, "skeleton", None)
    if skeleton is None:
        return None
    try:
        pts = skeleton(q)
        return pts[-1]
    except (AttributeError, IndexError, TypeError, ValueError, ArithmeticError):
        return None


def positional_manipulability(chain: Any, q: list[float]) -> float | None:
    """sqrt(det(J Jᵀ)) from a 3×n numerical Jacobian. None if the chain can't run."""
    n = len(q)
    if chain is None or getattr(chain, "dof", None) != n or n == 0:
        return None
    p0 = _ee_xyz(chain, q)
    if p0 is None:
        return None
    eps = 1e-5
    j = []
    for i in range(n):
        qp = list(q)
        qp[i] += eps
        p1 = _ee_xyz(chain, qp)
        if p1 is None:
            return None
        j.append([(p1[k] - p0[k]) / eps for k in range(3)])
    gram = [[0.0] * 3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            gram[r][c] = sum(j[i][r] * j[i][c] for i in range(n))
    det = (
        gram[0][0] * (gram[1][1] * gram[2][2] - gram[1][2] * gram[2][1])
        - gram[0][1] * (gram[1][0] * gram[2][2] - gram[1][2] * gram[2][0])
        + gram[0][2] * (gram[1][0] * gram[2][1] - gram[1][1] * gram[2][0])
    )
    return max(det, 0.0) ** 0.5


Reason = tuple[str, str, float]


def _singularity_reason(q: list[float], chain: Any | None) -> Reason | None:
    cal = phy_calibration()["singularity"]
    manip = positional_manipulability(chain, q)
    elbow = (
        len(q) == int(cal["elbow_lock_dof"])
        and abs(q[int(cal["elbow_lock_joint_index"])]) < float(cal["elbow_lock_rad"])
    )
    below = manip is not None and manip < float(cal["jacobian_floor"])
    if not below and not elbow:
        return None
    m = 0.0 if manip is None else manip
    detail = (
        f"positional manipulability {m:.4f} below threshold "
        f"{float(cal['ontology_manipulability']):.4f}"
    )
    return ("PHY.SINGULARITY", detail, 1.0)


def _tipover_reason(action: list[float]) -> Reason | None:
    cal = phy_calibration()["tipover"]
    if len(action) < int(cal["mobile_min_dof"]):
        return None
    accel = float(action[-1])
    if abs(accel) <= float(cal["base_accel_limit"]):
        return None
    zmp_y = float(cal["com_height_m"]) * accel / float(cal["g"])
    detail = (
        f"base accel {accel:.3f} m/s² exceeds {float(cal['base_accel_limit']):.3f}; "
        f"ZMP at (0.000, {zmp_y:.3f}) m"
    )
    return ("PHY.TIPOVER", detail, 1.0)


def _overload_reason(  # pylint: disable=too-many-locals
    action: list[float],
    q: list[float],
    tau_cap: list[float],
    names: list[str],
) -> Reason | None:
    cal = phy_calibration()["overload"]
    n = len(action)
    distal_from = max(0, n - int(cal["distal_joints"]))
    worst_i: int | None = None
    worst_tau = 0.0
    worst_nominal = 0.0
    for i, velocity in enumerate(action):
        inertia = (
            float(cal["distal_inertia"]) if i >= distal_from else float(cal["proximal_inertia"])
        )
        gravity = float(cal["gravity_coeff"]) * (n - 1 - i) * abs(math.sin(q[i]))
        tau = inertia * abs(float(velocity)) + gravity
        nominal = float(tau_cap[i])
        if tau > nominal and (worst_i is None or tau > worst_tau):
            worst_i = i
            worst_tau = tau
            worst_nominal = nominal
    if worst_i is None:
        return None
    detail = (
        f"joint {names[worst_i]} estimated torque {worst_tau:.2f} Nm "
        f"exceeds nominal {worst_nominal:.2f} Nm"
    )
    return ("PHY.OVERLOAD", detail, 1.0)


def extra_physical_reasons(
    action: list[float],
    joints: list[float],
    *,
    torque_max: list[float] | None = None,
    chain: Any | None = None,
    joint_names: list[str] | None = None,
) -> list[Reason]:
    """PHY.SINGULARITY / PHY.TIPOVER / PHY.OVERLOAD for the projected state."""
    n = len(action)
    if len(joints) == n:
        q = list(joints)
    else:
        q = list(joints) + [0.0] * max(0, n - len(joints))
    q = q[:n]
    names = joint_names or [f"j{i}" for i in range(n)]
    tau_cap = list(torque_max) if torque_max is not None else [50.0] * n
    if len(tau_cap) < n:
        tau_cap = tau_cap + [50.0] * (n - len(tau_cap))

    out: list[Reason] = []
    for reason in (
        _singularity_reason(q, chain),
        _tipover_reason(action),
        _overload_reason(action, q, tau_cap, names),
    ):
        if reason is not None:
            out.append(reason)
    return out
