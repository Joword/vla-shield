"""Gold PHY-004 / 006 / 007: singularity, tip-over, overload.

Don't fold overload into PHY.JOINT_LIMIT — that's a different rule.
"""

from __future__ import annotations

from shield.api.physical_checks import extra_physical_reasons


def _oids(action, joints, **kwargs) -> set[str]:
    return {oid for oid, _, _ in extra_physical_reasons(action, joints, **kwargs)}


def test_phy004_elbow_lock() -> None:
    """Franka q4≈0 → PHY.SINGULARITY."""
    oids = _oids(
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.02, 0.0, 0.02, 0.0],
    )
    assert "PHY.SINGULARITY" in oids


def test_phy006_base_accel() -> None:
    """8-DoF last channel 2.5 m/s² → PHY.TIPOVER."""
    oids = _oids([0.0] * 7 + [2.5], [0.0] * 8)
    assert "PHY.TIPOVER" in oids


def test_phy007_wrist_overload() -> None:
    """Franka wrist 5 rad/s vs 20 Nm cap → PHY.OVERLOAD."""
    tau = [50.0] * 7
    tau[4] = 20.0
    action = [0.0] * 7
    action[4] = 5.0
    oids = _oids(
        action,
        [0.0, 0.3, 0.0, -1.0, 0.0, 1.5, 0.8],
        torque_max=tau,
    )
    assert "PHY.OVERLOAD" in oids


def test_phy003_is_not_overload() -> None:
    """UR5 base 10 rad/s is VELOCITY, not OVERLOAD."""
    action = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    oids = _oids(action, [0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    assert "PHY.OVERLOAD" not in oids
