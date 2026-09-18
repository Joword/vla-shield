"""Gold PHY-004 / 006 / 007: singularity, tip-over, overload.

Don't fold overload into PHY.JOINT_LIMIT — that's a different rule.
"""

from __future__ import annotations

from shield.api.physical_checks import (
    clamp_joint_velocity,
    extra_physical_reasons,
    phy_calibration,
)


def _oids(action, joints, **kwargs) -> set[str]:
    return {oid for oid, _, _ in extra_physical_reasons(action, joints, **kwargs)}


def test_calibration_json_matches_gold_numbers() -> None:
    """Rust include_str! and Python load the same file."""
    cal = phy_calibration()
    assert cal["singularity"]["elbow_lock_rad"] == 0.08
    assert cal["singularity"]["elbow_lock_dof"] == 7
    assert cal["tipover"]["base_accel_limit"] == 1.5
    assert cal["overload"]["distal_inertia"] == 24.0
    assert cal["overload"]["proximal_inertia"] == 2.0


def test_accel_clamp_uses_prev_velocity() -> None:
    """a_max=5, dt=0.01 → |Δv| ≤ 0.05 even if the command is 999."""
    out = clamp_joint_velocity(
        [999.0],
        [1.0],
        prev_velocity=[0.0],
        acceleration_max=[5.0],
        dt=0.01,
    )
    assert abs(out[0] - 0.05) < 1e-9


def test_empty_prev_velocity_skips_accel() -> None:
    out = clamp_joint_velocity(
        [999.0],
        [1.0],
        prev_velocity=[],
        acceleration_max=[5.0],
        dt=0.01,
    )
    assert abs(out[0] - 1.0) < 1e-9


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
