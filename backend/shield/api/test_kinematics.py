"""URDF FK skeleton is a real chain, not the yaw/pitch sketch."""

from __future__ import annotations

from shield.api.kinematics import (
    default_urdf_for_dof,
    fk_skeleton_for,
    load_urdf_chain,
)


def test_ur5_zero_pose_reaches_forward() -> None:
    spec = default_urdf_for_dof(6)
    assert spec is not None
    path, root, ee = spec
    chain = load_urdf_chain(str(path), root, ee)
    assert chain.dof == 6
    skel = chain.skeleton([0.0] * 6)
    assert len(skel) == 7
    ee_xyz = skel[-1]
    # UR5 zero pose is not at the origin.
    assert abs(ee_xyz[0]) + abs(ee_xyz[1]) + abs(ee_xyz[2]) > 0.2
    via = fk_skeleton_for([0.0] * 6, chain)
    assert via[-1] == skel[-1]
