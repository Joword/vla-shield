"""URDF FK is a real chain, not the yaw/pitch sketch."""

from __future__ import annotations

from shield.api.kinematics import (
    confirm_aabb_hit,
    default_urdf_for_dof,
    fk_skeleton_for,
    load_urdf_chain,
    self_collision_pairs,
)


def test_panda_seven_dof_loads() -> None:
    """Panda fixture is 7 revolute joints, 8 skeleton points."""
    spec = default_urdf_for_dof(7)
    assert spec is not None
    path, root, ee = spec
    chain = load_urdf_chain(str(path), root, ee)
    assert chain.dof == 7
    skel = chain.skeleton([0.0] * 7)
    assert len(skel) == 8


def test_ur5_zero_pose_reaches_forward() -> None:
    """UR5 zero pose EE is out in front; fk_skeleton_for matches the chain."""
    spec = default_urdf_for_dof(6)
    assert spec is not None
    path, root, ee = spec
    chain = load_urdf_chain(str(path), root, ee)
    assert chain.dof == 6
    skel = chain.skeleton([0.0] * 6)
    assert len(skel) == 7
    ee_xyz = skel[-1]
    # UR5 zero pose sits out in front, not at the origin.
    assert abs(ee_xyz[0]) + abs(ee_xyz[1]) + abs(ee_xyz[2]) > 0.2
    via = fk_skeleton_for([0.0] * 6, chain)
    assert via[-1] == skel[-1]


def test_adjacent_links_are_skipped() -> None:
    boxes = [
        ("l0", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ("l1", [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]),
        ("l2", [0.4, 0.4, 0.4], [1.4, 1.4, 1.4]),
    ]
    assert self_collision_pairs(boxes) == []


def test_nonadjacent_overlap_is_self_collision() -> None:
    far = ([10.0, 10.0, 10.0], [11.0, 11.0, 11.0])
    boxes = [
        ("l0", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ("l1", far[0], far[1]),
        ("l2", far[0], far[1]),
        ("l3", [0.2, 0.2, 0.2], [0.8, 0.8, 0.8]),
    ]
    pairs = self_collision_pairs(boxes)
    assert pairs == [("l0", "l3")]


def test_confirm_deflate_drops_grazing_hit() -> None:
    assert confirm_aabb_hit(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.99, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        0.02,
    ) is False
    assert confirm_aabb_hit(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8],
        0.02,
    ) is True
