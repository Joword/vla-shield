"""Overlay topic contract without a ROS distro."""

from __future__ import annotations

from shield.ros2 import OVERLAY_BINDINGS, ShieldRosNode, rclpy_available


def test_bindings_match_overlay_contract() -> None:
    topics = [b["topic"] for b in OVERLAY_BINDINGS]
    assert topics == [
        "/vla_shield/action",
        "/joint_states",
        "/vla_shield/decision",
        "/vla_shield/telemetry",
    ]
    action = OVERLAY_BINDINGS[0]
    assert action["reliability"] == "reliable"
    assert action["depth"] == 1
    decision = OVERLAY_BINDINGS[2]
    assert decision["durability"] == "transient_local"
    assert decision["direction"] == "out"


def test_inprocess_node_pass_without_rclpy() -> None:
    node = ShieldRosNode()
    node.on_joint_state([0.0] * 6)
    out = node.on_action([0.05] * 6, sequence_id=9)
    assert out["decision"] == "PASS"
    assert out["topic"] == "/vla_shield/decision"
    assert rclpy_available() is node.rclpy_available
