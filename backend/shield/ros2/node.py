"""ROS overlay topic bindings. rclpy if present; otherwise in-process.

Same names/QoS as runtime/shield-ros2 overlay. A real node does:

    for b in OVERLAY_BINDINGS:
        if b["direction"] == "in":
            node.create_subscription(...)
        else:
            node.create_publisher(...)
"""

from __future__ import annotations

import importlib
from typing import Any

from ..api.evaluator import EvalInput, ShieldEvaluator

OVERLAY_BINDINGS: tuple[dict[str, Any], ...] = (
    {
        "topic": "/vla_shield/action",
        "msg": "vla_shield_msgs/msg/ActionProposal",
        "direction": "in",
        "reliability": "reliable",
        "durability": "volatile",
        "depth": 1,
    },
    {
        "topic": "/joint_states",
        "msg": "sensor_msgs/msg/JointState",
        "direction": "in",
        "reliability": "best_effort",
        "durability": "volatile",
        "depth": 1,
    },
    {
        "topic": "/vla_shield/decision",
        "msg": "vla_shield_msgs/msg/SafetyDecision",
        "direction": "out",
        "reliability": "reliable",
        "durability": "transient_local",
        "depth": 1,
    },
    {
        "topic": "/vla_shield/telemetry",
        "msg": "vla_shield_msgs/msg/RiskTelemetry",
        "direction": "out",
        "reliability": "reliable",
        "durability": "transient_local",
        "depth": 1,
    },
)


def rclpy_available() -> bool:
    """True when the optional ROS 2 Python client can be imported."""
    try:
        importlib.import_module("rclpy")
    except ImportError:
        return False
    return True


class ShieldRosNode:
    """In-process overlay. Empty `current_joints` uses the last joint_state."""

    def __init__(self, evaluator: ShieldEvaluator | None = None) -> None:
        self.evaluator = evaluator or ShieldEvaluator()
        self.last_joints: list[float] = []
        self.rclpy_available = rclpy_available()

    def on_joint_state(self, q: list[float]) -> None:
        """Latch `/joint_states` until the next action tick."""
        if q:
            self.last_joints = [float(v) for v in q]

    def on_action(  # pylint: disable=too-many-arguments
        self,
        action: list[float],
        *,
        current_joints: list[float] | None = None,
        prev_velocity: list[float] | None = None,
        sequence_id: int = 0,
        t_ns: int = 0,
        robot_id: str = "default-robot",
        obstacles: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Evaluate one `/vla_shield/action` and return a decision payload."""
        q = list(current_joints) if current_joints else list(self.last_joints)
        out = self.evaluator.evaluate(
            EvalInput(
                robot_id=robot_id,
                action=[float(v) for v in action],
                t_ns=t_ns,
                sequence_id=sequence_id,
                current_joints=q,
                obstacles=obstacles,
                prev_velocity=list(prev_velocity or []),
            )
        )
        return {
            "topic": "/vla_shield/decision",
            "sequence_id": out["sequence_id"],
            "decision": out["decision"],
            "ontology_ids": list(out["ontology_ids"]),
            "risk": out["risk"],
            "telemetry_topic": "/vla_shield/telemetry",
        }
