//! Overlay topic → QoS + message type. An `rclrs` node binds 1:1 from this
//! table; compiling `--features ros2` still needs a ROS overlay workspace.

use crate::qos::{
    ACTION_TOPIC, DECISION_TOPIC, JOINT_STATES_TOPIC, QosProfile, TELEMETRY_TOPIC,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    In,
    Out,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopicBinding {
    pub topic: &'static str,
    pub msg: &'static str,
    pub direction: Direction,
    pub qos: QosProfile,
}

/// Four topics [`crate::ShieldRosOverlay`] already speaks.
pub fn overlay_bindings() -> [TopicBinding; 4] {
    [
        TopicBinding {
            topic: ACTION_TOPIC,
            msg: "vla_shield_msgs/ActionProposal",
            direction: Direction::In,
            qos: QosProfile::action(),
        },
        TopicBinding {
            topic: JOINT_STATES_TOPIC,
            msg: "sensor_msgs/JointState",
            direction: Direction::In,
            qos: QosProfile::joint_state(),
        },
        TopicBinding {
            topic: DECISION_TOPIC,
            msg: "vla_shield_msgs/SafetyDecision",
            direction: Direction::Out,
            qos: QosProfile::decision(),
        },
        TopicBinding {
            topic: TELEMETRY_TOPIC,
            msg: "vla_shield_msgs/RiskTelemetry",
            direction: Direction::Out,
            qos: QosProfile::telemetry(),
        },
    ]
}

/// Same table an `rclrs` executor would `create_subscription` / `create_publisher` with.
pub fn rclrs_bind_table() -> [TopicBinding; 4] {
    overlay_bindings()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qos::{Durability, Reliability};

    #[test]
    fn bind_table_covers_overlay_topics() {
        let table = rclrs_bind_table();
        assert_eq!(table.len(), 4);
        assert_eq!(table[0].topic, ACTION_TOPIC);
        assert_eq!(table[0].qos.reliability, Reliability::Reliable);
        assert_eq!(table[1].topic, JOINT_STATES_TOPIC);
        assert_eq!(table[1].qos.reliability, Reliability::BestEffort);
        assert_eq!(table[2].topic, DECISION_TOPIC);
        assert_eq!(table[2].qos.durability, Durability::TransientLocal);
        assert_eq!(table[2].qos.depth, 1);
        assert_eq!(table[3].topic, TELEMETRY_TOPIC);
        assert_eq!(table[3].direction, Direction::Out);
    }
}
