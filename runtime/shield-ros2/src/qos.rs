//! Topic names and QoS for the ROS overlay.
//!
//! These match `ros2/vla_shield_msgs`. The in-process [`crate::overlay`]
//! uses the same names so an `rclrs` node can bind 1:1 later.

/// Incoming VLA command (`vla_shield_msgs/ActionProposal`).
pub const ACTION_TOPIC: &str = "/vla_shield/action";
/// Outgoing verdict (`vla_shield_msgs/SafetyDecision`).
pub const DECISION_TOPIC: &str = "/vla_shield/decision";
/// Monitor stream (`vla_shield_msgs/RiskTelemetry`).
pub const TELEMETRY_TOPIC: &str = "/vla_shield/telemetry";
/// Sibling of the action topic — current `q` for FK. Not in ActionProposal.msg.
pub const JOINT_STATES_TOPIC: &str = "/joint_states";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reliability {
    BestEffort,
    Reliable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    Volatile,
    TransientLocal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QosProfile {
    pub reliability: Reliability,
    pub durability: Durability,
    /// Keep-last history depth.
    pub depth: i32,
}

impl QosProfile {
    /// Commands: reliable, drop stale ticks.
    pub fn action() -> Self {
        Self {
            reliability: Reliability::Reliable,
            durability: Durability::Volatile,
            depth: 1,
        }
    }

    /// Verdicts: late joiners see the last decision.
    pub fn decision() -> Self {
        Self {
            reliability: Reliability::Reliable,
            durability: Durability::TransientLocal,
            depth: 1,
        }
    }

    /// Joint state: sensor-data-like.
    pub fn joint_state() -> Self {
        Self {
            reliability: Reliability::BestEffort,
            durability: Durability::Volatile,
            depth: 1,
        }
    }

    pub fn telemetry() -> Self {
        Self::decision()
    }
}
