//! ROS 2 glue. Default build has no ROS — the pipeline talks over channels
//! so you can test without a distro installed.

pub mod adapter;
pub mod bind;
pub mod config;
pub mod node;
pub mod overlay;
pub mod pipeline;
pub mod qos;
pub mod tf2;

pub use adapter::{
    classify_reasons, scene_from_aabbs, ActionProposal, SafetyDecision,
};
pub use bind::{overlay_bindings, rclrs_bind_table, Direction, TopicBinding};
pub use overlay::{RiskTelemetry, ShieldRosOverlay};
pub use qos::{
    ACTION_TOPIC, DECISION_TOPIC, JOINT_STATES_TOPIC, QosProfile, TELEMETRY_TOPIC,
};
pub use pipeline::SafetyPipeline;
