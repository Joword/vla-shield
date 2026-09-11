//! ROS 2 integration for the shield runtime.
//!
//! When the `ros2` feature is disabled (default), the crate exposes a
//! standalone pipeline that communicates via channels instead of ROS topics.
//! This allows testing without a ROS 2 installation.

pub mod adapter;
pub mod config;
pub mod node;
pub mod pipeline;
pub mod tf2;

pub use adapter::{
    classify_reasons, scene_from_aabbs, ActionProposal, SafetyDecision,
};
pub use pipeline::SafetyPipeline;
