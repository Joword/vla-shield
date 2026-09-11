//! ROS 2 glue. Default build has no ROS — the pipeline talks over channels
//! so you can test without a distro installed.

pub mod adapter;
pub mod config;
pub mod node;
pub mod pipeline;
pub mod tf2;

pub use adapter::{
    classify_reasons, scene_from_aabbs, ActionProposal, SafetyDecision,
};
pub use pipeline::SafetyPipeline;
