pub mod config;
pub mod pipeline;

/// This crate provides the main vlashield-Runtime entry point that integrates
/// all safety checks into a ROS 2 node pipeline.
///
/// When the `ros2` feature is disabled (default), the crate exposes a
/// standalone pipeline that communicates via channels instead of ROS topics.
/// This allows testing without a ROS 2 installation.
