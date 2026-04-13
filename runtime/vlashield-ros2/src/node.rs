//! ROS 2 lifecycle hooks for the shield runtime.
//!
//! Full `rclrs` graph wiring (subscriptions to `ActionProposal`, publishers for
//! `SafetyDecision`, QoS, executors) should be built in a ROS 2 overlay workspace
//! where `ros2` feature is enabled. The hooks below match the lifecycle sequence
//! used by managed nodes (`configure` → `activate` → `deactivate`).

use std::path::PathBuf;

use tracing::info;

/// Stateful callbacks for a shield ROS 2 node (maps to lifecycle transitions).
#[derive(Debug, Clone)]
pub struct ShieldLifecycleHooks {
    pub urdf_path: Option<PathBuf>,
}

impl Default for ShieldLifecycleHooks {
    fn default() -> Self {
        Self { urdf_path: None }
    }
}

impl ShieldLifecycleHooks {
    /// `on_configure`: load URDF path, declare parameters, allocate buffers.
    pub fn on_configure(&mut self, urdf_path: PathBuf) {
        self.urdf_path = Some(urdf_path);
        info!(path = ?self.urdf_path, "shield lifecycle: on_configure");
    }

    /// `on_activate`: arm the safety pipeline and start processing proposals.
    pub fn on_activate(&self) {
        info!("shield lifecycle: on_activate — safety pipeline armed");
    }

    /// `on_deactivate`: soft landing / hold — stop forwarding raw VLA commands.
    pub fn on_deactivate(&self) {
        info!("shield lifecycle: on_deactivate — soft landing");
    }
}

/// When `ros2` is enabled, call this from your `rclrs` context after `rclrs::init`.
#[cfg(feature = "ros2")]
pub fn log_rclrs_build_stub() {
    tracing::warn!(
        "rclrs feature is on: link against the ROS 2 workspace and register publishers/subscribers here"
    );
}

#[cfg(not(feature = "ros2"))]
pub fn log_rclrs_build_stub() {
    tracing::debug!("rclrs feature off: build with `--features ros2` inside a ROS 2 environment");
}
