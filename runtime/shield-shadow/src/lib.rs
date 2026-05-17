//! `shield-shadow` — Rust-native shadow path pre-check for the VLA-Shield async safety layer.
//!
//! # Purpose
//!
//! Before a VLA action reaches the physical actuators, the shadow simulator
//! runs a lightweight multi-step trajectory in the background (outside the
//! hard-real-time hot path).  It evaluates joint-limit approach, kinematic
//! forbidden zones, and—optionally—near-singularity conditions, returning a
//! `ShadowResult` that the arbiter can incorporate as a risk prior.
//!
//! # Design
//!
//! * **Async-safe**: the simulation is CPU-bound and does not touch the GPU
//!   or block ROS 2 timers.
//! * **Stale-safe**: if the simulation is still running when the next action
//!   arrives, the previous result is used (or the result is ignored).
//! * **Pluggable**: the `ShadowSimulator` trait allows swapping in higher-
//!   fidelity implementations (e.g. a CUDA physics core) later.

pub mod joint_space;
pub mod result;

use result::ShadowResult;
use shield_core::types::JointLimits;
use shield_urdf::{AxisAlignedBox, UrdfKinematicChain};

pub use joint_space::{simulate, JointSpaceShadowConfig};

/// Trait for pluggable shadow path simulators.
pub trait ShadowSimulator: Send + Sync {
    /// Run the simulation and return a risk summary.
    ///
    /// Implementations should be non-blocking and complete well within the
    /// inter-action period (typically 10–20 ms).
    fn simulate(
        &self,
        current_joints: &[f64],
        action: &[f64],
        limits: &JointLimits,
        urdf_chain: Option<&UrdfKinematicChain>,
        forbidden_zones: &[AxisAlignedBox],
    ) -> ShadowResult;
}

/// Default joint-space simulator backed by [`joint_space::simulate`].
pub struct JointSpaceSimulator {
    pub config: JointSpaceShadowConfig,
}

impl Default for JointSpaceSimulator {
    fn default() -> Self {
        JointSpaceSimulator {
            config: JointSpaceShadowConfig::default(),
        }
    }
}

impl ShadowSimulator for JointSpaceSimulator {
    fn simulate(
        &self,
        current_joints: &[f64],
        action: &[f64],
        limits: &JointLimits,
        urdf_chain: Option<&UrdfKinematicChain>,
        forbidden_zones: &[AxisAlignedBox],
    ) -> ShadowResult {
        joint_space::simulate(&self.config, current_joints, action, limits, urdf_chain, forbidden_zones)
    }
}
