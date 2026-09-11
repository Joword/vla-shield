//! Shadow path: cheap multi-step look-ahead, off the hot path.
//!
//! Runs in the background before the command hits actuators. Checks joint
//! limits, forbidden zones, optionally near-singularity. Result is a risk
//! prior for the arbiter — stale is fine, skip it.
//!
//! * CPU only. Don't block ROS 2 timers, don't touch the GPU.
//! * If the next action arrives first, keep the old result or drop it.
//! * `ShadowSimulator` is the swap point for a fancier backend later.

pub mod joint_space;
pub mod result;

use result::ShadowResult;
use shield_core::types::JointLimits;
use shield_urdf::{AxisAlignedBox, UrdfKinematicChain};

pub use joint_space::{simulate, JointSpaceShadowConfig};

/// Pluggable shadow sim. Swap in something fancier later.
pub trait ShadowSimulator: Send + Sync {
    /// Run it, return a risk summary.
    ///
    /// Stay well under the inter-action gap (typically 10–20 ms). Don't block.
    fn simulate(
        &self,
        current_joints: &[f64],
        action: &[f64],
        limits: &JointLimits,
        urdf_chain: Option<&UrdfKinematicChain>,
        forbidden_zones: &[AxisAlignedBox],
    ) -> ShadowResult;
}

/// Default joint-space sim. Just calls [`joint_space::simulate`].
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
