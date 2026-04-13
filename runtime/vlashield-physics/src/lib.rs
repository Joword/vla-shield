pub mod projection;

use vlashield_core::action::ActionVector;
use vlashield_core::error::Result;
use vlashield_core::scene::SceneGraph;
use vlashield_core::types::JointLimits;

/// Proposed dynamic state after applying an action.
#[derive(Debug, Clone)]
pub struct DynProposal {
    pub joint_positions: Vec<f64>,
    pub joint_velocities: Vec<f64>,
    pub ee_position: [f64; 3],
    pub ee_orientation: [f64; 4],
}

/// Context required for projection.
pub struct ProjectionContext<'a> {
    pub current_joints: &'a [f64],
    pub limits: &'a JointLimits,
    pub scene: &'a SceneGraph,
    pub dt: f64,
}

/// Trait for projecting abstract VLA actions into physical-space proposals.
pub trait PhysicalProjector: Send + Sync {
    fn project(&self, ctx: &ProjectionContext, action: &ActionVector) -> Result<DynProposal>;
}
