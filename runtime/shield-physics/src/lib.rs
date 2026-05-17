pub mod projection;
pub mod semantic;

use shield_core::action::ActionVector;
use shield_core::error::Result;
use shield_core::scene::SceneGraph;
use shield_core::types::JointLimits;

pub use semantic::{SemanticConstraint, SemanticConstraintMapper};

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
    /// When set, end-effector pose is computed via URDF forward kinematics.
    pub urdf_chain: Option<&'a shield_urdf::UrdfKinematicChain>,
    /// Cartesian forbidden regions in the same frame as FK (typically base link).
    pub forbidden_zones: &'a [shield_urdf::AxisAlignedBox],
    /// Active semantic constraints mapped from SEM.* ontology nodes.
    pub semantic_constraints: &'a [SemanticConstraint],
}

/// Trait for projecting abstract VLA actions into physical-space proposals.
pub trait PhysicalProjector: Send + Sync {
    fn project(&self, ctx: &ProjectionContext, action: &ActionVector) -> Result<DynProposal>;
}
