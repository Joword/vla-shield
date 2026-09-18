pub mod calibration;
pub mod checks;
pub mod projection;
pub mod semantic;

use shield_core::action::ActionVector;
use shield_core::error::Result;
use shield_core::scene::SceneGraph;
use shield_core::types::JointLimits;

pub use calibration::{phy_calibration, PhyCalibration};
pub use checks::{extra_physical_reasons, ontology_for_projection_error};
pub use semantic::{SemanticConstraint, SemanticConstraintMapper};

/// Joints + EE after one projected step.
#[derive(Debug, Clone)]
pub struct DynProposal {
    pub joint_positions: Vec<f64>,
    pub joint_velocities: Vec<f64>,
    pub ee_position: [f64; 3],
    pub ee_orientation: [f64; 4],
}

/// Stuff `project()` needs. Borrowed — don't stash this.
pub struct ProjectionContext<'a> {
    pub current_joints: &'a [f64],
    pub limits: &'a JointLimits,
    pub scene: &'a SceneGraph,
    pub dt: f64,
    /// When set, EE pose comes from URDF FK instead of the origin placeholder.
    pub urdf_chain: Option<&'a shield_urdf::UrdfKinematicChain>,
    /// Cartesian no-go boxes in the same frame as FK (usually base link).
    pub forbidden_zones: &'a [shield_urdf::AxisAlignedBox],
    /// Live SEM.* constraints (heat, humans, …).
    pub semantic_constraints: &'a [SemanticConstraint],
    /// Last tick's joint velocity. Empty → skip acceleration clamp.
    pub prev_velocity: &'a [f64],
}

/// Turn a VLA command into a physical proposal (joints + EE).
pub trait PhysicalProjector: Send + Sync {
    fn project(&self, ctx: &ProjectionContext, action: &ActionVector) -> Result<DynProposal>;
}
