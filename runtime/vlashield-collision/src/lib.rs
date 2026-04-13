pub mod broad_phase;

use vlashield_core::arbiter::CollisionReport;
use vlashield_core::scene::SceneGraph;
use vlashield_core::types::JointLimits;
use vlashield_physics::DynProposal;

/// Context for a collision precheck query.
pub struct CollisionContext<'a> {
    pub scene: &'a SceneGraph,
    pub limits: &'a JointLimits,
    /// Conservative inflation factor (meters) to account for perception delay.
    pub epsilon: f64,
}

/// Trait for collision precheck implementations.
pub trait CollisionPrechecker: Send + Sync {
    fn precheck(&self, ctx: &CollisionContext, proposal: &DynProposal) -> CollisionReport;
}
