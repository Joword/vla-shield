pub mod broad_phase;

use shield_core::arbiter::CollisionReport;
use shield_core::scene::SceneGraph;
use shield_core::types::JointLimits;
use shield_physics::DynProposal;

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
