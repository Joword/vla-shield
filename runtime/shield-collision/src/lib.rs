pub mod broad_phase;

use shield_core::arbiter::CollisionReport;
use shield_core::scene::SceneGraph;
use shield_core::types::{Aabb, JointLimits};
use shield_physics::DynProposal;
use shield_urdf::UrdfKinematicChain;

/// Context for a collision precheck query.
pub struct CollisionContext<'a> {
    pub scene: &'a SceneGraph,
    pub limits: &'a JointLimits,
    pub epsilon: f64,
    /// When set, link AABBs are produced from URDF forward kinematics
    /// instead of the joint-as-X placeholder.
    pub urdf_chain: Option<&'a UrdfKinematicChain>,
    /// Link AABBs already computed by the caller (uninflated, world frame).
    /// Lets the hot path run FK once and time it separately.
    pub link_aabbs: Option<&'a [(String, Aabb)]>,
}

impl<'a> CollisionContext<'a> {
    pub fn new(scene: &'a SceneGraph, limits: &'a JointLimits, epsilon: f64) -> Self {
        Self {
            scene,
            limits,
            epsilon,
            urdf_chain: None,
            link_aabbs: None,
        }
    }

    pub fn with_urdf(mut self, chain: &'a UrdfKinematicChain) -> Self {
        self.urdf_chain = Some(chain);
        self
    }

    pub fn with_link_aabbs(mut self, boxes: &'a [(String, Aabb)]) -> Self {
        self.link_aabbs = Some(boxes);
        self
    }
}

/// Trait for collision precheck implementations.
pub trait CollisionPrechecker: Send + Sync {
    fn precheck(&self, ctx: &CollisionContext, proposal: &DynProposal) -> CollisionReport;
}
