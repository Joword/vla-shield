use crate::{CollisionContext, CollisionPrechecker};
use vlashield_core::arbiter::{CollisionPair, CollisionReport};
use vlashield_core::types::Aabb;
use vlashield_physics::DynProposal;

/// AABB-based broad-phase collision prechecker.
///
/// Generates a conservative bounding box per robot link from the proposed joint
/// state, inflates it by `epsilon`, and tests against all scene entities.
pub struct AabbBroadPhase;

impl AabbBroadPhase {
    /// Placeholder: generate per-link AABBs from joint positions.
    /// Real implementation would use robot URDF / DH chain.
    fn link_aabbs(proposal: &DynProposal, epsilon: f64) -> Vec<(String, Aabb)> {
        let n = proposal.joint_positions.len();
        let mut aabbs = Vec::with_capacity(n);
        for i in 0..n {
            let p = proposal.joint_positions[i];
            let base = Aabb::new(
                [p - 0.05, -0.05, -0.05],
                [p + 0.05, 0.05, 0.05],
            );
            aabbs.push((format!("link_{i}"), base.inflated(epsilon)));
        }
        aabbs
    }
}

impl CollisionPrechecker for AabbBroadPhase {
    fn precheck(&self, ctx: &CollisionContext, proposal: &DynProposal) -> CollisionReport {
        let link_aabbs = Self::link_aabbs(proposal, ctx.epsilon);
        let mut pairs = Vec::new();

        for (link_name, link_aabb) in &link_aabbs {
            for entity in &ctx.scene.entities {
                if link_aabb.intersects(&entity.aabb) {
                    let dist = {
                        let lc = link_aabb.center();
                        let ec = entity.aabb.center();
                        (lc - ec).norm()
                    };
                    pairs.push(CollisionPair {
                        link: link_name.clone(),
                        obstacle: entity.id.clone(),
                        min_distance: dist,
                    });
                }
            }
        }

        CollisionReport {
            hit: !pairs.is_empty(),
            pairs,
            energy_lower_bound: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vlashield_core::scene::{Primitive, SceneEntity, SceneGraph};
    use vlashield_core::types::JointLimits;
    use vlashield_physics::DynProposal;

    fn test_scene() -> SceneGraph {
        SceneGraph {
            frame_id: "base_link".into(),
            revision: 1,
            entities: vec![SceneEntity {
                id: "shelf".into(),
                primitive: Primitive::Box { extents: [1.0, 0.4, 2.0] },
                pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                aabb: Aabb::new([-0.5, -0.2, -1.0], [0.5, 0.2, 1.0]),
                tags: vec![],
            }],
        }
    }

    #[test]
    fn detects_collision() {
        let checker = AabbBroadPhase;
        let scene = test_scene();
        let limits = JointLimits {
            names: vec!["j0".into()],
            position_min: vec![-3.14],
            position_max: vec![3.14],
            velocity_max: vec![1.0],
            acceleration_max: vec![5.0],
            torque_max: vec![50.0],
        };
        let ctx = CollisionContext {
            scene: &scene,
            limits: &limits,
            epsilon: 0.02,
        };
        let proposal = DynProposal {
            joint_positions: vec![0.0],
            joint_velocities: vec![0.1],
            ee_position: [0.0; 3],
            ee_orientation: [0.0, 0.0, 0.0, 1.0],
        };
        let report = checker.precheck(&ctx, &proposal);
        assert!(report.hit);
    }

    #[test]
    fn no_collision_when_far() {
        let checker = AabbBroadPhase;
        let scene = test_scene();
        let limits = JointLimits {
            names: vec!["j0".into()],
            position_min: vec![-6.0],
            position_max: vec![6.0],
            velocity_max: vec![1.0],
            acceleration_max: vec![5.0],
            torque_max: vec![50.0],
        };
        let ctx = CollisionContext {
            scene: &scene,
            limits: &limits,
            epsilon: 0.01,
        };
        let proposal = DynProposal {
            joint_positions: vec![5.0],
            joint_velocities: vec![0.0],
            ee_position: [0.0; 3],
            ee_orientation: [0.0, 0.0, 0.0, 1.0],
        };
        let report = checker.precheck(&ctx, &proposal);
        assert!(!report.hit);
    }
}
