use crate::{CollisionContext, CollisionPrechecker};
use shield_core::arbiter::{CollisionPair, CollisionReport};
use shield_core::types::Aabb;
use shield_physics::DynProposal;

/// AABB overlap sweep.
///
/// Link volumes come from URDF FK when `urdf_chain` is set. No chain? We drop
/// one conservative box on the projected EE — skipped if EE is still the
/// origin placeholder (that used to false-hit *and* false-miss).
pub struct AabbBroadPhase;

impl AabbBroadPhase {
    fn link_aabbs(ctx: &CollisionContext, proposal: &DynProposal) -> Vec<(String, Aabb)> {
        if let Some(boxes) = ctx.link_aabbs {
            return boxes
                .iter()
                .map(|(name, aabb)| (name.clone(), aabb.inflated(ctx.epsilon)))
                .collect();
        }
        if let Some(chain) = ctx.urdf_chain {
            if let Ok(boxes) = chain.link_world_aabbs(&proposal.joint_positions) {
                return boxes
                    .into_iter()
                    .map(|(name, aabb)| (name, aabb.inflated(ctx.epsilon)))
                    .collect();
            }
        }
        let ee = proposal.ee_position;
        let is_origin = ee[0].abs() < 1e-12 && ee[1].abs() < 1e-12 && ee[2].abs() < 1e-12;
        if is_origin {
            return Vec::new();
        }
        let r = 0.06 + ctx.epsilon;
        vec![(
            "ee".into(),
            Aabb::new(
                [ee[0] - r, ee[1] - r, ee[2] - r],
                [ee[0] + r, ee[1] + r, ee[2] + r],
            ),
        )]
    }
}

impl CollisionPrechecker for AabbBroadPhase {
    fn precheck(&self, ctx: &CollisionContext, proposal: &DynProposal) -> CollisionReport {
        let link_aabbs = Self::link_aabbs(ctx, proposal);
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
    use shield_core::scene::{Primitive, SceneEntity, SceneGraph};
    use shield_core::types::JointLimits;
    use shield_physics::DynProposal;
    use shield_urdf::{UrdfKinematicChain, UrdfRobot};

    fn test_scene() -> SceneGraph {
        SceneGraph {
            frame_id: "base_link".into(),
            revision: 1,
            entities: vec![SceneEntity {
                id: "shelf".into(),
                primitive: Primitive::Box {
                    extents: [1.0, 0.4, 2.0],
                },
                pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                aabb: Aabb::new([-0.5, -0.2, -1.0], [0.5, 0.2, 1.0]),
                tags: vec![],
            }],
        }
    }

    fn limits(n: usize) -> JointLimits {
        JointLimits {
            names: (0..n).map(|i| format!("j{i}")).collect(),
            position_min: vec![-3.14; n],
            position_max: vec![3.14; n],
            velocity_max: vec![1.0; n],
            acceleration_max: vec![5.0; n],
            torque_max: vec![50.0; n],
        }
    }

    #[test]
    fn urdf_fk_detects_collision_near_base() {
        let xml = r#"<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/>
  <link name="ee"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="ee"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>
"#;
        let robot = UrdfRobot::from_str(xml).unwrap();
        let chain = UrdfKinematicChain::from_robot(&robot, "base", "ee").unwrap();
        let checker = AabbBroadPhase;
        let scene = test_scene();
        let lim = limits(1);
        let ctx = CollisionContext::new(&scene, &lim, 0.02).with_urdf(&chain);
        let proposal = DynProposal {
            joint_positions: vec![0.0],
            joint_velocities: vec![0.0],
            ee_position: chain.ee_position(&[0.0]).unwrap(),
            ee_orientation: [0.0, 0.0, 0.0, 1.0],
        };
        let report = checker.precheck(&ctx, &proposal);
        assert!(report.hit, "synthesized link AABB should overlap the shelf");
    }

    #[test]
    fn no_collision_when_scene_empty() {
        let checker = AabbBroadPhase;
        let scene = SceneGraph::default();
        let lim = limits(1);
        let ctx = CollisionContext::new(&scene, &lim, 0.01);
        let proposal = DynProposal {
            joint_positions: vec![0.0],
            joint_velocities: vec![0.0],
            ee_position: [0.4, 0.0, 0.4],
            ee_orientation: [0.0, 0.0, 0.0, 1.0],
        };
        let report = checker.precheck(&ctx, &proposal);
        assert!(!report.hit);
    }

    #[test]
    fn ee_fallback_hits_obstacle() {
        let checker = AabbBroadPhase;
        let scene = test_scene();
        let lim = limits(1);
        let ctx = CollisionContext::new(&scene, &lim, 0.02);
        let proposal = DynProposal {
            joint_positions: vec![0.0],
            joint_velocities: vec![0.0],
            ee_position: [0.0, 0.0, 0.0],
            ee_orientation: [0.0, 0.0, 0.0, 1.0],
        };
        // Origin EE without URDF is ignored — the old joint-as-X trick both
        // false-hit and false-missed. Put the EE on the shelf instead.
        let proposal_hit = DynProposal {
            ee_position: [0.0, 0.0, 0.1],
            ..proposal
        };
        let report = checker.precheck(&ctx, &proposal_hit);
        assert!(report.hit);
    }
}
