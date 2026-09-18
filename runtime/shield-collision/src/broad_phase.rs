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
            return boxes.to_vec();
        }
        if let Some(chain) = ctx.urdf_chain {
            if let Ok(boxes) = chain.link_world_aabbs(&proposal.joint_positions) {
                return boxes;
            }
        }
        let ee = proposal.ee_position;
        let is_origin = ee[0].abs() < 1e-12 && ee[1].abs() < 1e-12 && ee[2].abs() < 1e-12;
        if is_origin {
            return Vec::new();
        }
        let r = 0.06;
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
        let raw_boxes = Self::link_aabbs(ctx, proposal);
        if raw_boxes.is_empty() {
            return CollisionReport {
                hit: false,
                pairs: Vec::new(),
                energy_lower_bound: 0.0,
            };
        }
        let link_aabbs: Vec<(String, Aabb)> = raw_boxes
            .iter()
            .map(|(n, a)| (n.clone(), a.inflated(ctx.epsilon)))
            .collect();
        let entities = &ctx.scene.entities;
        let mut pairs = Vec::new();
        if !entities.is_empty() {
            let mut link_min = Vec::with_capacity(link_aabbs.len());
            let mut link_max = Vec::with_capacity(link_aabbs.len());
            for (_, aabb) in &link_aabbs {
                link_min.push([aabb.min[0] as f32, aabb.min[1] as f32, aabb.min[2] as f32]);
                link_max.push([aabb.max[0] as f32, aabb.max[1] as f32, aabb.max[2] as f32]);
            }
            let mut obs_min = Vec::with_capacity(entities.len());
            let mut obs_max = Vec::with_capacity(entities.len());
            for entity in entities {
                obs_min.push([
                    entity.aabb.min[0] as f32,
                    entity.aabb.min[1] as f32,
                    entity.aabb.min[2] as f32,
                ]);
                obs_max.push([
                    entity.aabb.max[0] as f32,
                    entity.aabb.max[1] as f32,
                    entity.aabb.max[2] as f32,
                ]);
            }

            let n_obs = entities.len();
            let mask = shield_cuda::aabb_overlap_mask(&link_min, &link_max, &obs_min, &obs_max)
                .unwrap_or_else(|_| cpu_overlap_mask(&link_min, &link_max, &obs_min, &obs_max));

            let deflate = (ctx.epsilon * 0.25).max(0.0);
            for (i, (link_name, link_aabb)) in link_aabbs.iter().enumerate() {
                for (j, entity) in entities.iter().enumerate() {
                    if mask.get(i * n_obs + j).copied().unwrap_or(0) == 0 {
                        continue;
                    }
                    if !crate::narrow_phase::confirm_aabb_hit(link_aabb, &entity.aabb, deflate) {
                        continue;
                    }
                    let lc = link_aabb.center();
                    let ec = entity.aabb.center();
                    pairs.push(CollisionPair {
                        link: link_name.clone(),
                        obstacle: entity.id.clone(),
                        min_distance: (lc - ec).norm(),
                    });
                }
            }
        }

        pairs.extend(crate::narrow_phase::self_collision_pairs(&raw_boxes));

        CollisionReport {
            hit: !pairs.is_empty(),
            pairs,
            energy_lower_bound: 0.0,
        }
    }
}

fn cpu_overlap_mask(
    link_min: &[[f32; 3]],
    link_max: &[[f32; 3]],
    obs_min: &[[f32; 3]],
    obs_max: &[[f32; 3]],
) -> Vec<u8> {
    let n = link_min.len();
    let m = obs_min.len();
    let mut mask = vec![0u8; n * m];
    for i in 0..n {
        for j in 0..m {
            let overlap = link_min[i][0] <= obs_max[j][0]
                && link_max[i][0] >= obs_min[j][0]
                && link_min[i][1] <= obs_max[j][1]
                && link_max[i][1] >= obs_min[j][1]
                && link_min[i][2] <= obs_max[j][2]
                && link_max[i][2] >= obs_min[j][2];
            mask[i * m + j] = u8::from(overlap);
        }
    }
    mask
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

    #[test]
    fn overlap_mask_keeps_pair_identities() {
        let checker = AabbBroadPhase;
        let scene = SceneGraph {
            frame_id: "base_link".into(),
            revision: 1,
            entities: vec![
                SceneEntity {
                    id: "shelf".into(),
                    primitive: Primitive::Box {
                        extents: [1.0, 0.4, 2.0],
                    },
                    pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    aabb: Aabb::new([-0.5, -0.2, -1.0], [0.5, 0.2, 1.0]),
                    tags: vec![],
                },
                SceneEntity {
                    id: "bin".into(),
                    primitive: Primitive::Box {
                        extents: [0.4, 0.4, 0.4],
                    },
                    pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    aabb: Aabb::new([-0.2, -0.2, 0.0], [0.2, 0.2, 0.4]),
                    tags: vec![],
                },
            ],
        };
        let lim = limits(1);
        let ctx = CollisionContext::new(&scene, &lim, 0.02);
        let proposal = DynProposal {
            joint_positions: vec![0.0],
            joint_velocities: vec![0.0],
            ee_position: [0.0, 0.0, 0.1],
            ee_orientation: [0.0, 0.0, 0.0, 1.0],
        };
        let report = checker.precheck(&ctx, &proposal);
        assert!(report.hit);
        let mut ids: Vec<_> = report
            .pairs
            .iter()
            .map(|p| (p.link.as_str(), p.obstacle.as_str()))
            .collect();
        ids.sort();
        assert_eq!(ids, vec![("ee", "bin"), ("ee", "shelf")]);
    }

    #[test]
    fn ur5_nominal_poses_are_not_self_collision() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../dataset/urdf/ur5_simple.urdf");
        let robot = UrdfRobot::from_file(&path).expect("ur5");
        let chain =
            UrdfKinematicChain::from_robot(&robot, "base_link", "wrist_3_link").expect("chain");
        let checker = AabbBroadPhase;
        let scene = SceneGraph::default();
        let lim = limits(6);
        let ctx = CollisionContext::new(&scene, &lim, 0.02).with_urdf(&chain);
        for q in [
            vec![0.0; 6],
            vec![0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        ] {
            let proposal = DynProposal {
                joint_positions: q.clone(),
                joint_velocities: vec![0.0; 6],
                ee_position: chain.ee_position(&q).unwrap(),
                ee_orientation: [0.0, 0.0, 0.0, 1.0],
            };
            let report = checker.precheck(&ctx, &proposal);
            assert!(
                !report.hit,
                "q={q:?} pairs={:?}",
                report
                    .pairs
                    .iter()
                    .map(|p| (p.link.as_str(), p.obstacle.as_str()))
                    .collect::<Vec<_>>()
            );
        }

        let panda_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../dataset/urdf/panda_arm_simple.urdf");
        let panda = UrdfRobot::from_file(&panda_path).expect("panda");
        let pchain =
            UrdfKinematicChain::from_robot(&panda, "panda_link0", "panda_hand").expect("chain");
        let plim = limits(7);
        let pctx = CollisionContext::new(&scene, &plim, 0.02).with_urdf(&pchain);
        let q = vec![0.0, 0.3, 0.0, -1.5, 0.0, 1.5, 0.0];
        let proposal = DynProposal {
            joint_positions: q.clone(),
            joint_velocities: vec![0.0; 7],
            ee_position: pchain.ee_position(&q).unwrap(),
            ee_orientation: [0.0, 0.0, 0.0, 1.0],
        };
        let report = checker.precheck(&pctx, &proposal);
        assert!(
            !report.hit,
            "panda folded pairs={:?}",
            report
                .pairs
                .iter()
                .map(|p| (p.link.as_str(), p.obstacle.as_str()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn empty_scene_still_reports_self_collision() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mid = Aabb::new([10.0, 10.0, 10.0], [11.0, 11.0, 11.0]);
        let c = Aabb::new([0.2, 0.2, 0.2], [0.8, 0.8, 0.8]);
        let boxes = vec![
            ("l0".into(), a),
            ("l1".into(), mid),
            ("l2".into(), mid),
            ("l3".into(), c),
        ];
        let checker = AabbBroadPhase;
        let scene = SceneGraph::default();
        let lim = limits(1);
        let ctx = CollisionContext::new(&scene, &lim, 0.01).with_link_aabbs(&boxes);
        let proposal = DynProposal {
            joint_positions: vec![0.0],
            joint_velocities: vec![0.0],
            ee_position: [0.0, 0.0, 0.0],
            ee_orientation: [0.0, 0.0, 0.0, 1.0],
        };
        let report = checker.precheck(&ctx, &proposal);
        assert!(report.hit);
        assert!(report
            .pairs
            .iter()
            .any(|p| p.link == "l0" && p.obstacle == "l3"));
    }
}
