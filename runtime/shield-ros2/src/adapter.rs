//! Channel-friendly action / decision types. Same shape as `vla_shield_msgs`
//! without needing a ROS overlay.

use shield_core::action::ActionVector;
use shield_core::arbiter::{ArbiterDecision, ArbiterReason, SemanticRiskReport};
use shield_core::ontology::physical;
use shield_core::scene::{Primitive, SceneEntity, SceneGraph};
use shield_core::types::{Aabb, JointLimits};

use crate::pipeline::SafetyPipeline;

/// Incoming VLA command. Same shape as ROS `ActionProposal`.
#[derive(Debug, Clone)]
pub struct ActionProposal {
    pub sequence_id: u64,
    pub t_ns: u64,
    pub action: Vec<f32>,
    pub current_joints: Vec<f64>,
}

/// Outgoing verdict. Same shape as ROS `SafetyDecision`.
#[derive(Debug, Clone)]
pub struct SafetyDecision {
    pub sequence_id: u64,
    /// 0=PASS 1=BLOCK 2=CLAMP 3=WARN — don't max() these, WARN would win.
    pub decision: u8,
    pub ontology_ids: Vec<String>,
    pub risk_score: f32,
    pub safe_action: Vec<f32>,
}

impl SafetyDecision {
    pub const PASS: u8 = 0;
    pub const BLOCK: u8 = 1;
    pub const CLAMP: u8 = 2;
    pub const WARN: u8 = 3;

    pub fn label(&self) -> &'static str {
        match self.decision {
            Self::BLOCK => "BLOCK",
            Self::CLAMP => "CLAMP",
            Self::WARN => "WARN",
            _ => "PASS",
        }
    }
}

/// Named AABBs → scene graph, base frame.
pub fn scene_from_aabbs(obstacles: &[(String, Aabb)]) -> SceneGraph {
    SceneGraph {
        frame_id: "base_link".into(),
        revision: 1,
        entities: obstacles
            .iter()
            .map(|(id, aabb)| {
                let c = aabb.center();
                let h = aabb.half_extents();
                SceneEntity {
                    id: id.clone(),
                    primitive: Primitive::Box {
                        extents: [h.x * 2.0, h.y * 2.0, h.z * 2.0],
                    },
                    pose: [c.x, c.y, c.z, 0.0, 0.0, 0.0, 1.0],
                    aabb: *aabb,
                    tags: vec![],
                }
            })
            .collect(),
    }
}

pub fn classify_reasons(reasons: &[ArbiterReason]) -> u8 {
    // Numeric ROS constants are PASS=0 BLOCK=1 CLAMP=2 WARN=3, so we cannot
    // take max() — WARN would outrank BLOCK. Same precedence as RuleRegistry:
    // block > clamp > warn > pass.
    let mut block = false;
    let mut clamp = false;
    let mut warn = false;
    for r in reasons {
        let id = r.ontology_id.0.as_str();
        if id == physical::collision().0
            || id == physical::joint_limit().0
            || id == physical::forbidden_zone().0
            || id == physical::singularity().0
            || id == physical::tipover().0
            || id == physical::overload().0
        {
            block = true;
        } else if id == physical::velocity_limit().0 {
            clamp = true;
        } else {
            warn = true;
        }
    }
    if block {
        SafetyDecision::BLOCK
    } else if clamp {
        SafetyDecision::CLAMP
    } else if warn {
        SafetyDecision::WARN
    } else {
        SafetyDecision::PASS
    }
}

impl SafetyPipeline {
    /// One proposal vs `limits` + `scene`. Semantic report may be stale — that's ok.
    pub fn evaluate_proposal(
        &self,
        proposal: &ActionProposal,
        limits: &JointLimits,
        scene: &SceneGraph,
        semantic: &SemanticRiskReport,
    ) -> SafetyDecision {
        let action = ActionVector::new(
            proposal.t_ns,
            proposal.sequence_id,
            proposal.action.clone(),
        );
        let event = self.evaluate(
            &action,
            &proposal.current_joints,
            limits,
            scene,
            semantic,
            None,
        );
        match event.decision {
            ArbiterDecision::Pass { action, .. } => SafetyDecision {
                sequence_id: proposal.sequence_id,
                decision: SafetyDecision::PASS,
                ontology_ids: vec![],
                risk_score: 0.0,
                safe_action: action.data,
            },
            ArbiterDecision::Block {
                safe_fallback,
                reasons,
                ..
            } => {
                let decision = classify_reasons(&reasons);
                let risk = reasons.iter().map(|r| r.score).fold(0.0_f32, f32::max);
                let ids = reasons
                    .iter()
                    .map(|r| r.ontology_id.to_string())
                    .collect();
                let safe_action = if decision == SafetyDecision::CLAMP {
                    self.clamped_action(&action, &proposal.current_joints, limits, scene)
                        .unwrap_or(safe_fallback.data)
                } else {
                    safe_fallback.data
                };
                SafetyDecision {
                    sequence_id: proposal.sequence_id,
                    decision,
                    ontology_ids: ids,
                    risk_score: risk,
                    safe_action,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeConfig;
    use shield_core::types::JointLimits;

    fn limits(n: usize) -> JointLimits {
        JointLimits {
            names: (0..n).map(|i| format!("j{i}")).collect(),
            position_min: vec![-3.14; n],
            position_max: vec![3.14; n],
            velocity_max: vec![1.0; n],
            acceleration_max: vec![10.0; n],
            torque_max: vec![50.0; n],
        }
    }

    #[test]
    fn pass_on_small_command() {
        let pipe = SafetyPipeline::with_defaults(RuntimeConfig::default());
        let lim = limits(3);
        let scene = SceneGraph::default();
        let sem = SemanticRiskReport {
            sequence_id: 0,
            risk_score: 0.0,
            triggered: vec![],
            stale: true,
        };
        let out = pipe.evaluate_proposal(
            &ActionProposal {
                sequence_id: 1,
                t_ns: 0,
                action: vec![0.1, 0.0, -0.05],
                current_joints: vec![0.0, 0.0, 0.0],
            },
            &lim,
            &scene,
            &sem,
        );
        assert_eq!(out.decision, SafetyDecision::PASS);
        assert_eq!(out.label(), "PASS");
    }

    #[test]
    fn clamped_action_respects_position_limit() {
        let config = RuntimeConfig::default();
        let dt = config.dt;
        let pipe = SafetyPipeline::with_defaults(config);
        let mut lim = limits(1);
        lim.position_max = vec![0.001];
        lim.velocity_max = vec![1.0];
        let action = ActionVector::new(0, 1, vec![10.0]);
        let out = pipe
            .clamped_action(&action, &[0.0], &lim, &SceneGraph::default())
            .expect("projector accepts the state");
        // A plain velocity_max clamp would emit 1.0 rad/s and overshoot the stop.
        assert!(
            out[0] as f64 * dt <= 0.001 + 1e-9,
            "clamped action {out:?} drives past position_max"
        );
    }

    #[test]
    fn collision_blocks_when_ee_in_box() {
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
        let robot = shield_urdf::UrdfRobot::from_str(xml).unwrap();
        let chain =
            shield_urdf::UrdfKinematicChain::from_robot(&robot, "base", "ee").unwrap();
        let pipe = SafetyPipeline::with_defaults(RuntimeConfig::default()).with_urdf(chain);
        let lim = limits(1);
        let scene = scene_from_aabbs(&[(
            "shelf".into(),
            Aabb::new([-0.5, -0.2, -1.0], [0.5, 0.2, 1.0]),
        )]);
        let sem = SemanticRiskReport {
            sequence_id: 0,
            risk_score: 0.0,
            triggered: vec![],
            stale: true,
        };
        let out = pipe.evaluate_proposal(
            &ActionProposal {
                sequence_id: 2,
                t_ns: 0,
                action: vec![0.0],
                current_joints: vec![0.0],
            },
            &lim,
            &scene,
            &sem,
        );
        assert_eq!(out.decision, SafetyDecision::BLOCK);
        assert!(out.ontology_ids.iter().any(|id| id == "PHY.COLLISION"));
    }

    #[test]
    fn singularity_blocks_elbow_lock() {
        let pipe = SafetyPipeline::with_defaults(RuntimeConfig::default());
        let mut action = vec![0.0; 7];
        action[3] = 0.5;
        let out = pipe.evaluate_proposal(
            &ActionProposal {
                sequence_id: 3,
                t_ns: 0,
                action,
                current_joints: vec![0.0, 0.0, 0.0, 0.02, 0.0, 0.02, 0.0],
            },
            &limits(7),
            &SceneGraph::default(),
            &SemanticRiskReport::default(),
        );
        assert_eq!(out.decision, SafetyDecision::BLOCK);
        assert!(out.ontology_ids.iter().any(|id| id == "PHY.SINGULARITY"));
    }

    #[test]
    fn tipover_blocks_base_accel() {
        let pipe = SafetyPipeline::with_defaults(RuntimeConfig::default());
        let mut action = vec![0.0; 8];
        action[7] = 2.5;
        let out = pipe.evaluate_proposal(
            &ActionProposal {
                sequence_id: 4,
                t_ns: 0,
                action,
                current_joints: vec![0.0; 8],
            },
            &limits(8),
            &SceneGraph::default(),
            &SemanticRiskReport::default(),
        );
        assert_eq!(out.decision, SafetyDecision::BLOCK);
        assert!(out.ontology_ids.iter().any(|id| id == "PHY.TIPOVER"));
    }

    #[test]
    fn overload_blocks_wrist_spike() {
        let pipe = SafetyPipeline::with_defaults(RuntimeConfig::default());
        let mut lim = limits(7);
        lim.torque_max[4] = 20.0;
        let mut action = vec![0.0; 7];
        action[4] = 5.0;
        let out = pipe.evaluate_proposal(
            &ActionProposal {
                sequence_id: 5,
                t_ns: 0,
                action,
                current_joints: vec![0.0, 0.3, 0.0, -1.0, 0.0, 1.5, 0.8],
            },
            &lim,
            &SceneGraph::default(),
            &SemanticRiskReport::default(),
        );
        assert_eq!(out.decision, SafetyDecision::BLOCK);
        assert!(out.ontology_ids.iter().any(|id| id == "PHY.OVERLOAD"));
    }
}
