//! In-process ROS overlay: subscribe action / joint_states, publish decision.
//!
//! No `rclrs` required. Same topic names and message shapes as
//! `vla_shield_msgs`. When `--features ros2` is built inside a ROS workspace,
//! bind these names to real publishers/subscribers.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

use shield_core::arbiter::SemanticRiskReport;
use shield_core::ontology::physical;
use shield_core::scene::SceneGraph;
use shield_core::types::JointLimits;

use crate::adapter::{ActionProposal, SafetyDecision};
use crate::node::ShieldLifecycleHooks;
use crate::qos::{
    ACTION_TOPIC, DECISION_TOPIC, JOINT_STATES_TOPIC, QosProfile, TELEMETRY_TOPIC,
};
use crate::tf2::Tf2Validator;

/// Monitor-facing snapshot. Same fields as `vla_shield_msgs/RiskTelemetry`.
#[derive(Debug, Clone)]
pub struct RiskTelemetry {
    pub robot_id: String,
    pub sequence_id: u64,
    pub risk_score: f32,
    pub decision: u8,
    pub ontology_ids: Vec<String>,
    pub scene_revision: u64,
    pub latency_total_ms: f32,
}

/// Channel graph around [`ShieldLifecycleHooks`].
pub struct ShieldRosOverlay {
    hooks: ShieldLifecycleHooks,
    limits: JointLimits,
    scene: SceneGraph,
    semantic: SemanticRiskReport,
    last_joints: Vec<f64>,
    robot_id: String,
    tf2: Option<Tf2Validator>,
    actions: VecDeque<ActionProposal>,
    decisions: VecDeque<SafetyDecision>,
    telemetry: VecDeque<RiskTelemetry>,
}

impl ShieldRosOverlay {
    pub fn new(limits: JointLimits) -> Self {
        let n = limits.names.len();
        Self {
            hooks: ShieldLifecycleHooks::new(),
            last_joints: vec![0.0; n],
            limits,
            scene: SceneGraph::default(),
            semantic: SemanticRiskReport::default(),
            robot_id: "default-robot".into(),
            tf2: None,
            actions: VecDeque::new(),
            decisions: VecDeque::new(),
            telemetry: VecDeque::new(),
        }
    }

    pub fn with_robot_id(mut self, id: impl Into<String>) -> Self {
        self.robot_id = id.into();
        self
    }

    pub fn with_tf2(mut self, tf2: Tf2Validator) -> Self {
        self.tf2 = Some(tf2);
        self
    }

    pub fn topic_qos() -> [(&'static str, QosProfile); 4] {
        [
            (ACTION_TOPIC, QosProfile::action()),
            (JOINT_STATES_TOPIC, QosProfile::joint_state()),
            (DECISION_TOPIC, QosProfile::decision()),
            (TELEMETRY_TOPIC, QosProfile::telemetry()),
        ]
    }

    pub fn configure(
        &mut self,
        urdf_path: PathBuf,
        root_link: Option<String>,
        ee_link: Option<String>,
    ) -> Result<(), String> {
        self.hooks.on_configure(urdf_path, root_link, ee_link)
    }

    pub fn activate(&mut self) {
        self.hooks.on_activate();
    }

    pub fn deactivate(&mut self) {
        self.hooks.on_deactivate();
    }

    pub fn is_armed(&self) -> bool {
        self.hooks.is_armed()
    }

    pub fn set_scene(&mut self, scene: SceneGraph) {
        self.scene = scene;
    }

    pub fn set_semantic(&mut self, semantic: SemanticRiskReport) {
        self.semantic = semantic;
    }

    /// `/joint_states` — latched until the next action tick.
    pub fn publish_joint_state(&mut self, q: Vec<f64>) {
        if !q.is_empty() {
            self.last_joints = q;
        }
    }

    /// `/vla_shield/action`. Empty `current_joints` uses the last joint_state.
    pub fn publish_action(&mut self, mut proposal: ActionProposal) {
        if proposal.current_joints.is_empty() {
            proposal.current_joints = self.last_joints.clone();
        }
        self.actions.push_back(proposal);
        let depth = QosProfile::action().depth.max(1) as usize;
        while self.actions.len() > depth {
            self.actions.pop_front();
        }
    }

    /// One inbox action → evaluate → decision + telemetry outboxes.
    pub fn spin_once(&mut self) -> Option<SafetyDecision> {
        let proposal = self.actions.pop_front()?;
        let t0 = Instant::now();
        let mut decision = self.hooks.evaluate(
            &proposal,
            &self.limits,
            &self.scene,
            &self.semantic,
        );
        self.apply_tf2(&proposal, &mut decision);
        let latency_ms = t0.elapsed().as_secs_f64() as f32 * 1000.0;
        self.push_decision(decision.clone());
        self.push_telemetry(RiskTelemetry {
            robot_id: self.robot_id.clone(),
            sequence_id: decision.sequence_id,
            risk_score: decision.risk_score,
            decision: decision.decision,
            ontology_ids: decision.ontology_ids.clone(),
            scene_revision: self.scene.revision,
            latency_total_ms: latency_ms,
        });
        Some(decision)
    }

    pub fn take_decision(&mut self) -> Option<SafetyDecision> {
        self.decisions.pop_front()
    }

    pub fn take_telemetry(&mut self) -> Option<RiskTelemetry> {
        self.telemetry.pop_front()
    }

    fn apply_tf2(&self, proposal: &ActionProposal, decision: &mut SafetyDecision) {
        let Some(tf2) = &self.tf2 else {
            return;
        };
        let Some(pipe) = self.hooks.pipeline() else {
            return;
        };
        let Some(chain) = pipe.urdf_chain() else {
            return;
        };
        let Ok(ee) = chain.ee_position(&proposal.current_joints) else {
            return;
        };
        if !tf2.violates_forbidden_world(&ee) {
            return;
        }
        let oid = physical::forbidden_zone().to_string();
        if !decision.ontology_ids.iter().any(|id| id == &oid) {
            decision.ontology_ids.push(oid);
        }
        decision.decision = SafetyDecision::BLOCK;
        decision.risk_score = decision.risk_score.max(1.0);
        decision.safe_action = vec![0.0; proposal.action.len()];
    }

    fn push_decision(&mut self, d: SafetyDecision) {
        let depth = QosProfile::decision().depth.max(1) as usize;
        self.decisions.push_back(d);
        while self.decisions.len() > depth {
            self.decisions.pop_front();
        }
    }

    fn push_telemetry(&mut self, t: RiskTelemetry) {
        let depth = QosProfile::telemetry().depth.max(1) as usize;
        self.telemetry.push_back(t);
        while self.telemetry.len() > depth {
            self.telemetry.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qos::{Durability, Reliability};
    use nalgebra::Isometry3;
    use shield_urdf::AxisAlignedBox;

    fn limits(n: usize) -> JointLimits {
        JointLimits {
            names: (0..n).map(|i| format!("j{i}")).collect(),
            position_min: vec![-6.3; n],
            position_max: vec![6.3; n],
            velocity_max: vec![3.0; n],
            acceleration_max: vec![10.0; n],
            torque_max: vec![50.0; n],
        }
    }

    fn ur5() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../dataset/urdf/ur5_simple.urdf")
    }

    #[test]
    fn qos_matches_overlay_contract() {
        let map = ShieldRosOverlay::topic_qos();
        let binds = crate::overlay_bindings();
        assert_eq!(map[0].0, ACTION_TOPIC);
        assert_eq!(map[0].1.reliability, Reliability::Reliable);
        assert_eq!(map[1].0, JOINT_STATES_TOPIC);
        assert_eq!(map[1].1.reliability, Reliability::BestEffort);
        assert_eq!(map[2].1.durability, Durability::TransientLocal);
        assert_eq!(map[2].1.depth, 1);
        assert_eq!(binds[0].topic, map[0].0);
        assert_eq!(binds[2].topic, map[2].0);
    }

    #[test]
    fn inactive_spin_blocks() {
        let mut bus = ShieldRosOverlay::new(limits(6));
        bus.publish_joint_state(vec![0.0; 6]);
        bus.publish_action(ActionProposal {
            sequence_id: 1,
            t_ns: 0,
            action: vec![0.1; 6],
            current_joints: vec![],
            prev_velocity: vec![],
        });
        let out = bus.spin_once().expect("decision");
        assert_eq!(out.decision, SafetyDecision::BLOCK);
        assert!(out.ontology_ids.iter().any(|id| id == "SYS.NOT_ACTIVE"));
        let tel = bus.take_telemetry().expect("telemetry");
        assert_eq!(tel.sequence_id, 1);
        assert_eq!(tel.decision, SafetyDecision::BLOCK);
    }

    #[test]
    fn configure_activate_roundtrip_pass() {
        let mut bus = ShieldRosOverlay::new(limits(6));
        bus.configure(ur5(), Some("base_link".into()), Some("wrist_3_link".into()))
            .expect("urdf");
        bus.activate();
        assert!(bus.is_armed());
        bus.publish_joint_state(vec![0.0; 6]);
        bus.publish_action(ActionProposal {
            sequence_id: 9,
            t_ns: 0,
            action: vec![0.05; 6],
            current_joints: vec![],
            prev_velocity: vec![],
        });
        let out = bus.spin_once().expect("decision");
        assert_eq!(out.label(), "PASS");
        assert_eq!(out.sequence_id, 9);
        bus.deactivate();
        bus.publish_action(ActionProposal {
            sequence_id: 10,
            t_ns: 0,
            action: vec![0.05; 6],
            current_joints: vec![0.0; 6],
            prev_velocity: vec![],
        });
        let held = bus.spin_once().expect("held");
        assert_eq!(held.decision, SafetyDecision::BLOCK);
        assert!(held.ontology_ids.iter().any(|id| id == "SYS.NOT_ACTIVE"));
    }

    #[test]
    fn keep_last_drops_stale_actions() {
        let mut bus = ShieldRosOverlay::new(limits(6));
        bus.publish_action(ActionProposal {
            sequence_id: 1,
            t_ns: 0,
            action: vec![0.1; 6],
            current_joints: vec![0.0; 6],
            prev_velocity: vec![],
        });
        bus.publish_action(ActionProposal {
            sequence_id: 2,
            t_ns: 0,
            action: vec![0.2; 6],
            current_joints: vec![0.0; 6],
            prev_velocity: vec![],
        });
        let out = bus.spin_once().expect("latest");
        assert_eq!(out.sequence_id, 2);
        assert!(bus.spin_once().is_none());
    }

    #[test]
    fn tf2_world_box_blocks() {
        let mut bus = ShieldRosOverlay::new(limits(6)).with_tf2(Tf2Validator::new(
            Isometry3::identity(),
            vec![AxisAlignedBox {
                min: [-2.0, -2.0, -2.0],
                max: [2.0, 2.0, 2.0],
            }],
        ));
        bus.configure(ur5(), Some("base_link".into()), Some("wrist_3_link".into()))
            .expect("urdf");
        bus.activate();
        bus.publish_action(ActionProposal {
            sequence_id: 3,
            t_ns: 0,
            action: vec![0.0; 6],
            current_joints: vec![0.0; 6],
            prev_velocity: vec![],
        });
        let out = bus.spin_once().expect("tf2");
        assert_eq!(out.decision, SafetyDecision::BLOCK);
        assert!(out.ontology_ids.iter().any(|id| id == "PHY.FORBIDDEN_ZONE"));
    }
}
