//! ROS 2 lifecycle hooks for the shield runtime.
//!
//! Full `rclrs` graph wiring (subscriptions to `ActionProposal`, publishers for
//! `SafetyDecision`) belongs in a ROS 2 overlay. These hooks load the URDF
//! pipeline on `configure` and only evaluate while `active`.

use std::path::PathBuf;

use tracing::info;

use crate::config::RuntimeConfig;
use crate::pipeline::SafetyPipeline;
use crate::{ActionProposal, SafetyDecision};
use shield_core::arbiter::SemanticRiskReport;
use shield_core::scene::SceneGraph;
use shield_core::types::JointLimits;

/// Ontology id reported while the node is not active. Distinct from
/// `PHY.COLLISION`: nothing is colliding, the node simply refuses to forward.
pub const INACTIVE_ONTOLOGY_ID: &str = "SYS.NOT_ACTIVE";

/// Stateful callbacks for a shield ROS 2 node (maps to lifecycle transitions).
pub struct ShieldLifecycleHooks {
    pub urdf_path: Option<PathBuf>,
    pub root_link: String,
    pub ee_link: String,
    pipeline: Option<SafetyPipeline>,
    armed: bool,
}

impl Default for ShieldLifecycleHooks {
    fn default() -> Self {
        Self {
            urdf_path: None,
            root_link: "base_link".into(),
            ee_link: "wrist_3_link".into(),
            pipeline: None,
            armed: false,
        }
    }
}

impl ShieldLifecycleHooks {
    pub fn new() -> Self {
        Self::default()
    }

    /// `on_configure`: load URDF, allocate the safety pipeline.
    pub fn on_configure(
        &mut self,
        urdf_path: PathBuf,
        root_link: Option<String>,
        ee_link: Option<String>,
    ) -> Result<(), String> {
        if let Some(r) = root_link {
            self.root_link = r;
        }
        if let Some(e) = ee_link {
            self.ee_link = e;
        }
        self.urdf_path = Some(urdf_path.clone());
        let pipe = SafetyPipeline::from_urdf_file(
            RuntimeConfig::default(),
            &urdf_path,
            &self.root_link,
            &self.ee_link,
        )
        .map_err(|e| e.to_string())?;
        self.pipeline = Some(pipe);
        self.armed = false;
        info!(path = %urdf_path.display(), "shield lifecycle: on_configure");
        Ok(())
    }

    /// `on_activate`: arm the safety pipeline and start processing proposals.
    pub fn on_activate(&mut self) {
        self.armed = self.pipeline.is_some();
        info!(armed = self.armed, "shield lifecycle: on_activate");
    }

    /// `on_deactivate`: stop forwarding raw VLA commands.
    pub fn on_deactivate(&mut self) {
        self.armed = false;
        info!("shield lifecycle: on_deactivate — hold last safe action");
    }

    pub fn is_armed(&self) -> bool {
        self.armed
    }

    /// Evaluate a proposal while active. Inactive nodes return BLOCK + zeros.
    pub fn evaluate(
        &self,
        proposal: &ActionProposal,
        limits: &JointLimits,
        scene: &SceneGraph,
        semantic: &SemanticRiskReport,
    ) -> SafetyDecision {
        match (&self.pipeline, self.armed) {
            (Some(pipe), true) => pipe.evaluate_proposal(proposal, limits, scene, semantic),
            _ => SafetyDecision {
                sequence_id: proposal.sequence_id,
                decision: SafetyDecision::BLOCK,
                ontology_ids: vec![INACTIVE_ONTOLOGY_ID.into()],
                risk_score: 1.0,
                safe_action: vec![0.0; proposal.action.len()],
            },
        }
    }
}

/// When `ros2` is enabled, call this from your `rclrs` context after `rclrs::init`.
#[cfg(feature = "ros2")]
pub fn log_rclrs_build_stub() {
    tracing::warn!(
        "rclrs feature is on: link against the ROS 2 workspace and register publishers/subscribers here"
    );
}

#[cfg(not(feature = "ros2"))]
pub fn log_rclrs_build_stub() {
    tracing::debug!("rclrs feature off: build with `--features ros2` inside a ROS 2 environment");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ActionProposal;
    use shield_core::types::JointLimits;

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

    #[test]
    fn inactive_holds() {
        let hooks = ShieldLifecycleHooks::new();
        let out = hooks.evaluate(
            &ActionProposal {
                sequence_id: 1,
                t_ns: 0,
                action: vec![0.1; 6],
                current_joints: vec![0.0; 6],
            },
            &limits(6),
            &SceneGraph::default(),
            &SemanticRiskReport::default(),
        );
        assert_eq!(out.decision, SafetyDecision::BLOCK);
        assert!(!hooks.is_armed());
    }

    #[test]
    fn configure_ur5_and_pass() {
        let urdf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../dataset/urdf/ur5_simple.urdf");
        let mut hooks = ShieldLifecycleHooks::new();
        hooks
            .on_configure(urdf, Some("base_link".into()), Some("wrist_3_link".into()))
            .expect("urdf");
        hooks.on_activate();
        assert!(hooks.is_armed());
        let out = hooks.evaluate(
            &ActionProposal {
                sequence_id: 2,
                t_ns: 0,
                action: vec![0.05; 6],
                current_joints: vec![0.0; 6],
            },
            &limits(6),
            &SceneGraph::default(),
            &SemanticRiskReport::default(),
        );
        assert_eq!(out.label(), "PASS");
        hooks.on_deactivate();
        assert!(!hooks.is_armed());
    }
}
