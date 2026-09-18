use crate::config::RuntimeConfig;
use shield_collision::broad_phase::AabbBroadPhase;
use shield_collision::{CollisionContext, CollisionPrechecker};
use shield_core::action::ActionVector;
use shield_core::arbiter::{
    ArbiterDecision, ArbiterReason, CollisionReport, LatencyBreakdown, SafetyEvent,
    SemanticRiskReport,
};
use shield_core::scene::SceneGraph;
use shield_core::types::{JointLimits, RunMode};
use shield_physics::projection::KinematicClampProjector;
use shield_physics::{extra_physical_reasons, PhysicalProjector, ProjectionContext};
use shield_shadow::result::ShadowResult;
use shield_shadow::{JointSpaceSimulator, ShadowSimulator};
use shield_urdf::{AxisAlignedBox, UrdfKinematicChain, UrdfRobot};
use std::time::Instant;

/// Hot path: project → collide → arbiter. That's the loop.
pub struct SafetyPipeline {
    pub config: RuntimeConfig,
    projector: Box<dyn PhysicalProjector>,
    checker: Box<dyn CollisionPrechecker>,
    shadow_simulator: Option<Box<dyn ShadowSimulator>>,
    urdf_chain: Option<UrdfKinematicChain>,
    forbidden_zones: Vec<AxisAlignedBox>,
}

impl SafetyPipeline {
    pub fn new(
        config: RuntimeConfig,
        projector: Box<dyn PhysicalProjector>,
        checker: Box<dyn CollisionPrechecker>,
    ) -> Self {
        Self {
            config,
            projector,
            checker,
            shadow_simulator: Some(Box::new(JointSpaceSimulator::default())),
            urdf_chain: None,
            forbidden_zones: Vec::new(),
        }
    }

    /// Default projector + AABB checker. Pass a URDF chain if you've got one.
    pub fn with_defaults(config: RuntimeConfig) -> Self {
        Self::new(
            config,
            Box::new(KinematicClampProjector),
            Box::new(AabbBroadPhase),
        )
    }

    pub fn from_urdf_file(
        config: RuntimeConfig,
        path: impl AsRef<std::path::Path>,
        root_link: &str,
        ee_link: &str,
    ) -> Result<Self, shield_urdf::UrdfError> {
        let robot = UrdfRobot::from_file(path)?;
        let chain = UrdfKinematicChain::from_robot(&robot, root_link, ee_link)?;
        Ok(Self::with_defaults(config).with_urdf(chain))
    }

    pub fn with_urdf(mut self, chain: UrdfKinematicChain) -> Self {
        self.urdf_chain = Some(chain);
        self
    }

    pub fn with_forbidden_zones(mut self, zones: Vec<AxisAlignedBox>) -> Self {
        self.forbidden_zones = zones;
        self
    }

    pub fn urdf_chain(&self) -> Option<&UrdfKinematicChain> {
        self.urdf_chain.as_ref()
    }

    /// One-step clamp through the same projector the hot path uses, expressed
    /// back as a velocity command. Respects vel cap *and* position limits, so a
    /// CLAMP verdict can't emit something physics would have rejected. `None`
    /// if the projector refuses the state.
    pub fn clamped_action(
        &self,
        action: &ActionVector,
        current_joints: &[f64],
        limits: &JointLimits,
        scene: &SceneGraph,
        prev_velocity: &[f64],
    ) -> Option<Vec<f32>> {
        if self.config.dt <= 0.0 {
            return None;
        }
        let ctx = ProjectionContext {
            current_joints,
            limits,
            scene,
            dt: self.config.dt,
            urdf_chain: self.urdf_chain.as_ref(),
            forbidden_zones: &self.forbidden_zones,
            semantic_constraints: &[],
            prev_velocity,
        };
        let proposal = self.projector.project(&ctx, action).ok()?;
        Some(
            proposal
                .joint_positions
                .iter()
                .zip(current_joints)
                .map(|(pos, q)| ((pos - q) / self.config.dt) as f32)
                .collect(),
        )
    }

    /// Swap the default shadow sim. `None` turns it off.
    pub fn with_shadow_simulator(mut self, sim: Option<Box<dyn ShadowSimulator>>) -> Self {
        self.shadow_simulator = sim;
        self
    }

    /// One action through the hot path.
    ///
    /// `shadow` is last async sim result. `None` = nothing ready yet; that's fine.
    pub fn evaluate(
        &self,
        action: &ActionVector,
        current_joints: &[f64],
        limits: &JointLimits,
        scene: &SceneGraph,
        semantic: &SemanticRiskReport,
        shadow: Option<&ShadowResult>,
        prev_velocity: &[f64],
    ) -> SafetyEvent {
        let t0 = Instant::now();

        let ingest_done = Instant::now();
        let ingest_ms = ingest_done.duration_since(t0).as_secs_f64() * 1000.0;

        let chain_ref = self.urdf_chain.as_ref();
        let proj_ctx = ProjectionContext {
            current_joints,
            limits,
            scene,
            dt: self.config.dt,
            urdf_chain: chain_ref,
            forbidden_zones: &self.forbidden_zones,
            semantic_constraints: &[],
            prev_velocity,
        };

        let proposal = self.projector.project(&proj_ctx, action);
        let physics_done = Instant::now();
        let physics_ms = physics_done.duration_since(ingest_done).as_secs_f64() * 1000.0;

        // FK for collision boxes runs once here so we can report its cost
        // instead of stuffing it into `collision_ms`.
        let mut urdf_fk_ms = None;
        let link_boxes = match (&proposal, chain_ref) {
            (Ok(p), Some(chain)) => {
                let fk_start = Instant::now();
                let boxes = chain.link_world_aabbs(&p.joint_positions).ok();
                urdf_fk_ms = Some(fk_start.elapsed().as_secs_f64() * 1000.0);
                boxes
            }
            _ => None,
        };
        let fk_done = Instant::now();

        let collision_report = match &proposal {
            Ok(p) => {
                let mut ctx = CollisionContext::new(scene, limits, self.config.collision_epsilon);
                if let Some(chain) = chain_ref {
                    ctx = ctx.with_urdf(chain);
                }
                if let Some(boxes) = link_boxes.as_deref() {
                    ctx = ctx.with_link_aabbs(boxes);
                }
                self.checker.precheck(&ctx, p)
            }
            Err(_) => CollisionReport {
                hit: true,
                pairs: vec![],
                energy_lower_bound: f64::MAX,
            },
        };
        let collision_done = Instant::now();
        let collision_ms = collision_done.duration_since(fk_done).as_secs_f64() * 1000.0;

        let mut shadow_owned = shadow.cloned();
        let mut shadow_ms = if shadow.is_some() { Some(0.0) } else { None };
        if shadow_owned.is_none() {
            if let Some(sim) = &self.shadow_simulator {
                let t_shadow = Instant::now();
                let action_f64: Vec<f64> = action.data.iter().map(|v| *v as f64).collect();
                shadow_owned = Some(sim.simulate(current_joints, &action_f64, limits, None, &[]));
                shadow_ms = Some(t_shadow.elapsed().as_secs_f64() * 1000.0);
            }
        }

        let arbiter_start = Instant::now();
        let extra = extra_physical_reasons(
            action,
            current_joints,
            limits,
            chain_ref,
            proposal.as_ref().ok(),
        );
        let decision =
            self.arbiter_decide(action, &collision_report, semantic, shadow_owned.as_ref(), extra);
        let arbiter_ms = arbiter_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = Instant::now().duration_since(t0).as_secs_f64() * 1000.0;
        let latency = LatencyBreakdown {
            ingest_ms,
            urdf_fk_ms,
            physics_ms,
            collision_ms,
            tf2_ms: None,
            arbiter_ms,
            shadow_ms,
            total_ms,
        };

        let decision = match decision {
            ArbiterDecision::Pass { action, .. } => ArbiterDecision::Pass { action, latency },
            ArbiterDecision::Block {
                safe_fallback,
                reasons,
                ..
            } => ArbiterDecision::Block {
                safe_fallback,
                reasons,
                latency,
            },
        };

        SafetyEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            ts_ns: action.t_ns,
            robot_id: self.config.robot_id.clone(),
            sequence_id: action.sequence_id,
            decision,
            action_hash: action.hash_hex(),
            mode: self.config.mode,
        }
    }

    fn arbiter_decide(
        &self,
        action: &ActionVector,
        collision: &CollisionReport,
        semantic: &SemanticRiskReport,
        shadow: Option<&ShadowResult>,
        extra: Vec<ArbiterReason>,
    ) -> ArbiterDecision {
        let mut reasons = extra;

        if collision.hit {
            for pair in &collision.pairs {
                reasons.push(ArbiterReason {
                    ontology_id: shield_core::ontology::physical::collision(),
                    detail: format!("pair={}:{}", pair.link, pair.obstacle),
                    score: 1.0,
                });
            }
        }

        if self.config.mode != RunMode::PhysicsOnly && !semantic.stale {
            for oid in &semantic.triggered {
                reasons.push(ArbiterReason {
                    ontology_id: oid.clone(),
                    detail: String::new(),
                    score: semantic.risk_score,
                });
            }
        }

        // Shadow prior from the async pass. Stale is fine.
        if let Some(sr) = shadow {
            if sr.risk_score > 0.5 {
                for oid in &sr.triggered_ids {
                    // Don't double-count an id that's already in `reasons`.
                    if !reasons.iter().any(|r| &r.ontology_id == oid) {
                        reasons.push(ArbiterReason {
                            ontology_id: oid.clone(),
                            detail: format!(
                                "shadow_path: risk={:.2} steps={}",
                                sr.risk_score, sr.steps_evaluated
                            ),
                            score: sr.risk_score,
                        });
                    }
                }
            }
        }

        if reasons.is_empty() || self.config.mode == RunMode::Monitor {
            ArbiterDecision::Pass {
                action: action.clone(),
                latency: LatencyBreakdown::default(),
            }
        } else {
            let fallback = ActionVector::new(action.t_ns, action.sequence_id, vec![0.0; action.dim()]);
            ArbiterDecision::Block {
                safe_fallback: fallback,
                reasons,
                latency: LatencyBreakdown::default(),
            }
        }
    }
}
