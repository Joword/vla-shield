use crate::config::RuntimeConfig;
use shield_collision::CollisionPrechecker;
use shield_core::action::ActionVector;
use shield_core::arbiter::{
    ArbiterDecision, ArbiterReason, CollisionReport, LatencyBreakdown, SafetyEvent,
    SemanticRiskReport,
};
use shield_core::scene::SceneGraph;
use shield_core::types::{JointLimits, RunMode};
use shield_physics::{PhysicalProjector, ProjectionContext};
use shield_shadow::result::ShadowResult;
use shield_shadow::{JointSpaceSimulator, ShadowSimulator};
use std::time::Instant;

/// The main safety-check pipeline orchestrating projection → collision → arbiter.
pub struct SafetyPipeline {
    pub config: RuntimeConfig,
    projector: Box<dyn PhysicalProjector>,
    checker: Box<dyn CollisionPrechecker>,
    shadow_simulator: Option<Box<dyn ShadowSimulator>>,
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
        }
    }

    /// Override the default shadow simulator.
    pub fn with_shadow_simulator(mut self, sim: Option<Box<dyn ShadowSimulator>>) -> Self {
        self.shadow_simulator = sim;
        self
    }

    /// Run the full hot-path pipeline for a single action.
    ///
    /// `shadow` carries the result from the previous async shadow simulation
    /// pass (stale-safe: `None` means no shadow data available yet).
    pub fn evaluate(
        &self,
        action: &ActionVector,
        current_joints: &[f64],
        limits: &JointLimits,
        scene: &SceneGraph,
        semantic: &SemanticRiskReport,
        shadow: Option<&ShadowResult>,
    ) -> SafetyEvent {
        let t0 = Instant::now();

        let ingest_done = Instant::now();
        let ingest_ms = ingest_done.duration_since(t0).as_secs_f64() * 1000.0;

        let proj_ctx = ProjectionContext {
            current_joints,
            limits,
            scene,
            dt: self.config.dt,
            urdf_chain: None,
            forbidden_zones: &[],
            semantic_constraints: &[],
        };

        let proposal = self.projector.project(&proj_ctx, action);
        let physics_done = Instant::now();
        let physics_ms = physics_done.duration_since(ingest_done).as_secs_f64() * 1000.0;

        let collision_report = match &proposal {
            Ok(p) => {
                let ctx = shield_collision::CollisionContext {
                    scene,
                    limits,
                    epsilon: self.config.collision_epsilon,
                };
                self.checker.precheck(&ctx, p)
            }
            Err(_) => CollisionReport {
                hit: true,
                pairs: vec![],
                energy_lower_bound: f64::MAX,
            },
        };
        let collision_done = Instant::now();
        let collision_ms = collision_done.duration_since(physics_done).as_secs_f64() * 1000.0;

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
        let decision = self.arbiter_decide(action, &collision_report, semantic, shadow_owned.as_ref());
        let arbiter_ms = arbiter_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = Instant::now().duration_since(t0).as_secs_f64() * 1000.0;
        let latency = LatencyBreakdown {
            ingest_ms,
            urdf_fk_ms: None,
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
    ) -> ArbiterDecision {
        let mut reasons = Vec::new();

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

        // Incorporate shadow simulation risk prior (async, stale-safe).
        if let Some(sr) = shadow {
            if sr.risk_score > 0.5 {
                for oid in &sr.triggered_ids {
                    // Only add if not already present to avoid duplicates.
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
