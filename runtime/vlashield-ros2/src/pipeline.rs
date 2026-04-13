use crate::config::RuntimeConfig;
use vlashield_collision::CollisionPrechecker;
use vlashield_core::action::ActionVector;
use vlashield_core::arbiter::{
    ArbiterDecision, ArbiterReason, CollisionReport, LatencyBreakdown, SafetyEvent,
    SemanticRiskReport,
};
use vlashield_core::scene::SceneGraph;
use vlashield_core::types::{JointLimits, RunMode};
use vlashield_physics::{PhysicalProjector, ProjectionContext};
use std::time::Instant;

/// The main safety-check pipeline orchestrating projection → collision → arbiter.
pub struct SafetyPipeline {
    pub config: RuntimeConfig,
    projector: Box<dyn PhysicalProjector>,
    checker: Box<dyn CollisionPrechecker>,
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
        }
    }

    /// Run the full hot-path pipeline for a single action.
    pub fn evaluate(
        &self,
        action: &ActionVector,
        current_joints: &[f64],
        limits: &JointLimits,
        scene: &SceneGraph,
        semantic: &SemanticRiskReport,
    ) -> SafetyEvent {
        let t0 = Instant::now();

        let ingest_done = Instant::now();
        let ingest_ms = ingest_done.duration_since(t0).as_secs_f64() * 1000.0;

        let proj_ctx = ProjectionContext {
            current_joints,
            limits,
            scene,
            dt: self.config.dt,
        };

        let proposal = self.projector.project(&proj_ctx, action);
        let physics_done = Instant::now();
        let physics_ms = physics_done.duration_since(ingest_done).as_secs_f64() * 1000.0;

        let collision_report = match &proposal {
            Ok(p) => {
                let ctx = vlashield_collision::CollisionContext {
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

        let decision = self.arbiter_decide(action, &collision_report, semantic);

        let total_ms = Instant::now().duration_since(t0).as_secs_f64() * 1000.0;
        let latency = LatencyBreakdown {
            ingest_ms,
            physics_ms,
            collision_ms,
            semantic_ms: if semantic.stale { None } else { Some(0.0) },
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
    ) -> ArbiterDecision {
        let mut reasons = Vec::new();

        if collision.hit {
            for pair in &collision.pairs {
                reasons.push(ArbiterReason {
                    ontology_id: vlashield_core::ontology::physical::collision(),
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

        if reasons.is_empty() || self.config.mode == RunMode::Monitor {
            let dummy = LatencyBreakdown {
                ingest_ms: 0.0,
                physics_ms: 0.0,
                collision_ms: 0.0,
                semantic_ms: None,
                total_ms: 0.0,
            };
            ArbiterDecision::Pass {
                action: action.clone(),
                latency: dummy,
            }
        } else {
            let fallback = ActionVector::new(action.t_ns, action.sequence_id, vec![0.0; action.dim()]);
            let dummy = LatencyBreakdown {
                ingest_ms: 0.0,
                physics_ms: 0.0,
                collision_ms: 0.0,
                semantic_ms: None,
                total_ms: 0.0,
            };
            ArbiterDecision::Block {
                safe_fallback: fallback,
                reasons,
                latency: dummy,
            }
        }
    }
}
