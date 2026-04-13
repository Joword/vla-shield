use crate::action::ActionVector;
use crate::ontology::{OntologyId, Severity};
use crate::types::RunMode;
use serde::{Deserialize, Serialize};

/// Report from the collision precheck module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionReport {
    pub hit: bool,
    pub pairs: Vec<CollisionPair>,
    pub energy_lower_bound: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionPair {
    pub link: String,
    pub obstacle: String,
    pub min_distance: f64,
}

/// Report from the semantic risk module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRiskReport {
    pub sequence_id: u64,
    pub risk_score: f32,
    pub triggered: Vec<OntologyId>,
    pub stale: bool,
}

impl Default for SemanticRiskReport {
    fn default() -> Self {
        Self {
            sequence_id: 0,
            risk_score: 0.0,
            triggered: Vec::new(),
            stale: true,
        }
    }
}

/// Latency breakdown for structured logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    pub ingest_ms: f64,
    pub physics_ms: f64,
    pub collision_ms: f64,
    pub semantic_ms: Option<f64>,
    pub total_ms: f64,
}

/// Final decision from the arbiter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "UPPERCASE")]
pub enum ArbiterDecision {
    Pass {
        action: ActionVector,
        latency: LatencyBreakdown,
    },
    Block {
        safe_fallback: ActionVector,
        reasons: Vec<ArbiterReason>,
        latency: LatencyBreakdown,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbiterReason {
    pub ontology_id: OntologyId,
    pub detail: String,
    pub score: f32,
}

impl ArbiterDecision {
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass { .. })
    }

    pub fn is_block(&self) -> bool {
        matches!(self, Self::Block { .. })
    }
}

/// Structured safety event emitted on every decision (primarily blocks).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyEvent {
    pub event_id: String,
    pub ts_ns: u64,
    pub robot_id: String,
    pub sequence_id: u64,
    pub decision: ArbiterDecision,
    pub action_hash: String,
    pub mode: RunMode,
}

/// Trait for the central arbiter that combines all reports into a decision.
pub trait Arbiter: Send + Sync {
    fn decide(
        &self,
        mode: RunMode,
        action: &ActionVector,
        collision: &CollisionReport,
        semantic: &SemanticRiskReport,
        severity_threshold: Severity,
    ) -> ArbiterDecision;
}
