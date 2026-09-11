use crate::action::ActionVector;
use crate::ontology::{OntologyId, Severity};
use crate::types::RunMode;
use serde::{Deserialize, Serialize};

/// Broad-phase result. Empty `pairs` means nothing overlapped.
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

/// Semantic snapshot. If `stale` is true, ignore it and stay physics-only.
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

/// Per-stage latency in milliseconds. `None` = that stage didn't run
/// (no URDF → no `urdf_fk_ms`; shadow off → no `shadow_ms`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatencyBreakdown {
    /// Deserialize + validate the incoming action.
    pub ingest_ms: f64,
    /// URDF FK (+ manipulability if we bother).
    pub urdf_fk_ms: Option<f64>,
    /// Clamp joints / check forbidden zones.
    pub physics_ms: f64,
    /// AABB overlap sweep.
    pub collision_ms: f64,
    /// World-frame check (mobile base / multi-robot).
    pub tf2_ms: Option<f64>,
    /// Rank reasons and pick PASS vs BLOCK.
    pub arbiter_ms: f64,
    /// Shadow sim — off the hot path, so this can be stale.
    pub shadow_ms: Option<f64>,
    /// Wall clock, ingest → decision.
    pub total_ms: f64,
}

/// Pass or block. That's the whole verdict.
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

/// One log row per decision. Blocks are the interesting ones.
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

/// Smash collision + semantic + extras into one Pass/Block.
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
