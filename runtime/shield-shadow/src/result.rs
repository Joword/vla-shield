use serde::{Deserialize, Serialize};
use shield_core::ontology::OntologyId;

/// Result of a shadow path simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowResult {
    /// Composite risk score in `[0.0, 1.0]`.
    pub risk_score: f32,
    /// Ontology IDs triggered during the simulation (may be empty).
    pub triggered_ids: Vec<OntologyId>,
    /// Number of trajectory steps evaluated before a violation was found
    /// (or the full `steps` count if no violation occurred).
    pub steps_evaluated: usize,
    /// True when the simulation was cut short due to a hard violation.
    pub early_exit: bool,
}

impl ShadowResult {
    /// Convenience: construct a safe (no-violation) result.
    pub fn safe(steps: usize) -> Self {
        ShadowResult {
            risk_score: 0.0,
            triggered_ids: vec![],
            steps_evaluated: steps,
            early_exit: false,
        }
    }

    /// Convenience: construct a violation result with a single ontology ID.
    pub fn violation(id: OntologyId, step: usize, score: f32) -> Self {
        ShadowResult {
            risk_score: score,
            triggered_ids: vec![id],
            steps_evaluated: step,
            early_exit: true,
        }
    }
}
