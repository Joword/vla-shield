use serde::{Deserialize, Serialize};
use shield_core::ontology::OntologyId;

/// One shadow-sim pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowResult {
    /// Composite risk in `[0.0, 1.0]`.
    pub risk_score: f32,
    /// Ontology ids that fired (can be empty).
    pub triggered_ids: Vec<OntologyId>,
    /// Steps we actually evaluated. Full `steps` if nothing tripped.
    pub steps_evaluated: usize,
    /// True if we bailed early on a hard violation.
    pub early_exit: bool,
}

impl ShadowResult {
    /// Clean pass.
    pub fn safe(steps: usize) -> Self {
        ShadowResult {
            risk_score: 0.0,
            triggered_ids: vec![],
            steps_evaluated: steps,
            early_exit: false,
        }
    }

    /// One ontology id tripped. We stopped.
    pub fn violation(id: OntologyId, step: usize, score: f32) -> Self {
        ShadowResult {
            risk_score: score,
            triggered_ids: vec![id],
            steps_evaluated: step,
            early_exit: true,
        }
    }
}
