//! Python lists ↔ shield-core types.

use shield_core::action::ActionVector;
use shield_core::arbiter::{ArbiterDecision, LatencyBreakdown};
use shield_core::types::JointLimits;

/// `JointLimits` from flat Python lists.
///
/// Empty `acceleration_max` → 10× velocity. Empty `torque_max` → 50 Nm each.
pub fn make_joint_limits(
    names: Vec<String>,
    position_min: Vec<f64>,
    position_max: Vec<f64>,
    velocity_max: Vec<f64>,
    acceleration_max: Vec<f64>,
    torque_max: Vec<f64>,
) -> JointLimits {
    let n = names.len();
    let acceleration_max = if acceleration_max.is_empty() {
        velocity_max.iter().map(|v| v * 10.0).collect()
    } else {
        acceleration_max
    };
    let torque_max = if torque_max.is_empty() {
        vec![50.0; n]
    } else {
        torque_max
    };
    JointLimits {
        names,
        position_min,
        position_max,
        velocity_max,
        acceleration_max,
        torque_max,
    }
}

/// Flatten an `ArbiterDecision` into something Python can swallow.
pub struct PyDecisionSummary {
    pub decision: &'static str,
    pub reasons: Vec<(String, String, f32)>,
    pub latency: LatencyBreakdown,
}

impl From<ArbiterDecision> for PyDecisionSummary {
    fn from(d: ArbiterDecision) -> Self {
        match d {
            ArbiterDecision::Pass { latency, .. } => PyDecisionSummary {
                decision: "PASS",
                reasons: vec![],
                latency,
            },
            ArbiterDecision::Block {
                reasons, latency, ..
            } => {
                let rs = reasons
                    .into_iter()
                    .map(|r| (r.ontology_id.to_string(), r.detail, r.score))
                    .collect();
                PyDecisionSummary {
                    decision: "BLOCK",
                    reasons: rs,
                    latency,
                }
            }
        }
    }
}

/// Python `list[float]` → `ActionVector`.
pub fn vec_to_action(t_ns: u64, sequence_id: u64, data: Vec<f32>) -> ActionVector {
    ActionVector::new(t_ns, sequence_id, data)
}
