//! Conversions between Python-facing types and shield-core Rust types.

use shield_core::action::ActionVector;
use shield_core::arbiter::{ArbiterDecision, LatencyBreakdown};
use shield_core::types::JointLimits;

/// Build a `JointLimits` from flat Python lists.
///
/// # Arguments
/// * `names`           - Joint name strings.
/// * `position_min`    - Lower position limits (rad).
/// * `position_max`    - Upper position limits (rad).
/// * `velocity_max`    - Velocity caps (rad/s).
/// * `acceleration_max`- Acceleration caps (rad/s²); uses 10× velocity if empty.
/// * `torque_max`      - Torque caps (Nm); uses 50 Nm per joint if empty.
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

/// Summarise an `ArbiterDecision` into a Python-friendly dict-like structure.
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

/// Convert a raw Python `list[float]` action into an `ActionVector`.
pub fn vec_to_action(t_ns: u64, sequence_id: u64, data: Vec<f32>) -> ActionVector {
    ActionVector::new(t_ns, sequence_id, data)
}
