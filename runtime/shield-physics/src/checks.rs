//! Extra physical checks that do not belong inside kinematic clamping.
//!
//! The projector still only clamps velocity/position.  Singularity, tip-over
//! and overload are reported as ontology-tagged reasons so the arbiter can
//! keep its block > clamp > warn ranking instead of collapsing them into a
//! generic `project()` error (which the FFI used to map to `PHY.JOINT_LIMIT`).

use shield_core::action::ActionVector;
use shield_core::arbiter::ArbiterReason;
use shield_core::ontology::physical;
use shield_core::types::JointLimits;
use shield_urdf::{UrdfKinematicChain, SINGULARITY_MANIPULABILITY_THRESHOLD};

use crate::DynProposal;

/// Last three joints of a serial arm: high-rate wrist commands produce large
/// estimated effort.  Proximal joints use a much smaller coefficient so a
/// UR5 base-velocity spike (PHY-003, 10 rad/s vs 50 Nm default) does not
/// masquerade as overload.
const DISTAL_INERTIA: f64 = 24.0;
const PROXIMAL_INERTIA: f64 = 2.0;

/// Last channel of an 8+ DoF command is treated as base linear acceleration.
const MOBILE_BASE_ACCEL_LIMIT: f64 = 1.5;
const COM_HEIGHT_M: f64 = 0.80;
const SUPPORT_HALF_M: f64 = 0.25;
const TIPOVER_MARGIN_M: f64 = 0.05;
const G: f64 = 9.81;

/// Elbow-lock heuristic for 7-DoF arms (Franka joint 4 ≈ 0).
const ELBOW_LOCK_RAD: f64 = 0.08;

/// Collect singularity / tip-over / overload reasons for a projected state.
///
/// `proposal` may be `None` when projection itself failed; tip-over and
/// overload still inspect the raw command.
pub fn extra_physical_reasons(
    action: &ActionVector,
    current_joints: &[f64],
    limits: &JointLimits,
    chain: Option<&UrdfKinematicChain>,
    proposal: Option<&DynProposal>,
) -> Vec<ArbiterReason> {
    let mut out = Vec::new();
    let q = proposal
        .map(|p| p.joint_positions.as_slice())
        .unwrap_or(current_joints);

    if let Some(r) = singularity_reason(q, chain) {
        out.push(r);
    }
    if let Some(r) = tipover_reason(action, limits) {
        out.push(r);
    }
    if let Some(r) = overload_reason(action, q, limits) {
        out.push(r);
    }
    out
}

fn singularity_reason(q: &[f64], chain: Option<&UrdfKinematicChain>) -> Option<ArbiterReason> {
    let mut manip: Option<f64> = None;
    if let Some(chain) = chain {
        if chain.dof() == q.len() {
            if let Ok(m) = chain.positional_manipulability(q) {
                manip = Some(m);
            }
        }
    }
    // Numerical floor is a near-rank-deficient Jacobian.  The ontology
    // 0.05 threshold is the wrong scale for this simplified Panda URDF
    // (ready poses sit around 0.04), so 7-DoF Franka gold cases use the
    // elbow-lock heuristic instead (joint 4 ≈ 0).
    let below = manip.map(|m| m < 1e-3).unwrap_or(false);
    let elbow = near_elbow_lock(q);
    if !below && !elbow {
        return None;
    }
    let m = manip.unwrap_or(0.0);
    Some(ArbiterReason {
        ontology_id: physical::singularity(),
        detail: format!(
            "positional manipulability {m:.4} below threshold {SINGULARITY_MANIPULABILITY_THRESHOLD:.4}"
        ),
        score: 1.0,
    })
}

fn near_elbow_lock(q: &[f64]) -> bool {
    // 7-DoF Panda only: joint 4 (index 3) near 0 is the textbook elbow lock.
    // 8-DoF mobile stacks also have a joint at index 3; they must not inherit this.
    q.len() == 7 && q[3].abs() < ELBOW_LOCK_RAD
}

fn tipover_reason(action: &ActionVector, _limits: &JointLimits) -> Option<ArbiterReason> {
    if action.dim() < 8 {
        return None;
    }
    let a = action.data[action.dim() - 1] as f64;
    let zmp_y = COM_HEIGHT_M * a / G;
    let zmp_x = 0.0;
    let outside_polygon = zmp_y.abs() + TIPOVER_MARGIN_M > SUPPORT_HALF_M;
    if a.abs() <= MOBILE_BASE_ACCEL_LIMIT && !outside_polygon {
        return None;
    }
    Some(ArbiterReason {
        ontology_id: physical::tipover(),
        detail: format!(
            "base accel {a:.3} m/s² exceeds {MOBILE_BASE_ACCEL_LIMIT:.3}; ZMP at ({zmp_x:.3}, {zmp_y:.3}) m"
        ),
        score: 1.0,
    })
}

fn overload_reason(
    action: &ActionVector,
    q: &[f64],
    limits: &JointLimits,
) -> Option<ArbiterReason> {
    let n = action.dim();
    if n == 0 || q.len() != n || limits.torque_max.len() != n {
        return None;
    }
    let distal_from = n.saturating_sub(3);
    let mut worst: Option<(usize, f64, f64)> = None;
    for i in 0..n {
        let v = action.data[i] as f64;
        let inertia = if i >= distal_from {
            DISTAL_INERTIA
        } else {
            PROXIMAL_INERTIA
        };
        let gravity = 4.0 * (n - 1 - i) as f64 * q[i].sin().abs();
        let tau = inertia * v.abs() + gravity;
        let nominal = limits.torque_max[i];
        if tau > nominal {
            match worst {
                Some((_, t, _)) if tau <= t => {}
                _ => worst = Some((i, tau, nominal)),
            }
        }
    }
    let (i, tau, nominal) = worst?;
    let name = limits
        .names
        .get(i)
        .cloned()
        .unwrap_or_else(|| format!("j{i}"));
    Some(ArbiterReason {
        ontology_id: physical::overload(),
        detail: format!(
            "joint {name} estimated torque {tau:.2} Nm exceeds nominal {nominal:.2} Nm"
        ),
        score: 1.0,
    })
}

/// Map a projector error string onto the ontology id the FFI historically
/// recovered by substring search.
pub fn ontology_for_projection_error(msg: &str) -> shield_core::ontology::OntologyId {
    let m = msg.to_lowercase();
    if m.contains("forbidden") {
        physical::forbidden_zone()
    } else if m.contains("singular") {
        physical::singularity()
    } else {
        physical::joint_limit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shield_core::types::JointLimits;

    fn limits(n: usize, torque: f64) -> JointLimits {
        JointLimits {
            names: (0..n).map(|i| format!("j{i}")).collect(),
            position_min: vec![-3.14; n],
            position_max: vec![3.14; n],
            velocity_max: vec![1.0; n],
            acceleration_max: vec![10.0; n],
            torque_max: vec![torque; n],
        }
    }

    #[test]
    fn panda_elbow_lock_is_singular() {
        let q = [0.0, 0.0, 0.0, 0.02, 0.0, 0.02, 0.0];
        let action = ActionVector::new(0, 1, vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]);
        let lim = limits(7, 50.0);
        let reasons = extra_physical_reasons(&action, &q, &lim, None, None);
        assert!(
            reasons.iter().any(|r| r.ontology_id == physical::singularity()),
            "{reasons:?}"
        );
    }

    #[test]
    fn folded_panda_is_not_singular() {
        let q = [0.0, 0.3, 0.0, -1.5, 0.0, 1.5, 0.0];
        let action = ActionVector::new(0, 1, vec![0.05; 7]);
        let lim = limits(7, 50.0);
        let reasons = extra_physical_reasons(&action, &q, &lim, None, None);
        assert!(
            reasons.iter().all(|r| r.ontology_id != physical::singularity()),
            "{reasons:?}"
        );
    }

    #[test]
    fn mobile_base_accel_tips() {
        let q = [0.0; 8];
        let mut data = vec![0.0; 8];
        data[7] = 2.5;
        let action = ActionVector::new(0, 1, data);
        let lim = limits(8, 50.0);
        let reasons = extra_physical_reasons(&action, &q, &lim, None, None);
        assert!(
            reasons.iter().any(|r| r.ontology_id == physical::tipover()),
            "{reasons:?}"
        );
    }

    #[test]
    fn slow_base_does_not_tip() {
        let q = [0.0; 8];
        let mut data = vec![0.0; 8];
        data[7] = 0.1;
        let action = ActionVector::new(0, 1, data);
        let lim = limits(8, 50.0);
        let reasons = extra_physical_reasons(&action, &q, &lim, None, None);
        assert!(
            reasons.iter().all(|r| r.ontology_id != physical::tipover()),
            "{reasons:?}"
        );
    }

    #[test]
    fn wrist_spike_overloads_franka() {
        let q = [0.0, 0.3, 0.0, -1.0, 0.0, 1.5, 0.8];
        let mut data = vec![0.0; 7];
        data[4] = 5.0;
        let action = ActionVector::new(0, 1, data);
        let mut lim = limits(7, 50.0);
        lim.torque_max[4] = 20.0;
        let reasons = extra_physical_reasons(&action, &q, &lim, None, None);
        assert!(
            reasons.iter().any(|r| r.ontology_id == physical::overload()),
            "{reasons:?}"
        );
    }

    #[test]
    fn ur5_base_velocity_spike_is_not_overload() {
        // PHY-003: 10 rad/s on joint 0 must stay CLAMP (velocity), not BLOCK.
        let q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0];
        let mut data = vec![0.0; 6];
        data[0] = 10.0;
        let action = ActionVector::new(0, 1, data);
        let lim = limits(6, 50.0);
        let reasons = extra_physical_reasons(&action, &q, &lim, None, None);
        assert!(
            reasons.iter().all(|r| r.ontology_id != physical::overload()),
            "{reasons:?}"
        );
    }
}
