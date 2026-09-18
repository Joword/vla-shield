//! Extra physics that does *not* belong inside the velocity/position clamp.
//!
//! Projector still only clamps. Singularity / tip-over / overload come back as
//! tagged reasons so the arbiter can keep BLOCK > CLAMP > WARN. Don't fold them
//! into a generic `project()` error — FFI used to slam that into `PHY.JOINT_LIMIT`.

use crate::calibration::phy_calibration;
use crate::DynProposal;
use shield_core::action::ActionVector;
use shield_core::arbiter::ArbiterReason;
use shield_core::ontology::physical;
use shield_core::types::JointLimits;
use shield_urdf::UrdfKinematicChain;

/// Singularity / tip-over / overload reasons for a projected state.
///
/// `proposal` can be `None` if projection itself failed; tip-over and overload
/// still look at the raw command.
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
    let cal = &phy_calibration().singularity;
    let mut manip: Option<f64> = None;
    if let Some(chain) = chain {
        if chain.dof() == q.len() {
            if let Ok(m) = chain.positional_manipulability(q) {
                manip = Some(m);
            }
        }
    }
    let below = manip.map(|m| m < cal.jacobian_floor).unwrap_or(false);
    let elbow = near_elbow_lock(q);
    if !below && !elbow {
        return None;
    }
    let m = manip.unwrap_or(0.0);
    Some(ArbiterReason {
        ontology_id: physical::singularity(),
        detail: format!(
            "positional manipulability {m:.4} below threshold {:.4}",
            cal.ontology_manipulability
        ),
        score: 1.0,
    })
}

fn near_elbow_lock(q: &[f64]) -> bool {
    let cal = &phy_calibration().singularity;
    q.len() == cal.elbow_lock_dof
        && q
            .get(cal.elbow_lock_joint_index)
            .map(|v| v.abs() < cal.elbow_lock_rad)
            .unwrap_or(false)
}

fn tipover_reason(action: &ActionVector, _limits: &JointLimits) -> Option<ArbiterReason> {
    let cal = &phy_calibration().tipover;
    if action.dim() < cal.mobile_min_dof {
        return None;
    }
    let a = action.data[action.dim() - 1] as f64;
    let zmp_y = cal.com_height_m * a / cal.g;
    let zmp_x = 0.0;
    let outside_polygon = zmp_y.abs() + cal.margin_m > cal.support_half_m;
    if a.abs() <= cal.base_accel_limit && !outside_polygon {
        return None;
    }
    Some(ArbiterReason {
        ontology_id: physical::tipover(),
        detail: format!(
            "base accel {a:.3} m/s² exceeds {:.3}; ZMP at ({zmp_x:.3}, {zmp_y:.3}) m",
            cal.base_accel_limit
        ),
        score: 1.0,
    })
}

fn overload_reason(
    action: &ActionVector,
    q: &[f64],
    limits: &JointLimits,
) -> Option<ArbiterReason> {
    let cal = &phy_calibration().overload;
    let n = action.dim();
    if n == 0 || q.len() != n || limits.torque_max.len() != n {
        return None;
    }
    let distal_from = n.saturating_sub(cal.distal_joints);
    let mut worst: Option<(usize, f64, f64)> = None;
    for i in 0..n {
        let v = action.data[i] as f64;
        let inertia = if i >= distal_from {
            cal.distal_inertia
        } else {
            cal.proximal_inertia
        };
        let gravity = cal.gravity_coeff * (n - 1 - i) as f64 * q[i].sin().abs();
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

/// Map a projector error string onto the ontology id the FFI used to recover
/// by substring search. Don't slam everything into JOINT_LIMIT.
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
