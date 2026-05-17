//! Joint-space shadow path simulation.
//!
//! Interpolates from the current joint configuration toward the integrated
//! velocity command over `steps` intermediate points, checking joint limits
//! and optional forbidden Cartesian zones at each step.
//!
//! This module mirrors the Python `ShadowSimPredictor` reference implementation
//! but runs natively in Rust and can be called from the async safety layer
//! without touching the hot-path budget.

use crate::result::ShadowResult;
use shield_core::ontology::physical;
use shield_core::types::JointLimits;
use shield_urdf::{AxisAlignedBox, UrdfKinematicChain};

/// Configuration for a joint-space shadow simulation pass.
#[derive(Debug, Clone)]
pub struct JointSpaceShadowConfig {
    /// Number of interpolation steps (minimum 2).
    pub steps: usize,
    /// Integration time step in seconds.
    pub dt: f64,
    /// Safety margin (radians) inside joint limits before raising a flag.
    pub limit_margin_rad: f64,
}

impl Default for JointSpaceShadowConfig {
    fn default() -> Self {
        JointSpaceShadowConfig {
            steps: 8,
            dt: 0.01,
            limit_margin_rad: 0.005,
        }
    }
}

/// Simulate a shadow trajectory in joint space and return a risk summary.
///
/// # Arguments
/// * `config`         - Simulation parameters.
/// * `current_joints` - Current joint positions (rad).
/// * `action`         - Joint velocity command (rad/s).
/// * `limits`         - URDF joint limits.
/// * `urdf_chain`     - Optional FK chain for Cartesian zone checks.
/// * `forbidden_zones`- Cartesian forbidden zones in the base-link frame.
pub fn simulate(
    config: &JointSpaceShadowConfig,
    current_joints: &[f64],
    action: &[f64],
    limits: &JointLimits,
    urdf_chain: Option<&UrdfKinematicChain>,
    forbidden_zones: &[AxisAlignedBox],
) -> ShadowResult {
    let ndof = current_joints.len();
    if action.len() != ndof {
        return ShadowResult::violation(physical::joint_limit(), 0, 1.0);
    }

    let steps = config.steps.max(2);

    for step in 1..=steps {
        let alpha = step as f64 / steps as f64;

        // Integrate velocity toward the final commanded position.
        let mut q: Vec<f64> = current_joints
            .iter()
            .zip(action.iter())
            .map(|(q0, v)| q0 + alpha * config.dt * v)
            .collect();

        // Clamp to limits.
        for i in 0..ndof {
            q[i] = q[i].clamp(limits.position_min[i], limits.position_max[i]);
        }

        // Check for joint limit approach (within margin).
        for i in 0..ndof {
            let margin = config.limit_margin_rad;
            if q[i] <= limits.position_min[i] + margin
                || q[i] >= limits.position_max[i] - margin
            {
                tracing::debug!(
                    step,
                    joint = limits.names[i],
                    q = q[i],
                    "shadow: joint limit approach"
                );
                return ShadowResult::violation(physical::joint_limit(), step, 0.8);
            }
        }

        // Optional: Cartesian forbidden-zone check via FK.
        if let Some(chain) = urdf_chain {
            if chain.dof() == ndof {
                if let Ok(ee) = chain.ee_position(&q) {
                    for zone in forbidden_zones {
                        if zone.contains(&ee) {
                            tracing::debug!(step, ?ee, "shadow: EE in forbidden zone");
                            return ShadowResult::violation(
                                physical::forbidden_zone(),
                                step,
                                1.0,
                            );
                        }
                    }
                }
            }
        }
    }

    ShadowResult::safe(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shield_core::types::JointLimits;

    fn make_limits(n: usize) -> JointLimits {
        JointLimits {
            names: (0..n).map(|i| format!("j{i}")).collect(),
            position_min: vec![-3.14; n],
            position_max: vec![3.14; n],
            velocity_max: vec![1.0; n],
            acceleration_max: vec![10.0; n],
            torque_max: vec![50.0; n],
        }
    }

    #[test]
    fn safe_trajectory() {
        let cfg = JointSpaceShadowConfig::default();
        let q0 = vec![0.0, 0.0, 0.0];
        let action = vec![0.1, -0.1, 0.05];
        let limits = make_limits(3);
        let result = simulate(&cfg, &q0, &action, &limits, None, &[]);
        assert_eq!(result.risk_score, 0.0);
        assert!(result.triggered_ids.is_empty());
    }

    #[test]
    fn detects_limit_approach() {
        let cfg = JointSpaceShadowConfig {
            steps: 4,
            dt: 1.0,
            limit_margin_rad: 0.1,
        };
        // Drive joint 0 hard toward its upper limit (3.14).
        let q0 = vec![3.0, 0.0];
        let action = vec![5.0, 0.0]; // large positive velocity
        let limits = make_limits(2);
        let result = simulate(&cfg, &q0, &action, &limits, None, &[]);
        assert!(result.risk_score > 0.0);
        assert!(!result.triggered_ids.is_empty());
    }
}
