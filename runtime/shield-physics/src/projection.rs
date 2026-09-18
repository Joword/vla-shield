use crate::semantic::SemanticConstraintMapper;
use crate::{DynProposal, PhysicalProjector, ProjectionContext};
use shield_core::action::ActionVector;
use shield_core::error::{Result, shieldError};

/// Dumb-but-safe projector: treat `action.data` as joint vel, clamp to
/// `velocity_max`, integrate one step, clamp position. That's it.
pub struct KinematicClampProjector;

impl PhysicalProjector for KinematicClampProjector {
    fn project(&self, ctx: &ProjectionContext, action: &ActionVector) -> Result<DynProposal> {
        let ndof = ctx.current_joints.len();
        if action.dim() != ndof {
            return Err(shieldError::DimensionMismatch {
                expected: ndof,
                got: action.dim(),
            });
        }

        let mut proposed_pos = Vec::with_capacity(ndof);
        let mut clamped_vel = Vec::with_capacity(ndof);

        for i in 0..ndof {
            let raw_vel = action.data[i] as f64;
            let v_max = ctx.limits.velocity_max[i];
            let mut vel = raw_vel.clamp(-v_max, v_max);
            if ctx.prev_velocity.len() == ndof && ctx.dt > 0.0 {
                let a_max = ctx
                    .limits
                    .acceleration_max
                    .get(i)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                if a_max.is_finite() {
                    let prev = ctx.prev_velocity[i];
                    let da = a_max * ctx.dt;
                    vel = vel.clamp(prev - da, prev + da);
                }
            }

            let pos = ctx.current_joints[i] + vel * ctx.dt;
            let pos = pos.clamp(ctx.limits.position_min[i], ctx.limits.position_max[i]);

            clamped_vel.push(vel);
            proposed_pos.push(pos);
        }

        let (ee_position, ee_orientation) = if let Some(chain) = ctx.urdf_chain {
            if chain.dof() > ndof {
                return Err(shieldError::DimensionMismatch {
                    expected: chain.dof(),
                    got: ndof,
                });
            }
            let ee = chain
                .ee_position(&proposed_pos)
                .map_err(|e| shieldError::Projection(e.to_string()))?;
            let quat = chain
                .ee_orientation_quat(&proposed_pos)
                .map_err(|e| shieldError::Projection(e.to_string()))?;
            (ee, quat)
        } else {
            ([0.0; 3], [0.0, 0.0, 0.0, 1.0])
        };

        // URDF Cartesian no-go boxes.
        for zone in ctx.forbidden_zones {
            if zone.contains(&ee_position) {
                return Err(shieldError::Projection(
                    "end-effector in forbidden Cartesian zone (PHY.FORBIDDEN_ZONE)".into(),
                ));
            }
        }

        // SEM.* exclusion zones (heat, forbidden region, …).
        let sem_mapper = SemanticConstraintMapper::new(ctx.semantic_constraints);
        for (zone, oid) in sem_mapper.exclusion_zones() {
            if zone.contains(&ee_position) {
                return Err(shieldError::Projection(format!(
                    "end-effector in semantic exclusion zone ({})",
                    oid
                )));
            }
        }

        Ok(DynProposal {
            joint_positions: proposed_pos,
            joint_velocities: clamped_vel,
            ee_position,
            ee_orientation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shield_core::scene::SceneGraph;
    use shield_core::types::JointLimits;

    fn make_limits(ndof: usize) -> JointLimits {
        JointLimits {
            names: (0..ndof).map(|i| format!("joint_{i}")).collect(),
            position_min: vec![-3.14; ndof],
            position_max: vec![3.14; ndof],
            velocity_max: vec![1.0; ndof],
            acceleration_max: vec![5.0; ndof],
            torque_max: vec![50.0; ndof],
        }
    }

    #[test]
    fn clamp_within_limits() {
        let proj = KinematicClampProjector;
        let limits = make_limits(3);
        let scene = SceneGraph::default();
        let current = vec![0.0, 0.0, 0.0];
        let ctx = ProjectionContext {
            current_joints: &current,
            limits: &limits,
            scene: &scene,
            dt: 0.01,
            urdf_chain: None,
            forbidden_zones: &[],
            semantic_constraints: &[],
            prev_velocity: &[],
        };
        let action = ActionVector::new(0, 1, vec![0.5, -0.5, 0.0]);
        let prop = proj.project(&ctx, &action).unwrap();
        assert!((prop.joint_positions[0] - 0.005).abs() < 1e-9);
        assert!((prop.joint_positions[1] - (-0.005)).abs() < 1e-9);
    }

    #[test]
    fn velocity_clamping() {
        let proj = KinematicClampProjector;
        let limits = make_limits(2);
        let scene = SceneGraph::default();
        let current = vec![0.0, 0.0];
        let ctx = ProjectionContext {
            current_joints: &current,
            limits: &limits,
            scene: &scene,
            dt: 0.01,
            urdf_chain: None,
            forbidden_zones: &[],
            semantic_constraints: &[],
            prev_velocity: &[],
        };
        let action = ActionVector::new(0, 1, vec![999.0, -999.0]);
        let prop = proj.project(&ctx, &action).unwrap();
        assert!((prop.joint_velocities[0] - 1.0).abs() < 1e-9);
        assert!((prop.joint_velocities[1] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn dimension_mismatch() {
        let proj = KinematicClampProjector;
        let limits = make_limits(3);
        let scene = SceneGraph::default();
        let current = vec![0.0, 0.0, 0.0];
        let ctx = ProjectionContext {
            current_joints: &current,
            limits: &limits,
            scene: &scene,
            dt: 0.01,
            urdf_chain: None,
            forbidden_zones: &[],
            semantic_constraints: &[],
            prev_velocity: &[],
        };
        let action = ActionVector::new(0, 1, vec![0.5, -0.5]);
        assert!(proj.project(&ctx, &action).is_err());
    }

    #[test]
    fn acceleration_clamp_uses_prev_velocity() {
        let proj = KinematicClampProjector;
        let limits = make_limits(1);
        let scene = SceneGraph::default();
        let current = vec![0.0];
        let prev = [0.0];
        let ctx = ProjectionContext {
            current_joints: &current,
            limits: &limits,
            scene: &scene,
            dt: 0.01,
            urdf_chain: None,
            forbidden_zones: &[],
            semantic_constraints: &[],
            prev_velocity: &prev,
        };
        // a_max = 5, dt = 0.01 → |Δv| ≤ 0.05 even if the command is 999.
        let action = ActionVector::new(0, 1, vec![999.0]);
        let prop = proj.project(&ctx, &action).unwrap();
        assert!((prop.joint_velocities[0] - 0.05).abs() < 1e-9);
    }
}
