use crate::semantic::SemanticConstraintMapper;
use crate::{DynProposal, PhysicalProjector, ProjectionContext};
use shield_core::action::ActionVector;
use shield_core::error::{Result, shieldError};

/// Conservative kinematic projector that clamps action deltas to joint limits.
///
/// This is the simplest projector: it treats `action.data` as joint-space velocity
/// commands and clamps them to `limits.velocity_max`, then integrates one step.
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
            let vel = raw_vel.clamp(-v_max, v_max);

            let pos = ctx.current_joints[i] + vel * ctx.dt;
            let pos = pos.clamp(ctx.limits.position_min[i], ctx.limits.position_max[i]);

            clamped_vel.push(vel);
            proposed_pos.push(pos);
        }

        let (ee_position, ee_orientation) = if let Some(chain) = ctx.urdf_chain {
            if chain.dof() != ndof {
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

        // Check URDF-derived Cartesian forbidden zones.
        for zone in ctx.forbidden_zones {
            if zone.contains(&ee_position) {
                return Err(shieldError::Projection(
                    "end-effector in forbidden Cartesian zone (PHY.FORBIDDEN_ZONE)".into(),
                ));
            }
        }

        // Check semantic exclusion zones (SEM.HEAT_SOURCE, SEM.FORBIDDEN_REGION, etc.).
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
        };
        let action = ActionVector::new(0, 1, vec![0.5, -0.5]);
        assert!(proj.project(&ctx, &action).is_err());
    }
}
