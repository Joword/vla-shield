//! Forward kinematics and a simple positional manipulability measure.

use nalgebra::{Isometry3, Rotation3, Translation3, Unit, UnitQuaternion, Vector3};

use crate::error::UrdfError;
use crate::urdf_loader::{JointSpec, UrdfRobot};

/// Ordered revolute chain from URDF (base link -> end-effector link).
#[derive(Debug, Clone)]
pub struct UrdfKinematicChain {
    joints: Vec<JointSpec>,
}

impl UrdfKinematicChain {
    /// Build chain from a parsed robot.
    pub fn from_robot(robot: &UrdfRobot, root_link: &str, ee_link: &str) -> Result<Self, UrdfError> {
        let joints = robot.chain_to(root_link, ee_link)?;
        Ok(Self { joints })
    }

    pub fn dof(&self) -> usize {
        self.joints.len()
    }

    /// End-effector isometry in the root link frame (same convention as ROS chain product).
    pub fn forward_isometry(&self, q: &[f64]) -> Result<Isometry3<f64>, UrdfError> {
        if q.len() != self.dof() {
            return Err(UrdfError::DimensionMismatch {
                expected: self.dof(),
                got: q.len(),
            });
        }
        let mut world = Isometry3::identity();
        for (i, j) in self.joints.iter().enumerate() {
            world *= joint_transform(j, q[i]);
        }
        Ok(world)
    }

    /// End-effector position `[x, y, z]` in the root link frame.
    pub fn ee_position(&self, q: &[f64]) -> Result<[f64; 3], UrdfError> {
        let iso = self.forward_isometry(q)?;
        let t = iso.translation.vector;
        Ok([t.x, t.y, t.z])
    }

    /// Unit quaternion `[x, y, z, w]` for end-effector orientation (root frame).
    pub fn ee_orientation_quat(&self, q: &[f64]) -> Result<[f64; 4], UrdfError> {
        let iso = self.forward_isometry(q)?;
        let q = iso.rotation.quaternion();
        Ok([q.i, q.j, q.k, q.w])
    }

    /// Positional manipulability `sqrt(det(J J^T))` using numerical Jacobian (3 x n).
    pub fn positional_manipulability(&self, q: &[f64]) -> Result<f64, UrdfError> {
        if q.len() != self.dof() {
            return Err(UrdfError::DimensionMismatch {
                expected: self.dof(),
                got: q.len(),
            });
        }
        let n = q.len();
        if n == 0 {
            return Ok(0.0);
        }
        let eps = 1e-5;
        let p0 = self.ee_position(q)?;
        let v0 = Vector3::new(p0[0], p0[1], p0[2]);
        let mut j_dyn = nalgebra::DMatrix::zeros(3, n);
        for j in 0..n {
            let mut qp = q.to_vec();
            qp[j] += eps;
            let p1 = self.ee_position(&qp)?;
            let v1 = Vector3::new(p1[0], p1[1], p1[2]);
            let col = (v1 - v0) / eps;
            j_dyn.set_column(j, &col);
        }
        let gram = &j_dyn * j_dyn.transpose();
        Ok(gram.determinant().max(0.0).sqrt())
    }
}

/// Default threshold below which the arm is treated as near a kinematic singularity.
pub const SINGULARITY_MANIPULABILITY_THRESHOLD: f64 = 1e-4;

fn joint_transform(j: &JointSpec, q: f64) -> Isometry3<f64> {
    let origin = isometry_from_xyz_rpy(j.origin_xyz, j.origin_rpy);
    let axis = Vector3::new(j.axis[0], j.axis[1], j.axis[2]);
    let axis = Unit::new_normalize(axis);
    let motion = Isometry3::from_parts(
        Translation3::identity(),
        UnitQuaternion::from_axis_angle(&axis, q),
    );
    origin * motion
}

fn isometry_from_xyz_rpy(xyz: [f64; 3], rpy: [f64; 3]) -> Isometry3<f64> {
    let t = Translation3::new(xyz[0], xyz[1], xyz[2]);
    let r = Rotation3::from_euler_angles(rpy[0], rpy[1], rpy[2]);
    Isometry3::from_parts(t, r.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::urdf_loader::UrdfRobot;

    const MINIMAL_ARM: &str = r#"<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="ee"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="link2"/>
    <child link="ee"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>
"#;

    #[test]
    fn fk_three_dof_planar() {
        let robot = UrdfRobot::from_str(MINIMAL_ARM).unwrap();
        let chain = UrdfKinematicChain::from_robot(&robot, "base", "ee").unwrap();
        let q = [0.0_f64, 0.0, 0.0];
        let p = chain.ee_position(&q).unwrap();
        assert!((p[0] - 1.5).abs() < 1e-9);
        assert!(p[1].abs() < 1e-9);
    }

    #[test]
    fn load_dataset_panda_urdf() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../dataset/urdf/panda_arm_simple.urdf");
        let robot = UrdfRobot::from_file(&path).expect("panda_arm_simple.urdf");
        let chain =
            UrdfKinematicChain::from_robot(&robot, "panda_link0", "panda_hand").expect("chain");
        assert_eq!(chain.dof(), 6);
        let _ = chain.ee_position(&[0.0; 6]).unwrap();
    }
}
