//! FK plus a cheap positional manipulability number.

use std::collections::HashMap;

use nalgebra::{Isometry3, Rotation3, Translation3, Unit, UnitQuaternion, Vector3};
use shield_core::types::Aabb;

use crate::error::UrdfError;
use crate::urdf_loader::{JointSpec, UrdfRobot};

/// Ordered revolute chain: base link → EE.
#[derive(Debug, Clone)]
pub struct UrdfKinematicChain {
    joints: Vec<JointSpec>,
    /// Link-frame AABBs keyed by name (root + every child on the chain).
    link_aabbs: HashMap<String, Aabb>,
}

impl UrdfKinematicChain {
    /// Build from a parsed robot.
    pub fn from_robot(robot: &UrdfRobot, root_link: &str, ee_link: &str) -> Result<Self, UrdfError> {
        let joints = robot.chain_to(root_link, ee_link)?;
        Ok(Self {
            joints,
            link_aabbs: robot.link_aabbs.clone(),
        })
    }

    pub fn dof(&self) -> usize {
        self.joints.len()
    }

    pub fn joints(&self) -> &[JointSpec] {
        &self.joints
    }

    pub fn root_link(&self) -> Option<&str> {
        self.joints.first().map(|j| j.parent.as_str())
    }

    pub fn ee_link(&self) -> Option<&str> {
        self.joints.last().map(|j| j.child.as_str())
    }

    /// Link frames from root (identity) through each child, in chain order.
    pub fn link_frames(&self, q: &[f64]) -> Result<Vec<(String, Isometry3<f64>)>, UrdfError> {
        if q.len() != self.dof() {
            return Err(UrdfError::DimensionMismatch {
                expected: self.dof(),
                got: q.len(),
            });
        }
        let mut frames = Vec::with_capacity(self.joints.len() + 1);
        let mut world = Isometry3::identity();
        if let Some(root) = self.root_link() {
            frames.push((root.to_string(), world));
        }
        for (i, j) in self.joints.iter().enumerate() {
            world *= joint_transform(j, q[i]);
            frames.push((j.child.clone(), world));
        }
        Ok(frames)
    }

    /// Waypoints `[root, j1_child, …, ee]` in the root frame.
    pub fn skeleton(&self, q: &[f64]) -> Result<Vec<[f64; 3]>, UrdfError> {
        Ok(self
            .link_frames(q)?
            .into_iter()
            .map(|(_, iso)| {
                let t = iso.translation.vector;
                [t.x, t.y, t.z]
            })
            .collect())
    }

    /// World AABBs for every link that actually has geometry.
    pub fn link_world_aabbs(&self, q: &[f64]) -> Result<Vec<(String, Aabb)>, UrdfError> {
        let frames = self.link_frames(q)?;
        let mut out = Vec::with_capacity(frames.len());
        for (name, iso) in frames {
            if let Some(local) = self.link_aabbs.get(&name) {
                out.push((name, local.transformed(&iso)));
            }
        }
        Ok(out)
    }

    /// EE isometry in the root link frame. Same product ROS uses.
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

    /// EE xyz in the root link frame.
    pub fn ee_position(&self, q: &[f64]) -> Result<[f64; 3], UrdfError> {
        let iso = self.forward_isometry(q)?;
        let t = iso.translation.vector;
        Ok([t.x, t.y, t.z])
    }

    /// Unit quat `[x, y, z, w]` for EE orientation (root frame).
    pub fn ee_orientation_quat(&self, q: &[f64]) -> Result<[f64; 4], UrdfError> {
        let iso = self.forward_isometry(q)?;
        let q = iso.rotation.quaternion();
        Ok([q.i, q.j, q.k, q.w])
    }

    /// Positional manipulability `sqrt(det(J J^T))` via a numerical 3×n Jacobian.
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

/// Below this, treat the arm as near a singularity.
/// Matches `dataset/ontology/rules_physical.json` (`min_manipulability`: 0.05).
pub const SINGULARITY_MANIPULABILITY_THRESHOLD: f64 = 0.05;

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
        assert_eq!(chain.dof(), 7);
        let _ = chain.ee_position(&[0.0; 7]).unwrap();
        let skel = chain.skeleton(&[0.0; 7]).unwrap();
        assert_eq!(skel.len(), 8);
        let aabbs = chain.link_world_aabbs(&[0.0; 7]).unwrap();
        assert!(!aabbs.is_empty());
    }

    #[test]
    fn panda_elbow_lock_has_low_manipulability() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../dataset/urdf/panda_arm_simple.urdf");
        let robot = UrdfRobot::from_file(&path).expect("panda");
        let chain =
            UrdfKinematicChain::from_robot(&robot, "panda_link0", "panda_hand").expect("chain");
        let stretched = chain
            .positional_manipulability(&[0.0, 0.0, 0.0, 0.02, 0.0, 0.02, 0.0])
            .unwrap();
        let folded = chain
            .positional_manipulability(&[0.0, 0.3, 0.0, -1.5, 0.0, 1.5, 0.0])
            .unwrap();
        assert!(
            stretched < folded,
            "stretched={stretched} folded={folded}"
        );
    }

    #[test]
    fn ur5_link_aabbs_move_with_q() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../dataset/urdf/ur5_simple.urdf");
        let robot = UrdfRobot::from_file(&path).expect("ur5");
        let chain =
            UrdfKinematicChain::from_robot(&robot, "base_link", "wrist_3_link").expect("chain");
        let z = chain.link_world_aabbs(&[0.0; 6]).unwrap();
        let spun = chain.link_world_aabbs(&[1.57, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(z.len(), spun.len());
        let ee_z = z.last().unwrap().1.center();
        let ee_s = spun.last().unwrap().1.center();
        let dx = (ee_z.x - ee_s.x).abs();
        let dy = (ee_z.y - ee_s.y).abs();
        assert!(dx + dy > 0.05, "yaw should move the distal AABB");
    }
}
