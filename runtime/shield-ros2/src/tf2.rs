//! World-frame validation using a `world -> base_link` isometry (tf2-style).

use nalgebra::{Isometry3, Point3};
use shield_urdf::AxisAlignedBox;

/// Applies `world_from_base` so forbidden regions defined in **world** coordinates
/// can be checked against end-effector pose expressed in **base**.
#[derive(Debug, Clone)]
pub struct Tf2Validator {
    pub world_from_base: Isometry3<f64>,
    pub forbidden_in_world: Vec<AxisAlignedBox>,
}

impl Tf2Validator {
    pub fn new(world_from_base: Isometry3<f64>, forbidden_in_world: Vec<AxisAlignedBox>) -> Self {
        Self {
            world_from_base,
            forbidden_in_world,
        }
    }

    /// Transform a point from base frame to world frame.
    pub fn ee_in_world(&self, ee_base: &[f64; 3]) -> [f64; 3] {
        let p = Point3::new(ee_base[0], ee_base[1], ee_base[2]);
        let pw = self.world_from_base.transform_point(&p);
        [pw.x, pw.y, pw.z]
    }

    /// True if the EE position (base frame) lies inside any forbidden AABB in world frame.
    pub fn violates_forbidden_world(&self, ee_base: &[f64; 3]) -> bool {
        let w = self.ee_in_world(ee_base);
        self.forbidden_in_world.iter().any(|b| b.contains(&w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};

    #[test]
    fn identity_tf_matches_base() {
        let v = Tf2Validator::new(
            Isometry3::identity(),
            vec![AxisAlignedBox {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            }],
        );
        assert!(v.violates_forbidden_world(&[0.5, 0.5, 0.5]));
        assert!(!v.violates_forbidden_world(&[2.0, 2.0, 2.0]));
    }

    #[test]
    fn translation_moves_envelope() {
        let t = Translation3::new(10.0, 0.0, 0.0);
        let iso = Isometry3::from_parts(t, UnitQuaternion::identity());
        let v = Tf2Validator::new(
            iso,
            vec![AxisAlignedBox {
                min: [9.0, 0.0, 0.0],
                max: [11.0, 1.0, 1.0],
            }],
        );
        assert!(v.violates_forbidden_world(&[0.0, 0.0, 0.0]));
    }
}
