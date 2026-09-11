use nalgebra::{Isometry3, Point3, Vector3};
use serde::{Deserialize, Serialize};

/// Pose: xyz + unit quaternion.
pub type Pose = Isometry3<f64>;

/// Axis-aligned box. Broad-phase lives on these.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    pub fn center(&self) -> Vector3<f64> {
        Vector3::new(
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        )
    }

    pub fn half_extents(&self) -> Vector3<f64> {
        Vector3::new(
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        )
    }

    /// Grow by `eps` on every side. Conservative on purpose.
    pub fn inflated(&self, eps: f64) -> Self {
        Self {
            min: [self.min[0] - eps, self.min[1] - eps, self.min[2] - eps],
            max: [self.max[0] + eps, self.max[1] + eps, self.max[2] + eps],
        }
    }

    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    /// Tight box around points. Empty iterator → `None`.
    pub fn from_points(pts: impl IntoIterator<Item = [f64; 3]>) -> Option<Self> {
        let mut iter = pts.into_iter();
        let first = iter.next()?;
        let mut min = first;
        let mut max = first;
        for p in iter {
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        Some(Self { min, max })
    }

    /// World AABB after a rigid transform. Corners only — a bit fat after rotation.
    pub fn transformed(&self, iso: &Isometry3<f64>) -> Self {
        let corners = [
            [self.min[0], self.min[1], self.min[2]],
            [self.min[0], self.min[1], self.max[2]],
            [self.min[0], self.max[1], self.min[2]],
            [self.min[0], self.max[1], self.max[2]],
            [self.max[0], self.min[1], self.min[2]],
            [self.max[0], self.min[1], self.max[2]],
            [self.max[0], self.max[1], self.min[2]],
            [self.max[0], self.max[1], self.max[2]],
        ];
        let pts = corners.map(|c| {
            let p = iso * Point3::new(c[0], c[1], c[2]);
            [p.x, p.y, p.z]
        });
        Self::from_points(pts).expect("8 corners")
    }

    /// Smallest box that covers both.
    pub fn union(&self, other: &Aabb) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// Capsule-ish box from origin to `xyz`, padded by `radius`.
    pub fn along_segment(xyz: [f64; 3], radius: f64) -> Self {
        Self {
            min: [
                xyz[0].min(0.0) - radius,
                xyz[1].min(0.0) - radius,
                xyz[2].min(0.0) - radius,
            ],
            max: [
                xyz[0].max(0.0) + radius,
                xyz[1].max(0.0) + radius,
                xyz[2].max(0.0) + radius,
            ],
        }
    }
}

/// Per-joint limits from the URDF (or whoever filled them in).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLimits {
    pub names: Vec<String>,
    pub position_min: Vec<f64>,
    pub position_max: Vec<f64>,
    pub velocity_max: Vec<f64>,
    pub acceleration_max: Vec<f64>,
    pub torque_max: Vec<f64>,
}

/// How hard we enforce. `Disabled` is an escape hatch — you have to opt in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Physics + collision + semantic. The real thing.
    Production,
    /// Skip VLM/LLM. Physics + collision only.
    PhysicsOnly,
    /// Log it, never block. For data collection.
    Monitor,
    /// Everything off. Explicit opt-in, don't leave this on.
    Disabled,
}

impl Default for RunMode {
    fn default() -> Self {
        Self::Production
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_intersection() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
        let c = Aabb::new([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn aabb_inflation() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let inflated = a.inflated(0.1);
        assert!((inflated.min[0] - (-0.1)).abs() < 1e-9);
        assert!((inflated.max[0] - 1.1).abs() < 1e-9);
    }

    #[test]
    fn aabb_transform_rotates_corners() {
        let a = Aabb::new([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]);
        let iso = Isometry3::from_parts(
            nalgebra::Translation3::new(1.0, 0.0, 0.0),
            nalgebra::UnitQuaternion::identity(),
        );
        let w = a.transformed(&iso);
        assert!((w.center().x - 1.0).abs() < 1e-9);
    }
}
