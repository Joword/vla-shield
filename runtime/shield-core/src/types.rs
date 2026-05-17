use nalgebra::{Isometry3, Vector3};
use serde::{Deserialize, Serialize};

/// 6-DOF pose (position + unit quaternion).
pub type Pose = Isometry3<f64>;

/// Axis-Aligned Bounding Box for broad-phase collision.
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

    /// Inflate the AABB by `eps` in all directions (conservative margin).
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
}

/// Per-joint kinematic limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLimits {
    pub names: Vec<String>,
    pub position_min: Vec<f64>,
    pub position_max: Vec<f64>,
    pub velocity_max: Vec<f64>,
    pub acceleration_max: Vec<f64>,
    pub torque_max: Vec<f64>,
}

/// Operating mode of the safety runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Full checking pipeline (physics + collision + semantic).
    Production,
    /// Physics + collision only; VLM/LLM skipped.
    PhysicsOnly,
    /// Log violations but never block (for data collection).
    Monitor,
    /// All checks disabled (escape hatch, requires explicit opt-in).
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
}
