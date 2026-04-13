//! Axis-aligned forbidden zones in Cartesian space (base or world frame).

/// Axis-aligned box `[min, max]` per axis.
#[derive(Debug, Clone)]
pub struct AxisAlignedBox {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl AxisAlignedBox {
    pub fn contains(&self, p: &[f64; 3]) -> bool {
        point_in_aabb(p, self)
    }
}

/// True if point lies inside the closed AABB.
pub fn point_in_aabb(p: &[f64; 3], b: &AxisAlignedBox) -> bool {
    p[0] >= b.min[0]
        && p[0] <= b.max[0]
        && p[1] >= b.min[1]
        && p[1] <= b.max[1]
        && p[2] >= b.min[2]
        && p[2] <= b.max[2]
}
