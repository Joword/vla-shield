//! Axis-aligned no-go boxes in Cartesian space (base or world).

/// Axis-aligned box, `[min, max]` per axis.
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

/// Closed AABB: on the face still counts as inside.
pub fn point_in_aabb(p: &[f64; 3], b: &AxisAlignedBox) -> bool {
    p[0] >= b.min[0]
        && p[0] <= b.max[0]
        && p[1] >= b.min[1]
        && p[1] <= b.max[1]
        && p[2] >= b.min[2]
        && p[2] <= b.max[2]
}
