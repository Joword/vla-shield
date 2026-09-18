//! Narrow-phase confirm + self-collision after the AABB mask.
//!
//! We don't have meshes. Confirm a broad-phase hit by re-testing slightly
//! deflated boxes. Self-collision skips adjacent links in chain order.

use shield_core::arbiter::CollisionPair;
use shield_core::types::Aabb;

/// How many chain neighbors to skip (2 = parent and grandparent).
/// Wrist AABBs on a serial arm sit inside each other's synthesized boxes.
pub const SELF_SKIP_ADJACENT: usize = 2;
/// Synthesized origin-to-next boxes graze at joints (`geom.rs` SYNTH_RADIUS).
const SELF_DEFLATE: f64 = 0.045;

/// Keep a broad-phase pair if the boxes still overlap after shrinking `deflate`.
pub fn confirm_aabb_hit(a: &Aabb, b: &Aabb, deflate: f64) -> bool {
    if deflate <= 0.0 {
        return a.intersects(b);
    }
    let a2 = Aabb {
        min: [a.min[0] + deflate, a.min[1] + deflate, a.min[2] + deflate],
        max: [a.max[0] - deflate, a.max[1] - deflate, a.max[2] - deflate],
    };
    let b2 = Aabb {
        min: [b.min[0] + deflate, b.min[1] + deflate, b.min[2] + deflate],
        max: [b.max[0] - deflate, b.max[1] - deflate, b.max[2] - deflate],
    };
    if a2.max[0] < a2.min[0] || b2.max[0] < b2.min[0] {
        return a.intersects(b);
    }
    a2.intersects(&b2)
}

/// Non-adjacent link vs link, uninflated boxes.
pub fn self_collision_pairs(boxes: &[(String, Aabb)]) -> Vec<CollisionPair> {
    let n = boxes.len();
    let mut pairs = Vec::new();
    if n < SELF_SKIP_ADJACENT + 2 {
        return pairs;
    }
    for i in 0..n {
        for j in (i + 1 + SELF_SKIP_ADJACENT)..n {
            if confirm_aabb_hit(&boxes[i].1, &boxes[j].1, SELF_DEFLATE) {
                let lc = boxes[i].1.center();
                let oc = boxes[j].1.center();
                pairs.push(CollisionPair {
                    link: boxes[i].0.clone(),
                    obstacle: boxes[j].0.clone(),
                    min_distance: (lc - oc).norm(),
                });
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_links_are_skipped() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
        let c = Aabb::new([0.4, 0.4, 0.4], [1.4, 1.4, 1.4]);
        let boxes = vec![("l0".into(), a), ("l1".into(), b), ("l2".into(), c)];
        assert!(self_collision_pairs(&boxes).is_empty());
    }

    #[test]
    fn nonadjacent_overlap_is_self_collision() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let far = Aabb::new([10.0, 10.0, 10.0], [11.0, 11.0, 11.0]);
        let c = Aabb::new([0.2, 0.2, 0.2], [0.8, 0.8, 0.8]);
        let boxes = vec![
            ("l0".into(), a),
            ("l1".into(), far),
            ("l2".into(), far),
            ("l3".into(), c),
        ];
        let pairs = self_collision_pairs(&boxes);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].link, "l0");
        assert_eq!(pairs[0].obstacle, "l3");
    }

    #[test]
    fn deflate_drops_grazing_overlap() {
        let a = Aabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = Aabb::new([0.99, 0.0, 0.0], [2.0, 1.0, 1.0]);
        assert!(confirm_aabb_hit(&a, &b, 0.0));
        assert!(!confirm_aabb_hit(&a, &b, 0.02));
    }
}
