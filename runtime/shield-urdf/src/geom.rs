//! URDF `<collision>` / `<visual>` shapes → link-frame AABBs.

use std::collections::HashMap;

use nalgebra::{Isometry3, Rotation3, Translation3};
use quick_xml::events::Event;
use quick_xml::Reader;
use shield_core::types::Aabb;

use crate::error::UrdfError;
use crate::urdf_loader::JointSpec;

const SYNTH_RADIUS: f64 = 0.045;

#[derive(Clone, Copy)]
enum ShapeKind {
    Collision,
    Visual,
}

/// Prefer `<collision>` boxes/cylinders/spheres; fall back to `<visual>`.
/// Result is a conservative AABB in each **link** frame.
pub fn parse_link_aabbs(xml: &str) -> Result<HashMap<String, Aabb>, UrdfError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();

    let mut current_link = String::new();
    let mut shape: Option<ShapeKind> = None;
    let mut origin_xyz = [0.0; 3];
    let mut origin_rpy = [0.0; 3];
    let mut out: HashMap<String, Aabb> = HashMap::new();
    let mut from_collision: HashMap<String, bool> = HashMap::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let is_empty = matches!(
                    // Empty and Start both land here; End is a different arm.
                    e.name().as_ref(),
                    _
                );
                let _ = is_empty;
                match e.name().as_ref() {
                    b"link" => {
                        current_link.clear();
                        for a in e.attributes() {
                            let a = a.map_err(|err| UrdfError::Xml(err.to_string()))?;
                            if a.key.as_ref() == b"name" {
                                current_link = String::from_utf8_lossy(&a.value).into_owned();
                            }
                        }
                    }
                    b"collision" if !current_link.is_empty() => {
                        shape = Some(ShapeKind::Collision);
                        origin_xyz = [0.0; 3];
                        origin_rpy = [0.0; 3];
                    }
                    b"visual" if !current_link.is_empty() => {
                        shape = Some(ShapeKind::Visual);
                        origin_xyz = [0.0; 3];
                        origin_rpy = [0.0; 3];
                    }
                    b"origin" if shape.is_some() => {
                        for a in e.attributes() {
                            let a = a.map_err(|err| UrdfError::Xml(err.to_string()))?;
                            let s = String::from_utf8_lossy(&a.value);
                            let p: Vec<&str> = s.split_whitespace().collect();
                            if a.key.as_ref() == b"xyz" {
                                if let Some(v) = parse_vec3(&p) {
                                    origin_xyz = v;
                                }
                            } else if a.key.as_ref() == b"rpy" {
                                if let Some(v) = parse_vec3(&p) {
                                    origin_rpy = v;
                                }
                            }
                        }
                    }
                    b"box" if shape.is_some() => {
                        for a in e.attributes() {
                            let a = a.map_err(|err| UrdfError::Xml(err.to_string()))?;
                            if a.key.as_ref() == b"size" {
                                let s = String::from_utf8_lossy(&a.value);
                                let p: Vec<&str> = s.split_whitespace().collect();
                                if let Some(size) = parse_vec3(&p) {
                                    let local = Aabb::new(
                                        [-size[0] * 0.5, -size[1] * 0.5, -size[2] * 0.5],
                                        [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5],
                                    );
                                    absorb(
                                        &mut out,
                                        &mut from_collision,
                                        &current_link,
                                        local,
                                        origin_xyz,
                                        origin_rpy,
                                        matches!(shape, Some(ShapeKind::Collision)),
                                    );
                                }
                            }
                        }
                    }
                    b"cylinder" if shape.is_some() => {
                        let mut radius = 0.0;
                        let mut length = 0.0;
                        for a in e.attributes() {
                            let a = a.map_err(|err| UrdfError::Xml(err.to_string()))?;
                            if a.key.as_ref() == b"radius" {
                                radius = String::from_utf8_lossy(&a.value).parse().unwrap_or(0.0);
                            } else if a.key.as_ref() == b"length" {
                                length = String::from_utf8_lossy(&a.value).parse().unwrap_or(0.0);
                            }
                        }
                        if radius > 0.0 && length > 0.0 {
                            let local = Aabb::new(
                                [-radius, -radius, -length * 0.5],
                                [radius, radius, length * 0.5],
                            );
                            absorb(
                                &mut out,
                                &mut from_collision,
                                &current_link,
                                local,
                                origin_xyz,
                                origin_rpy,
                                matches!(shape, Some(ShapeKind::Collision)),
                            );
                        }
                    }
                    b"sphere" if shape.is_some() => {
                        for a in e.attributes() {
                            let a = a.map_err(|err| UrdfError::Xml(err.to_string()))?;
                            if a.key.as_ref() == b"radius" {
                                let r: f64 =
                                    String::from_utf8_lossy(&a.value).parse().unwrap_or(0.0);
                                if r > 0.0 {
                                    let local = Aabb::new([-r, -r, -r], [r, r, r]);
                                    absorb(
                                        &mut out,
                                        &mut from_collision,
                                        &current_link,
                                        local,
                                        origin_xyz,
                                        origin_rpy,
                                        matches!(shape, Some(ShapeKind::Collision)),
                                    );
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => match e.name().as_ref() {
                b"link" => current_link.clear(),
                b"collision" | b"visual" => shape = None,
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => return Err(UrdfError::Xml(e.to_string())),
            _ => {}
        }
        buf.clear();
    }
    Ok(out)
}

fn absorb(
    out: &mut HashMap<String, Aabb>,
    from_collision: &mut HashMap<String, bool>,
    link: &str,
    geom_aabb: Aabb,
    origin_xyz: [f64; 3],
    origin_rpy: [f64; 3],
    is_collision: bool,
) {
    if link.is_empty() {
        return;
    }
    let iso = isometry_from_xyz_rpy(origin_xyz, origin_rpy);
    let in_link = geom_aabb.transformed(&iso);
    match out.get(link) {
        None => {
            out.insert(link.to_string(), in_link);
            from_collision.insert(link.to_string(), is_collision);
        }
        Some(existing) => {
            let had_collision = *from_collision.get(link).unwrap_or(&false);
            if is_collision && !had_collision {
                out.insert(link.to_string(), in_link);
                from_collision.insert(link.to_string(), true);
            } else if is_collision == had_collision {
                out.insert(link.to_string(), existing.union(&in_link));
            }
        }
    }
}

fn parse_vec3(parts: &[&str]) -> Option<[f64; 3]> {
    if parts.len() != 3 {
        return None;
    }
    Some([
        parts[0].parse().ok()?,
        parts[1].parse().ok()?,
        parts[2].parse().ok()?,
    ])
}

fn isometry_from_xyz_rpy(xyz: [f64; 3], rpy: [f64; 3]) -> Isometry3<f64> {
    let t = Translation3::new(xyz[0], xyz[1], xyz[2]);
    let r = Rotation3::from_euler_angles(rpy[0], rpy[1], rpy[2]);
    Isometry3::from_parts(t, r.into())
}

/// No collision meshes? Fake a capsule-ish AABB along each joint origin so
/// broad-phase still has volume to chew on.
///
/// A link is the union of segments to *all* its kids, so HashMap iteration
/// order can't change the envelope. Branching robots (grippers, dual arms)
/// must not get a different box run to run.
pub fn synthesize_link_aabbs(joints: &HashMap<String, JointSpec>) -> HashMap<String, Aabb> {
    let mut by_parent: HashMap<&str, Vec<&JointSpec>> = HashMap::new();
    for j in joints.values() {
        by_parent.entry(j.parent.as_str()).or_default().push(j);
    }

    let mut out: HashMap<String, Aabb> = HashMap::new();
    let absorb_span = |out: &mut HashMap<String, Aabb>, link: &str, span: [f64; 3]| {
        let box_ = Aabb::along_segment(span, SYNTH_RADIUS);
        match out.get(link) {
            Some(existing) => {
                let merged = existing.union(&box_);
                out.insert(link.to_string(), merged);
            }
            None => {
                out.insert(link.to_string(), box_);
            }
        }
    };

    for j in joints.values() {
        absorb_span(&mut out, &j.parent, j.origin_xyz);
        match by_parent.get(j.child.as_str()) {
            Some(kids) if !kids.is_empty() => {
                for kid in kids {
                    absorb_span(&mut out, &j.child, kid.origin_xyz);
                }
            }
            // Distal link: stub a short forward segment so the tip isn't a point.
            _ => absorb_span(&mut out, &j.child, [0.06, 0.0, 0.0]),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint(name: &str, parent: &str, child: &str, xyz: [f64; 3]) -> (String, JointSpec) {
        (
            name.to_string(),
            JointSpec {
                name: name.to_string(),
                parent: parent.to_string(),
                child: child.to_string(),
                origin_xyz: xyz,
                origin_rpy: [0.0; 3],
                axis: [0.0, 0.0, 1.0],
                limit_lower: -3.14,
                limit_upper: 3.14,
            },
        )
    }

    /// Two kids → cover both, no matter which HashMap dumps first.
    #[test]
    fn branching_parent_covers_every_child() {
        let joints: HashMap<String, JointSpec> = [
            joint("j1", "wrist", "finger_a", [0.1, 0.0, 0.0]),
            joint("j2", "wrist", "finger_b", [0.0, -0.2, 0.0]),
        ]
        .into_iter()
        .collect();

        let out = synthesize_link_aabbs(&joints);
        let wrist = out.get("wrist").expect("wrist box");
        assert!(wrist.max[0] >= 0.1 - f64::EPSILON, "missing finger_a span");
        assert!(wrist.min[1] <= -0.2 + f64::EPSILON, "missing finger_b span");
    }
}

/// Parsed geom wins per link; synthesized boxes fill the gaps.
pub fn merge_geoms(
    parsed: HashMap<String, Aabb>,
    joints: &HashMap<String, JointSpec>,
) -> HashMap<String, Aabb> {
    let mut out = synthesize_link_aabbs(joints);
    for (k, v) in parsed {
        out.insert(k, v);
    }
    out
}
