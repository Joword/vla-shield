//! Minimal URDF loader: revolute joints and link connectivity for serial chains.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use quick_xml::events::Event;
use quick_xml::Reader;

use crate::error::UrdfError;

/// One revolute joint along the kinematic chain (parent link -> child link).
#[derive(Debug, Clone)]
pub struct JointSpec {
    pub name: String,
    pub parent: String,
    pub child: String,
    /// Fixed transform from parent link frame to joint frame at q = 0 (xyz meters, rpy radians).
    pub origin_xyz: [f64; 3],
    pub origin_rpy: [f64; 3],
    /// Joint axis in joint frame (normalized).
    pub axis: [f64; 3],
    pub limit_lower: f64,
    pub limit_upper: f64,
}

/// Parsed robot with named joints (revolute only; fixed joints are skipped in chain build).
#[derive(Debug, Clone)]
pub struct UrdfRobot {
    pub name: String,
    pub joints: HashMap<String, JointSpec>,
    pub root_link: String,
}

impl UrdfRobot {
    /// Load URDF from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, UrdfError> {
        let text = fs::read_to_string(path.as_ref())?;
        Self::from_str(&text)
    }

    /// Parse URDF XML string.
    pub fn from_str(xml: &str) -> Result<Self, UrdfError> {
        parse_urdf(xml)
    }

    /// Build an ordered kinematic chain from `root_link` to `ee_link` using only revolute joints.
    pub fn chain_to(&self, root_link: &str, ee_link: &str) -> Result<Vec<JointSpec>, UrdfError> {
        chain_between(&self.joints, root_link, ee_link)
    }
}

fn parse_vec3(parts: &[&str]) -> Option<[f64; 3]> {
    if parts.len() != 3 {
        return None;
    }
    let x = parts[0].parse().ok()?;
    let y = parts[1].parse().ok()?;
    let z = parts[2].parse().ok()?;
    Some([x, y, z])
}

fn parse_urdf(xml: &str) -> Result<UrdfRobot, UrdfError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let mut robot_name: Option<String> = None;
    let mut joints: HashMap<String, JointSpec> = HashMap::new();
    let mut in_joint = false;
    let mut current: Option<JointBuilder> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"robot" => {
                for a in e.attributes() {
                    let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                    if a.key.as_ref() == b"name" {
                        robot_name = Some(
                            String::from_utf8_lossy(&a.value).trim().to_string(),
                        );
                    }
                }
            }
            Ok(Event::Start(ref e)) if e.name().as_ref() == b"joint" => {
                let mut name = String::new();
                let mut jtype = String::new();
                for a in e.attributes() {
                    let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                    match a.key.as_ref() {
                        b"name" => name = String::from_utf8_lossy(&a.value).into_owned(),
                        b"type" => jtype = String::from_utf8_lossy(&a.value).into_owned(),
                        _ => {}
                    }
                }
                if jtype == "revolute" || jtype == "continuous" {
                    in_joint = true;
                    current = Some(JointBuilder {
                        name,
                        parent: String::new(),
                        child: String::new(),
                        origin_xyz: [0.0; 3],
                        origin_rpy: [0.0; 3],
                        axis: [0.0, 0.0, 1.0],
                        limit_lower: -std::f64::consts::PI,
                        limit_upper: std::f64::consts::PI,
                    });
                } else {
                    in_joint = true;
                    current = None;
                }
            }
            Ok(Event::Start(ref e))
                if in_joint && current.is_some() && e.name().as_ref() == b"parent" =>
            {
                if let Some(ref mut j) = current {
                    for a in e.attributes() {
                        let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                        if a.key.as_ref() == b"link" {
                            j.parent = String::from_utf8_lossy(&a.value).into_owned();
                        }
                    }
                }
            }
            Ok(Event::Start(ref e))
                if in_joint && current.is_some() && e.name().as_ref() == b"child" =>
            {
                if let Some(ref mut j) = current {
                    for a in e.attributes() {
                        let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                        if a.key.as_ref() == b"link" {
                            j.child = String::from_utf8_lossy(&a.value).into_owned();
                        }
                    }
                }
            }
            Ok(Event::Start(ref e))
                if in_joint && current.is_some() && e.name().as_ref() == b"origin" =>
            {
                if let Some(ref mut j) = current {
                    for a in e.attributes() {
                        let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                        if a.key.as_ref() == b"xyz" {
                            let s = String::from_utf8_lossy(&a.value);
                            let p: Vec<&str> = s.split_whitespace().collect();
                            if let Some(v) = parse_vec3(&p) {
                                j.origin_xyz = v;
                            }
                        } else if a.key.as_ref() == b"rpy" {
                            let s = String::from_utf8_lossy(&a.value);
                            let p: Vec<&str> = s.split_whitespace().collect();
                            if let Some(v) = parse_vec3(&p) {
                                j.origin_rpy = v;
                            }
                        }
                    }
                }
            }
            Ok(Event::Start(ref e))
                if in_joint && current.is_some() && e.name().as_ref() == b"axis" =>
            {
                if let Some(ref mut j) = current {
                    for a in e.attributes() {
                        let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                        if a.key.as_ref() == b"xyz" {
                            let s = String::from_utf8_lossy(&a.value);
                            let p: Vec<&str> = s.split_whitespace().collect();
                            if let Some(v) = parse_vec3(&p) {
                                j.axis = v;
                            }
                        }
                    }
                }
            }
            Ok(Event::Start(ref e))
                if in_joint && current.is_some() && e.name().as_ref() == b"limit" =>
            {
                if let Some(ref mut j) = current {
                    for a in e.attributes() {
                        let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                        if a.key.as_ref() == b"lower" {
                            if let Ok(v) = String::from_utf8_lossy(&a.value).parse() {
                                j.limit_lower = v;
                            }
                        } else if a.key.as_ref() == b"upper" {
                            if let Ok(v) = String::from_utf8_lossy(&a.value).parse() {
                                j.limit_upper = v;
                            }
                        }
                    }
                }
            }
            Ok(Event::Empty(ref e)) if in_joint && current.is_some() && e.name().as_ref() == b"parent" => {
                if let Some(ref mut j) = current {
                    for a in e.attributes() {
                        let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                        if a.key.as_ref() == b"link" {
                            j.parent = String::from_utf8_lossy(&a.value).into_owned();
                        }
                    }
                }
            }
            Ok(Event::Empty(ref e)) if in_joint && current.is_some() && e.name().as_ref() == b"child" => {
                if let Some(ref mut j) = current {
                    for a in e.attributes() {
                        let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                        if a.key.as_ref() == b"link" {
                            j.child = String::from_utf8_lossy(&a.value).into_owned();
                        }
                    }
                }
            }
            Ok(Event::Empty(ref e)) if in_joint && current.is_some() && e.name().as_ref() == b"origin" => {
                if let Some(ref mut j) = current {
                    for a in e.attributes() {
                        let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                        if a.key.as_ref() == b"xyz" {
                            let s = String::from_utf8_lossy(&a.value);
                            let p: Vec<&str> = s.split_whitespace().collect();
                            if let Some(v) = parse_vec3(&p) {
                                j.origin_xyz = v;
                            }
                        } else if a.key.as_ref() == b"rpy" {
                            let s = String::from_utf8_lossy(&a.value);
                            let p: Vec<&str> = s.split_whitespace().collect();
                            if let Some(v) = parse_vec3(&p) {
                                j.origin_rpy = v;
                            }
                        }
                    }
                }
            }
            Ok(Event::Empty(ref e)) if in_joint && current.is_some() && e.name().as_ref() == b"axis" => {
                if let Some(ref mut j) = current {
                    for a in e.attributes() {
                        let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                        if a.key.as_ref() == b"xyz" {
                            let s = String::from_utf8_lossy(&a.value);
                            let p: Vec<&str> = s.split_whitespace().collect();
                            if let Some(v) = parse_vec3(&p) {
                                j.axis = v;
                            }
                        }
                    }
                }
            }
            Ok(Event::Empty(ref e)) if in_joint && current.is_some() && e.name().as_ref() == b"limit" => {
                if let Some(ref mut j) = current {
                    for a in e.attributes() {
                        let a = a.map_err(|e| UrdfError::Xml(e.to_string()))?;
                        if a.key.as_ref() == b"lower" {
                            if let Ok(v) = String::from_utf8_lossy(&a.value).parse() {
                                j.limit_lower = v;
                            }
                        } else if a.key.as_ref() == b"upper" {
                            if let Ok(v) = String::from_utf8_lossy(&a.value).parse() {
                                j.limit_upper = v;
                            }
                        }
                    }
                }
            }
            Ok(Event::End(ref e)) if e.name().as_ref() == b"joint" => {
                in_joint = false;
                if let Some(jb) = current.take() {
                    if jb.parent.is_empty() || jb.child.is_empty() {
                        continue;
                    }
                    let name = jb.name.clone();
                    let axis = nalgebra::Vector3::new(jb.axis[0], jb.axis[1], jb.axis[2]);
                    let n = axis.norm();
                    let axis = if n > 1e-12 {
                        [axis.x / n, axis.y / n, axis.z / n]
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    joints.insert(
                        name.clone(),
                        JointSpec {
                            name,
                            parent: jb.parent,
                            child: jb.child,
                            origin_xyz: jb.origin_xyz,
                            origin_rpy: jb.origin_rpy,
                            axis,
                            limit_lower: jb.limit_lower,
                            limit_upper: jb.limit_upper,
                        },
                    );
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(UrdfError::Xml(e.to_string())),
            _ => {}
        }
        buf.clear();
    }

    let name = robot_name.unwrap_or_else(|| "robot".to_string());
    let root_link = find_root_link(&joints)?;

    Ok(UrdfRobot {
        name,
        joints,
        root_link,
    })
}

struct JointBuilder {
    name: String,
    parent: String,
    child: String,
    origin_xyz: [f64; 3],
    origin_rpy: [f64; 3],
    axis: [f64; 3],
    limit_lower: f64,
    limit_upper: f64,
}

fn find_root_link(joints: &HashMap<String, JointSpec>) -> Result<String, UrdfError> {
    let mut children = std::collections::HashSet::new();
    for j in joints.values() {
        children.insert(j.child.clone());
    }
    let mut roots: Vec<&str> = joints
        .values()
        .map(|j| j.parent.as_str())
        .filter(|p| !children.contains(*p))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    roots.sort_unstable();
    if roots.len() != 1 {
        return Err(UrdfError::Structure(format!(
            "expected one root link, found {:?}",
            roots
        )));
    }
    Ok(roots[0].to_string())
}

/// Walk from ee backward to root; reverse to base -> tip.
fn chain_between(
    joints: &HashMap<String, JointSpec>,
    root: &str,
    ee: &str,
) -> Result<Vec<JointSpec>, UrdfError> {
    let mut by_child: HashMap<&str, &JointSpec> = HashMap::new();
    for j in joints.values() {
        by_child.insert(j.child.as_str(), j);
    }

    let mut ordered_rev: Vec<JointSpec> = Vec::new();
    let mut cur = ee;
    while cur != root {
        let j = by_child
            .get(cur)
            .ok_or_else(|| {
                UrdfError::Structure(format!("no joint found whose child is link '{cur}'"))
            })?;
        ordered_rev.push((*j).clone());
        cur = j.parent.as_str();
    }
    ordered_rev.reverse();
    Ok(ordered_rev)
}
