use crate::types::Aabb;
use serde::{Deserialize, Serialize};

/// Shape of a scene entity. AABB is what the hot path actually uses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "primitive", rename_all = "snake_case")]
pub enum Primitive {
    Sphere { radius: f64 },
    Box { extents: [f64; 3] },
    Cylinder { radius: f64, height: f64 },
    Mesh { path: String },
}

/// One thing in the scene. `aabb` is the collision volume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEntity {
    pub id: String,
    pub primitive: Primitive,
    /// Pose: `[x, y, z, qx, qy, qz, qw]`.
    pub pose: [f64; 7],
    pub aabb: Aabb,
    /// Tags for SEM.* queries — `"fragile"`, `"heat_source"`, etc.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Scene: static + dynamic entities in one bag.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SceneGraph {
    pub frame_id: String,
    pub revision: u64,
    pub entities: Vec<SceneEntity>,
}

impl SceneGraph {
    pub fn entity_by_id(&self, id: &str) -> Option<&SceneEntity> {
        self.entities.iter().find(|e| e.id == id)
    }

    pub fn entities_with_tag(&self, tag: &str) -> Vec<&SceneEntity> {
        self.entities
            .iter()
            .filter(|e| e.tags.iter().any(|t| t == tag))
            .collect()
    }
}
