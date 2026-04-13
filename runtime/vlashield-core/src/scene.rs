use crate::types::Aabb;
use serde::{Deserialize, Serialize};

/// Geometric primitive for scene entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "primitive", rename_all = "snake_case")]
pub enum Primitive {
    Sphere { radius: f64 },
    Box { extents: [f64; 3] },
    Cylinder { radius: f64, height: f64 },
    Mesh { path: String },
}

/// A single entity in the scene graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEntity {
    pub id: String,
    pub primitive: Primitive,
    /// Pose as [x, y, z, qx, qy, qz, qw].
    pub pose: [f64; 7],
    pub aabb: Aabb,
    /// Semantic tags for semantic-risk queries (e.g. "fragile", "heat_source").
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Scene graph: a collection of static and dynamic entities.
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
