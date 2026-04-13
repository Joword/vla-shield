use serde::{Deserialize, Serialize};

/// Stable ontology identifier following `DOMAIN.CODE` pattern (e.g. `PHY.COLLISION`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OntologyId(pub String);

impl OntologyId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn domain(&self) -> &str {
        self.0.split('.').next().unwrap_or(&self.0)
    }

    pub fn is_physical(&self) -> bool {
        self.domain() == "PHY"
    }

    pub fn is_semantic(&self) -> bool {
        self.domain() == "SEM"
    }
}

impl std::fmt::Display for OntologyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// A single node in the safety ontology tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyNode {
    pub id: OntologyId,
    pub severity: Severity,
    pub hard_block: bool,
    pub title: String,
    pub description: String,
    #[serde(default)]
    pub parents: Vec<OntologyId>,
}

/// Physical safety ontology constants.
pub mod physical {
    use super::OntologyId;

    pub fn collision() -> OntologyId {
        OntologyId::new("PHY.COLLISION")
    }
    pub fn tipover() -> OntologyId {
        OntologyId::new("PHY.TIPOVER")
    }
    pub fn overload() -> OntologyId {
        OntologyId::new("PHY.OVERLOAD")
    }
}

/// Semantic safety ontology constants.
pub mod semantic {
    use super::OntologyId;

    pub fn fragile() -> OntologyId {
        OntologyId::new("SEM.FRAGILE")
    }
    pub fn heat_source() -> OntologyId {
        OntologyId::new("SEM.HEAT_SOURCE")
    }
    pub fn forbidden_region() -> OntologyId {
        OntologyId::new("SEM.FORBIDDEN_REGION")
    }
    pub fn liquid_electrical() -> OntologyId {
        OntologyId::new("SEM.LIQUID_ELECTRICAL")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ontology_id_domain_parsing() {
        let id = OntologyId::new("PHY.COLLISION");
        assert_eq!(id.domain(), "PHY");
        assert!(id.is_physical());
        assert!(!id.is_semantic());
    }

    #[test]
    fn severity_ordering() {
        assert!(Severity::Info < Severity::Critical);
        assert!(Severity::Medium < Severity::High);
    }

    #[test]
    fn node_serialization_roundtrip() {
        let node = OntologyNode {
            id: physical::collision(),
            severity: Severity::High,
            hard_block: true,
            title: "Collision".into(),
            description: "Imminent link-object or object-human impact".into(),
            parents: vec![],
        };
        let json = serde_json::to_string(&node).unwrap();
        let restored: OntologyNode = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.id, node.id);
        assert!(restored.hard_block);
    }
}
