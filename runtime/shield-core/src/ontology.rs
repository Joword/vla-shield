use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Stable id, `DOMAIN.CODE` style — `PHY.COLLISION`, `SEM.HEAT_SOURCE`, etc.
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

/// One node in the ontology tree. `hard_block` means BLOCK, not a suggestion.
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

/// What the rule does: hard stop, clamp, or just yell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuleAction {
    Block,
    Clamp,
    Warn,
}

/// One rule from `dataset/ontology/rules_*.json`.
///
/// Maps a node to a trigger, a threshold blob, and `block | clamp | warn`.
/// `{placeholders}` in `explanation_template` get filled in at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleEntry {
    pub rule_id: OntologyId,
    pub trigger_condition: String,
    /// Threshold blob. Shape depends on the rule; it's just JSON.
    pub threshold: Value,
    pub action: RuleAction,
    pub severity: Severity,
    #[serde(default)]
    pub hard_block: bool,
    pub explanation_template: String,
    #[serde(default)]
    pub applies_to: Vec<String>,
    #[serde(default)]
    pub disabled: bool,
}

impl RuleEntry {
    /// Parse a JSON array of rules (the `rules_*.json` files).
    pub fn load_from_str(json: &str) -> Result<Vec<Self>, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Empty `applies_to` = every platform. Otherwise match the id.
    pub fn applies_to_platform(&self, platform: &str) -> bool {
        self.applies_to.is_empty() || self.applies_to.iter().any(|p| p == platform)
    }
}

/// PHY.* ids. Don't invent new strings — reuse these.
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
    pub fn velocity_limit() -> OntologyId {
        OntologyId::new("PHY.VELOCITY_LIMIT")
    }
    pub fn joint_limit() -> OntologyId {
        OntologyId::new("PHY.JOINT_LIMIT")
    }
    pub fn singularity() -> OntologyId {
        OntologyId::new("PHY.SINGULARITY")
    }
    pub fn forbidden_zone() -> OntologyId {
        OntologyId::new("PHY.FORBIDDEN_ZONE")
    }
}

/// SEM.* ids. Same deal: reuse, don't typo a new one.
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
    pub fn human_proximity() -> OntologyId {
        OntologyId::new("SEM.HUMAN_PROXIMITY")
    }
    pub fn sharp_object() -> OntologyId {
        OntologyId::new("SEM.SHARP_OBJECT")
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
