//! Semantic constraint mapper: converts SEM.* ontology labels into
//! deterministic geometric or velocity constraints usable by the projector.
//!
//! # Design
//!
//! Semantic risks (e.g. "heat source", "human proximity") are inherently
//! scene-specific and cannot be encoded in a universal URDF.  This module
//! bridges the gap by allowing an operator or a VFV result to inject
//! runtime constraints that are physically comparable to the URDF-derived ones.
//!
//! For example, a "heat source detected at (0.4, 0.2, 0.8)" becomes an AABB
//! exclusion zone around that point; "human proximity" becomes a velocity cap.

use shield_core::ontology::OntologyId;
use shield_urdf::AxisAlignedBox;

/// A velocity cap constraint derived from a semantic risk node.
#[derive(Debug, Clone)]
pub struct VelocityCapConstraint {
    /// Maximum allowed EE speed (m/s) while the constraint is active.
    pub max_ee_speed_ms: f64,
    pub source_ontology_id: OntologyId,
}

/// A Cartesian exclusion zone derived from a semantic annotation.
#[derive(Debug, Clone)]
pub struct SemanticZone {
    pub zone: AxisAlignedBox,
    pub label: String,
    pub source_ontology_id: OntologyId,
}

/// A single semantic constraint that the projector enforces.
#[derive(Debug, Clone)]
pub enum SemanticConstraint {
    /// Exclude a Cartesian region from the reachable workspace.
    ExclusionZone(SemanticZone),
    /// Reduce maximum end-effector velocity while the constraint is active.
    VelocityCap(VelocityCapConstraint),
}

impl SemanticConstraint {
    pub fn ontology_id(&self) -> &OntologyId {
        match self {
            SemanticConstraint::ExclusionZone(z) => &z.source_ontology_id,
            SemanticConstraint::VelocityCap(v) => &v.source_ontology_id,
        }
    }

    /// Build an exclusion-zone constraint for `SEM.HEAT_SOURCE`.
    pub fn heat_source_zone(center: [f64; 3], radius: f64) -> Self {
        let zone = AxisAlignedBox {
            min: [center[0] - radius, center[1] - radius, center[2] - radius],
            max: [center[0] + radius, center[1] + radius, center[2] + radius],
        };
        SemanticConstraint::ExclusionZone(SemanticZone {
            zone,
            label: "heat_source".to_string(),
            source_ontology_id: OntologyId::new("SEM.HEAT_SOURCE"),
        })
    }

    /// Build an exclusion-zone constraint for `SEM.FORBIDDEN_REGION`.
    pub fn forbidden_region(zone: AxisAlignedBox, label: impl Into<String>) -> Self {
        SemanticConstraint::ExclusionZone(SemanticZone {
            zone,
            label: label.into(),
            source_ontology_id: OntologyId::new("SEM.FORBIDDEN_REGION"),
        })
    }

    /// Build a velocity-cap constraint for `SEM.HUMAN_PROXIMITY`.
    pub fn human_proximity_cap(max_ee_speed_ms: f64) -> Self {
        SemanticConstraint::VelocityCap(VelocityCapConstraint {
            max_ee_speed_ms,
            source_ontology_id: OntologyId::new("SEM.HUMAN_PROXIMITY"),
        })
    }

    /// Build an exclusion-zone constraint for `SEM.LIQUID_ELECTRICAL`.
    pub fn liquid_electrical_zone(zone: AxisAlignedBox, label: impl Into<String>) -> Self {
        SemanticConstraint::ExclusionZone(SemanticZone {
            zone,
            label: label.into(),
            source_ontology_id: OntologyId::new("SEM.LIQUID_ELECTRICAL"),
        })
    }
}

/// Maps a slice of `SemanticConstraint`s into extraction helpers for the projector.
pub struct SemanticConstraintMapper<'a> {
    constraints: &'a [SemanticConstraint],
}

impl<'a> SemanticConstraintMapper<'a> {
    pub fn new(constraints: &'a [SemanticConstraint]) -> Self {
        SemanticConstraintMapper { constraints }
    }

    /// Collect all active exclusion zones as `AxisAlignedBox` references.
    pub fn exclusion_zones(&self) -> Vec<(&AxisAlignedBox, &OntologyId)> {
        self.constraints
            .iter()
            .filter_map(|c| match c {
                SemanticConstraint::ExclusionZone(z) => {
                    Some((&z.zone, &z.source_ontology_id))
                }
                _ => None,
            })
            .collect()
    }

    /// Return the most restrictive velocity cap across all active constraints.
    /// Returns `None` when no velocity caps are active.
    pub fn effective_velocity_cap(&self) -> Option<f64> {
        self.constraints
            .iter()
            .filter_map(|c| match c {
                SemanticConstraint::VelocityCap(v) => Some(v.max_ee_speed_ms),
                _ => None,
            })
            .reduce(f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_source_zone_shape() {
        let c = SemanticConstraint::heat_source_zone([1.0, 0.0, 0.5], 0.2);
        match &c {
            SemanticConstraint::ExclusionZone(z) => {
                assert!((z.zone.min[0] - 0.8).abs() < 1e-9);
                assert!((z.zone.max[0] - 1.2).abs() < 1e-9);
            }
            _ => panic!("expected ExclusionZone"),
        }
    }

    #[test]
    fn velocity_cap_min_reduction() {
        let constraints = vec![
            SemanticConstraint::human_proximity_cap(0.3),
            SemanticConstraint::human_proximity_cap(0.5),
        ];
        let mapper = SemanticConstraintMapper::new(&constraints);
        assert_eq!(mapper.effective_velocity_cap(), Some(0.3));
    }

    #[test]
    fn no_caps_returns_none() {
        let constraints: Vec<SemanticConstraint> = vec![];
        let mapper = SemanticConstraintMapper::new(&constraints);
        assert!(mapper.effective_velocity_cap().is_none());
    }
}
