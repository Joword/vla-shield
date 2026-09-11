//! SEM.* labels → boxes / velocity caps the projector can actually enforce.
//!
//! "Heat source" isn't in the URDF. An operator (or VFV) injects a box at
//! runtime; "human nearby" becomes a speed cap. Same shape as the kinematic
//! constraints, just tagged SEM.* so BLOCK still beats CLAMP.

use shield_core::ontology::OntologyId;
use shield_urdf::AxisAlignedBox;

/// EE speed cap from a SEM.* node.
#[derive(Debug, Clone)]
pub struct VelocityCapConstraint {
    /// Max EE speed (m/s) while this is live.
    pub max_ee_speed_ms: f64,
    pub source_ontology_id: OntologyId,
}

/// Cartesian no-go box from a semantic annotation.
#[derive(Debug, Clone)]
pub struct SemanticZone {
    pub zone: AxisAlignedBox,
    pub label: String,
    pub source_ontology_id: OntologyId,
}

/// One SEM.* constraint the projector actually enforces.
#[derive(Debug, Clone)]
pub enum SemanticConstraint {
    /// Keep the EE out of this box.
    ExclusionZone(SemanticZone),
    /// Slow the EE down while this is live.
    VelocityCap(VelocityCapConstraint),
}

impl SemanticConstraint {
    pub fn ontology_id(&self) -> &OntologyId {
        match self {
            SemanticConstraint::ExclusionZone(z) => &z.source_ontology_id,
            SemanticConstraint::VelocityCap(v) => &v.source_ontology_id,
        }
    }

    /// `SEM.HEAT_SOURCE` box around `center`.
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

    /// `SEM.FORBIDDEN_REGION` — operator-drawn no-go.
    pub fn forbidden_region(zone: AxisAlignedBox, label: impl Into<String>) -> Self {
        SemanticConstraint::ExclusionZone(SemanticZone {
            zone,
            label: label.into(),
            source_ontology_id: OntologyId::new("SEM.FORBIDDEN_REGION"),
        })
    }

    /// `SEM.HUMAN_PROXIMITY` speed cap.
    pub fn human_proximity_cap(max_ee_speed_ms: f64) -> Self {
        SemanticConstraint::VelocityCap(VelocityCapConstraint {
            max_ee_speed_ms,
            source_ontology_id: OntologyId::new("SEM.HUMAN_PROXIMITY"),
        })
    }

    /// `SEM.LIQUID_ELECTRICAL` box. Don't drip on the electronics.
    pub fn liquid_electrical_zone(zone: AxisAlignedBox, label: impl Into<String>) -> Self {
        SemanticConstraint::ExclusionZone(SemanticZone {
            zone,
            label: label.into(),
            source_ontology_id: OntologyId::new("SEM.LIQUID_ELECTRICAL"),
        })
    }
}

/// Helpers to pull boxes / caps out of a constraint slice.
pub struct SemanticConstraintMapper<'a> {
    constraints: &'a [SemanticConstraint],
}

impl<'a> SemanticConstraintMapper<'a> {
    pub fn new(constraints: &'a [SemanticConstraint]) -> Self {
        SemanticConstraintMapper { constraints }
    }

    /// Live exclusion boxes + the ontology id that put them there.
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

    /// Tightest speed cap, or `None` if nobody asked.
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
