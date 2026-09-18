//! Coefficients shared with `dataset/ontology/phy_calibration.json`.
//!
//! Fitted to the bundled URDFs + gold PHY-004 / 006 / 007 — not a dynamometer
//! ID. `include_str!` so Python and Rust cannot drift without a test failing.

use serde::Deserialize;
use std::sync::OnceLock;

const RAW: &str = include_str!("../../../dataset/ontology/phy_calibration.json");

#[derive(Debug, Clone, Deserialize)]
pub struct PhyCalibration {
    pub singularity: SingularityCal,
    pub tipover: TipoverCal,
    pub overload: OverloadCal,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SingularityCal {
    pub jacobian_floor: f64,
    pub ontology_manipulability: f64,
    pub elbow_lock_rad: f64,
    pub elbow_lock_dof: usize,
    pub elbow_lock_joint_index: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TipoverCal {
    pub mobile_min_dof: usize,
    pub base_accel_limit: f64,
    pub com_height_m: f64,
    pub support_half_m: f64,
    pub margin_m: f64,
    pub g: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OverloadCal {
    pub distal_inertia: f64,
    pub proximal_inertia: f64,
    pub gravity_coeff: f64,
    pub distal_joints: usize,
}

fn parsed() -> &'static PhyCalibration {
    static CELL: OnceLock<PhyCalibration> = OnceLock::new();
    CELL.get_or_init(|| {
        serde_json::from_str(RAW).expect("phy_calibration.json must parse")
    })
}

pub fn phy_calibration() -> &'static PhyCalibration {
    parsed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_loads_and_matches_gold_numbers() {
        let c = phy_calibration();
        assert_eq!(c.singularity.elbow_lock_rad, 0.08);
        assert_eq!(c.singularity.elbow_lock_dof, 7);
        assert_eq!(c.tipover.base_accel_limit, 1.5);
        assert_eq!(c.overload.distal_inertia, 24.0);
        assert_eq!(c.overload.proximal_inertia, 2.0);
    }
}
