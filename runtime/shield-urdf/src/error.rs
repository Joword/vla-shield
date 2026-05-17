//! Error types for URDF loading and kinematics.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum UrdfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("XML parse error: {0}")]
    Xml(String),

    #[error("URDF structure error: {0}")]
    Structure(String),

    #[error("Joint angle dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}
