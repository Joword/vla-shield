use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FfiError {
    #[error("dimension mismatch: expected {expected} dof, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("projection failed: {0}")]
    Projection(String),

    #[error("pipeline not configured: {0}")]
    NotConfigured(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("core error: {0}")]
    Core(#[from] shield_core::error::shieldError),
}

impl From<FfiError> for PyErr {
    fn from(e: FfiError) -> Self {
        PyRuntimeError::new_err(e.to_string())
    }
}
