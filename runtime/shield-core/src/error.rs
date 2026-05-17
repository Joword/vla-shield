use thiserror::Error;

#[derive(Debug, Error)]
pub enum shieldError {
    #[error("projection failed: {0}")]
    Projection(String),

    #[error("collision check failed: {0}")]
    Collision(String),

    #[error("semantic risk service unavailable: {0}")]
    SemanticUnavailable(String),

    #[error("arbiter timeout after {ms}ms")]
    Timeout { ms: u64 },

    #[error("invalid action dimension: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, shieldError>;
