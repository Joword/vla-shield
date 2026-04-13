use vlashield_core::ontology::Severity;
use vlashield_core::types::RunMode;
use serde::{Deserialize, Serialize};

/// Top-level runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub robot_id: String,
    pub mode: RunMode,
    /// Minimum severity to trigger a hard block (inclusive).
    pub block_threshold: Severity,
    /// Maximum allowed age (ms) for a semantic risk report before it is
    /// considered stale and the arbiter falls back to physics-only mode.
    pub semantic_staleness_ms: u64,
    /// Conservative inflation factor for AABB collision checks.
    pub collision_epsilon: f64,
    /// Control loop period (seconds).
    pub dt: f64,
    pub mysql_url: String,
    pub redis_url: String,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            robot_id: "default-robot".into(),
            mode: RunMode::Production,
            block_threshold: Severity::High,
            semantic_staleness_ms: 200,
            collision_epsilon: 0.02,
            dt: 0.01,
            mysql_url: "mysql://root:password@localhost:3306/vlashield".into(),
            redis_url: "redis://127.0.0.1:6379".into(),
        }
    }
}
