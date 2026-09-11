use shield_core::ontology::Severity;
use shield_core::types::RunMode;
use serde::{Deserialize, Serialize};

/// Knobs for the runtime. Defaults are "a robot on localhost".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub robot_id: String,
    pub mode: RunMode,
    /// Min severity that triggers a hard block (inclusive).
    pub block_threshold: Severity,
    /// Semantic report older than this (ms) is stale → physics-only.
    pub semantic_staleness_ms: u64,
    /// AABB inflation. Conservative on purpose.
    pub collision_epsilon: f64,
    /// Control-loop period (seconds).
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
            mysql_url: "mysql://root:password@localhost:3306/shield".into(),
            redis_url: "redis://127.0.0.1:6379".into(),
        }
    }
}
