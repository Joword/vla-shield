use vlashield_core::arbiter::{ArbiterDecision, SafetyEvent};
use vlashield_core::types::RunMode;
use sqlx::mysql::MySqlPool;

/// MySQL-backed persistent store for safety events and action logs.
pub struct MySqlStore {
    pool: MySqlPool,
}

impl MySqlStore {
    pub async fn new(url: &str) -> Result<Self, sqlx::Error> {
        let pool = MySqlPool::connect(url).await?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &MySqlPool {
        &self.pool
    }

    pub async fn insert_action_log(
        &self,
        robot_id: &str,
        sequence_id: u64,
        t_ns: u64,
        decision: &str,
        risk_score: f32,
        model_id: &str,
        action_dim: usize,
        action_hash: &str,
        run_mode: RunMode,
        latency_total_ms: f64,
        latency_detail: &str,
        meta_json: &str,
    ) -> Result<(), sqlx::Error> {
        let mode_str = serde_json::to_value(&run_mode)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "production".into());
        sqlx::query(
            r#"
            INSERT INTO actions_log
                (robot_id, sequence_id, t_ns, decision, risk_score, model_id,
                 action_dim, action_hash, run_mode, latency_total_ms, latency_detail, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(robot_id)
        .bind(sequence_id as i64)
        .bind(t_ns as i64)
        .bind(decision)
        .bind(risk_score)
        .bind(model_id)
        .bind(action_dim as i32)
        .bind(action_hash)
        .bind(&mode_str)
        .bind(latency_total_ms)
        .bind(latency_detail)
        .bind(meta_json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn insert_safety_event(&self, event: &SafetyEvent) -> Result<(), sqlx::Error> {
        let payload = serde_json::to_string(event).unwrap_or_default();

        let (decision_str, risk_score, ontology_ids, reasons_json, latency_total) =
            match &event.decision {
                ArbiterDecision::Pass { latency, .. } => {
                    ("PASS", 0.0_f32, "[]".to_string(), "[]".to_string(), latency.total_ms)
                }
                ArbiterDecision::Block {
                    reasons, latency, ..
                } => {
                    let oids: Vec<&str> = reasons.iter().map(|r| r.ontology_id.0.as_str()).collect();
                    let max_score = reasons.iter().map(|r| r.score).fold(0.0_f32, f32::max);
                    (
                        "BLOCK",
                        max_score,
                        serde_json::to_string(&oids).unwrap_or_default(),
                        serde_json::to_string(reasons).unwrap_or_default(),
                        latency.total_ms,
                    )
                }
            };

        let mode_str = serde_json::to_value(&event.mode)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "production".into());

        sqlx::query(
            r#"
            INSERT INTO safety_events
                (id, robot_id, sequence_id, ts_ns, decision, risk_score,
                 action_hash, run_mode, ontology_ids, reasons, latency_total_ms, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&event.event_id)
        .bind(&event.robot_id)
        .bind(event.sequence_id as i64)
        .bind(event.ts_ns as i64)
        .bind(decision_str)
        .bind(risk_score)
        .bind(&event.action_hash)
        .bind(&mode_str)
        .bind(&ontology_ids)
        .bind(&reasons_json)
        .bind(latency_total)
        .bind(&payload)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
