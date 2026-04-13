use redis::AsyncCommands;
use vlashield_core::arbiter::ArbiterDecision;

/// Redis client for real-time risk scores and telemetry streaming.
pub struct RedisClient {
    client: redis::Client,
}

impl RedisClient {
    pub fn new(url: &str) -> Result<Self, redis::RedisError> {
        let client = redis::Client::open(url)?;
        Ok(Self { client })
    }

    pub async fn connection(
        &self,
    ) -> Result<redis::aio::MultiplexedConnection, redis::RedisError> {
        self.client.get_multiplexed_async_connection().await
    }

    /// Publish the latest risk score (overwrites, 1s TTL).
    pub async fn set_risk_score(
        &self,
        robot_id: &str,
        score: f32,
    ) -> Result<(), redis::RedisError> {
        let mut conn = self.connection().await?;
        let key = format!("risk:{robot_id}");
        conn.set_ex(&key, score.to_string(), 1).await?;
        Ok(())
    }

    /// Publish arbiter state hash (5s TTL).
    pub async fn set_arbiter_state(
        &self,
        robot_id: &str,
        decision: &ArbiterDecision,
    ) -> Result<(), redis::RedisError> {
        let mut conn = self.connection().await?;
        let key = format!("arbiter:{robot_id}");
        let mode = if decision.is_pass() { "PASS" } else { "BLOCK" };
        let json = serde_json::to_string(decision).unwrap_or_default();
        redis::pipe()
            .hset(&key, "mode", mode)
            .hset(&key, "detail", &json)
            .expire(&key, 5)
            .exec_async(&mut conn)
            .await?;
        Ok(())
    }

    /// Append a telemetry entry to the Redis stream for WebSocket fan-out.
    pub async fn publish_telemetry(
        &self,
        robot_id: &str,
        payload: &str,
    ) -> Result<(), redis::RedisError> {
        let mut conn = self.connection().await?;
        let key = format!("stream:telemetry:{robot_id}");
        redis::cmd("XADD")
            .arg(&key)
            .arg("MAXLEN")
            .arg("~")
            .arg("10000")
            .arg("*")
            .arg("data")
            .arg(payload)
            .exec_async(&mut conn)
            .await?;
        Ok(())
    }
}
