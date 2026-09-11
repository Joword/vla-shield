use redis::AsyncCommands;
use shield_core::arbiter::ArbiterDecision;

/// Redis: live risk scores + telemetry fan-out.
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

    /// Latest risk score. Overwrites, 1s TTL.
    pub async fn set_risk_score(
        &self,
        robot_id: &str,
        score: f32,
    ) -> Result<(), redis::RedisError> {
        let mut conn = self.connection().await?;
        let key = format!("risk:{robot_id}");
        conn.set_ex::<_, _, ()>(&key, score.to_string(), 1).await?;
        Ok(())
    }

    /// Arbiter state hash. 5s TTL.
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

    /// Append one telemetry row to the stream for WebSocket fan-out.
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
