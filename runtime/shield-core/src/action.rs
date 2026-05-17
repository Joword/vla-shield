use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Action vector at time `t` in the robot's command space.
///
/// The dimensionality is determined by the robot configuration and VLA model.
/// On the hot path we avoid heap allocation by pre-sizing `data`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionVector {
    /// Timestamp in nanoseconds (monotonic clock preferred).
    pub t_ns: u64,
    /// Monotonically increasing per-robot sequence counter.
    pub sequence_id: u64,
    /// Raw action data (joint deltas, EE deltas, or normalized VLA output).
    pub data: Vec<f32>,
    /// Identifier of the model that produced this action.
    #[serde(default)]
    pub model_id: String,
}

impl ActionVector {
    pub fn new(t_ns: u64, sequence_id: u64, data: Vec<f32>) -> Self {
        Self {
            t_ns,
            sequence_id,
            data,
            model_id: String::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// SHA-256 hash of the raw data bytes for dedup / integrity.
    pub fn hash_hex(&self) -> String {
        let bytes: Vec<u8> = self
            .data
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let digest = Sha256::digest(&bytes);
        format!("sha256:{:x}", digest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_vector_basic() {
        let a = ActionVector::new(1_000_000, 1, vec![0.1, 0.2, 0.3]);
        assert_eq!(a.dim(), 3);
        assert!(a.hash_hex().starts_with("sha256:"));
    }

    #[test]
    fn deterministic_hash() {
        let a = ActionVector::new(0, 0, vec![1.0, 2.0]);
        let b = ActionVector::new(999, 999, vec![1.0, 2.0]);
        assert_eq!(a.hash_hex(), b.hash_hex());
    }
}
