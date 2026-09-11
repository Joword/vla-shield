use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Command at time `t`. Length = robot DoF (or whatever the VLA spit out).
/// Pre-size `data` on the hot path so we don't realloc every tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionVector {
    /// Timestamp in ns. Prefer a monotonic clock.
    pub t_ns: u64,
    /// Per-robot seq. Must go up.
    pub sequence_id: u64,
    /// The numbers. Joint vel, EE delta, or whatever the model emits.
    pub data: Vec<f32>,
    /// Which model produced this. Empty is fine.
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

    /// SHA-256 of the payload bytes. Dedup / integrity, not a security boundary.
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
