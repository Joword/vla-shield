//! `shield-ffi` — PyO3 bridge exposing the VLA-Shield hot-path pipeline to Python.
//!
//! # Architecture
//!
//! ```text
//! Python VLA model
//!      │  action: list[float]  (or numpy array cast to list)
//!      ▼
//! PyShieldPipeline.evaluate(action, t_ns, sequence_id)
//!      │  (PyO3 FFI — copy-based; zero-copy variant via buffer protocol is planned)
//!      ▼
//! Rust: KinematicClampProjector → AabbBroadPhase → inline arbiter
//!      │
//!      ▼
//! PyDecision { decision: "PASS"|"BLOCK", reasons: [...], latency: {...} }
//! ```
//!
//! # Zero-copy note
//! Full GPU zero-copy (sharing `tensor.data_ptr()`) requires a CUDA-aware
//! allocator and pinned-memory contract between Python and Rust.  The current
//! implementation copies the action slice across the FFI boundary.  A
//! `zero_copy` feature flag is reserved for the CUDA extension module.

pub mod convert;
pub mod error;

use convert::{make_joint_limits, vec_to_action, PyDecisionSummary};
use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::Mutex;
use shield_collision::broad_phase::AabbBroadPhase;
use shield_collision::{CollisionContext, CollisionPrechecker};
use shield_core::arbiter::{ArbiterDecision, ArbiterReason, LatencyBreakdown, SemanticRiskReport};
use shield_core::ontology::physical;
use shield_core::scene::SceneGraph;
use shield_core::types::{JointLimits, RunMode};
use shield_physics::projection::KinematicClampProjector;
use shield_physics::{PhysicalProjector, ProjectionContext};
#[cfg(feature = "cuda")]
use shield_cuda::CudaCtx;

/// Python-visible decision result.
#[pyclass(name = "Decision")]
#[derive(Debug, Clone)]
pub struct PyDecision {
    #[pyo3(get)]
    pub decision: String,
    /// List of (ontology_id, detail, score) tuples.
    #[pyo3(get)]
    pub reasons: Vec<(String, String, f32)>,
    /// Latency breakdown as a dict.
    pub latency_raw: LatencyBreakdown,
}

#[pymethods]
impl PyDecision {
    fn latency<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ingest_ms", self.latency_raw.ingest_ms)?;
        d.set_item("urdf_fk_ms", self.latency_raw.urdf_fk_ms)?;
        d.set_item("physics_ms", self.latency_raw.physics_ms)?;
        d.set_item("collision_ms", self.latency_raw.collision_ms)?;
        d.set_item("tf2_ms", self.latency_raw.tf2_ms)?;
        d.set_item("arbiter_ms", self.latency_raw.arbiter_ms)?;
        d.set_item("shadow_ms", self.latency_raw.shadow_ms)?;
        d.set_item("total_ms", self.latency_raw.total_ms)?;
        Ok(d)
    }

    fn is_pass(&self) -> bool {
        self.decision == "PASS"
    }

    fn is_block(&self) -> bool {
        self.decision == "BLOCK"
    }

    fn __repr__(&self) -> String {
        format!(
            "Decision(decision={:?}, reasons={}, total_ms={:.3})",
            self.decision,
            self.reasons.len(),
            self.latency_raw.total_ms,
        )
    }
}

impl From<PyDecisionSummary> for PyDecision {
    fn from(s: PyDecisionSummary) -> Self {
        PyDecision {
            decision: s.decision.to_string(),
            reasons: s.reasons,
            latency_raw: s.latency,
        }
    }
}

/// Python-visible shield pipeline.
///
/// ```python
/// from shield_ffi import ShieldPipeline
///
/// pipeline = ShieldPipeline(
///     joint_names=["j1","j2","j3"],
///     position_min=[-3.14]*3,
///     position_max=[3.14]*3,
///     velocity_max=[1.0]*3,
///     dt=0.01,
///     collision_epsilon=0.02,
/// )
/// result = pipeline.evaluate(
///     action=[0.1, -0.2, 0.05],
///     current_joints=[0.0, 0.0, 0.0],
///     t_ns=0,
///     sequence_id=1,
/// )
/// print(result.decision)  # "PASS" or "BLOCK"
/// ```
#[pyclass(name = "ShieldPipeline")]
pub struct PyShieldPipeline {
    limits: Arc<JointLimits>,
    /// Pre-computed `velocity_max` cast to f32, matching the dtype consumed
    /// by the CUDA / CPU clamp backend.  Cached once at construction so we
    /// do not pay the cast cost on every `evaluate` call.
    #[cfg(feature = "cuda")]
    velocity_max_f32: Arc<Vec<f32>>,
    projector: KinematicClampProjector,
    checker: AabbBroadPhase,
    dt: f64,
    collision_epsilon: f64,
    /// Per-pipeline CUDA context owning cached device buffers, pinned host
    /// staging buffers, and a private CUDA stream.  Held behind a `Mutex`
    /// because `clamp_into` mutates the cached buffers and PyO3 invokes
    /// methods through `&self`.
    #[cfg(feature = "cuda")]
    cuda_ctx: Mutex<CudaCtx>,
}

#[pymethods]
impl PyShieldPipeline {
    #[new]
    #[pyo3(signature = (
        joint_names,
        position_min,
        position_max,
        velocity_max,
        dt = 0.01,
        collision_epsilon = 0.02,
        acceleration_max = vec![],
        torque_max = vec![],
    ))]
    fn new(
        joint_names: Vec<String>,
        position_min: Vec<f64>,
        position_max: Vec<f64>,
        velocity_max: Vec<f64>,
        dt: f64,
        collision_epsilon: f64,
        acceleration_max: Vec<f64>,
        torque_max: Vec<f64>,
    ) -> PyResult<Self> {
        let limits = make_joint_limits(
            joint_names,
            position_min,
            position_max,
            velocity_max,
            acceleration_max,
            torque_max,
        );
        #[cfg(feature = "cuda")]
        let velocity_max_f32: Vec<f32> =
            limits.velocity_max.iter().map(|v| *v as f32).collect();
        #[cfg(feature = "cuda")]
        let cuda_ctx = Mutex::new(
            CudaCtx::new(limits.names.len()).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "shield-cuda context init failed: {e}"
                ))
            })?,
        );
        Ok(PyShieldPipeline {
            limits: Arc::new(limits),
            #[cfg(feature = "cuda")]
            velocity_max_f32: Arc::new(velocity_max_f32),
            projector: KinematicClampProjector,
            checker: AabbBroadPhase,
            dt,
            collision_epsilon,
            #[cfg(feature = "cuda")]
            cuda_ctx,
        })
    }

    /// Evaluate a single action vector.  Returns a `Decision` object.
    ///
    /// Parameters
    /// ----------
    /// action : list[float] | numpy.ndarray[float32]
    ///     Raw VLA action vector (joint-space velocities by default).
    ///     Lists are converted to a `Vec<f32>` by PyO3 (one allocation, scalar
    ///     iteration).  Prefer :py:meth:`evaluate_numpy` when the caller
    ///     already holds a contiguous `numpy.ndarray` to avoid that copy.
    /// current_joints : list[float] | numpy.ndarray[float64]
    ///     Current joint positions in radians.
    /// t_ns : int
    ///     Action timestamp in nanoseconds.
    /// sequence_id : int
    ///     Monotonic per-robot action counter.
    #[pyo3(signature = (action, current_joints, t_ns = 0, sequence_id = 0))]
    fn evaluate(
        &self,
        action: Vec<f32>,
        current_joints: Vec<f64>,
        t_ns: u64,
        sequence_id: u64,
    ) -> PyResult<PyDecision> {
        self.evaluate_impl(&action, &current_joints, t_ns, sequence_id)
    }

    /// Zero-copy evaluate that borrows directly from contiguous numpy arrays.
    ///
    /// Saves the per-call Python list → `Vec` conversion that ``evaluate``
    /// pays.  Requires both inputs to be **contiguous** ndarrays of the
    /// expected dtype (``float32`` for ``action``, ``float64`` for
    /// ``current_joints``); non-contiguous or wrong-dtype inputs raise.
    ///
    /// In typical VLA hot paths this saves ~1–3 µs per call (mostly on
    /// ``current_joints`` which is a 6–14-element f64 list).
    #[pyo3(signature = (action, current_joints, t_ns = 0, sequence_id = 0))]
    fn evaluate_numpy(
        &self,
        action: PyReadonlyArray1<'_, f32>,
        current_joints: PyReadonlyArray1<'_, f64>,
        t_ns: u64,
        sequence_id: u64,
    ) -> PyResult<PyDecision> {
        let action_slice = action.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "evaluate_numpy: action must be a contiguous numpy.ndarray[float32]",
            )
        })?;
        let current_slice = current_joints.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "evaluate_numpy: current_joints must be a contiguous numpy.ndarray[float64]",
            )
        })?;
        self.evaluate_impl(action_slice, current_slice, t_ns, sequence_id)
    }

    fn __repr__(&self) -> String {
        format!(
            "ShieldPipeline(dof={}, dt={}, epsilon={})",
            self.limits.names.len(),
            self.dt,
            self.collision_epsilon
        )
    }
}

impl PyShieldPipeline {
    /// Backend evaluation routine shared by `evaluate` and `evaluate_numpy`.
    ///
    /// * `action_in`     – raw, **unclamped** action commanded by the VLA.
    /// * `current_joints`– current joint positions used for one-step projection.
    ///
    /// Pre-detection runs on the unclamped input so the arbiter can surface
    /// `PHY.VELOCITY_LIMIT` / `PHY.JOINT_LIMIT` even when the CUDA pre-clamp
    /// silently truncates the raw command.
    fn evaluate_impl(
        &self,
        action_in: &[f32],
        current_joints: &[f64],
        t_ns: u64,
        sequence_id: u64,
    ) -> PyResult<PyDecision> {
        use std::time::Instant;

        let t0 = Instant::now();
        let ndof = self.limits.names.len();

        // Working copy of the action that downstream stages will see; starts
        // as a verbatim copy of the raw input and is overwritten by the CUDA
        // clamp when that feature is active.  Keeping the original `action_in`
        // alive lets pre-detection observe the *unclamped* values.
        let mut action_clamped: Vec<f32> = action_in.to_vec();

        #[cfg(feature = "cuda")]
        {
            if action_in.len() == ndof {
                if let Ok(mut ctx) = self.cuda_ctx.lock() {
                    let _ = ctx.clamp_into(
                        action_in,
                        self.velocity_max_f32.as_slice(),
                        &mut action_clamped,
                    );
                }
            }
        }

        // Pre-detection on the UNCLAMPED action so VELOCITY_LIMIT / JOINT_LIMIT
        // reasons are not hidden by the CUDA pre-clamp.
        let mut pre_reasons: Vec<ArbiterReason> = Vec::new();
        if action_in.len() == ndof && current_joints.len() == ndof {
            for i in 0..ndof {
                let raw = action_in[i] as f64;
                let vmax = self.limits.velocity_max[i];
                if raw.abs() > vmax {
                    pre_reasons.push(ArbiterReason {
                        ontology_id: physical::velocity_limit(),
                        detail: format!(
                            "joint={} requested={:.3} exceeds limit={:.3} rad/s",
                            self.limits.names[i], raw, vmax
                        ),
                        score: 0.6,
                    });
                }
                let projected = current_joints[i] + raw.clamp(-vmax, vmax) * self.dt;
                let lo = self.limits.position_min[i];
                let hi = self.limits.position_max[i];
                if projected < lo || projected > hi {
                    pre_reasons.push(ArbiterReason {
                        ontology_id: physical::joint_limit(),
                        detail: format!(
                            "joint={} projected={:.3} out_of [{:.3}, {:.3}] rad",
                            self.limits.names[i], projected, lo, hi
                        ),
                        score: 1.0,
                    });
                }
            }
        }

        let av = vec_to_action(t_ns, sequence_id, action_clamped);
        let ingest_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let scene = SceneGraph::default();
        let proj_ctx = ProjectionContext {
            current_joints,
            limits: &self.limits,
            scene: &scene,
            dt: self.dt,
            urdf_chain: None,
            forbidden_zones: &[],
            semantic_constraints: &[],
        };

        let physics_start = Instant::now();
        let proposal = self.projector.project(&proj_ctx, &av);
        let physics_ms = physics_start.elapsed().as_secs_f64() * 1000.0;

        let collision_start = Instant::now();
        let collision_report = match &proposal {
            Ok(p) => {
                let ctx = CollisionContext {
                    scene: &scene,
                    limits: &self.limits,
                    epsilon: self.collision_epsilon,
                };
                self.checker.precheck(&ctx, p)
            }
            Err(_) => shield_core::arbiter::CollisionReport {
                hit: true,
                pairs: vec![],
                energy_lower_bound: f64::MAX,
            },
        };
        let collision_ms = collision_start.elapsed().as_secs_f64() * 1000.0;

        let arbiter_start = Instant::now();
        let mut reasons: Vec<ArbiterReason> = pre_reasons;
        if collision_report.hit {
            for pair in &collision_report.pairs {
                reasons.push(ArbiterReason {
                    ontology_id: physical::collision(),
                    detail: format!("pair={}:{}", pair.link, pair.obstacle),
                    score: 1.0,
                });
            }
        }
        // Propagate physics projection errors as PHY.JOINT_LIMIT / PHY.FORBIDDEN_ZONE.
        if let Err(ref e) = proposal {
            let oid = if e.to_string().to_lowercase().contains("forbidden") {
                physical::forbidden_zone()
            } else {
                physical::joint_limit()
            };
            let already = reasons.iter().any(|r| r.ontology_id == oid);
            if !already {
                reasons.push(ArbiterReason {
                    ontology_id: oid,
                    detail: e.to_string(),
                    score: 1.0,
                });
            }
        }
        let arbiter_ms = arbiter_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let latency = LatencyBreakdown {
            ingest_ms,
            urdf_fk_ms: None,
            physics_ms,
            collision_ms,
            tf2_ms: None,
            arbiter_ms,
            shadow_ms: None,
            total_ms,
        };

        let decision = if reasons.is_empty() {
            ArbiterDecision::Pass {
                action: av,
                latency,
            }
        } else {
            ArbiterDecision::Block {
                safe_fallback: shield_core::action::ActionVector::new(
                    t_ns,
                    sequence_id,
                    vec![0.0f32; current_joints.len()],
                ),
                reasons,
                latency,
            }
        };

        let summary = PyDecisionSummary::from(decision);
        Ok(PyDecision::from(summary))
    }
}

/// Register the module with Python.
#[pymodule]
fn shield_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyShieldPipeline>()?;
    m.add_class::<PyDecision>()?;
    Ok(())
}
