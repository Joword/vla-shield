//! `shield-ffi` — PyO3 bridge exposing the VLA-Shield hot-path pipeline to Python.
//!
//! # Architecture
//!
//! ```text
//! Python VLA model
//!      │  action: list[float]  or  numpy.ndarray[float32]  (contiguous)
//!      ▼
//! PyShieldPipeline.evaluate(...)          ← copies list → Vec
//! PyShieldPipeline.evaluate_numpy(...)    ← zero-copy borrow of ndarray
//!      │  both share evaluate_impl(&[f32], &[f64], ...)
//!      ▼
//! Optional CUDA pre-clamp (small-n stays on CPU; see shield-cuda)
//!      ▼
//! Rust: KinematicClampProjector → URDF FK AABB broad-phase → inline arbiter
//!      │
//!      ▼
//! PyDecision { decision: "PASS"|"BLOCK", reasons: [...], latency: {...} }
//! ```
//!
//! # Copying vs zero-copy
//!
//! * **Host zero-copy** is already implemented: `evaluate_numpy` borrows
//!   contiguous `numpy.ndarray` buffers via PyO3 (`PyReadonlyArray1`).
//! * **Device zero-copy** (`tensor.data_ptr()` shared with CUDA) is not
//!   implemented.  Typical VLA DoF (6–14) is below the GPU bypass threshold,
//!   so a device round-trip would not pay off.

pub mod convert;
pub mod error;

use convert::{make_joint_limits, vec_to_action, PyDecisionSummary};
use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};
#[cfg(feature = "cuda")]
use shield_cuda::CudaCtx;
use shield_collision::broad_phase::AabbBroadPhase;
use shield_collision::{CollisionContext, CollisionPrechecker};
use shield_core::arbiter::{ArbiterDecision, ArbiterReason, LatencyBreakdown};
use shield_core::ontology::physical;
use shield_core::scene::{Primitive, SceneEntity, SceneGraph};
use shield_core::types::{Aabb, JointLimits};
use shield_physics::projection::KinematicClampProjector;
use shield_physics::{PhysicalProjector, ProjectionContext};
use shield_urdf::{AxisAlignedBox, UrdfKinematicChain, UrdfRobot};

/// `(id, min_x, min_y, min_z, max_x, max_y, max_z, ontology_id)` as sent from Python.
type Obstacle = (String, f64, f64, f64, f64, f64, f64, String);

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
        let d = PyDict::new_bound(py);
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
    urdf_chain: Option<UrdfKinematicChain>,
    scene: Mutex<SceneState>,
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
        urdf_xml = None,
        urdf_path = None,
        root_link = None,
        ee_link = None,
        obstacles = vec![],
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
        urdf_xml: Option<String>,
        urdf_path: Option<String>,
        root_link: Option<String>,
        ee_link: Option<String>,
        obstacles: Vec<Obstacle>,
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
        let urdf_chain = load_urdf_chain(urdf_xml, urdf_path, root_link, ee_link).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("URDF load failed: {e}"))
        })?;
        Ok(PyShieldPipeline {
            limits: Arc::new(limits),
            #[cfg(feature = "cuda")]
            velocity_max_f32: Arc::new(velocity_max_f32),
            projector: KinematicClampProjector,
            checker: AabbBroadPhase,
            dt,
            collision_epsilon,
            urdf_chain,
            scene: Mutex::new(scene_state_from_obstacles(&obstacles)),
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

    /// Replace scene obstacles. Each tuple is
    /// ``(id, min_x, min_y, min_z, max_x, max_y, max_z, ontology_id)``.
    /// ``PHY.FORBIDDEN_ZONE`` entries become end-effector zone checks;
    /// any other ontology id becomes a collision body.
    fn set_obstacles(&self, obstacles: Vec<Obstacle>) -> PyResult<()> {
        let mut scene = self
            .scene
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("scene lock poisoned"))?;
        *scene = scene_state_from_obstacles(&obstacles);
        Ok(())
    }

    /// Cartesian skeleton (root → EE) from the loaded URDF, or empty if none.
    fn skeleton(&self, joints: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        let Some(chain) = &self.urdf_chain else {
            return Ok(vec![]);
        };
        let pts = chain.skeleton(&joints).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        })?;
        Ok(pts.into_iter().map(|p| p.to_vec()).collect())
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
        // pre-clamp when that feature is active.  CudaCtx skips the GPU for
        // n < min_gpu_n (default 64), so 6–14 DoF arms stay on a scalar loop.
        // Keeping the original `action_in` alive lets pre-detection observe
        // the *unclamped* values.
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

        let scene_guard = self.scene.lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("scene lock poisoned")
        })?;
        let scene = &scene_guard.scene;
        let proj_ctx = ProjectionContext {
            current_joints,
            limits: &self.limits,
            scene,
            dt: self.dt,
            urdf_chain: self.urdf_chain.as_ref(),
            forbidden_zones: &scene_guard.forbidden,
            semantic_constraints: &[],
        };

        let physics_start = Instant::now();
        let proposal = self.projector.project(&proj_ctx, &av);
        let physics_ms = physics_start.elapsed().as_secs_f64() * 1000.0;

        // FK for the collision boxes runs once here so its cost is reported
        // separately instead of hiding inside `collision_ms`.
        let mut urdf_fk_ms = None;
        let link_boxes = match (&proposal, self.urdf_chain.as_ref()) {
            (Ok(p), Some(chain)) => {
                let fk_start = Instant::now();
                let boxes = chain.link_world_aabbs(&p.joint_positions).ok();
                urdf_fk_ms = Some(fk_start.elapsed().as_secs_f64() * 1000.0);
                boxes
            }
            _ => None,
        };

        let collision_start = Instant::now();
        let collision_report = match &proposal {
            Ok(p) => {
                let mut ctx =
                    CollisionContext::new(scene, &self.limits, self.collision_epsilon);
                if let Some(chain) = self.urdf_chain.as_ref() {
                    ctx = ctx.with_urdf(chain);
                }
                if let Some(boxes) = link_boxes.as_deref() {
                    ctx = ctx.with_link_aabbs(boxes);
                }
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
            urdf_fk_ms,
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

/// Obstacles split by ontology: `PHY.FORBIDDEN_ZONE` boxes are end-effector
/// point checks in the projector, everything else is a collision body.
/// Keeping both in one lock keeps the two views consistent per evaluation.
struct SceneState {
    scene: SceneGraph,
    forbidden: Vec<AxisAlignedBox>,
}

const FORBIDDEN_ZONE_ID: &str = "PHY.FORBIDDEN_ZONE";

fn scene_state_from_obstacles(items: &[Obstacle]) -> SceneState {
    let mut entities = Vec::new();
    let mut forbidden = Vec::new();
    for (id, x0, y0, z0, x1, y1, z1, ontology_id) in items {
        let aabb = Aabb::new([*x0, *y0, *z0], [*x1, *y1, *z1]);
        if ontology_id == FORBIDDEN_ZONE_ID {
            forbidden.push(AxisAlignedBox {
                min: aabb.min,
                max: aabb.max,
            });
            continue;
        }
        let c = aabb.center();
        let h = aabb.half_extents();
        entities.push(SceneEntity {
            id: id.clone(),
            primitive: Primitive::Box {
                extents: [h.x * 2.0, h.y * 2.0, h.z * 2.0],
            },
            pose: [c.x, c.y, c.z, 0.0, 0.0, 0.0, 1.0],
            aabb,
            tags: vec![],
        });
    }
    SceneState {
        scene: SceneGraph {
            frame_id: "base_link".into(),
            revision: 1,
            entities,
        },
        forbidden,
    }
}

fn load_urdf_chain(
    urdf_xml: Option<String>,
    urdf_path: Option<String>,
    root_link: Option<String>,
    ee_link: Option<String>,
) -> Result<Option<UrdfKinematicChain>, String> {
    let robot = if let Some(xml) = urdf_xml.filter(|s| !s.trim().is_empty()) {
        UrdfRobot::from_str(&xml).map_err(|e| e.to_string())?
    } else if let Some(path) = urdf_path.filter(|s| !s.trim().is_empty()) {
        UrdfRobot::from_file(&path).map_err(|e| e.to_string())?
    } else {
        return Ok(None);
    };
    let root = root_link.unwrap_or_else(|| robot.root_link.clone());
    let ee = match ee_link.filter(|s| !s.is_empty()) {
        Some(e) => e,
        None => leaf_link(&robot).ok_or_else(|| "URDF has no leaf link".to_string())?,
    };
    UrdfKinematicChain::from_robot(&robot, &root, &ee)
        .map(Some)
        .map_err(|e| e.to_string())
}

fn leaf_link(robot: &UrdfRobot) -> Option<String> {
    let parents: std::collections::HashSet<&str> =
        robot.joints.values().map(|j| j.parent.as_str()).collect();
    let mut leaves: Vec<String> = robot
        .joints
        .values()
        .map(|j| j.child.clone())
        .filter(|c| !parents.contains(c.as_str()))
        .collect();
    leaves.sort();
    leaves.into_iter().next()
}

/// Register the module with Python.
#[pymodule]
fn shield_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyShieldPipeline>()?;
    m.add_class::<PyDecision>()?;
    Ok(())
}
