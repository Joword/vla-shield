//! Optional CUDA acceleration layer for VLA-Shield.
//!
//! Two API tiers are exposed:
//!
//! 1. **Stateless one-shot** — [`clamp_action_cuda`].
//!    Convenient for tests and ad-hoc calls; allocates / copies / frees per
//!    invocation, so unsuited for hot-path workloads.
//!
//! 2. **Stateful context** — [`CudaCtx`] + [`CudaCtx::clamp_into`].
//!    The context owns three cached device buffers, three pinned host
//!    staging buffers, and one persistent CUDA stream.  Reusing the same
//!    [`CudaCtx`] across calls eliminates the per-call `cudaMalloc`,
//!    enables `cudaMemcpyAsync`, and lets independent pipelines avoid
//!    serialising on the default stream.
//!
//! Both tiers share the same C ABI on the C++/CUDA side so the CPU
//! fallback (compiled when `nvcc` is absent) is fully ABI-compatible —
//! there is no `#[cfg]` plumbing required at the call site.
//!
//! ## Small-n CPU bypass
//!
//! Typical VLA arms are 6–14 DoF.  Launching a CUDA kernel plus two
//! asynchronous H↔D copies for a dozen floats is slower than a scalar
//! loop.  [`CudaCtx::clamp_into`] therefore stays on the CPU whenever
//! `n < min_gpu_n` (default [`DEFAULT_MIN_GPU_N`] = 64, overridable via
//! `SHIELD_CUDA_MIN_GPU_N` or [`CudaCtx::set_min_gpu_n`]).  Set the
//! threshold to `0` to force the C backend for A/B measurements.

use std::os::raw::c_void;
use thiserror::Error;

extern "C" {
    fn shield_cuda_clamp(
        input: *const f32,
        limit: *const f32,
        output: *mut f32,
        n: usize,
    ) -> i32;

    fn shield_cuda_ctx_create(initial_capacity: usize, out_ctx: *mut *mut c_void) -> i32;
    fn shield_cuda_ctx_destroy(ctx: *mut c_void);
    fn shield_cuda_ctx_clamp(
        ctx: *mut c_void,
        input: *const f32,
        limit: *const f32,
        output: *mut f32,
        n: usize,
    ) -> i32;
}

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("dimension mismatch: input={input} limit={limit}")]
    DimensionMismatch { input: usize, limit: usize },
    #[error("output buffer too small: need={need} got={got}")]
    OutputTooSmall { need: usize, got: usize },
    #[error("CUDA backend returned error code {0}")]
    Backend(i32),
    #[error("CUDA context allocation failed (code {0})")]
    CtxAlloc(i32),
}

/// Vectors shorter than this skip the GPU / C-ABI hop on the hot path.
///
/// 64 is well above any current serial-arm DoF (6–14) and well below
/// batched / wide-vector workloads where a kernel starts to pay off.
pub const DEFAULT_MIN_GPU_N: usize = 64;

/// Which implementation served the last [`CudaCtx::clamp_into`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClampPath {
    /// In-process scalar loop; no C ABI, no kernel launch.
    Cpu,
    /// Crossed into the C backend (real CUDA kernel or CPU stub).
    Backend,
}

/// Read `SHIELD_CUDA_MIN_GPU_N`, falling back to [`DEFAULT_MIN_GPU_N`].
pub fn min_gpu_n_from_env() -> usize {
    match std::env::var("SHIELD_CUDA_MIN_GPU_N") {
        Ok(s) if !s.is_empty() => s.parse().unwrap_or(DEFAULT_MIN_GPU_N),
        _ => DEFAULT_MIN_GPU_N,
    }
}

/// Scalar clamp matching `clamp_stub.cpp` and the CUDA kernel:
/// `out[i]` is `in[i]` clipped to `[-limit[i], limit[i]]`.
pub fn clamp_cpu(input: &[f32], limit: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), limit.len());
    debug_assert!(output.len() >= input.len());
    for i in 0..input.len() {
        let mut x = input[i];
        let l = limit[i];
        if x > l {
            x = l;
        }
        if x < -l {
            x = -l;
        }
        output[i] = x;
    }
}

fn check_lengths(input: &[f32], limit: &[f32], output: &[f32]) -> Result<usize, CudaError> {
    if input.len() != limit.len() {
        return Err(CudaError::DimensionMismatch {
            input: input.len(),
            limit: limit.len(),
        });
    }
    if output.len() < input.len() {
        return Err(CudaError::OutputTooSmall {
            need: input.len(),
            got: output.len(),
        });
    }
    Ok(input.len())
}

/// Stateless one-shot clamp — convenient for tests, **not** the hot path.
///
/// Always crosses the C ABI so tests can exercise the compiled backend.
/// Prefer [`CudaCtx::clamp_into`] on the hot path (it applies the small-n
/// CPU bypass).
pub fn clamp_action_cuda(input: &[f32], limit: &[f32]) -> Result<Vec<f32>, CudaError> {
    if input.len() != limit.len() {
        return Err(CudaError::DimensionMismatch {
            input: input.len(),
            limit: limit.len(),
        });
    }
    let mut output = vec![0.0_f32; input.len()];
    let code = unsafe {
        shield_cuda_clamp(
            input.as_ptr(),
            limit.as_ptr(),
            output.as_mut_ptr(),
            input.len(),
        )
    };
    if code != 0 {
        return Err(CudaError::Backend(code));
    }
    Ok(output)
}

/// Persistent CUDA context owning cached device buffers, pinned host staging
/// buffers, and a private CUDA stream.
///
/// Designed to be created once per pipeline and reused across every action
/// evaluation.  Internally, the backend grows its buffers transparently when
/// a larger `n` arrives.
///
/// Safe to send across threads (held behind a `Mutex` in `shield-ffi`) — the
/// underlying C ABI only mutates its private device/host buffers and stream.
pub struct CudaCtx {
    handle: *mut c_void,
    capacity_hint: usize,
    min_gpu_n: usize,
    last_path: ClampPath,
}

unsafe impl Send for CudaCtx {}
// Not `Sync`: callers must serialise access (e.g. via Mutex) because
// `shield_cuda_ctx_clamp` mutates the cached pinned/device buffers.

impl CudaCtx {
    /// Create a new context, optionally pre-allocating buffers for `dof`
    /// floats.  Pass `dof = 0` to defer allocation until the first call.
    ///
    /// `min_gpu_n` is taken from `SHIELD_CUDA_MIN_GPU_N` or
    /// [`DEFAULT_MIN_GPU_N`].
    pub fn new(dof: usize) -> Result<Self, CudaError> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        let code = unsafe { shield_cuda_ctx_create(dof, &mut handle) };
        if code != 0 || handle.is_null() {
            return Err(CudaError::CtxAlloc(code));
        }
        Ok(CudaCtx {
            handle,
            capacity_hint: dof,
            min_gpu_n: min_gpu_n_from_env(),
            last_path: ClampPath::Cpu,
        })
    }

    /// Override the small-n bypass threshold.  `0` forces the C backend
    /// for every non-empty call (used by A/B micro-benchmarks).
    pub fn set_min_gpu_n(&mut self, n: usize) {
        self.min_gpu_n = n;
    }

    /// Builder-style alias of [`Self::set_min_gpu_n`].
    pub fn with_min_gpu_n(mut self, n: usize) -> Self {
        self.min_gpu_n = n;
        self
    }

    pub fn min_gpu_n(&self) -> usize {
        self.min_gpu_n
    }

    /// Last known buffer capacity hint, in floats.  Mostly informative.
    /// Unchanged by CPU-bypass calls (device buffers were not touched).
    pub fn capacity_hint(&self) -> usize {
        self.capacity_hint
    }

    /// Which path served the most recent [`Self::clamp_into`] call.
    pub fn last_path(&self) -> ClampPath {
        self.last_path
    }

    /// Clamp `input` against `limit` and write the result into `output`
    /// **without allocating**.  All three slices must have the same length.
    ///
    /// When `n < min_gpu_n` this is a scalar loop and never crosses into
    /// C++/CUDA.  Otherwise it reuses the cached device buffers / pinned
    /// host buffers / stream so the only per-call overhead is two `memcpy`s
    /// into pinned memory and the asynchronous H↔D transfers themselves.
    pub fn clamp_into(
        &mut self,
        input: &[f32],
        limit: &[f32],
        output: &mut [f32],
    ) -> Result<(), CudaError> {
        let n = check_lengths(input, limit, output)?;
        if n == 0 {
            self.last_path = ClampPath::Cpu;
            return Ok(());
        }
        if n < self.min_gpu_n {
            clamp_cpu(input, limit, output);
            self.last_path = ClampPath::Cpu;
            return Ok(());
        }
        let code = unsafe {
            shield_cuda_ctx_clamp(
                self.handle,
                input.as_ptr(),
                limit.as_ptr(),
                output.as_mut_ptr(),
                n,
            )
        };
        if code != 0 {
            return Err(CudaError::Backend(code));
        }
        if n > self.capacity_hint {
            self.capacity_hint = n;
        }
        self.last_path = ClampPath::Backend;
        Ok(())
    }

    /// Convenience wrapper that returns a freshly-allocated `Vec`.  Prefer
    /// [`Self::clamp_into`] in hot paths to avoid the allocation.
    pub fn clamp(&mut self, input: &[f32], limit: &[f32]) -> Result<Vec<f32>, CudaError> {
        let mut out = vec![0.0_f32; input.len()];
        self.clamp_into(input, limit, &mut out)?;
        Ok(out)
    }
}

impl Drop for CudaCtx {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { shield_cuda_ctx_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Returns whether this build uses the actual CUDA kernel backend.
pub fn is_cuda_kernel_enabled() -> bool {
    cfg!(has_cuda_kernel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_basic_stateless() {
        let out = clamp_action_cuda(&[2.0, -3.0, 0.2], &[1.0, 1.5, 1.0]).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (-1.5)).abs() < 1e-6);
        assert!((out[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn clamp_cpu_matches_stateless_backend() {
        let input = [2.0, -3.0, 0.2, 0.0, -0.4];
        let limit = [1.0, 1.5, 1.0, 1.0, 0.3];
        let backend = clamp_action_cuda(&input, &limit).unwrap();
        let mut cpu = [0.0_f32; 5];
        clamp_cpu(&input, &limit, &mut cpu);
        for i in 0..5 {
            assert!((cpu[i] - backend[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn ctx_small_n_stays_on_cpu() {
        let mut ctx = CudaCtx::new(8).expect("ctx_create");
        assert_eq!(ctx.min_gpu_n(), min_gpu_n_from_env());
        let mut out = vec![0.0_f32; 3];
        ctx.clamp_into(&[2.0, -3.0, 0.2], &[1.0, 1.5, 1.0], &mut out)
            .unwrap();
        assert_eq!(ctx.last_path(), ClampPath::Cpu);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (-1.5)).abs() < 1e-6);
        assert!((out[2] - 0.2).abs() < 1e-6);
        // Device capacity is unchanged: we never crossed into the backend.
        assert_eq!(ctx.capacity_hint(), 8);
    }

    #[test]
    fn ctx_force_backend_matches_cpu() {
        let input = [5.0, -5.0, 0.5, 1.2, -0.2, 0.0, 3.0, -3.0];
        let limit = [1.0_f32; 8];
        let mut cpu_out = [0.0_f32; 8];
        clamp_cpu(&input, &limit, &mut cpu_out);

        let mut ctx = CudaCtx::new(8)
            .expect("ctx_create")
            .with_min_gpu_n(0);
        let mut backend_out = [0.0_f32; 8];
        ctx.clamp_into(&input, &limit, &mut backend_out).unwrap();
        assert_eq!(ctx.last_path(), ClampPath::Backend);
        for i in 0..8 {
            assert!((cpu_out[i] - backend_out[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn ctx_large_n_uses_backend() {
        let n = DEFAULT_MIN_GPU_N;
        let mut ctx = CudaCtx::new(n).expect("ctx_create");
        let input: Vec<f32> = (0..n).map(|k| (k as f32) - (n as f32) / 2.0).collect();
        let limit = vec![2.0_f32; n];
        let mut out = vec![0.0_f32; n];
        ctx.clamp_into(&input, &limit, &mut out).unwrap();
        assert_eq!(ctx.last_path(), ClampPath::Backend);
        for v in &out {
            assert!(*v >= -2.0 - 1e-6 && *v <= 2.0 + 1e-6);
        }
    }

    #[test]
    fn ctx_clamp_into_reuses_buffers() {
        let mut ctx = CudaCtx::new(8).expect("ctx_create");
        let mut out = vec![0.0_f32; 3];
        ctx.clamp_into(&[2.0, -3.0, 0.2], &[1.0, 1.5, 1.0], &mut out)
            .unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (-1.5)).abs() < 1e-6);
        assert!((out[2] - 0.2).abs() < 1e-6);

        for _ in 0..16 {
            ctx.clamp_into(&[5.0, -5.0, 0.5], &[1.0, 1.0, 1.0], &mut out)
                .unwrap();
            assert!((out[0] - 1.0).abs() < 1e-6);
            assert!((out[1] - (-1.0)).abs() < 1e-6);
            assert!((out[2] - 0.5).abs() < 1e-6);
            assert_eq!(ctx.last_path(), ClampPath::Cpu);
        }
    }

    #[test]
    fn ctx_grows_when_capacity_exceeded() {
        // Force the backend so capacity_hint tracks device-buffer growth.
        let mut ctx = CudaCtx::new(2)
            .expect("ctx_create")
            .with_min_gpu_n(0);
        assert_eq!(ctx.capacity_hint(), 2);
        let mut small = vec![0.0_f32; 2];
        ctx.clamp_into(&[3.0, -3.0], &[1.0, 1.0], &mut small).unwrap();
        let mut big = vec![0.0_f32; 6];
        ctx.clamp_into(
            &[2.0, -2.0, 0.5, 1.5, -1.5, 0.0],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &mut big,
        )
        .unwrap();
        assert!(ctx.capacity_hint() >= 6);
        assert_eq!(ctx.last_path(), ClampPath::Backend);
    }

    #[test]
    fn ctx_dimension_mismatch() {
        let mut ctx = CudaCtx::new(0).expect("ctx_create");
        let mut out = vec![0.0_f32; 3];
        let err = ctx.clamp_into(&[1.0, 2.0], &[1.0], &mut out).unwrap_err();
        assert!(matches!(err, CudaError::DimensionMismatch { .. }));
    }
}
