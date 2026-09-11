//! Optional CUDA clamp. Skip it for a 7-DoF arm.
//!
//! Two APIs:
//!
//! 1. **One-shot** — [`clamp_action_cuda`]. Fine for tests. Allocates /
//!    copies / frees every call. Don't put this on the hot path.
//! 2. **Context** — [`CudaCtx`] + [`CudaCtx::clamp_into`]. Owns three device
//!    buffers, three pinned host buffers, one stream. Reuse it and you skip
//!    `cudaMalloc` per tick. Independent pipelines don't serialize on the
//!    default stream.
//!
//! Same C ABI on both the real kernel and the CPU stub (no nvcc). Call sites
//! don't need `#[cfg]`.
//!
//! ## Small-n CPU bypass
//!
//! n < 64 stays on CPU — launching a kernel for a 7-DoF arm is slower than
//! just clamping. Override via `SHIELD_CUDA_MIN_GPU_N` or
//! [`CudaCtx::set_min_gpu_n`]. Set `0` to force the C backend for A/B.

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

    fn shield_cuda_aabb_hits(
        link_min: *const f32,
        link_max: *const f32,
        n_links: usize,
        obs_min: *const f32,
        obs_max: *const f32,
        n_obs: usize,
        hits: *mut u8,
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
/// 64 is way above any serial-arm DoF (6–14) and still below batched /
/// wide-vector work where a kernel starts to pay.
pub const DEFAULT_MIN_GPU_N: usize = 64;

/// Which impl served the last [`CudaCtx::clamp_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClampPath {
    /// In-process scalar loop. No C ABI, no kernel.
    Cpu,
    /// Went through the C backend (real kernel or CPU stub).
    Backend,
}

/// `SHIELD_CUDA_MIN_GPU_N`, or [`DEFAULT_MIN_GPU_N`] if unset / junk.
pub fn min_gpu_n_from_env() -> usize {
    match std::env::var("SHIELD_CUDA_MIN_GPU_N") {
        Ok(s) if !s.is_empty() => s.parse().unwrap_or(DEFAULT_MIN_GPU_N),
        _ => DEFAULT_MIN_GPU_N,
    }
}

/// Scalar clamp matching `clamp_stub.cpp` and the CUDA kernel:
/// `out[i]` = `in[i]` clipped to `[-limit[i], limit[i]]`.
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

/// One-shot clamp. Fine for tests, **not** the hot path.
///
/// Always crosses the C ABI so tests actually hit the compiled backend.
/// Hot path wants [`CudaCtx::clamp_into`] (small-n CPU bypass).
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

/// Persistent CUDA ctx: cached device + pinned host buffers + a private stream.
///
/// Create once per pipeline, reuse every tick. Backend grows buffers if a
/// bigger `n` shows up.
///
/// `Send` but not `Sync` — `shield-ffi` holds it behind a `Mutex`. The C ABI
/// mutates its own buffers/stream.
pub struct CudaCtx {
    handle: *mut c_void,
    capacity_hint: usize,
    min_gpu_n: usize,
    last_path: ClampPath,
}

unsafe impl Send for CudaCtx {}
// Not `Sync`: callers must serialize (Mutex) because `shield_cuda_ctx_clamp`
// mutates the cached pinned/device buffers.

impl CudaCtx {
    /// New ctx. `dof` pre-allocates that many floats; `0` waits until first use.
    ///
    /// `min_gpu_n` comes from `SHIELD_CUDA_MIN_GPU_N` or [`DEFAULT_MIN_GPU_N`].
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

    /// Override the small-n bypass. `0` forces the C backend every non-empty
    /// call (A/B benches).
    pub fn set_min_gpu_n(&mut self, n: usize) {
        self.min_gpu_n = n;
    }

    /// Builder alias of [`Self::set_min_gpu_n`].
    pub fn with_min_gpu_n(mut self, n: usize) -> Self {
        self.min_gpu_n = n;
        self
    }

    pub fn min_gpu_n(&self) -> usize {
        self.min_gpu_n
    }

    /// Last known device-buffer capacity, in floats. Informative.
    /// CPU-bypass calls don't touch this (device wasn't involved).
    pub fn capacity_hint(&self) -> usize {
        self.capacity_hint
    }

    /// Which path served the last [`Self::clamp_into`].
    pub fn last_path(&self) -> ClampPath {
        self.last_path
    }

    /// Clamp `input` against `limit` into `output`. No alloc. Same length.
    ///
    /// n < `min_gpu_n` → scalar loop, never leaves Rust. Otherwise reuse the
    /// cached buffers/stream; per-call cost is two memcpys into pinned memory
    /// plus the async H↔D copies.
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

    /// Allocates a `Vec`. Hot path wants [`Self::clamp_into`].
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

/// Did this build actually compile the CUDA kernel?
pub fn is_cuda_kernel_enabled() -> bool {
    cfg!(has_cuda_kernel)
}

/// Packed AABB: `[min_x, min_y, min_z]` / `[max_x, max_y, max_z]` per box.
///
/// `hits[i] = 1` if link `i` overlaps any obstacle. Typical VLA scenes are a
/// handful of links × a handful of obstacles, so this is a host loop on both
/// backends (same C ABI).
pub fn aabb_hits(
    link_min: &[[f32; 3]],
    link_max: &[[f32; 3]],
    obs_min: &[[f32; 3]],
    obs_max: &[[f32; 3]],
    hits: &mut [u8],
) -> Result<(), CudaError> {
    if link_min.len() != link_max.len() {
        return Err(CudaError::DimensionMismatch {
            input: link_min.len(),
            limit: link_max.len(),
        });
    }
    if obs_min.len() != obs_max.len() {
        return Err(CudaError::DimensionMismatch {
            input: obs_min.len(),
            limit: obs_max.len(),
        });
    }
    if hits.len() < link_min.len() {
        return Err(CudaError::OutputTooSmall {
            need: link_min.len(),
            got: hits.len(),
        });
    }
    let n = link_min.len();
    let m = obs_min.len();
    let code = unsafe {
        shield_cuda_aabb_hits(
            link_min.as_ptr() as *const f32,
            link_max.as_ptr() as *const f32,
            n,
            obs_min.as_ptr() as *const f32,
            obs_max.as_ptr() as *const f32,
            m,
            hits.as_mut_ptr(),
        )
    };
    if code != 0 {
        return Err(CudaError::Backend(code));
    }
    Ok(())
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
        // Device capacity unchanged: we never crossed into the backend.
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
        // Force the backend so capacity_hint actually tracks device-buffer growth.
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

    #[test]
    fn aabb_hits_detects_overlap() {
        let link_min = [[0.0_f32, 0.0, 0.0]];
        let link_max = [[1.0_f32, 1.0, 1.0]];
        let obs_min = [[0.5_f32, 0.5, 0.5]];
        let obs_max = [[1.5_f32, 1.5, 1.5]];
        let mut hits = [0u8; 1];
        aabb_hits(&link_min, &link_max, &obs_min, &obs_max, &mut hits).unwrap();
        assert_eq!(hits[0], 1);

        let miss_min = [[2.0_f32, 2.0, 2.0]];
        let miss_max = [[3.0_f32, 3.0, 3.0]];
        aabb_hits(&link_min, &link_max, &miss_min, &miss_max, &mut hits).unwrap();
        assert_eq!(hits[0], 0);
    }
}
