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

/// Stateless one-shot clamp — convenient for tests, **not** the hot path.
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
}

unsafe impl Send for CudaCtx {}
// Not `Sync`: callers must serialise access (e.g. via Mutex) because
// `shield_cuda_ctx_clamp` mutates the cached pinned/device buffers.

impl CudaCtx {
    /// Create a new context, optionally pre-allocating buffers for `dof`
    /// floats.  Pass `dof = 0` to defer allocation until the first call.
    pub fn new(dof: usize) -> Result<Self, CudaError> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        let code = unsafe { shield_cuda_ctx_create(dof, &mut handle) };
        if code != 0 || handle.is_null() {
            return Err(CudaError::CtxAlloc(code));
        }
        Ok(CudaCtx {
            handle,
            capacity_hint: dof,
        })
    }

    /// Last known buffer capacity hint, in floats.  Mostly informative.
    pub fn capacity_hint(&self) -> usize {
        self.capacity_hint
    }

    /// Clamp `input` against `limit` and write the result into `output`
    /// **without allocating**.  All three slices must have the same length.
    ///
    /// Reuses the cached device buffers / pinned host buffers / stream so
    /// the only per-call overhead is two `memcpy`s into pinned memory and
    /// the asynchronous H↔D transfers themselves.
    pub fn clamp_into(
        &mut self,
        input: &[f32],
        limit: &[f32],
        output: &mut [f32],
    ) -> Result<(), CudaError> {
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
        let n = input.len();
        if n == 0 {
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
    fn ctx_clamp_into_reuses_buffers() {
        let mut ctx = CudaCtx::new(8).expect("ctx_create");
        let mut out = vec![0.0_f32; 3];
        ctx.clamp_into(&[2.0, -3.0, 0.2], &[1.0, 1.5, 1.0], &mut out).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (-1.5)).abs() < 1e-6);
        assert!((out[2] - 0.2).abs() < 1e-6);

        // Re-run several times to exercise buffer reuse.
        for _ in 0..16 {
            ctx.clamp_into(&[5.0, -5.0, 0.5], &[1.0, 1.0, 1.0], &mut out).unwrap();
            assert!((out[0] - 1.0).abs() < 1e-6);
            assert!((out[1] - (-1.0)).abs() < 1e-6);
            assert!((out[2] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn ctx_grows_when_capacity_exceeded() {
        let mut ctx = CudaCtx::new(2).expect("ctx_create");
        assert_eq!(ctx.capacity_hint(), 2);
        // First call within capacity.
        let mut small = vec![0.0_f32; 2];
        ctx.clamp_into(&[3.0, -3.0], &[1.0, 1.0], &mut small).unwrap();
        // Trigger growth to 6 floats.
        let mut big = vec![0.0_f32; 6];
        ctx.clamp_into(
            &[2.0, -2.0, 0.5, 1.5, -1.5, 0.0],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &mut big,
        )
        .unwrap();
        assert!(ctx.capacity_hint() >= 6);
    }

    #[test]
    fn ctx_dimension_mismatch() {
        let mut ctx = CudaCtx::new(0).expect("ctx_create");
        let mut out = vec![0.0_f32; 3];
        let err = ctx.clamp_into(&[1.0, 2.0], &[1.0], &mut out).unwrap_err();
        assert!(matches!(err, CudaError::DimensionMismatch { .. }));
    }
}
