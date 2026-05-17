# shield-cuda

Optional CUDA acceleration layer for VLA-Shield.

## Layered design

The crate intentionally separates **Rust → C → C++ → CUDA kernel** so each
hop has a single responsibility and the kernel translation unit stays minimal.

```
src/lib.rs                               (Rust, safe)
       │ extern "C"  shield_cuda_clamp        ← stateless one-shot
       │ extern "C"  shield_cuda_ctx_*        ← stateful context API
       ▼
src/kernels/cuda_host.cpp                (C++, host side)
       │ ShieldCudaCtx { d_input, d_limit, d_output,        ← cudaMalloc 1×
       │                 h_input, h_limit, h_output,        ← cudaMallocHost (pinned)
       │                 stream }                            ← cudaStreamCreate 1×
       │ cudaMemcpyAsync host↔device on the ctx stream
       │ extern "C"  shield_cuda_launch_clamp
       ▼
src/kernels/clamp_kernel.cu              (CUDA, device side)
       │ __global__ clamp_kernel<<<blocks, threads, 0, stream>>>
       ▼
                GPU
```

When `nvcc` is unavailable the C++ host file is skipped and only
`src/kernels/clamp_stub.cpp` is compiled.  The stub exports **the same C ABI**
(stateless + stateful) and is therefore a drop-in replacement; nothing in the
Rust layer or `shield-ffi` needs to know which backend is active.

## Two API tiers

### 1. Stateless one-shot (tests, ad-hoc)

```rust
use shield_cuda::clamp_action_cuda;
let out = clamp_action_cuda(&[2.0, -3.0, 0.2], &[1.0, 1.5, 1.0])?;
```

Allocates / copies / frees per call.  Convenient but **not** hot-path safe.

### 2. Stateful context (hot path, preferred)

```rust
use shield_cuda::CudaCtx;

let mut ctx = CudaCtx::new(dof)?;       // allocates device + pinned buffers once
let mut out = vec![0.0_f32; dof];
for action in stream {
    ctx.clamp_into(&action, &limit, &mut out)?;   // no allocations on hot path
}
```

A `CudaCtx` owns:

* three **cached device buffers** sized for the current `dof` (one `cudaMalloc`
  per buffer, reused across every call);
* three **pinned host staging buffers** allocated with `cudaMallocHost` so the
  H↔D transfers can run asynchronously and without page-locking on the fly;
* one **persistent CUDA stream** so multiple pipelines do not serialise on the
  default stream.

When a call arrives with `n > capacity`, all six buffers are transparently
reallocated to fit; the new capacity is sticky.  The Rust side enforces
`input.len() == limit.len()` and `output.len() ≥ input.len()` before crossing
the FFI boundary.

## Build behavior

`build.rs` probes `nvcc --version`:

| Condition | Compiled | Backend |
|---|---|---|
| `nvcc` available, `CUDA_DISABLE` unset | `clamp_kernel.cu` + `cuda_host.cpp` | real GPU |
| `nvcc` available, `CUDA_DISABLE=1`     | `clamp_stub.cpp`                   | CPU fallback |
| `nvcc` missing                         | `clamp_stub.cpp`                   | CPU fallback |

In the CUDA branch, `cargo:rustc-link-lib=cudart` is emitted so the final
binary links the CUDA runtime, and `cargo:rustc-cfg=has_cuda_kernel` enables
[`is_cuda_kernel_enabled()`] to reflect the active backend at runtime.

## FFI integration

`shield-ffi` exposes a `cuda` feature:

```bash
cd runtime
cargo build -p shield-ffi --features cuda
```

When enabled, `PyShieldPipeline` constructs a `Mutex<CudaCtx>` on
initialisation and reuses it on every `evaluate()` call — there is no
`cudaMalloc` on the hot path.

## Built-in micro-benchmark

```bash
cd runtime
cargo run -p shield-cuda --example bench_clamp --release -- --iters 100000 --dof 8
```

Reports p50 / p95 / p99 / mean / max in microseconds for both the stateless
path and the cached-context path, plus the median speedup.

## Testing

```bash
cd runtime
cargo test -p shield-cuda                # CPU fallback (no nvcc required)
CUDA_DISABLE=1 cargo test -p shield-cuda # forces CPU even when nvcc is present
cargo test -p shield-cuda                # CUDA backend (when nvcc is on PATH)
```

The integration tests in `tests/ctx.rs` exercise: basic clamp correctness,
1000-call reuse, lazy capacity growth, dimension-mismatch and short-output
error paths, and the empty-input no-op.
