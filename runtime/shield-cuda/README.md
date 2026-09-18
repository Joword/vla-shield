# shield-cuda

Optional CUDA acceleration for VLA-Shield **action clamp** and **AABB overlap**.

## Layered design

The crate separates **Rust → C → C++ → CUDA kernel** so each hop has a single
responsibility and the kernel translation units stay minimal.

```
src/lib.rs                               (Rust, safe)
       │ extern "C"  shield_cuda_clamp / ctx_*     ← clamp
       │ extern "C"  shield_cuda_aabb_mask / hits  ← N×M overlap
       ▼
src/kernels/cuda_host.cpp                (C++, host side)
       │ ShieldCudaCtx { d_*, h_* pinned, stream }  ← clamp, malloc 1×
       │ AabbDeviceScratch { d_lmin/lmax/omin/omax, d_mask }
       │   grow-only device buffers, mutexed; GPU when pairs ≥ 64
       │ extern "C"  shield_cuda_launch_clamp
       │ extern "C"  shield_cuda_launch_aabb_mask
       ▼
src/kernels/clamp_kernel.cu              one thread per DoF
src/kernels/aabb_kernel.cu               one thread per (link, obstacle) pair
       ▼
                GPU
```

When `nvcc` is unusable the C++ host files are skipped and only
`src/kernels/clamp_stub.cpp` is compiled. The stub exports **the same C ABI**
(clamp + AABB) and is a drop-in replacement; nothing in the Rust layer,
`shield-collision`, or `shield-ffi` needs to know which backend is active.

`build.rs` **tries** the CUDA compile and **catches** host-compiler failures.
On Windows, nvcc needs MSVC `cl.exe`; if it is missing (typical MinGW box)
that is a handled exception → CPU stub. The crate never fails the workspace
build because of CUDA.

## Clamp API

### 1. Stateless one-shot (tests, ad-hoc)

```rust
use shield_cuda::clamp_action_cuda;
let out = clamp_action_cuda(&[2.0, -3.0, 0.2], &[1.0, 1.5, 1.0])?;
```

Allocates / copies / frees per call. Convenient but **not** hot-path safe.

### 2. Stateful context (hot path, preferred)

```rust
use shield_cuda::CudaCtx;

let mut ctx = CudaCtx::new(dof)?;       // allocates device + pinned buffers once
let mut out = vec![0.0_f32; dof];
for action in stream {
    ctx.clamp_into(&action, &limit, &mut out)?;   // no allocations on hot path
}
```

A `CudaCtx` owns three cached device buffers, three pinned host staging
buffers, and one persistent CUDA stream. When `n > capacity`, buffers grow and
the new capacity is sticky.

## AABB overlap

Packed boxes: `[min_x, min_y, min_z]` / `[max_x, max_y, max_z]` per box.

```rust
use shield_cuda::aabb_overlap_mask;

// mask[i * n_obs + j] = 1 iff link i overlaps obstacle j
let mask = aabb_overlap_mask(&link_min, &link_max, &obs_min, &obs_max)?;
```

`AabbBroadPhase` calls this so collision pair reasons stay `(link, obstacle)`.
`aabb_hits` is the per-link OR-reduction of a mask row (kept for tests).

| Pairs `N×M` | What runs |
|---|---|
| 0 | no-op |
| 1–63 | host nested loop (typical 7-link × few obstacles) |
| ≥ 64 | CUDA kernel when nvcc built the crate; host fill if malloc/launch fails |

## Small-n CPU bypass (clamp only)

Typical VLA arms are 6–14 DoF. A kernel launch plus two H↔D copies is slower
than a scalar clamp at that size, so `CudaCtx::clamp_into` **never crosses
into C++/CUDA** when `n < min_gpu_n`.

| `n` | Default (`min_gpu_n = 64`) | Forced (`min_gpu_n = 0`) |
|---|---|---|
| 6–14 (serial arm) | CPU scalar loop | C backend (kernel or stub) |
| ≥ 64 | C backend | C backend |

Override:

* env `SHIELD_CUDA_MIN_GPU_N` (read at `CudaCtx::new`)
* `ctx.set_min_gpu_n(0)` / `--force-gpu` on `bench_clamp`

`last_path()` reports `ClampPath::Cpu` or `ClampPath::Backend` after each call.
The C++ host layer itself always runs the GPU path when invoked, so A/B
benches remain honest.

## Build behavior

`build.rs` runs `nvcc` via `cc::Build::try_compile`. Failures — including a
missing Windows host compiler `cl.exe` — are caught and turned into the CPU
stub. They are not crate build errors.

| Condition | Compiled | Backend |
|---|---|---|
| `nvcc` compile succeeds, `CUDA_DISABLE` unset | `clamp_kernel.cu` + `aabb_kernel.cu` + `cuda_host.cpp` | real GPU |
| `nvcc` compile throws (e.g. no `cl.exe`) | `clamp_stub.cpp` | CPU fallback (handled) |
| `CUDA_DISABLE=1` | `clamp_stub.cpp` | CPU fallback |
| `nvcc` missing | `clamp_stub.cpp` | CPU fallback |

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
`cudaMalloc` on the clamp hot path. Collision always links `shield-cuda`
(stub if no nvcc) so AABB mask works without that feature flag.

## Built-in micro-benchmarks

```bash
cd runtime
cargo run -p shield-cuda --example bench_clamp --release -- --iters 100000 --dof 8
cargo run -p shield-cuda --example bench_clamp --release -- --iters 20000 --dof 256
cargo run -p shield-cuda --example bench_clamp --release -- --iters 100000 --dof 8 --force-gpu

cargo run -p shield-cuda --example bench_aabb --release -- --iters 20000 --n-links 8 --n-obs 8
cargo run -p shield-cuda --example bench_aabb --release -- --iters 5000 --n-links 32 --n-obs 32
```

`bench_clamp` reports p50 / p95 / p99 / mean / max (µs) for CPU bypass, forced
C backend, default `CudaCtx` policy, and stateless one-shot. At `--dof 8`
CPU should win.

`bench_aabb` compares a Rust nested loop against `aabb_overlap_mask`. At
8×8 = 64 pairs the C backend may take the kernel path.

## Testing

```bash
cd runtime
cargo test -p shield-cuda -p shield-collision   # CPU fallback (no nvcc required)
CUDA_DISABLE=1 cargo test -p shield-cuda        # forces CPU even when nvcc is present
cargo test -p shield-cuda                       # CUDA backend (when nvcc + cl.exe)
```
