# VLA-Shield

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20Jazzy-22314E?logo=ros)](https://docs.ros.org/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=next.js)](https://nextjs.org/)
[![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia&logoColor=white)](#cuda-acceleration-layer)

**VLA-Shield** is a model-agnostic, real-time safety filter layer for Vision-Language-Action (VLA) policies. Unlike **training-time** alignment (e.g. SafeVLA-style constraints on the policy), VLA-Shield operates as a **decoupled runtime middleware** that intercepts raw action outputs, projects them into physical dynamics space, and enforces **hard** safety constraints — within a **&lt; 5 ms** hot-path budget — **without modifying** the base VLA model weights.

---

## What Makes VLA-Shield Different

| Approach | Modifies Model? | Latency | Deterministic? | Hardware Requirement |
|----------|:---------------:|:-------:|:--------------:|:---:|
| RLHF / DPO alignment | Yes | Training-time | No | Multi-GPU training cluster |
| SafeVLA (training-time safety) | Yes | Training / inference | No (soft constraints) | Multi-GPU training cluster |
| Safety-CHORES (arXiv:2503.03480) | Yes (fine-tuning) | Inference-time | No | GPU + retraining |
| **VLA-Shield (ours)** | **No** | **&lt; 5 ms runtime** | **Yes (URDF + rule engine)** | **CPU is enough; GPU optional** |

### Project Highlights

- **🧱 Zero-touch on the policy** — slots in front of any VLA (OpenVLA, RT-2, Octo, Diffusion Policy) without a single parameter change.
- **⚡ Sub-5 ms hot path** — written in Rust with per-stage `LatencyBreakdown` (`ingest / urdf_fk / physics / collision / tf2 / arbiter / shadow / total`) so latency budget violations are observable, not guessed.
- **📜 Executable safety ontology** — 13 ontology nodes (`PHY.*` × 7, `SEM.*` × 6) backed by JSON rule files (`dataset/ontology/rules_*.json`) carrying `trigger_condition`, typed `threshold`, `action ∈ {block, clamp, warn}`, `severity`, and a `{placeholder}`-driven `explanation_template`. The same JSON is the source of truth for the Rust arbiter, the FastAPI `/v1/rules` endpoint, and the React `RuleViewer`.
- **🛡️ Hot + Async split** — hard, real-time checks live on the Rust hot path; expensive predictors (VLM-based VFV, multi-step shadow simulation) run asynchronously and feed the arbiter as **stale-safe** priors.
- **🚀 Multi-tier acceleration** — Python → PyO3 (zero-copy from `numpy.ndarray`) → Rust → optional C++ host layer → CUDA kernel; every hop has one responsibility and the layer above transparently falls back when the layer below is unavailable.
- **🎮 Per-pipeline CUDA context** — when GPU is enabled, `shield-cuda` holds cached device buffers, pinned host staging buffers, and a private CUDA stream per pipeline; **no `cudaMalloc` ever happens on the hot path**.
- **🔄 Always-on CPU fallback** — same C ABI on both backends, so the runtime keeps working on machines without `nvcc` (or with `CUDA_DISABLE=1`); a Python evaluator falls back further when even the Rust extension is missing.
- **🗺️ Real-time 3D digital twin** — Next.js + React monitor renders live action, shadow trajectory, latency stacked bar, why-blocked panel, and active rule set, all driven by a Redis-Stream → WebSocket pipe.
- **📦 Three deployment targets** — full server stack (Docker Compose), edge Jetson (ARM64 compose), and the bare-metal `cargo build --workspace` path. ROS 2 binding is feature-gated; the runtime can run completely outside a ROS environment.

---

## Architecture

```
                          ╔══════════════════════════════════════════════════╗
                          ║                  VLA Policy Model                 ║
                          ║   OpenVLA · RT-2 · Octo · Diffusion Policy · …    ║
                          ╚═══════════════════════ │ ═════════════════════════╝
                                                   │  raw action  (N × DoF)
                                                   ▼
                          ┌──────────────────────────────────────────────────┐
                          │     shield_ffi.ShieldPipeline   (PyO3 bridge)     │
                          │   • evaluate(list[float])                         │
                          │   • evaluate_numpy(ndarray)  ← zero-copy hot path │
                          └─────────────────────── │ ────────────────────────┘
                                                   │  &[f32], &[f64]
                                                   ▼
   ╔═════════════════════════════════════════════════════════════════════════════════╗
   ║                         ⚡  HOT PATH — Rust (< 5 ms p99)                          ║
   ║                                                                                 ║
   ║   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    ║
   ║   │ shield-cuda  │   │ shield-      │   │ shield-      │   │ shield-      │    ║
   ║   │  (optional)  │   │ physics      │   │ urdf         │   │ collision    │    ║
   ║   │              │   │              │   │              │   │              │    ║
   ║   │ pre-clamp    │──▶│ kinematic    │──▶│ FK +         │──▶│ AABB broad   │──┐ ║
   ║   │ ctx-cached   │   │ projector +  │   │ singularity  │   │ phase        │  │ ║
   ║   │ pinned host  │   │ semantic     │   │ + forbidden  │   │ pre-check    │  │ ║
   ║   │ + stream     │   │ constraints  │   │ zones        │   │              │  │ ║
   ║   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘  │ ║
   ║                                                                              │ ║
   ║                                ┌─────────────────────────────────────────────┘ ║
   ║                                ▼                                                ║
   ║                ┌─────────────────────────────────────────┐                      ║
   ║                │            Arbiter (rule-driven)         │                      ║
   ║                │  ┌────────────────────────────────────┐  │                      ║
   ║                │  │ rules_physical.json + rules_       │  │                      ║
   ║                │  │ semantic.json  →  PHY.* / SEM.*    │  │                      ║
   ║                │  │ action ∈ {block | clamp | warn}    │  │                      ║
   ║                │  │ explanation_template {placeholders}│  │                      ║
   ║                │  └────────────────────────────────────┘  │                      ║
   ║                └────────────────────┬────────────────────┘                      ║
   ╚═════════════════════════════════════ │ ══════════════════════════════════════════╝
                                          ▼
                          PASS (clamped action)  ◀───┐    BLOCK (safe fallback)
                                          │           │            │
              ┌───────────────────────────┼───────────┴────────────┴───────────┐
              ▼                           ▼                                    ▼
   ┌──────────────────┐     ┌──────────────────────┐          ┌──────────────────────┐
   │ ROS 2 Lifecycle  │     │ Safety Event         │          │ Redis Stream         │
   │ (Fast-DDS QoS)   │     │ → MySQL audit log    │          │ → WebSocket → UI     │
   │ activate /       │     │ (event_id, latency,  │          │ (telemetry, latency  │
   │ deactivate hooks │     │  reasons[], action)  │          │  breakdown, reasons) │
   └──────────────────┘     └──────────────────────┘          └──────────┬───────────┘
                                                                          ▼
                                                          ┌──────────────────────────┐
                                                          │  Monitor UI              │
                                                          │  Next.js + React + Three │
                                                          │  ┌────────────────────┐  │
                                                          │  │ LatencyChart       │  │
                                                          │  │ WhyBlocked         │  │
                                                          │  │ RuleViewer         │  │
                                                          │  │ SceneView (3D)     │  │
                                                          │  └────────────────────┘  │
                                                          └──────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │                       ⏳  ASYNC PATH — off the hot-path budget                    │
   │                                                                                 │
   │   ┌──────────────────────────────┐         ┌──────────────────────────────┐     │
   │   │  shield-shadow                │         │  Visual Feedback Verifier    │     │
   │   │  joint-space roll-forward     │         │  (Python · backend.shield)   │     │
   │   │  multi-step risk prior        │         │  VLM / CLIP semantic risk    │     │
   │   │  (JointSpaceSimulator)        │         │  → SEM.* triggers            │     │
   │   └──────────────────────────────┘         └──────────────────────────────┘     │
   │                                                                                 │
   │   Both results are fed back into the arbiter as STALE-SAFE priors —              │
   │   if the async pass hasn't finished yet, the hot path simply proceeds.           │
   └─────────────────────────────────────────────────────────────────────────────────┘
```

### Layer & Stack

| Layer | Crate / Module | Purpose | Tech |
|---|---|---|---|
| Python FFI | `shield-ffi` | PyO3 + numpy zero-copy bridge; per-pipeline `Mutex<CudaCtx>` | Rust · PyO3 0.22 · numpy |
| CUDA acceleration | `shield-cuda` | Rust → C++ host → CUDA kernel; auto CPU fallback | Rust · C++ · CUDA |
| Domain core | `shield-core` | Ontology, action, arbiter, scene-graph types | Rust |
| Physics | `shield-physics` | Kinematic projector + semantic constraint mapper | Rust · nalgebra |
| Kinematics | `shield-urdf` | URDF parse, forward kinematics, forbidden zones | Rust · quick-xml |
| Collision | `shield-collision` | AABB broad-phase pre-check | Rust |
| Shadow | `shield-shadow` | Async joint-space roll-forward predictor | Rust |
| ROS 2 glue | `shield-ros2` | Lifecycle hooks, tf2 validator, pipeline orchestrator | Rust · rclrs (optional) |
| Storage | `shield-io` | MySQL persistence + Redis Stream telemetry | Rust · sqlx · redis |
| Backend API | `backend/shield/api` | FastAPI REST + WebSocket; rule engine; evaluator | Python · FastAPI · numpy |
| VFV | `backend/shield/vfv` | Visual Feedback Verification reference predictor | Python · PyTorch |
| Monitor | `monitor/` | Real-time 3D digital twin dashboard | Next.js 14 · React · Three.js · Zustand |

---

## Quick Start

### 1 · Rust runtime

```bash
cd runtime
cargo build --workspace
cargo test  --workspace
```

Optional ROS 2 Rust bindings:

```bash
cargo build -p shield-ros2 --features ros2     # requires ROS 2 + Rust overlay
```

### 2 · Python backend

```bash
cd backend
pip install -e ".[dev]"
pytest                                          # 17 tests including /v1/evaluate smoke
```

### 3 · Safety monitor (Next.js)

```bash
cd monitor
yarn install && yarn dev
```

### 4 · ROS 2 messages (requires ROS 2 SDK)

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --packages-select vla_shield_msgs
```

### 5 · Docker — full stack

```bash
cp deploy/.env.example deploy/.env
docker compose -f deploy/docker-compose.yml up -d
```

### 6 · Edge — Jetson / ARM64 (API + DB only)

```bash
cp deploy/edge/.env.jetson.example deploy/edge/.env
docker compose -f deploy/edge/docker-compose.jetson.yml up -d
```

### 7 · Python ↔ Rust extension (`shield_ffi`)

```bash
cd runtime/shield-ffi
maturin develop --release                       # CPU clamp backend
maturin develop --release --features cuda       # enables shield-cuda (needs nvcc)
```

---

## CUDA Acceleration Layer

`shield-cuda` is an **optional** crate that accelerates the action-clamp stage on GPU. It builds **without** CUDA installed (a CPU-ABI-compatible fallback is compiled instead), so the rest of the workspace never breaks.

```
src/lib.rs                  Rust safe wrapper (CudaCtx + clamp_into / clamp_action_cuda)
    │ extern "C"
    ▼
src/kernels/cuda_host.cpp   C++ host glue:
    │                         · ShieldCudaCtx { d_*, h_* (pinned), stream }
    │                         · cudaMalloc once · cudaMemcpyAsync · cudaFree on Drop
    │ extern "C"               · grows buffers transparently when n > capacity
    ▼
src/kernels/clamp_kernel.cu CUDA kernel + thin launcher
    ▼
                            GPU
```

| Mode | Trigger | Backend compiled |
|---|---|---|
| Real GPU | `nvcc` on PATH, default | `clamp_kernel.cu` + `cuda_host.cpp` (cudart linked) |
| Forced CPU | `CUDA_DISABLE=1 cargo build` | `clamp_stub.cpp` |
| No CUDA toolkit | `nvcc` missing | `clamp_stub.cpp` |

Run the built-in micro-benchmark:

```bash
cd runtime
cargo run -p shield-cuda --example bench_clamp --release -- --iters 100000 --dof 8
```

---

## Benchmarking

Two complementary suites live under `benchmark/`:

```bash
# Latency stress-test against the live FastAPI server (HTTP path)
python benchmark/run_latency.py --dof 6 --n-actions 10000

# Latency through the Rust FFI directly (skip HTTP), zero-copy numpy by default
python benchmark/run_latency.py --use-ffi --dof 8 --n-actions 50000
python benchmark/run_latency.py --use-ffi --no-numpy           # A/B vs list path

# Safety recall / precision over the 22-scenario gold set
python benchmark/run_safety.py --scenarios dataset/scenarios/scenarios.jsonl

# Zero-copy vs list-path micro-benchmark on the FFI directly
python benchmark/bench_zero_copy.py --dof 8 --iters 100000
```

`run_latency.py` reports per-stage `p50 / p95 / p99 / mean / max` plus a `budget_violation_rate (> 5 ms)` headline; `run_safety.py` reports `block_recall`, `false_stop_rate`, `hard_block_precision`, and `accuracy`.

---

## Repository Structure

```
vla-shield/
├── README.md
├── LICENSE                                  (Apache 2.0)
│
├── runtime/                                 Rust real-time shield runtime
│   ├── shield-core/                         Ontology, action, arbiter, scene
│   ├── shield-urdf/                         URDF parse, FK, forbidden zones
│   ├── shield-physics/                      Kinematic projector + semantic constraints
│   ├── shield-collision/                    AABB broad-phase pre-check
│   ├── shield-shadow/                       Async joint-space roll-forward simulator
│   ├── shield-cuda/                         Optional GPU clamp — kernel + C++ host + CPU stub
│   ├── shield-ffi/                          PyO3 bridge with numpy zero-copy
│   ├── shield-ros2/                         ROS 2 lifecycle hooks, tf2, pipeline
│   └── shield-io/                           MySQL + Redis I/O
│
├── backend/                                 Python backend (API + ML + evaluation)
│   ├── shield/
│   │   ├── api/
│   │   │   ├── app.py                       FastAPI REST + WebSocket
│   │   │   ├── evaluator.py                 Hot-path evaluator (FFI + Py fallback)
│   │   │   ├── rule_engine.py               RuleRegistry loaded from rules_*.json
│   │   │   └── deps.py                      Redis + MySQL factories
│   │   ├── vfv/                             Visual Feedback Verification predictors
│   │   ├── evaluation/                      Metrics (precision / recall / FPR / F1)
│   │   ├── data/                            Dataset smoke path + validation
│   │   └── schemas.py                       Pydantic models (single source of truth)
│   ├── migrations/                          MySQL DDL + ontology seed
│   └── tests/                               17 tests (schemas, metrics, /v1/evaluate)
│
├── monitor/                                 Real-time safety monitor UI
│   └── src/
│       ├── app/                             Next.js App Router
│       ├── components/
│       │   ├── LatencyChart.tsx             Stacked-bar + sparkline latency view
│       │   ├── RuleViewer.tsx               Live rule table fed by /v1/rules
│       │   ├── WhyBlocked.tsx               Block reasons with trigger + explanation
│       │   ├── RiskGauge.tsx
│       │   └── SceneView.tsx                Three.js shadow trajectory
│       ├── hooks/                           WebSocket telemetry hook
│       └── store/                           Zustand state
│
├── ros2/
│   └── vla_shield_msgs/                     Custom .msg / .srv for ROS 2
│
├── dataset/
│   ├── ontology/
│   │   ├── physical.json + semantic.json    Ontology node definitions
│   │   ├── rules_physical.json              7 PHY.* executable rules
│   │   ├── rules_semantic.json              6 SEM.* executable rules
│   │   └── rule_schema.json                 JSON Schema for the above
│   ├── scenarios/
│   │   ├── scenarios.jsonl                  22 gold scenarios (PHY + SEM + COMBO + PASS)
│   │   └── scenario_spec.md
│   ├── red_team/                            Red-team JSONL schema + samples
│   └── urdf/                                Minimal URDF fixtures for tests
│
├── benchmark/
│   ├── protocol.md                          Metric & methodology spec
│   ├── run_latency.py                       HTTP / FFI / numpy A·B latency suite
│   ├── run_safety.py                        22-scenario recall / precision suite
│   └── bench_zero_copy.py                   list vs numpy FFI micro-bench
│
├── docs/
│   └── openapi/shield-ops-v1.yaml           REST + WebSocket schema
│
└── deploy/
    ├── Dockerfile                           Multi-stage (backend + monitor)
    ├── docker-compose.yml                   MySQL · Redis · API · Monitor
    ├── .env.example
    └── edge/                                Jetson-oriented compose + env template
```

---

## Changelog

Each entry summarises **what changed** during that update — code only, no roadmap fluff.

### v0.4 — Zero-copy FFI & runtime evaluator hardening

- **`shield_ffi.evaluate_numpy`** — new PyO3 method that borrows `&[f32]` / `&[f64]` directly from contiguous `numpy.ndarray`, saving the Python-list → `Vec` conversion (~ 3–5 µs / call on 8-DoF). Shared core extracted into private `evaluate_impl`.
- **Fixed hidden CUDA bug** — `PHY.VELOCITY_LIMIT` was silenced after CUDA pre-clamp because pre-detection read the already-clamped buffer. Refactor now keeps the unclamped input alive for detection.
- **Backend evaluator** auto-detects `evaluate_numpy` and feeds `np.ascontiguousarray(...)`; transparently falls back to the list path or to the Python fallback when the extension is missing.
- **`benchmark/bench_zero_copy.py`** — list vs numpy A/B with p50 / p95 / p99 and median speedup; `run_latency.py` gains `--no-numpy` for the same comparison through the HTTP / FFI path.
- **Test count**: 16 → 17 (new `test_evaluate_numpy_zero_copy_path_matches_list_path`).

### v0.3 — CUDA context: cached buffers, pinned host, persistent stream

- **`ShieldCudaCtx`** — owns three cached device buffers, three `cudaMallocHost` pinned host staging buffers, and one private `cudaStream_t`; transparently grows when `n > capacity`.
- **Two-tier C ABI** — `shield_cuda_clamp` (stateless one-shot) + `shield_cuda_ctx_{create,destroy,clamp}` (hot-path). CPU fallback (`clamp_stub.cpp`) implements **the same ABI**, so the Rust side never special-cases.
- **Rust safe wrapper `CudaCtx`** — `clamp_into(input, limit, &mut output)` does zero allocations on the hot path; `Drop` releases device + pinned + stream.
- **`shield-ffi` integration** — `PyShieldPipeline` constructs `Mutex<CudaCtx>` once per pipeline; **no `cudaMalloc` is ever called on the hot path**.
- **6 integration tests** in `runtime/shield-cuda/tests/ctx.rs` + `cargo run -p shield-cuda --example bench_clamp` micro-bench.

### v0.2 — CUDA host layer split (Rust → C → C++ → CUDA)

- Split `clamp.cu` into **`clamp_kernel.cu`** (pure `__global__` kernel + thin launcher) and **`cuda_host.cpp`** (C++ host glue: `cudaMalloc / cudaMemcpyAsync / cudaFree` via RAII `DeviceBuffer`).
- Fixed a real correctness bug: the previous one-file `clamp.cu` launched the kernel on raw **host pointers**, which is UB on real GPUs; the new host layer round-trips through device memory.
- `build.rs` now compiles the kernel + host pair together when `nvcc` is found, links `cudart`, and supports `CUDA_DISABLE=1` to force the CPU fallback even on GPU machines.
- `rustc-check-cfg(cfg(has_cuda_kernel))` to silence the Rust 1.80+ unknown-cfg warning.
- Replaced 2024-edition `unsafe extern "C" { … }` syntax with 2021-compatible form so the workspace builds on the project's pinned edition.

### v0.1.5 — `vlashield` → `shield` workspace-wide rename

- Renamed every `vlashield-*` crate / folder / file / identifier (Rust crate names, Python package, Cargo deps, Docker compose env, OpenAPI spec, monitor package.json — 63 files touched, 12 directories renamed). Marketing string "VLA-Shield" preserved.
- Python imports normalised to module top per PEP 8; deferred imports inside functions removed.

### v0.1.4 — Plan-review fix pass (closing the loop)

- **Executable rule engine** — new `backend/shield/api/rule_engine.py` loads `rules_*.json` into a `RuleRegistry`; the evaluator now uses rule-driven action (`block | clamp | warn`) and fills `{joint_name}` / `{requested}` / `{limit}` slots from runtime values.
- **`/v1/evaluate` & `/v1/rules`** REST endpoints added; OpenAPI schema updated.
- **Length-tolerant `current_joints`** — pad / truncate to match action length instead of crashing the FFI.
- **Atomic Redis pipeline** (`SETEX + HSET + EXPIRE` in one transaction) replaces the per-call multi-round-trip; **MySQL pool** now closed in lifespan teardown.
- **Backend evaluator preserves FFI per-stage latency**; only fills in `shadow_ms` and `total_ms`.
- **Benchmark scenarios** now map scenario-id strings (`PHY-001`) to integer sequence-ids and pass `current_joints` through.
- **API regression suite** — `test_evaluate_api.py` (FastAPI `TestClient` + fake-redis + aiomysql stub) covering `/v1/rules`, `/v1/evaluate` PASS / BLOCK, missing `current_joints`, empty action, rule-template rendering.

### v0.1.3 — `shield-ffi` rule-driven pre-detection

- Pre-detection loop in FFI explicitly emits `PHY.VELOCITY_LIMIT` and `PHY.JOINT_LIMIT` when raw input exceeds the limit, even though the projector silently clamps; projector errors are now routed to `FORBIDDEN_ZONE` / `JOINT_LIMIT` deterministically.

### v0.1.2 — Hot/Async split + monitor enhancements

- **`shield-shadow`** new Rust crate: `JointSpaceSimulator` runs multi-step roll-forward as an async risk prior; the `SafetyPipeline` invokes it stale-safely.
- **`shield-physics::semantic`** — `SemanticConstraintMapper` maps `SEM.HEAT_SOURCE` / `FORBIDDEN_REGION` / `LIQUID_ELECTRICAL` to AABB exclusion zones and `SEM.HUMAN_PROXIMITY` to a velocity cap, consumed by `KinematicClampProjector`.
- **`LatencyBreakdown`** expanded to 8 fields (`ingest_ms · urdf_fk_ms · physics_ms · collision_ms · tf2_ms · arbiter_ms · shadow_ms · total_ms`) end-to-end through Rust, OpenAPI, Pydantic, and the monitor's Zustand store.
- **Monitor** — new `LatencyChart` (stacked bar + sparkline + budget marker), new `RuleViewer` (filter / expand, fed by `/v1/rules`), upgraded `WhyBlocked` (trigger condition + runtime-filled explanation).
- **DB seed fix** — added `PHY.JOINT_LIMIT`, `PHY.SINGULARITY`, `PHY.FORBIDDEN_ZONE` nodes that were missing from `data.sql`.

### v0.1.1 — Benchmark suite & scenario gold set

- `benchmark/protocol.md` + `benchmark/run_latency.py` + `benchmark/run_safety.py` for HTTP and FFI paths.
- `dataset/scenarios/scenarios.jsonl` — 22 gold scenarios covering all 13 ontology nodes (`PHY-*`, `SEM-*`, `COMBO-*`, `PASS-*`).

### v0.1 — Initial scaffold

- Six baseline Rust crates (`shield-core / urdf / physics / collision / ros2 / io`), FastAPI backend, Next.js monitor, MySQL + Redis I/O, Docker + edge compose, ontology JSON (13 nodes), red-team JSONL schema.

---

## Data

**Manual samples** (`dataset/red_team/samples.jsonl`): bilingual (EN / ZH) entries for smoke testing.

**Scenario gold set** (`dataset/scenarios/scenarios.jsonl`): 22 high-risk scenarios with `injected_action`, `current_joints`, `expected_decision`, and `risk_tags` covering all 13 ontology nodes.

**Initialize a working red-team JSONL from samples:**

```bash
cd backend
pip install -e ".[data]"
python -m shield.data.download
python -m shield.data.validate --data ../dataset/red_team/public.jsonl
```

---

## Citation

```bibtex
@misc{vla-shield-2026,
  title        = {VLA-Shield: A Decoupled Real-Time Safety Filter Layer
                  with Semantic-to-Physics Projection for
                  Vision-Language-Action Policies},
  author       = {The VLA-Shield Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/your-org/vla-shield}},
  note         = {Technical design v0.4}
}
```

---

## License

Licensed under **Apache License 2.0**. See [LICENSE](LICENSE) for details.
