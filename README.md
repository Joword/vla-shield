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
- **🎮 Per-pipeline CUDA context** — when GPU is enabled, `shield-cuda` holds cached device buffers, pinned host staging buffers, and a private CUDA stream per pipeline; **no `cudaMalloc` ever happens on the hot path**. Vectors shorter than `min_gpu_n` (default 64) **never leave the CPU**, so 6–14 DoF arms skip kernel launch entirely.
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
   ║   │ n<64 → CPU   │   │ projector +  │   │ singularity  │   │ phase        │  │ ║
   ║   │ else GPU ctx │   │ semantic     │   │ + forbidden  │   │ pre-check    │  │ ║
   ║   │ pinned+stream│   │ constraints  │   │ zones        │   │              │  │ ║
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
                                                          │  │ SceneView skeleton │  │
                                                          │  │ + shadow + zones   │  │
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

| Mode | Trigger | What runs |
|---|---|---|
| Real GPU | `nvcc` on PATH, default | `clamp_kernel.cu` + `cuda_host.cpp` (cudart linked) |
| Forced CPU stub | `CUDA_DISABLE=1 cargo build` | `clamp_stub.cpp` |
| No CUDA toolkit | `nvcc` missing | `clamp_stub.cpp` |
| Small-n bypass | `n < min_gpu_n` (default 64) | Rust scalar loop — **no C ABI, no kernel** |
| Force backend | `SHIELD_CUDA_MIN_GPU_N=0` or `set_min_gpu_n(0)` | C ABI even for tiny `n` (A/B benches) |

Run the built-in micro-benchmark:

```bash
cd runtime
# Typical arm DoF — CPU bypass should beat the GPU hop
cargo run -p shield-cuda --example bench_clamp --release -- --iters 100000 --dof 8
# Same DoF, force the C backend for the A/B number
cargo run -p shield-cuda --example bench_clamp --release -- --iters 100000 --dof 8 --force-gpu
# Wide vector — backend is allowed by default (n ≥ 64)
cargo run -p shield-cuda --example bench_clamp --release -- --iters 20000 --dof 256
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
│       │   └── SceneView.tsx                Three.js skeleton + shadow path + zones
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
  howpublished = {\url{https://github.com/Joword/vla-shield}}
}
```

---

## License

Licensed under **Apache License 2.0**. See [LICENSE](LICENSE) for details.
