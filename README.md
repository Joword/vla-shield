# VLA-Shield

[![Version](https://img.shields.io/badge/version-0.6.4-0A7EA4)](https://github.com/Joword/vla-shield) [![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20Jazzy-22314E?logo=ros)](https://docs.ros.org/) [![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust)](https://www.rust-lang.org/) [![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=next.js)](https://nextjs.org/) [![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia&logoColor=white)](#cuda-acceleration-layer)

**VLA-Shield** is a model-agnostic runtime safety filter for Vision-Language-Action (VLA) policies. It sits after the policy and before the robot: intercept the raw action, project it into kinematics / dynamics, enforce **hard** ontology rules, and emit PASS / CLAMP / BLOCK — without touching model weights.

![Figure 1. VLA-Shield sits between an unchanged VLA policy and the robot / ops stack.](docs/figures/placement.png)

<p align="center"><em>Figure 1. Placement: unchanged policy weights in, PASS / CLAMP / BLOCK out. The &lt; 5&nbsp;ms number is a hot-path budget, not a published latency curve.</em></p>

This is **0.6.4** — a lab-runnable stack (gold **22 / 22**, four replay traces). It is not a measured Jetson closed loop.

---

## Why this exists

Training-time alignment (RLHF, DPO, SafeVLA) changes the policy. VLA-Shield does not. It is middleware you can put in front of OpenVLA, RT-2, Octo, or Diffusion Policy on a CPU — GPU is optional.

- **Zero-touch on the policy** — no fine-tune, no weight patch, no extra training cluster.
- **Hard rules, not a hope** — 13 ontology nodes (`PHY.*` × 7, `SEM.*` × 6) with JSON `trigger_condition`, typed `threshold`, and `action ∈ {block, clamp, warn}`.
- **Hot path vs async priors** — URDF FK, joint / velocity / acceleration envelopes, AABB broad-phase + narrow confirm, singularity / tip-over / overload run synchronously. Shadow roll-forward and VFV stay off the budget and feed **stale-safe** priors.
- **Gold set you can rerun** — 22 scenarios in `dataset/scenarios/scenarios.jsonl` (in-process evaluator matches all 22). `traces.jsonl` replays four labeled traces through the same path.
- **Operators see why** — Redis Stream → WebSocket → Next.js monitor (why-blocked, rule table, 3D twin).

| Approach | Modifies model? | When it runs | Deterministic? | Hardware |
|----------|:---------------:|:------------:|:--------------:|:--------:|
| RLHF / DPO | Yes | Training | No | Multi-GPU cluster |
| SafeVLA | Yes | Training / inference | Soft constraints | Multi-GPU cluster |
| Safety-CHORES | Yes (fine-tune) | Inference | No | GPU + retraining |
| **VLA-Shield** | **No** | **Runtime (budget &lt; 5 ms)** | **URDF + rule engine** | **CPU; GPU optional** |

---

## What 0.6.4 actually contains

| Piece | Status |
|---|---|
| Hot path | URDF FK AABBs → optional CUDA N×M mask → deflate confirm + self-collision (skip two adjacent links) → arbiter |
| Dynamics | `ActionProposal.prev_velocity`: empty skips accel; otherwise each joint is clamped to `prev ± a_max * dt` after the velocity cap |
| PHY extras | `PHY.SINGULARITY` / `TIPOVER` / `OVERLOAD` from `dataset/ontology/phy_calibration.json` (same file Rust `include_str!`s and Python loads). Fitted to shipped URDFs + gold PHY-004/006/007 — not a dynamometer ID |
| VFV | Hints first, then RGB cues from `dataset/ontology/vfv_cues.json` (floors aligned with `rules_semantic.json` `min_score`). CLIP only if `SHIELD_VFV_CLIP=1` |
| ROS 2 | In-process `ShieldRosOverlay` plus bind table (`overlay_bindings()` / `shield.ros2.OVERLAY_BINDINGS`). No distro required to test. `rclrs` still needs a ROS overlay workspace (`--features ros2`) |
| CUDA | Same C ABI with or without nvcc. Missing Windows `cl.exe` is a **handled exception** → `clamp_stub.cpp`; the crate still builds |
| Not in this release | Jetson hot-path p99 &lt; 5 ms, mesh narrow-phase, a live `rclrs` graph on this machine |

---

## Architecture

Runtime is a **filter**, not a second policy. Panel (a) is the stack a command actually walks; panel (b) is the five-stage hot path plus async priors that must not stall it.

![Figure 2. Runtime architecture: layered stack and five-stage hot path.](docs/figures/architecture.png)

<p align="center"><em>Figure 2. (a) Layered stack with optional CUDA clamp for n ≥ 64. (b) Clamp → project → FK → collision → decide; shadow / VFV remain stale-safe.</em></p>

| Layer | Crate / module | Role |
|---|---|---|
| Python FFI | `shield-ffi` | PyO3 + numpy zero-copy; optional `prev_velocity`; per-pipeline CUDA context |
| CUDA (optional) | `shield-cuda` | Action clamp + AABB overlap mask; CPU ABI fallback |
| Domain | `shield-core` | Ontology, action, arbiter |
| Physics | `shield-physics` | Kinematic projector (vel + accel) + PHY.SINGULARITY / TIPOVER / OVERLOAD |
| Kinematics | `shield-urdf` | URDF parse, FK, forbidden zones |
| Collision | `shield-collision` | AABB broad-phase, deflate confirm, self-collision |
| Shadow | `shield-shadow` | Async joint-space roll-forward |
| ROS 2 | `shield-ros2` | Overlay, QoS bind table, lifecycle, tf2 (rclrs feature-gated) |
| Storage | `shield-io` | MySQL audit + Redis telemetry |
| Backend | `backend/shield/api` | FastAPI + Python evaluator fallback |
| VFV | `backend/shield/vfv` | Visual semantic risk (off hot path) |
| ROS Python | `backend/shield/ros2` | Overlay bindings; in-process node if rclpy is missing |
| Monitor | `monitor/` | Next.js + Three.js digital twin |

---

## Gold set (real counts)

22 rows in `dataset/scenarios/scenarios.jsonl`. Decisions: **BLOCK 13 · CLAMP 1 · WARN 3 · PASS 5**. Families: PHY 8, SEM 6, COMBO 3, PASS 5. All **13** ontology nodes appear at least once. PASS rows exist so false-stop rate is measurable.

![Figure 3. Gold-set coverage from scenarios.jsonl.](docs/figures/gold-set.png)

<p align="center"><em>Figure 3. Expected-decision mix, scenario family, and ontology coverage. Not a field study; not a latency plot.</em></p>

In-process `test_gold.py`: **22 / 22** expected decisions, with `risk_tags ⊆ ontology_ids`. `test_traces.py` replays four traces in `dataset/scenarios/traces.jsonl` (PASS, collision BLOCK, velocity CLAMP, joint BLOCK). Latency vs the 5 ms budget is `benchmark/run_latency.py --use-ffi`, not a figure here.

---

## Monitor

![Figure 4. Safety monitor (concept mock).](docs/figures/monitor.png)

<p align="center"><em>Figure 4. Concept mock of the Next.js console — risk gauge, 3D twin, why-blocked, stacked latency, live rules. Not a live capture from this repo.</em></p>

---

## Quick Start

### 1 · Rust runtime

```bash
cd runtime
cargo build --workspace
cargo test  --workspace
```

On Windows without MSVC `link.exe`, use the gnu toolchain (does **not** change rustup's default):

```bash
# MinGW gcc on PATH
set RUSTUP_TOOLCHAIN=stable-x86_64-pc-windows-gnu
cargo test --workspace
```

Optional ROS 2 Rust bindings:

```bash
cargo build -p shield-ros2 --features ros2     # requires ROS 2 + Rust overlay
```

Without a distro, `ShieldRosOverlay` is the testable graph: subscribe
`/vla_shield/action` + `/joint_states`, publish `/vla_shield/decision`
(keep-last 1). Same topic names as `vla_shield_msgs`. Bind table:
`overlay_bindings()` / Python `shield.ros2.OVERLAY_BINDINGS`.

### 2 · Python backend

```bash
cd backend
pip install -e ".[dev]"
pytest
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

`ActionProposal.msg` carries `float32[] prev_velocity` (empty = skip acceleration clamp).

### 5 · Docker — full stack

```bash
cp deploy/.env.example deploy/.env
docker compose -f deploy/docker-compose.yml up -d
```

### 6 · Edge — Jetson / ARM64 (API + DB only)

Compose templates live under `deploy/edge/`. There is **no** published p99 from a Jetson in this repo.

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

On Windows without MSVC `link.exe`:

```bash
set RUSTUP_TOOLCHAIN=stable-x86_64-pc-windows-gnu
maturin build --release
pip install --force-reinstall --no-deps ../target/wheels/shield_ffi-0.6.4-*.whl
```

---

## CUDA Acceleration Layer

`shield-cuda` accelerates **action clamp** and **AABB overlap**. It builds **without** CUDA installed (a CPU-ABI-compatible fallback is compiled instead), so the rest of the workspace never breaks.

```
src/lib.rs                  Rust: CudaCtx clamp + aabb_overlap_mask
    │ extern "C"
    ▼
src/kernels/cuda_host.cpp   C++ host glue (cached clamp ctx; AABB H↔D)
    │
    ├─ clamp_kernel.cu      per-element clamp
    └─ aabb_kernel.cu       N×M AABB mask (GPU when pairs ≥ 64)
```

`AabbBroadPhase` calls `aabb_overlap_mask` so collision pair reasons stay `(link, obstacle)`. Tiny scenes stay on the host loop; dense scenes take the kernel when nvcc built the crate. `gpu_kernel_compiled()` / `cpu_stub_compiled()` report which backend `build.rs` actually linked.

| Mode | Trigger | What runs |
|---|---|---|
| Real GPU | `nvcc` compile succeeds | `clamp_kernel.cu` + `aabb_kernel.cu` + `cuda_host.cpp` |
| Host-compiler miss | `nvcc` throws (e.g. no `cl.exe` on Windows) | `clamp_stub.cpp` — handled exception, crate still builds |
| Forced CPU stub | `CUDA_DISABLE=1 cargo build` | `clamp_stub.cpp` |
| No CUDA toolkit | `nvcc` missing | `clamp_stub.cpp` |
| Small-n bypass | `n < min_gpu_n` (default 64) | Rust scalar loop — **no C ABI, no kernel** |
| Force backend | `SHIELD_CUDA_MIN_GPU_N=0` or `set_min_gpu_n(0)` | C ABI even for tiny `n` (A/B benches) |

```bash
cd runtime
cargo run -p shield-cuda --example bench_clamp --release -- --iters 100000 --dof 8
cargo run -p shield-cuda --example bench_clamp --release -- --iters 100000 --dof 8 --force-gpu
cargo run -p shield-cuda --example bench_aabb --release -- --iters 20000 --n-links 8 --n-obs 8
cargo run -p shield-cuda --example bench_aabb --release -- --iters 5000 --n-links 32 --n-obs 32
```

---

## Benchmarking

```bash
# Latency against the live FastAPI server (HTTP path)
python benchmark/run_latency.py --dof 6 --n-actions 10000

# Latency through the Rust FFI (skip HTTP); numpy zero-copy by default
python benchmark/run_latency.py --use-ffi --dof 8 --n-actions 50000
python benchmark/run_latency.py --use-ffi --no-numpy

# Safety recall / precision over the 22-scenario gold set
python benchmark/run_safety.py --scenarios dataset/scenarios/scenarios.jsonl

python benchmark/bench_zero_copy.py --dof 8 --iters 100000
```

`run_latency.py` reports per-stage `p50 / p95 / p99 / mean / max` plus `budget_violation_rate (> 5 ms)`. `run_safety.py` reports `block_recall`, `false_stop_rate`, `hard_block_precision`, and `accuracy`.

---

## Repository Structure

```
vla-shield/
├── README.md
├── LICENSE                                  (Apache 2.0)
├── runtime/                                 Rust hot-path crates (9)
├── backend/                                 FastAPI + Python evaluator + VFV + ROS overlay
├── monitor/                                 Next.js safety console
├── ros2/vla_shield_msgs/                    Custom .msg / .srv
├── dataset/
│   ├── ontology/                            PHY.* / SEM.* nodes, rules, phy_calibration.json, vfv_cues.json
│   ├── scenarios/scenarios.jsonl            22 gold scenarios
│   ├── scenarios/traces.jsonl               4 replay traces
│   ├── red_team/                            Red-team JSONL + samples
│   └── urdf/                                Minimal URDF fixtures
├── benchmark/                               Latency + gold-set safety suites
├── docs/
│   ├── figures/                             README figures (placement, architecture, gold set, monitor)
│   └── openapi/shield-ops-v1.yaml           REST + WebSocket schema
└── deploy/                                  Docker Compose + Jetson edge templates
```

---

## Data

**Manual samples** (`dataset/red_team/samples.jsonl`): bilingual (EN / ZH) entries for smoke testing.

**Scenario gold set** (`dataset/scenarios/scenarios.jsonl`): 22 high-risk scenarios with `injected_action`, `current_joints`, `expected_decision`, and `risk_tags` covering all 13 ontology nodes.

**Calibration** (`dataset/ontology/phy_calibration.json`, `vfv_cues.json`): coefficients shared by Rust and Python. Override cue floors with `SHIELD_VFV_CUES`.

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
