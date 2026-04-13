# VLA-Shield

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20Jazzy-22314E?logo=ros)](https://docs.ros.org/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=next.js)](https://nextjs.org/)

**VLA-Shield** is a model-agnostic, real-time safety filter layer for Vision-Language-Action (VLA) policies. Unlike **training-time** alignment (e.g. SafeVLA-style constraints on the policy), VLA-Shield operates as a **decoupled runtime middleware** that intercepts raw action outputs, projects them into physical dynamics space, and enforces **hard** safety constraints --- within a **&lt; 5 ms** latency budget --- **without modifying** the base VLA model weights.

> **Status:** Full project scaffold with working types, traits, and tests. Implementation follows the [milestones](#milestones) below.

---

## Key Differentiators

| Approach | Modifies Model? | Latency | Deterministic? |
|----------|:---------------:|:-------:|:--------------:|
| RLHF/DPO alignment | Yes | Training-time | No |
| SafeVLA (training-time safety) | Yes | Training / inference | No (soft constraints) |
| Safety-CHORES (arXiv:2503.03480) | Yes (fine-tuning) | Inference-time | No |
| **VLA-Shield (ours)** | **No** | **&lt; 5 ms runtime** | **Yes (URDF / geometry core)** |

- **Model-Agnostic** --- plug into any VLA (OpenVLA, RT-2, Octo) without retraining.
- **Semantic-to-Physics Projection** --- maps action vectors to joint / Cartesian checks.
- **URDF-based Kinematic Verification** --- forward kinematics and joint limits from the robot model (not a learned cost alone).
- **ROS 2 Lifecycle Hooks** --- `ShieldLifecycleHooks` maps to configure / activate / deactivate for highest-priority interception patterns.
- **Shadow Path Pre-check** --- lightweight joint-space roll-forward before dispatch.
- **Emergency Soft Landing** --- when VLM-based secondary verification flags ambiguous semantic risk, trigger a graceful deceleration protocol.

---

## Architecture

```
VLA Model ──> VLAShield-Runtime (Rust) ──> ROS 2 Control Stack
                 │                                │
          Semantic-to-Physics              Safe Action
           Projection                     or Soft Landing
          URDF FK + Singularity hint              │
          tf2 World-frame zone check              ▼
          Shadow Path Pre-check              Robot Actuators
          Collision / Kinematics
                 │
          Arbiter Decision
                 │
            Safety Event ──> MySQL
                 │
            Redis Stream ──> WebSocket ──> Monitor UI (Next.js + Three.js)
```

| Layer | Tech Stack | Key Metric |
|-------|-----------|------------|
| **Shield Runtime** | Rust, ROS 2, Fast-DDS | Filter latency &lt; 5 ms |
| **Physics Core** | URDF FK, AABB broad-phase | Collision / FK checks |
| **Visual Verifier** | VLM / CLIP (optional path) | Semantic risk (VFV) |
| **Backend** | Python, FastAPI, PyTorch | API + evaluation |
| **Storage** | MySQL, Redis | Action / event log |
| **Monitor** | Next.js, Three.js, Tailwind | Real-time telemetry &gt; 30 FPS |

---

## Quick Start

### Clone

```bash
git clone https://github.com/Joword/vla-shield.git
cd vla-shield
```

### Rust Runtime

```bash
cd runtime
cargo build --workspace
cargo test --workspace
```

Optional ROS 2 Rust bindings: `cargo build -p vlashield-ros2 --features ros2` (requires a ROS 2 + Rust overlay).

### Python Backend

```bash
cd backend
pip install -e ".[dev]"
pytest
```

### Safety Monitor (Next.js)

```bash
cd monitor
yarn install
yarn dev
```

### ROS 2 Messages (requires ROS 2 SDK)

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --packages-select vla_shield_msgs
```

### Docker (all services)

```bash
cp deploy/.env.example deploy/.env
docker compose -f deploy/docker-compose.yml up -d
```

### Edge (Jetson / ARM64, API + DB only)

```bash
cp deploy/edge/.env.jetson.example deploy/edge/.env
docker compose -f deploy/edge/docker-compose.jetson.yml up -d
```

---

## Repository Structure

```
vla-shield/
├── README.md
├── LICENSE                          (Apache 2.0)
│
├── runtime/                         Rust real-time shield runtime
│   ├── vlashield-core/              Core types, ontology, action, arbiter
│   ├── vlashield-urdf/              URDF parse, FK, forbidden zones
│   ├── vlashield-physics/           Semantic-to-physics projection, kinematic clamping
│   ├── vlashield-collision/         Shadow path pre-check, AABB broad-phase
│   ├── vlashield-ros2/              ROS 2 pipeline, lifecycle hooks, tf2 validator
│   └── vlashield-io/                MySQL + Redis I/O layer
│
├── backend/                         Python backend (API + ML + evaluation)
│   ├── vlashield/
│   │   ├── api/                     FastAPI REST + WebSocket server
│   │   ├── vfv/                     Visual Feedback Verification (VLM + shadow sim)
│   │   ├── evaluation/              Safety benchmark metrics
│   │   ├── data/                    Dataset smoke path + validation
│   │   └── schemas.py               Pydantic models (single source of truth)
│   ├── migrations/                  MySQL DDL (data.sql)
│   └── tests/
│
├── monitor/                         Real-time safety monitor UI
│   └── src/
│       ├── app/                     Next.js App Router
│       ├── components/              RiskGauge, WhyBlocked, SceneView (shadow trajectory)
│       ├── hooks/                   WebSocket telemetry hook
│       └── store/                   Zustand state management
│
├── ros2/
│   └── vla_shield_msgs/             Custom .msg / .srv for ROS 2
│
├── dataset/                         Shared data (ontology + red-team + URDF samples)
│   ├── ontology/                    Physical & semantic safety node definitions
│   ├── red_team/                    Schema, samples, generated data
│   └── urdf/                        Minimal URDF fixtures for tests
│
└── deploy/                          Deployment configuration
    ├── Dockerfile                   Multi-stage (backend + monitor)
    ├── docker-compose.yml           MySQL, Redis, API, monitor
    ├── .env.example
    └── edge/                        Jetson-oriented compose + env template
```

---

## Research Phases

| Phase | Period | Focus |
|-------|--------|-------|
| **Phase 1** | Month 1-3 | Physical safety ontology + URDF-linked constraints; embodied risk data |
| **Phase 2** | Month 4-6 | Rust shield runtime + ROS 2 lifecycle + tf2 world-frame checks + shadow path |
| **Phase 3** | Month 7-9 | VLM pre-execution visual verification (VFV) + shadow trajectory reference |
| **Phase 4** | Month 10-12 | Digital-twin monitor + edge deploy (Jetson / RPi) + latency evaluation |

---

## Milestones

| Version | Deliverables |
|---------|-------------|
| **v0.1-alpha** `VLA-Shield-Ontology` | Safety knowledge graph, physical constraint protocol, red-team dataset |
| **v0.5-beta** `Shield-Runtime-RS` | Rust ROS 2 plugin, action interception, latency benchmarks |
| **v1.0** `VLA-Shield-Agent` | Full shield suite with physics + visual verification, one-click deploy (Jetson/RPi/NUC) |

---

## Data

**Manual samples** (`dataset/red_team/samples.jsonl`): bilingual (EN/ZH) entries for smoke testing.

**Embodied / robotics safety datasets** for red-team JSONL are integrated in Phase 1. Until then, initialize a working file from samples:

```bash
cd backend
pip install -e ".[data]"
python -m vlashield.data.download
python -m vlashield.data.validate --data ../dataset/red_team/public.jsonl
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
  note         = {Technical design v0.1-draft}
}
```

---

## Contributing

- **[Contributing Guide](docs/CONTRIBUTING.md)** --- development setup, PR requirements, code style.
- **[Code of Conduct](docs/CODE_OF_CONDUCT.md)** --- Contributor Covenant v2.1.
- **[Security Policy](docs/SECURITY.md)** --- responsible disclosure.

---

## License

Licensed under **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built on the open-source robotics, Rust, and ML ecosystems. This work targets **deployment-time**, **hard** safety guarantees orthogonally to training-time approaches such as **SafeVLA** and related alignment methods.
