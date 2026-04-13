# VLA-Shield

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble%20%7C%20Jazzy-22314E?logo=ros)](https://docs.ros.org/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=next.js)](https://nextjs.org/)

**VLA-Shield** is a model-agnostic, real-time safety filter layer for Vision-Language-Action (VLA) policies. Unlike internal alignment approaches (RLHF/DPO), VLA-Shield operates as a **decoupled runtime middleware** that intercepts raw action outputs, projects them into physical dynamics space, and enforces hard safety constraints --- all within a **< 5 ms** latency budget --- without modifying the base VLA model.

> **Status:** Full project scaffold with working types, traits, and tests. Implementation follows the [milestones](#milestones) below.

---

## Key Differentiators

| Approach | Modifies Model? | Latency | Deterministic? |
|----------|:---------------:|:-------:|:--------------:|
| RLHF/DPO alignment | Yes | Training-time | No |
| Safety-CHORES (arXiv:2503.03480) | Yes (fine-tuning) | Inference-time | No |
| **VLA-Shield (ours)** | **No** | **< 5 ms runtime** | **Yes (physics core)** |

- **Model-Agnostic** --- plug into any VLA (OpenVLA, RT-2, Octo) without retraining.
- **Semantic-to-Physics Projection** --- translates abstract action vectors into physical-space coordinates for deterministic collision/kinematics checking.
- **Shadow Path Pre-check** --- lightweight physics core runs a millisecond-level "shadow trajectory" before the real command is dispatched.
- **Emergency Soft Landing** --- when VLM-based secondary verification detects ambiguous risk (e.g., distinguishing "pour water" vs. "pour oil"), triggers a graceful deceleration protocol.
- **Off-policy Shielding** --- filter decisions serve as external penalty signals to guide base model preferences without altering architecture.

---

## Architecture

```
VLA Model ──> VLAShield-Runtime (Rust) ──> ROS 2 Control Stack
                 │                                │
          Semantic-to-Physics              Safe Action
           Projection                     or Soft Landing
          Shadow Path Pre-check                   │
          Collision / Kinematics                   ▼
                 │                           Robot Actuators
          Arbiter Decision
                 │
            Safety Event ──> MySQL + Qdrant
                 │
            Redis Stream ──> WebSocket ──> Monitor UI (Next.js + Three.js)
```

| Layer | Tech Stack | Key Metric |
|-------|-----------|------------|
| **Shield Runtime** | Rust, ROS 2, Fast-DDS | Filter latency < 5 ms |
| **Physics Core** | Kinematics engine, AABB broad-phase | Collision prediction > 98% |
| **Visual Verifier** | CLIP, VLM predictor | Semantic risk detection > 95% |
| **Backend** | Python, FastAPI, PyTorch | API + Safe-RL training |
| **Storage** | MySQL, Redis, Qdrant | 1M+ action log retrieval |
| **Monitor** | Next.js, Three.js, Tailwind | Real-time shadow trajectory @ 30 FPS |

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

---

## Repository Structure

```
vla-shield/
├── README.md
├── LICENSE                          (Apache 2.0)
│
├── runtime/                         Rust real-time shield runtime
│   ├── vlashield-core/              Core types, ontology, action, arbiter
│   ├── vlashield-physics/           Semantic-to-physics projection, kinematic clamping
│   ├── vlashield-collision/         Shadow path pre-check, AABB broad-phase
│   ├── vlashield-ros2/              ROS 2 pipeline, Fast-DDS integration
│   └── vlashield-io/                MySQL + Redis I/O layer
│
├── backend/                         Python backend (API + ML + evaluation)
│   ├── vlashield/
│   │   ├── api/                     FastAPI REST + WebSocket server
│   │   ├── vfv/                     Visual Feedback Verification (VLM)
│   │   ├── training/                Off-policy shielding, Safe-RL penalty
│   │   ├── evaluation/              Safety benchmark metrics
│   │   ├── data/                    Dataset download + validation
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
├── dataset/                         Shared data (ontology + red-team)
│   ├── ontology/                    Physical & semantic safety node definitions
│   └── red_team/                    Schema, samples, downloaded data
│
└── deploy/                          Deployment configuration
    ├── Dockerfile                   Multi-stage (backend + monitor)
    ├── docker-compose.yml           MySQL, Redis, Qdrant, API, monitor
    └── .env.example
```

---

## Research Phases

| Phase | Period | Focus |
|-------|--------|-------|
| **Phase 1** | Month 1-3 | Multi-dimensional safety ontology + dynamic scene risk dataset |
| **Phase 2** | Month 4-6 | Rust shield runtime + semantic-to-physics projection + shadow path pre-check |
| **Phase 3** | Month 7-9 | VLM-based pre-execution visual verification + off-policy shielding feedback |
| **Phase 4** | Month 10-12 | Real-time monitor with shadow trajectory rendering + deployment on Jetson/RPi |

---

## Milestones

| Version | Deliverables |
|---------|-------------|
| **v0.1-alpha** `VLA-Shield-Ontology` | Safety knowledge graph, physical constraint protocol, red-team dataset |
| **v0.5-beta** `Shield-Runtime-RS` | Rust ROS 2 plugin, action interception, latency benchmarks |
| **v1.0** `VLA-Shield-Agent` | Full shield suite with physics + visual verification, one-click deploy (Jetson/RPi/NUC) |

---

## Data

**Manual samples** (`dataset/red_team/samples.jsonl`): 10 bilingual (EN/ZH) entries for smoke testing.

**Public datasets** (downloaded via backend):

```bash
cd backend
pip install -e ".[data]"
python -m vlashield.data.download --source all --max-samples 5000
python -m vlashield.data.validate --data ../dataset/red_team/public.jsonl
```

Sources: [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails), [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF).

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

Built on the open-source robotics, Rust, and ML ecosystems. Inspired by the safety challenges identified in end-to-end VLA deployment research.
