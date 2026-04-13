# Contributing to VLA-Shield

Thank you for your interest in contributing! This document provides guidelines for contributing to the VLA-Shield project.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a feature branch from `main`.
3. Make your changes, ensuring tests pass.
4. Submit a pull request.

## Development Setup

### Rust runtime (runtime/)

```bash
cd runtime
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

Optional ROS 2 bindings (requires ROS 2 SDK + Rust overlay):

```bash
cargo build -p vlashield-ros2 --features ros2
```

### Python backend (backend/)

```bash
cd backend
pip install -e ".[dev]"
pytest
ruff check .
mypy vlashield/
```

### Safety monitor UI (monitor/)

```bash
cd monitor
yarn install
yarn dev
yarn lint
```

### Infrastructure (local)

```bash
cp deploy/.env.example deploy/.env
docker compose -f deploy/docker-compose.yml up -d
```

### Edge deployment (Jetson / ARM64)

```bash
cp deploy/edge/.env.jetson.example deploy/edge/.env
docker compose -f deploy/edge/docker-compose.jetson.yml up -d
```

## Project Structure

| Directory | Language | Purpose |
|-----------|----------|---------|
| `runtime/vlashield-core` | Rust | Core types, ontology, action, arbiter |
| `runtime/vlashield-urdf` | Rust | URDF parsing, forward kinematics, forbidden zones |
| `runtime/vlashield-physics` | Rust | Physical projection with URDF FK integration |
| `runtime/vlashield-collision` | Rust | AABB broad-phase collision pre-check |
| `runtime/vlashield-ros2` | Rust | ROS 2 pipeline, lifecycle hooks, tf2 validator |
| `runtime/vlashield-io` | Rust | MySQL + Redis I/O |
| `backend/vlashield` | Python | FastAPI server, VFV, evaluation |
| `monitor/` | TypeScript | Next.js + Three.js dashboard |
| `ros2/vla_shield_msgs` | ROS IDL | Custom messages and services |
| `dataset/` | JSON/URDF | Ontology, red-team data, URDF test fixtures |

## Pull Request Requirements

- All CI checks must pass (fmt, clippy, tests, lint).
- New features should include tests.
- Update documentation if the public API changes.
- Sign your commits with DCO (`git commit -s`) or use the GitHub DCO app.

## Code Style

- **Rust**: Follow `rustfmt` defaults. Use `clippy` without warnings.
- **Python**: Follow `ruff` defaults. Type annotations required for public APIs.
- **TypeScript**: Follow ESLint / Next.js config in `monitor/`.

## Reporting Issues

- Use **GitHub Issues** for bugs and feature requests.
- Use **GitHub Discussions** for questions and design proposals.
- For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
