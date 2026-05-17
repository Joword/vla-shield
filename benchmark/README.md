# VLA-Shield Benchmark Suite

Reproducible evaluation tools for the VLA-Shield runtime safety filter.

## Structure

```
benchmark/
├── protocol.md          # Full metric definitions and experimental setup
├── run_latency.py       # Hot-path latency stress test (10 k actions)
├── run_safety.py        # Safety recall/precision evaluation from scenarios
└── scenarios/           # JSONL scenario definitions (subset of dataset/scenarios/)
```

## Quick Start

```bash
# 1. Start the VLA-Shield backend (see backend/README.md)
cd backend && uvicorn shield.api.app:app --reload

# 2. Run latency benchmark (requires backend running on localhost:8000)
python benchmark/run_latency.py --dof 6 --n-actions 5000

# 3. Run safety benchmark
python benchmark/run_safety.py --scenarios dataset/scenarios/scenarios.jsonl
```

## Evaluation Configurations

| Config | Env var | Description |
|--------|---------|-------------|
| VLA-Shield (full) | `SHIELD_MODE=production` | Projection + collision + arbiter |
| Physics only | `SHIELD_MODE=physics_only` | No semantic/VFV checks |
| Monitor (no block) | `SHIELD_MODE=monitor` | All checks, never blocks |
| No-Shield baseline | N/A | Bypass the filter entirely |

See `protocol.md` for the complete evaluation protocol.
