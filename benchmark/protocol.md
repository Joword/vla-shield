# VLA-Shield Benchmark Protocol v0.1

## Overview

This document defines the reproducible evaluation protocol for VLA-Shield.
All experiments compare three configurations:

| Config | Description |
|--------|-------------|
| **No-Shield** | Raw VLA model output forwarded directly to the robot |
| **Training-time only** | Safety constraints embedded in the model via fine-tuning (no runtime filter) |
| **VLA-Shield** | Decoupled runtime safety filter (this project) applied post-model |

---

## Evaluation Metrics

### Latency (hot-path)

| Metric | Definition |
|--------|-----------|
| `p50_total_ms` | Median end-to-end filter latency (action receipt → decision publish) |
| `p95_total_ms` | 95th-percentile total latency |
| `p99_total_ms` | 99th-percentile total latency |
| `p95_urdf_fk_ms` | 95th-percentile URDF FK + singularity check time |
| `p95_physics_ms` | 95th-percentile physical projection time |
| `p95_collision_ms` | 95th-percentile collision precheck time |
| `p95_arbiter_ms` | 95th-percentile arbiter decision time |
| `budget_violation_rate` | Fraction of episodes where `total_ms > 5.0` |

### Safety

| Metric | Definition |
|--------|-----------|
| `block_recall` | True-positive rate: fraction of injected high-risk actions correctly blocked |
| `false_stop_rate` | False-positive rate: fraction of safe actions incorrectly blocked |
| `near_miss_reduction` | Percentage reduction in near-miss events vs No-Shield baseline |
| `hard_block_precision` | Fraction of `BLOCK` decisions that match ground-truth violations |

---

## Experimental Setup

### Hardware Baseline

- **Target**: UR5e (6-DOF, 1 kHz control loop)
- **Host**: Intel i9-13900K, 32 GB DDR5, Ubuntu 22.04 + ROS 2 Humble
- **GPU** (for VFV only): RTX 4090, CUDA 12.4

### Software Versions

- `shield-core` ≥ 0.1.0
- FastAPI ≥ 0.110, Python ≥ 3.11
- ROS 2 Humble (rclrs 0.4)

---

## Latency Benchmark Procedure

Run `benchmark/run_latency.py`:

```bash
python benchmark/run_latency.py \
    --endpoint http://localhost:8000 \
    --dof 6 \
    --n-actions 10000 \
    --concurrency 1 \
    --output results/latency_$(date +%Y%m%d).jsonl
```

### Injection Mode

Actions are drawn from a uniform distribution in `[-1, 1]^DoF` to simulate a
wide velocity command distribution.  Each action is timestamped with the
current nanosecond clock and submitted synchronously (one at a time) to
measure pure filter latency without queueing effects.

---

## Safety Benchmark Procedure

Run `benchmark/run_safety.py`:

```bash
python benchmark/run_safety.py \
    --scenarios dataset/scenarios/scenarios.jsonl \
    --endpoint http://localhost:8000 \
    --output results/safety_$(date +%Y%m%d).jsonl
```

### Injection Mode

Each scenario in `scenarios.jsonl` specifies an `injected_action` vector
and an `expected_decision`.  The script submits the action and compares the
shield decision against the ground truth.

---

## Reporting

Results should be reported in a Markdown or LaTeX table with:

1. Mean ± std for all latency metrics (10 independent runs of 10 k actions each).
2. 95 % confidence intervals for safety metrics (bootstrapped, n = 1000).
3. Hardware/software version string.

Reference results for the UR5e baseline should be committed to
`results/reference_ur5e_YYYYMMDD.jsonl`.
