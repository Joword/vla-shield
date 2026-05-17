"""Micro-benchmark: list vs numpy zero-copy paths on `shield_ffi.ShieldPipeline`.

Runs N identical actions through `pipeline.evaluate` (list path) and
`pipeline.evaluate_numpy` (zero-copy ndarray path), reports p50 / p95 / p99
per-call wall time, and the median speedup of the numpy path.

Usage::

    python benchmark/bench_zero_copy.py --dof 8 --iters 100000

Requires the Rust extension to be built::

    cd runtime/shield-ffi && maturin develop --release
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

try:
    import shield_ffi  # type: ignore[import-not-found]
except ImportError:
    shield_ffi = None  # type: ignore[assignment]


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def _report(label: str, samples: list[float]) -> float:
    p50 = _percentile(samples, 50.0)
    p95 = _percentile(samples, 95.0)
    p99 = _percentile(samples, 99.0)
    mean = statistics.fmean(samples)
    print(
        f"  {label:24s}  p50={p50:.3f}  p95={p95:.3f}  p99={p99:.3f}  "
        f"mean={mean:.3f}  max={max(samples):.3f}  [µs]"
    )
    return p50


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dof", type=int, default=8)
    ap.add_argument("--iters", type=int, default=50_000)
    args = ap.parse_args()

    if shield_ffi is None:
        sys.exit(
            "shield_ffi not found. Build it with:\n"
            "  cd runtime/shield-ffi && maturin develop --release"
        )

    pipeline = shield_ffi.ShieldPipeline(
        joint_names=[f"j{i}" for i in range(args.dof)],
        position_min=[-3.14] * args.dof,
        position_max=[3.14] * args.dof,
        velocity_max=[1.0] * args.dof,
        dt=0.01,
        collision_epsilon=0.02,
    )
    if not hasattr(pipeline, "evaluate_numpy"):
        sys.exit(
            "shield_ffi.ShieldPipeline lacks evaluate_numpy — "
            "rebuild after pulling the numpy zero-copy patch."
        )

    rng = np.random.default_rng(0)
    raw_actions = rng.uniform(-1.5, 1.5, size=(args.iters, args.dof)).astype(np.float32)
    current_list = [0.0] * args.dof
    current_np = np.zeros(args.dof, dtype=np.float64)

    # Warmup.
    for i in range(64):
        pipeline.evaluate(raw_actions[i % args.iters].tolist(), current_list, t_ns=0, sequence_id=i)
        pipeline.evaluate_numpy(raw_actions[i % args.iters], current_np, t_ns=0, sequence_id=i)

    print(
        f"\nshield_ffi zero-copy bench  dof={args.dof}  iters={args.iters}\n"
    )

    list_samples: list[float] = []
    for i in range(args.iters):
        action_list = raw_actions[i].tolist()
        t0 = time.perf_counter()
        pipeline.evaluate(action_list, current_list, t_ns=0, sequence_id=i)
        list_samples.append((time.perf_counter() - t0) * 1_000_000.0)

    numpy_samples: list[float] = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        pipeline.evaluate_numpy(raw_actions[i], current_np, t_ns=0, sequence_id=i)
        numpy_samples.append((time.perf_counter() - t0) * 1_000_000.0)

    print("list path (evaluate)       — Python list → Vec<f32> conversion per call")
    p50_list = _report("evaluate", list_samples)

    print("\nnumpy path (evaluate_numpy) — borrows &[f32] directly from ndarray")
    p50_numpy = _report("evaluate_numpy", numpy_samples)

    speedup = p50_list / p50_numpy if p50_numpy > 0 else float("nan")
    print(f"\nmedian speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
