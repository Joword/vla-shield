"""
Latency stress-test for the VLA-Shield hot-path pipeline.

Submits N random actions to the /v1/evaluate endpoint (or directly via the
Rust FFI module if --use-ffi is set) and collects per-stage latency metrics.

Usage
-----
    python benchmark/run_latency.py --dof 6 --n-actions 10000 --output results/latency.jsonl

Outputs a JSONL file where each line is a LatencyBreakdown JSON object with an
additional ``action_idx`` field.  Summary statistics (p50/p95/p99) are printed
to stdout at the end.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

try:
    import shield_ffi  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional Rust extension
    shield_ffi = None  # type: ignore[assignment]


def _random_action(dof: int, scale: float = 1.0) -> list[float]:
    return [random.uniform(-scale, scale) for _ in range(dof)]


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0–100) of a sorted list."""
    if not data:
        return float("nan")
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(data_sorted):
        return data_sorted[-1]
    return data_sorted[f] + (k - f) * (data_sorted[c] - data_sorted[f])


def run_http(
    endpoint: str,
    dof: int,
    n_actions: int,
    robot_id: str,
    output: Path | None,  # noqa: ARG001 - kept to mirror run_ffi signature
) -> list[dict]:
    """Run latency benchmark via the FastAPI HTTP endpoint."""
    url = f"{endpoint.rstrip('/')}/v1/evaluate"
    results: list[dict] = []

    for idx in range(n_actions):
        action = _random_action(dof)
        payload = json.dumps({
            "robot_id": robot_id,
            "action": action,
            "t_ns": time.time_ns(),
            "sequence_id": idx,
        }).encode()

        t_wall = time.perf_counter()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            print(f"[WARN] action {idx}: request failed — {exc}", file=sys.stderr)
            continue

        wall_ms = (time.perf_counter() - t_wall) * 1000.0
        latency = body.get("latency", {})
        latency["action_idx"] = idx
        latency["wall_ms"] = wall_ms
        results.append(latency)

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{n_actions} actions submitted …", file=sys.stderr)

    return results


def run_ffi(
    dof: int,
    n_actions: int,
    output: Path | None,  # noqa: ARG001
    *,
    use_numpy: bool = True,
) -> list[dict]:
    """Run latency benchmark directly via the Rust FFI module (no HTTP overhead).

    When ``use_numpy`` is True (default), feeds the pipeline through the
    zero-copy ``evaluate_numpy`` path that borrows directly from contiguous
    ``numpy.ndarray`` buffers; otherwise uses the original list-based
    ``evaluate`` for an A/B comparison.
    """
    if shield_ffi is None:
        sys.exit(
            "shield_ffi not found. Build it with:\n"
            "  cd runtime/shield-ffi && maturin develop"
        )

    limits_kwargs = dict(
        joint_names=[f"j{i}" for i in range(dof)],
        position_min=[-3.14] * dof,
        position_max=[3.14] * dof,
        velocity_max=[1.0] * dof,
        dt=0.01,
        collision_epsilon=0.02,
    )
    pipeline = shield_ffi.ShieldPipeline(**limits_kwargs)
    evaluate_numpy = getattr(pipeline, "evaluate_numpy", None)
    if use_numpy and evaluate_numpy is None:
        print(
            "[WARN] shield_ffi.ShieldPipeline lacks evaluate_numpy; falling back to list path",
            file=sys.stderr,
        )

    current_list = [0.0] * dof
    current_np = np.zeros(dof, dtype=np.float64)
    results: list[dict] = []

    for idx in range(n_actions):
        action_list = _random_action(dof)
        if use_numpy and evaluate_numpy is not None:
            action_np = np.asarray(action_list, dtype=np.float32)
            t_wall = time.perf_counter()
            decision = evaluate_numpy(action_np, current_np, t_ns=time.time_ns(), sequence_id=idx)
        else:
            t_wall = time.perf_counter()
            decision = pipeline.evaluate(action_list, current_list, t_ns=time.time_ns(), sequence_id=idx)
        wall_ms = (time.perf_counter() - t_wall) * 1000.0

        lat = decision.latency()
        lat["action_idx"] = idx
        lat["wall_ms"] = wall_ms
        lat["decision"] = decision.decision
        results.append(lat)

        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{n_actions} actions evaluated …", file=sys.stderr)

    return results


def print_summary(results: list[dict]) -> None:
    def col(key: str) -> list[float]:
        return [r[key] for r in results if isinstance(r.get(key), (int, float))]

    print("\n=== Latency Summary ===")
    for metric in ["total_ms", "physics_ms", "collision_ms", "arbiter_ms", "wall_ms"]:
        vals = col(metric)
        if not vals:
            continue
        print(
            f"  {metric:20s}  p50={_percentile(vals, 50):.3f}  "
            f"p95={_percentile(vals, 95):.3f}  p99={_percentile(vals, 99):.3f}  "
            f"mean={statistics.mean(vals):.3f}  max={max(vals):.3f}  [ms]"
        )

    total = col("total_ms")
    if total:
        over_budget = sum(1 for v in total if v > 5.0)
        print(
            f"\n  budget_violation_rate (>5 ms): "
            f"{over_budget}/{len(total)} = {over_budget / len(total) * 100:.2f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="VLA-Shield latency benchmark")
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--dof", type=int, default=6)
    ap.add_argument("--n-actions", type=int, default=10_000)
    ap.add_argument("--robot-id", default="bench-robot-01")
    ap.add_argument("--use-ffi", action="store_true", help="Use Rust FFI instead of HTTP")
    ap.add_argument(
        "--no-numpy",
        action="store_true",
        help="(--use-ffi only) Use the list-based evaluate path instead of the "
             "zero-copy numpy path, for A/B comparison.",
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    print(f"VLA-Shield latency benchmark: dof={args.dof} n={args.n_actions}", file=sys.stderr)

    if args.use_ffi:
        results = run_ffi(args.dof, args.n_actions, args.output, use_numpy=not args.no_numpy)
    else:
        results = run_http(args.endpoint, args.dof, args.n_actions, args.robot_id, args.output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults written to {args.output}", file=sys.stderr)

    print_summary(results)


if __name__ == "__main__":
    main()
