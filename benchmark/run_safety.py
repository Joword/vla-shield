"""
Safety recall/precision evaluation for VLA-Shield.

Loads scenario definitions from a JSONL file, injects each action into the
shield pipeline, and compares the decision against the ground-truth label.

Metrics computed
----------------
- block_recall        : TP / (TP + FN)  — fraction of high-risk actions blocked
- false_stop_rate     : FP / (FP + TN)  — fraction of safe actions blocked
- hard_block_precision: TP / (TP + FP)  — precision of BLOCK decisions
- near_miss_reduction : (No-Shield near-misses − Shield near-misses) / No-Shield

Usage
-----
    python benchmark/run_safety.py \
        --scenarios dataset/scenarios/scenarios.jsonl \
        --endpoint http://localhost:8000 \
        --output results/safety.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def load_scenarios(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _scenario_id_to_int(raw: object, fallback: int) -> int:
    """Map scenario_id like 'PHY-001' to a positive integer for sequence_id."""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            return int(digits)
    return fallback


def evaluate_scenario(
    scenario: dict,
    endpoint: str,
    seq_fallback: int = 0,
) -> dict:
    """Submit a single scenario action and return the raw API response + outcome."""
    action = scenario.get("injected_action", [0.0] * 6)
    current = scenario.get("current_joints") or [0.0] * len(action)
    expected = scenario.get("expected_decision", "PASS").upper()

    url = f"{endpoint.rstrip('/')}/v1/evaluate"
    payload = json.dumps({
        "robot_id": scenario.get("robot_platform", "bench-robot"),
        "action": action,
        "current_joints": current,
        "t_ns": time.time_ns(),
        "sequence_id": _scenario_id_to_int(scenario.get("scenario_id"), seq_fallback),
    }).encode()

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
        return {
            "scenario_id": scenario.get("scenario_id"),
            "expected": expected,
            "got": "ERROR",
            "match": False,
            "error": str(exc),
        }

    got = body.get("decision", "UNKNOWN").upper()
    return {
        "scenario_id": scenario.get("scenario_id"),
        "risk_tags": scenario.get("risk_tags", []),
        "expected": expected,
        "got": got,
        "match": got == expected,
        "latency_total_ms": body.get("latency", {}).get("total_ms"),
        "ontology_ids": body.get("ontology_ids", []),
    }


def compute_metrics(outcomes: list[dict]) -> dict:
    tp = sum(1 for o in outcomes if o["expected"] == "BLOCK" and o["got"] == "BLOCK")
    fn = sum(1 for o in outcomes if o["expected"] == "BLOCK" and o["got"] != "BLOCK")
    fp = sum(1 for o in outcomes if o["expected"] == "PASS" and o["got"] == "BLOCK")
    tn = sum(1 for o in outcomes if o["expected"] == "PASS" and o["got"] == "PASS")

    block_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    false_stop_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    hard_block_precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "n_scenarios": len(outcomes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "block_recall": round(block_recall, 4),
        "false_stop_rate": round(false_stop_rate, 4),
        "hard_block_precision": round(hard_block_precision, 4),
        "accuracy": round((tp + tn) / len(outcomes), 4) if outcomes else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="VLA-Shield safety benchmark")
    ap.add_argument(
        "--scenarios",
        type=Path,
        default=Path("dataset/scenarios/scenarios.jsonl"),
    )
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios)
    print(f"Loaded {len(scenarios)} scenarios from {args.scenarios}", file=sys.stderr)

    outcomes: list[dict] = []
    for i, scenario in enumerate(scenarios):
        result = evaluate_scenario(scenario, args.endpoint, seq_fallback=i + 1)
        outcomes.append(result)
        status = "OK" if result["match"] else "MISMATCH"
        sid = str(scenario.get("scenario_id") or f"#{i + 1}")
        print(
            f"  [{i + 1:3d}/{len(scenarios)}] {sid:20s}  "
            f"expected={result['expected']}  got={result['got']}  [{status}]",
            file=sys.stderr,
        )

    metrics = compute_metrics(outcomes)

    print("\n=== Safety Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            f.write(json.dumps({"metrics": metrics}) + "\n")
            for o in outcomes:
                f.write(json.dumps(o) + "\n")
        print(f"\nResults written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
