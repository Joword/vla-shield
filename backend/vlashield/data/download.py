#!/usr/bin/env python3
"""Dataset preparation for VLA-Shield red-team / evaluation data.

Embodied robotics safety benchmarks (trajectory labels, URDF-scenario violations)
will be integrated in Phase 1. This module provides a smoke path that copies the
repository's bilingual samples into ``public.jsonl`` for local validation.

Usage (from backend/):
  python -m vlashield.data.download
  python -m vlashield.data.download --output ../dataset/red_team/public.jsonl
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[3] / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare red-team JSONL for VLA-Shield. "
            "Full embodied-safety datasets (robot sim traces, labeled hazards) "
            "are planned for Phase 1 integration."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination JSONL (default: dataset/red_team/public.jsonl)",
    )
    parser.add_argument(
        "--from-samples",
        action="store_true",
        help="Copy dataset/red_team/samples.jsonl to the output path",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else DATASET_DIR / "red_team" / "public.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.from_samples:
        samples = DATASET_DIR / "red_team" / "samples.jsonl"
        if not samples.is_file():
            raise FileNotFoundError(f"Missing {samples}")
        shutil.copyfile(samples, output)
        print(f"Copied {samples} -> {output}")
    else:
        print(
            "VLA-Shield Phase 1 will add embodied robotics safety datasets "
            "(simulation trajectories, URDF-scenario constraints).\n"
            "For smoke tests, run with --from-samples to copy samples.jsonl, or "
            "author entries matching dataset/red_team/schema.json manually."
        )
        if output.exists():
            print(f"Leaving existing file unchanged: {output}")
        else:
            samples = DATASET_DIR / "red_team" / "samples.jsonl"
            if samples.is_file():
                shutil.copyfile(samples, output)
                print(f"Initialized {output} from {samples}")
            else:
                output.write_text("", encoding="utf-8")
                print(f"Created empty {output}")

    print(f"\nValidate with:  python -m vlashield.data.validate --data {output}")


if __name__ == "__main__":
    main()
