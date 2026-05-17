#!/usr/bin/env python3
"""Validate red-team JSONL using the RedTeamEntry Pydantic model and report statistics.

Usage (run from backend/):
  python -m shield.data.validate                                           # validate samples.jsonl
  python -m shield.data.validate --data ../dataset/red_team/public.jsonl   # validate downloaded data
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from shield.schemas import RedTeamEntry

DATASET_DIR = Path(__file__).resolve().parents[3] / "dataset"


def validate(data_path: Path) -> int:
    """Validate every line in a JSONL file as a RedTeamEntry. Returns error count."""
    lines = data_path.read_text(encoding="utf-8").strip().splitlines()

    errors = 0
    tag_counter: Counter[str] = Counter()
    context_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    outcome_counter: Counter[str] = Counter()

    for i, line in enumerate(lines, 1):
        raw = json.loads(line)
        try:
            entry = RedTeamEntry.model_validate(raw)
        except ValidationError as e:
            print(f"  Line {i} ({raw.get('id', '?')}): {e.error_count()} error(s)")
            for err in e.errors():
                print(f"    - {'.'.join(str(x) for x in err['loc'])}: {err['msg']}")
            errors += 1
            continue

        for tag in entry.risk_tags:
            tag_counter[tag] += 1
        context_counter[entry.task_context or "unknown"] += 1
        split_counter[entry.split] += 1
        source_counter[entry.source] += 1
        outcome_counter[entry.expected_outcome] += 1

    print(f"\nFile              : {data_path}")
    print(f"Total entries     : {len(lines)}")
    print(f"Valid             : {len(lines) - errors}")
    print(f"Errors            : {errors}")
    print(f"\nSource distribution   : {dict(source_counter.most_common())}")
    print(f"Split distribution    : {dict(split_counter)}")
    print(f"Outcome distribution  : {dict(outcome_counter)}")
    print(f"Context distribution  : {dict(context_counter.most_common())}")
    print(f"Risk tag frequency    : {dict(tag_counter.most_common())}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VLA-Shield red-team dataset")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="JSONL data path (default: dataset/red_team/samples.jsonl)",
    )
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else DATASET_DIR / "red_team" / "samples.jsonl"

    if not data_path.exists():
        sys.exit(f"Data not found: {data_path}")

    errors = validate(data_path)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
