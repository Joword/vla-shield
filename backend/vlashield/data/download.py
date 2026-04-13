#!/usr/bin/env python3
"""Download and convert public safety datasets into VLA-Shield red-team JSONL format.

Sources:
  - BeaverTails (PKU-Alignment/BeaverTails): 330k+ safety-labeled prompts
    with 14 harm categories including physical harm.
  - PKU-SafeRLHF (PKU-Alignment/PKU-SafeRLHF): safety preference pairs
    with fine-grained harm annotations.

Usage (run from backend/):
  python -m vlashield.data.download                      # default: BeaverTails, 2000 samples
  python -m vlashield.data.download --source all          # both sources
  python -m vlashield.data.download --max-samples 5000
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from vlashield.schemas import RedTeamEntry

DATASET_DIR = Path(__file__).resolve().parents[3] / "dataset"

ONTOLOGY_MAP: dict[str, list[str]] = {
    "Physical Harm": ["PHY.COLLISION", "PHY.OVERLOAD"],
    "Physical_harm": ["PHY.COLLISION", "PHY.OVERLOAD"],
    "Hate/Toxicity": ["SEM.HUMAN_PROXIMITY"],
    "Violence": ["PHY.COLLISION", "SEM.HUMAN_PROXIMITY"],
    "Dangerous_suggestions": ["SEM.HEAT_SOURCE", "SEM.LIQUID_ELECTRICAL", "SEM.SHARP_OBJECT"],
    "Dangerous Suggestions": ["SEM.HEAT_SOURCE", "SEM.LIQUID_ELECTRICAL", "SEM.SHARP_OBJECT"],
    "Theft": ["SEM.FORBIDDEN_REGION"],
    "Property_damage": ["SEM.FRAGILE"],
    "self-harm": ["PHY.COLLISION"],
    "Self-Harm": ["PHY.COLLISION"],
}

PHYSICAL_CATEGORIES = {
    "Physical Harm",
    "Physical_harm",
    "Violence",
    "Dangerous_suggestions",
    "Dangerous Suggestions",
    "Property_damage",
    "Self-Harm",
    "self-harm",
}

CONTEXT_KEYWORDS: dict[str, str] = {
    "kitchen": "kitchen",
    "cook": "kitchen",
    "knife": "kitchen",
    "stove": "kitchen",
    "oven": "kitchen",
    "boil": "kitchen",
    "pour": "kitchen",
    "elderly": "eldercare",
    "grandma": "eldercare",
    "old person": "eldercare",
    "lab": "lab",
    "chemical": "lab",
    "experiment": "lab",
    "warehouse": "warehouse",
    "shelf": "warehouse",
    "heavy": "warehouse",
}


def guess_context(text: str) -> str:
    lower = text.lower()
    for keyword, ctx in CONTEXT_KEYWORDS.items():
        if keyword in lower:
            return ctx
    return "general"


def stable_id(text: str, prefix: str = "rt") -> str:
    h = hashlib.md5(text.encode()).hexdigest()[:6]
    return f"{prefix}-{h}"


def _write_entry(f, entry: RedTeamEntry) -> None:  # type: ignore[no-untyped-def]
    f.write(entry.model_dump_json() + "\n")


def convert_beavertails(max_samples: int, output: Path) -> int:
    from datasets import load_dataset

    print("[BeaverTails] Loading dataset from HuggingFace...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")

    count = 0
    seen: set[str] = set()

    with open(output, "a", encoding="utf-8") as f:
        for row in ds:
            if count >= max_samples:
                break

            prompt = row.get("prompt", "")
            if not prompt or prompt in seen:
                continue

            is_safe = row.get("is_safe", True)
            category = row.get("category", {})

            if isinstance(category, dict):
                triggered = [k for k, v in category.items() if v and k in PHYSICAL_CATEGORIES]
            elif isinstance(category, str):
                triggered = [category] if category in PHYSICAL_CATEGORIES else []
            else:
                triggered = []

            if is_safe and not triggered:
                if count < max_samples * 0.15:
                    entry = RedTeamEntry(
                        id=stable_id(prompt),
                        instruction=prompt.strip(),
                        task_context=guess_context(prompt),
                        risk_tags=[],
                        expected_outcome="proceed_safe",
                        annotator="beavertails-auto",
                        source="BeaverTails",
                    )
                    _write_entry(f, entry)
                    seen.add(prompt)
                    count += 1
                continue

            if not triggered and not is_safe:
                continue

            risk_tags: list[str] = []
            for cat in triggered:
                risk_tags.extend(ONTOLOGY_MAP.get(cat, []))
            risk_tags = sorted(set(risk_tags)) or ["PHY.COLLISION"]

            entry = RedTeamEntry(
                id=stable_id(prompt),
                instruction=prompt.strip(),
                task_context=guess_context(prompt),
                risk_tags=risk_tags,
                expected_outcome="refuse_or_safe_alternative",
                annotator="beavertails-auto",
                source="BeaverTails",
            )
            _write_entry(f, entry)
            seen.add(prompt)
            count += 1

    print(f"[BeaverTails] Wrote {count} entries")
    return count


def convert_pku_saferlhf(max_samples: int, output: Path) -> int:
    from datasets import load_dataset

    print("[PKU-SafeRLHF] Loading dataset from HuggingFace...")
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

    count = 0
    seen: set[str] = set()

    with open(output, "a", encoding="utf-8") as f:
        for row in ds:
            if count >= max_samples:
                break

            prompt = row.get("prompt", "")
            if not prompt or prompt in seen:
                continue

            is_safe_0 = row.get("is_response_0_safe", True)
            is_safe_1 = row.get("is_response_1_safe", True)

            if is_safe_0 and is_safe_1:
                continue

            risk_tags: list[str] = []
            lower_prompt = prompt.lower()
            if any(w in lower_prompt for w in ["hit", "punch", "attack", "hurt", "kill", "slam"]):
                risk_tags.append("PHY.COLLISION")
            if any(w in lower_prompt for w in ["burn", "fire", "heat", "boil"]):
                risk_tags.append("SEM.HEAT_SOURCE")
            if any(w in lower_prompt for w in ["knife", "stab", "cut", "blade", "sharp"]):
                risk_tags.append("SEM.SHARP_OBJECT")
            if any(w in lower_prompt for w in ["water", "pour", "liquid", "spill"]):
                risk_tags.append("SEM.LIQUID_ELECTRICAL")
            if any(w in lower_prompt for w in ["break", "smash", "destroy", "shatter"]):
                risk_tags.append("SEM.FRAGILE")
            if any(w in lower_prompt for w in ["enter", "trespass", "restricted", "forbidden"]):
                risk_tags.append("SEM.FORBIDDEN_REGION")

            if not risk_tags:
                risk_tags = ["PHY.COLLISION"]

            risk_tags = sorted(set(risk_tags))

            entry = RedTeamEntry(
                id=stable_id(prompt, prefix="sr"),
                instruction=prompt.strip(),
                task_context=guess_context(prompt),
                risk_tags=risk_tags,
                expected_outcome="refuse_or_safe_alternative",
                annotator="pku-saferlhf-auto",
                source="PKU-SafeRLHF",
            )
            _write_entry(f, entry)
            seen.add(prompt)
            count += 1

    print(f"[PKU-SafeRLHF] Wrote {count} entries")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public safety datasets for VLA-Shield")
    parser.add_argument(
        "--source",
        choices=["beavertails", "pku-saferlhf", "all"],
        default="beavertails",
    )
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else DATASET_DIR / "red_team" / "public.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        output.unlink()
        print(f"Removed existing {output}")

    total = 0
    if args.source in ("beavertails", "all"):
        total += convert_beavertails(args.max_samples, output)
    if args.source in ("pku-saferlhf", "all"):
        total += convert_pku_saferlhf(args.max_samples, output)

    print(f"\nDone. Total entries: {total}")
    print(f"Output: {output}")
    print(f"\nValidate with:  python -m vlashield.data.validate --data {output}")


if __name__ == "__main__":
    main()
