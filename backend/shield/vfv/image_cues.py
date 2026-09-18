"""Cheap RGB heuristics. No weights.

Default visual prior. CLIP is opt-in via SHIELD_VFV_CLIP=1. Dummy 4×4
black frames (tests) are ignored on purpose.

Pixel-fraction floors and emit scores come from
`dataset/ontology/vfv_cues.json` (aligned with rules_semantic.json
`min_score`). Override the path with SHIELD_VFV_CUES.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_MIN_SIDE_DEFAULT = 32
_DEFAULT_CUES_PATH = (
    Path(__file__).resolve().parents[3] / "dataset" / "ontology" / "vfv_cues.json"
)


@dataclass(frozen=True)
class CueSpec:
    """Min pixel fraction to fire, and the score we emit (rule floor)."""

    fraction: float
    min_score: float


@dataclass(frozen=True)
class CueConfig:
    min_side_px: int = _MIN_SIDE_DEFAULT
    human_velocity_threshold: float = 0.3
    cues: dict[str, CueSpec] = field(default_factory=dict)

    def spec(self, ontology_id: str) -> CueSpec | None:
        return self.cues.get(ontology_id)


_BUILTIN = CueConfig(
    min_side_px=_MIN_SIDE_DEFAULT,
    human_velocity_threshold=0.3,
    cues={
        "SEM.HEAT_SOURCE": CueSpec(0.08, 0.7),
        "SEM.SHARP_OBJECT": CueSpec(0.10, 0.6),
        "SEM.LIQUID_ELECTRICAL": CueSpec(0.12, 0.65),
        "SEM.HUMAN_PROXIMITY": CueSpec(0.18, 0.5),
    },
)


def _cue_spec_from_dict(raw: dict) -> CueSpec:
    return CueSpec(
        fraction=float(raw.get("fraction", 0.1)),
        min_score=float(raw.get("min_score", 0.6)),
    )


def load_cue_config(path: str | os.PathLike[str] | None = None) -> CueConfig:
    """JSON file, else SHIELD_VFV_CUES, else shipped vfv_cues.json, else builtin."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    env = os.environ.get("SHIELD_VFV_CUES", "").strip()
    if env:
        candidates.append(Path(env))
    candidates.append(_DEFAULT_CUES_PATH)
    for p in candidates:
        try:
            if not p.is_file():
                continue
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        raw_cues = data.get("cues") if isinstance(data, dict) else None
        if not isinstance(raw_cues, dict):
            continue
        cues = {
            str(oid): _cue_spec_from_dict(spec)
            for oid, spec in raw_cues.items()
            if isinstance(spec, dict)
        }
        if not cues:
            continue
        return CueConfig(
            min_side_px=int(data.get("min_side_px", _MIN_SIDE_DEFAULT)),
            human_velocity_threshold=float(data.get("human_velocity_threshold", 0.3)),
            cues=cues,
        )
    return _BUILTIN


_CONFIG = load_cue_config()


def cue_config() -> CueConfig:
    """Process-wide cue floors (tests may call load_cue_config directly)."""
    return _CONFIG


def _emit(frac: float, spec: CueSpec) -> float:
    """Fire at `spec.min_score`, then rise with how much the fraction clears the floor."""
    extra = max(0.0, frac - spec.fraction)
    return min(0.95, spec.min_score + 0.5 * extra)


def image_cue_scores(
    image: np.ndarray | None,
    config: CueConfig | None = None,
) -> dict[str, float]:
    """SEM.* scores from a uint8 HWC RGB frame. Tiny / missing images → {}."""
    cfg = config or _CONFIG
    if image is None or getattr(image, "ndim", 0) != 3:
        return {}
    h, w = int(image.shape[0]), int(image.shape[1])
    if min(h, w) < cfg.min_side_px:
        return {}
    rgb = np.asarray(image[..., :3], dtype=np.float32)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    heat = float(np.mean((r > 170.0) & (g < 110.0) & (b < 90.0)))
    sharp = float(np.mean((r > 160.0) & (g > 160.0) & (b < 80.0)))
    liquid = float(np.mean((b > 140.0) & (g > 100.0) & (r < 90.0)))
    human = float(
        np.mean(
            (r > 90.0)
            & (r < 220.0)
            & (g > 40.0)
            & (g < 180.0)
            & (b > 30.0)
            & (b < 150.0)
            & (r > g)
            & (g > b)
        )
    )

    scores: dict[str, float] = {}
    mapping = (
        ("SEM.HEAT_SOURCE", heat),
        ("SEM.SHARP_OBJECT", sharp),
        ("SEM.LIQUID_ELECTRICAL", liquid),
        ("SEM.HUMAN_PROXIMITY", human),
    )
    for oid, frac in mapping:
        spec = cfg.spec(oid)
        if spec is None or frac <= spec.fraction:
            continue
        scores[oid] = _emit(frac, spec)
    return scores
