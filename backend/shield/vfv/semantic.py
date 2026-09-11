"""Deterministic semantic VFV: scene hints + optional CLIP.

This is the production-shaped visual path that the Dummy predictor was
standing in for:

1. **Scene hints** (operator / dataset ``risk_tags`` / explicit ``SEM.*`` ids)
   are the primary, testable signal.  No model download.
2. **Keyword phrases** inside those hints (not the short task name) can
   fire the same ids.
3. **CLIP** is optional (``SHIELD_VFV_CLIP=1``).  When the image is missing
   or CLIP cannot load, scoring is skipped and hints still work.

Physical ``PHY.*`` tags are ignored here — they belong to the kinematics
hot path.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

import numpy as np

from shield.vfv.clip_backend import try_load_clip
from shield.vfv.image_cues import image_cue_scores
from shield.vfv.predictor import VFVPredictor, VFVResult

# Phrase lists are matched against *hints*, not against the short task field
# (so a PASS scenario with task="welding" does not auto-fire HEAT_SOURCE).
_HINT_PHRASES: dict[str, tuple[str, ...]] = {
    "SEM.HEAT_SOURCE": ("heat source", "heater", "hot plate", "torch", "oven"),
    "SEM.HUMAN_PROXIMITY": ("human", "person", "operator", "pedestrian"),
    "SEM.FRAGILE": ("fragile", "glass", "ceramic"),
    "SEM.SHARP_OBJECT": ("sharp", "knife", "blade", "scissors"),
    "SEM.LIQUID_ELECTRICAL": ("liquid", "spill", "electrical panel", "water near electric"),
    "SEM.FORBIDDEN_REGION": ("forbidden region", "keep-out", "cabinet interior", "off-limits"),
}

_HINT_SCORE = 0.85
_KEYWORD_SCORE = 0.78


class SemanticVFVPredictor(VFVPredictor):
    """Hint-first semantic risk predictor with optional CLIP image scores."""

    def __init__(self, clip_scorer: Callable[[np.ndarray], dict[str, float]] | None = None):
        self._clip = try_load_clip() if clip_scorer is None else clip_scorer

    @property
    def backend(self) -> str:
        return "clip" if self._clip is not None else "hints"

    def predict(
        self,
        image: np.ndarray,
        action: list[float],
        language_task: str,
        current_joints: list[float] | None = None,
        scene_hints: Iterable[str] | None = None,
    ) -> VFVResult:
        del current_joints  # reserved for proximity-from-FK later
        scores: dict[str, float] = {}
        hints = [str(h).strip() for h in (scene_hints or []) if str(h).strip()]
        blob = " ".join(hints).lower()

        for hint in hints:
            upper = hint.upper()
            if upper.startswith("PHY."):
                continue
            if upper.startswith("SEM."):
                scores[upper] = max(scores.get(upper, 0.0), _HINT_SCORE)

        for oid, phrases in _HINT_PHRASES.items():
            if any(re.search(rf"\b{re.escape(p)}\b", blob) for p in phrases):
                scores[oid] = max(scores.get(oid, 0.0), _KEYWORD_SCORE)

        clip_used = False
        if (
            self._clip is not None
            and image is not None
            and getattr(image, "ndim", 0) == 3
            and min(image.shape[:2]) >= 32
        ):
            try:
                clip_scores = self._clip(image)
            except Exception:
                clip_scores = {}
            for oid, val in clip_scores.items():
                if val >= 0.25:
                    scores[oid] = max(scores.get(oid, 0.0), float(val))
                    clip_used = True

        cue_used = False
        for oid, val in image_cue_scores(image).items():
            scores[oid] = max(scores.get(oid, 0.0), float(val))
            cue_used = True

        # Human proximity: only warn-level unless EE command is aggressive.
        # The rule itself is `warn`; leave scoring to the registry.
        if "SEM.HUMAN_PROXIMITY" in scores and action:
            ee_speed = float(max(abs(v) for v in action))
            if ee_speed < 0.05:
                scores.pop("SEM.HUMAN_PROXIMITY", None)

        triggered = sorted(scores)
        hazard = max(scores.values(), default=0.0)
        return VFVResult(
            hazard_score=hazard,
            triggered_ontology_ids=triggered,
            scores=dict(scores),
            backend="clip" if clip_used else ("image" if cue_used else "hints"),
        )
