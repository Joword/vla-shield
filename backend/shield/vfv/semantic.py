"""Semantic VFV: hints first, then image cues, CLIP if you opted in.

- Scene hints / risk_tags / explicit SEM.* ids are the testable signal.
- Phrases inside those hints can fire the same ids. Don't scan the short
  task name — a PASS row with task="welding" must not auto-fire HEAT_SOURCE.
- CLIP only if SHIELD_VFV_CLIP=1. Missing image / failed load → hints still work.

PHY.* tags are ignored here; that's the kinematics path.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

import numpy as np

from shield.vfv.clip_backend import try_load_clip
from shield.vfv.image_cues import cue_config, image_cue_scores
from shield.vfv.predictor import VFVPredictor, VFVResult

# Match phrases against *hints*, not the short task field
# (a PASS scenario with task="welding" must not auto-fire HEAT_SOURCE).
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
    """Hints first. Image cues always-on. CLIP only if loaded."""

    def __init__(self, clip_scorer: Callable[[np.ndarray], dict[str, float]] | None = None):
        self._clip = try_load_clip() if clip_scorer is None else clip_scorer

    @property
    def backend(self) -> str:
        """clip if weights loaded, else hints."""
        return "clip" if self._clip is not None else "hints"

    def predict(  # pylint: disable=too-many-locals
        self,
        image: np.ndarray,
        action: list[float],
        language_task: str,
        current_joints: list[float] | None = None,
        scene_hints: Iterable[str] | None = None,
    ) -> VFVResult:
        del current_joints  # later: proximity from FK
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
            except (TypeError, ValueError, RuntimeError):
                clip_scores = {}
            for oid, val in clip_scores.items():
                if val >= 0.25:
                    scores[oid] = max(scores.get(oid, 0.0), float(val))
                    clip_used = True

        cue_used = False
        for oid, val in image_cue_scores(image).items():
            scores[oid] = max(scores.get(oid, 0.0), float(val))
            cue_used = True

        # Drop SEM.HUMAN_PROXIMITY if EE command is below the rule's
        # velocity_threshold_ms (default 0.3). The rule is `warn`.
        v_cap = cue_config().human_velocity_threshold
        if "SEM.HUMAN_PROXIMITY" in scores and action:
            ee_speed = float(max(abs(v) for v in action))
            if ee_speed < v_cap:
                scores.pop("SEM.HUMAN_PROXIMITY", None)

        triggered = sorted(scores)
        hazard = max(scores.values(), default=0.0)
        return VFVResult(
            hazard_score=hazard,
            triggered_ontology_ids=triggered,
            scores=dict(scores),
            backend="clip" if clip_used else ("image" if cue_used else "hints"),
        )
