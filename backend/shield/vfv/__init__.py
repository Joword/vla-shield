"""Visual Feedback Verification (VFV) — semantic risk from hints / optional CLIP."""

from shield.vfv.predictor import (
    DummyVFVPredictor,
    ShadowSimPredictor,
    UrdfShadowConfig,
    VFVPredictor,
    VFVResult,
)
from shield.vfv.semantic import SemanticVFVPredictor
from shield.vfv.image_cues import image_cue_scores

__all__ = [
    "DummyVFVPredictor",
    "SemanticVFVPredictor",
    "image_cue_scores",
    "ShadowSimPredictor",
    "UrdfShadowConfig",
    "VFVPredictor",
    "VFVResult",
]
