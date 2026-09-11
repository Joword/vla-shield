"""Image-cue VFV fires on saturated heat-colored frames, not on dummy 4×4."""

from __future__ import annotations

import numpy as np

from shield.vfv.image_cues import image_cue_scores
from shield.vfv.semantic import SemanticVFVPredictor


def test_dummy_image_is_ignored() -> None:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert image_cue_scores(img) == {}


def test_red_frame_heatsource() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 220
    img[..., 1] = 20
    img[..., 2] = 20
    scores = image_cue_scores(img)
    assert "SEM.HEAT_SOURCE" in scores
    vfv = SemanticVFVPredictor(clip_scorer=lambda _im: {})
    result = vfv.predict(img, action=[0.0] * 6, language_task="inspect")
    assert "SEM.HEAT_SOURCE" in result.triggered_ontology_ids
    assert result.backend == "image"
