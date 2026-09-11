"""Cheap RGB heuristics. No weights.

Default visual prior. CLIP is opt-in via SHIELD_VFV_CLIP=1. Dummy 4×4
black frames (tests) are ignored on purpose.
"""

from __future__ import annotations

import numpy as np

_MIN_SIDE = 32


def image_cue_scores(image: np.ndarray | None) -> dict[str, float]:
    """SEM.* scores from a uint8 HWC RGB frame. Tiny / missing images → {}."""
    if image is None or getattr(image, "ndim", 0) != 3:
        return {}
    h, w = int(image.shape[0]), int(image.shape[1])
    if min(h, w) < _MIN_SIDE:
        return {}
    rgb = np.asarray(image[..., :3], dtype=np.float32)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    n = float(r.size) or 1.0

    heat = float(np.mean((r > 170.0) & (g < 110.0) & (b < 90.0)))
    # Yellow-ish metal → SEM.SHARP_OBJECT.
    sharp = float(np.mean((r > 160.0) & (g > 160.0) & (b < 80.0)))
    liquid = float(np.mean((b > 140.0) & (g > 100.0) & (r < 90.0)))
    # Skin-ish. Weak human cue — don't treat it as a detector.
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
    if heat > 0.08:
        scores["SEM.HEAT_SOURCE"] = min(0.9, 0.45 + heat)
    if sharp > 0.10:
        scores["SEM.SHARP_OBJECT"] = min(0.85, 0.4 + sharp)
    if liquid > 0.12:
        scores["SEM.LIQUID_ELECTRICAL"] = min(0.85, 0.4 + liquid)
    if human > 0.18:
        scores["SEM.HUMAN_PROXIMITY"] = min(0.8, 0.35 + human * 0.5)
    del n
    return scores
