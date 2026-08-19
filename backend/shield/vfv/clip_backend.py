"""Optional CLIP backend for visual semantic scoring.

Loaded only when ``SHIELD_VFV_CLIP=1``.  Missing weights or a failed
import degrade to ``None`` so the keyword/hint path stays the default
hot path (no multi-hundred-MB download in tests or CPU-only deploys).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

_PROMPTS: dict[str, str] = {
    "SEM.HEAT_SOURCE": "a heat source, heater, hot plate, torch or oven near a robot arm",
    "SEM.HUMAN_PROXIMITY": "a person standing close to a robot manipulator",
    "SEM.FRAGILE": "a fragile glass or ceramic object being grasped by a robot",
    "SEM.SHARP_OBJECT": "a knife or other sharp object in a robot workspace",
    "SEM.LIQUID_ELECTRICAL": "liquid spilling near electrical equipment",
    "SEM.FORBIDDEN_REGION": "a robot arm entering a cabinet or marked keep-out zone",
}


def clip_enabled() -> bool:
    return os.environ.get("SHIELD_VFV_CLIP", "").strip() in {"1", "true", "TRUE", "yes"}


def try_load_clip() -> Any | None:
    """Return a callable ``score(image_hwc_uint8) -> dict[str, float]`` or None."""
    if not clip_enabled():
        return None
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        return None

    try:
        model_id = os.environ.get("SHIELD_VFV_CLIP_MODEL", "openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
        model.eval()
    except Exception:
        return None

    labels = list(_PROMPTS.keys())
    texts = [_PROMPTS[k] for k in labels]

    def score(image: np.ndarray) -> dict[str, float]:
        if image.ndim != 3 or image.shape[2] < 3:
            return {}
        rgb = image[:, :, :3]
        inputs = processor(text=texts, images=rgb, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs)
            probs = out.logits_per_image.softmax(dim=1).squeeze(0).tolist()
        if not isinstance(probs, list):
            probs = [float(probs)]
        return {lab: float(p) for lab, p in zip(labels, probs)}

    return score
