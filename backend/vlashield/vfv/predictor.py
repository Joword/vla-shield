"""VFV predictor: given (image, action, language), estimate risk of consequence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class VFVResult:
    hazard_score: float
    triggered_ontology_ids: list[str]
    predicted_frame: np.ndarray | None = None


class VFVPredictor(ABC):
    """Base class for Visual Feedback Verification predictors."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        action: list[float],
        language_task: str,
    ) -> VFVResult:
        """Predict safety risk of executing `action` given current `image` and task."""
        ...


class DummyVFVPredictor(VFVPredictor):
    """Always-safe placeholder for testing."""

    def predict(
        self,
        image: np.ndarray,
        action: list[float],
        language_task: str,
    ) -> VFVResult:
        return VFVResult(hazard_score=0.0, triggered_ontology_ids=[])
