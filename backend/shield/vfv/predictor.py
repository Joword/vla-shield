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


@dataclass
class UrdfShadowConfig:
    """Minimal URDF-style joint limits for joint-space shadow simulation (Python reference)."""

    dof: int
    joint_limit_lower: list[float]
    joint_limit_upper: list[float]


class VFVPredictor(ABC):
    """Base class for Visual Feedback Verification predictors."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        action: list[float],
        language_task: str,
        current_joints: list[float] | None = None,
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
        current_joints: list[float] | None = None,
    ) -> VFVResult:
        return VFVResult(hazard_score=0.0, triggered_ontology_ids=[])


class ShadowSimPredictor(VFVPredictor):
    """Joint-space shadow trajectory: interpolate from current joints toward the integrated command.

    Mirrors the Rust runtime's "shadow path" idea: sample intermediate joint vectors and
    clamp to URDF limits. This is a lightweight Python reference, not a full dynamics sim.
    """

    def __init__(
        self,
        urdf_config: UrdfShadowConfig,
        *,
        steps: int = 8,
        dt: float = 0.01,
    ) -> None:
        self._cfg = urdf_config
        self._steps = max(2, steps)
        self._dt = dt
        if len(urdf_config.joint_limit_lower) != urdf_config.dof:
            raise ValueError("joint_limit_lower length must match dof")
        if len(urdf_config.joint_limit_upper) != urdf_config.dof:
            raise ValueError("joint_limit_upper length must match dof")

    def predict(
        self,
        image: np.ndarray,
        action: list[float],
        language_task: str,
        current_joints: list[float] | None = None,
    ) -> VFVResult:
        del image, language_task  # reserved for VLM-based VFV
        if len(action) != self._cfg.dof:
            return VFVResult(hazard_score=1.0, triggered_ontology_ids=["PHY.JOINT_LIMIT"])

        q0 = current_joints if current_joints is not None else [0.0] * self._cfg.dof
        if len(q0) != self._cfg.dof:
            return VFVResult(hazard_score=1.0, triggered_ontology_ids=["PHY.JOINT_LIMIT"])

        lower = np.array(self._cfg.joint_limit_lower, dtype=np.float64)
        upper = np.array(self._cfg.joint_limit_upper, dtype=np.float64)
        triggered: list[str] = []

        for s in range(1, self._steps + 1):
            alpha = s / self._steps
            q = np.array(q0, dtype=np.float64) + alpha * self._dt * np.array(action, dtype=np.float64)
            q = np.clip(q, lower, upper)
            if np.any(q <= lower + 1e-9) or np.any(q >= upper - 1e-9):
                triggered.append("PHY.JOINT_LIMIT")

        hazard = 0.8 if "PHY.JOINT_LIMIT" in triggered else 0.0
        return VFVResult(hazard_score=hazard, triggered_ontology_ids=sorted(set(triggered)))
