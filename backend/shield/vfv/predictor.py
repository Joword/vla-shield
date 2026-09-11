"""VFV: (image, action, language) → hazard score + ontology ids."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VFVResult:
    """Hazard + triggered SEM/PHY ids. trajectory is joint-space for the monitor."""
    hazard_score: float
    triggered_ontology_ids: list[str]
    predicted_frame: np.ndarray | None = None
    # Includes q0. Monitor draws this as the shadow polyline.
    trajectory: list[list[float]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    backend: str = "none"


@dataclass
class UrdfShadowConfig:
    """Just joint limits for the Python shadow sim. Not a full URDF."""

    dof: int
    joint_limit_lower: list[float]
    joint_limit_upper: list[float]


class VFVPredictor(ABC):
    """image + action + task → VFVResult."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        action: list[float],
        language_task: str,
        current_joints: list[float] | None = None,
        scene_hints: list[str] | None = None,
    ) -> VFVResult:
        """Hazard + triggered ids for this action. scene_hints is optional."""
        raise NotImplementedError


class DummyVFVPredictor(VFVPredictor):
    """Always PASS. Tests / when you don't want VFV."""

    def predict(
        self,
        image: np.ndarray,
        action: list[float],
        language_task: str,
        current_joints: list[float] | None = None,
        scene_hints: list[str] | None = None,
    ) -> VFVResult:
        del image, language_task, current_joints, scene_hints
        return VFVResult(hazard_score=0.0, triggered_ontology_ids=[], backend="dummy")


class ShadowSimPredictor(VFVPredictor):
    """Roll joints forward in joint space, clamp to URDF limits.

    Same idea as the Rust shadow path. Python reference, not a dynamics sim.
    Visual scoring lives on SemanticVFVPredictor.
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
        scene_hints: list[str] | None = None,
    ) -> VFVResult:
        del image, language_task, scene_hints  # visual path is SemanticVFVPredictor
        if len(action) != self._cfg.dof:
            return VFVResult(hazard_score=1.0, triggered_ontology_ids=["PHY.JOINT_LIMIT"])

        q0 = current_joints if current_joints is not None else [0.0] * self._cfg.dof
        if len(q0) != self._cfg.dof:
            return VFVResult(hazard_score=1.0, triggered_ontology_ids=["PHY.JOINT_LIMIT"])

        lower = np.array(self._cfg.joint_limit_lower, dtype=np.float64)
        upper = np.array(self._cfg.joint_limit_upper, dtype=np.float64)
        triggered: list[str] = []
        trajectory: list[list[float]] = [list(map(float, q0))]

        for s in range(1, self._steps + 1):
            alpha = s / self._steps
            q = np.array(q0, dtype=np.float64) + alpha * self._dt * np.array(
                action, dtype=np.float64
            )
            q = np.clip(q, lower, upper)
            trajectory.append(q.astype(float).tolist())
            if np.any(q <= lower + 1e-9) or np.any(q >= upper - 1e-9):
                triggered.append("PHY.JOINT_LIMIT")

        hazard = 0.8 if "PHY.JOINT_LIMIT" in triggered else 0.0
        return VFVResult(
            hazard_score=hazard,
            triggered_ontology_ids=sorted(set(triggered)),
            trajectory=trajectory,
            backend="shadow",
        )
