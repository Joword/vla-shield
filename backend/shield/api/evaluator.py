"""Lightweight runtime evaluator used by /v1/evaluate.

This module provides a backend-side evaluation path that can run with or
without the Rust PyO3 extension:

- Preferred: `shield_ffi.ShieldPipeline` (Rust hot-path)
- Fallback: deterministic Python evaluator (joint/velocity checks)

It also computes a shadow-path prior from the Python reference predictor so the
API can emit `shadow_ms` and shadow-driven ontology triggers.

The reasons surfaced through both paths are post-processed by
:class:`shield.api.rule_engine.RuleRegistry`, which means severity, action
(`block | clamp | warn`) and human-readable explanation text are sourced from
``dataset/ontology/rules_*.json`` — keeping the data plane and the displayed
documentation in lock-step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shield.api.rule_engine import RuleRegistry
from shield.vfv.predictor import ShadowSimPredictor, UrdfShadowConfig

try:
    import shield_ffi  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional runtime dependency
    shield_ffi = None


REPO_ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_DIR = REPO_ROOT / "dataset" / "ontology"


@dataclass
class EvalInput:
    robot_id: str
    action: list[float]
    t_ns: int
    sequence_id: int
    current_joints: list[float]


def _coerce_joints(current_joints: list[float], dof: int) -> list[float]:
    """Pad / truncate ``current_joints`` to match ``dof`` (length of action)."""
    if not current_joints:
        return [0.0] * dof
    if len(current_joints) == dof:
        return current_joints
    if len(current_joints) > dof:
        return current_joints[:dof]
    return list(current_joints) + [0.0] * (dof - len(current_joints))


class ShieldEvaluator:
    """Runtime evaluator facade used by the FastAPI endpoint."""

    def __init__(
        self,
        dt: float = 0.01,
        rule_registry: RuleRegistry | None = None,
    ) -> None:
        self._dt = dt
        self._ffi_pipelines: dict[int, Any] = {}
        self._shadow_predictors: dict[int, ShadowSimPredictor] = {}
        self._rules = rule_registry or RuleRegistry.load(ONTOLOGY_DIR)

    @property
    def rules(self) -> RuleRegistry:
        return self._rules

    def _get_limits(self, dof: int) -> dict[str, list[float]]:
        return {
            "joint_names": [f"j{i}" for i in range(dof)],
            "position_min": [-3.14] * dof,
            "position_max": [3.14] * dof,
            "velocity_max": [1.0] * dof,
        }

    def _get_shadow_predictor(self, dof: int) -> ShadowSimPredictor:
        predictor = self._shadow_predictors.get(dof)
        if predictor is not None:
            return predictor
        limits = self._get_limits(dof)
        predictor = ShadowSimPredictor(
            UrdfShadowConfig(
                dof=dof,
                joint_limit_lower=limits["position_min"],
                joint_limit_upper=limits["position_max"],
            ),
            steps=8,
            dt=self._dt,
        )
        self._shadow_predictors[dof] = predictor
        return predictor

    def _get_ffi_pipeline(self, dof: int) -> Any | None:
        if shield_ffi is None:
            return None
        pipeline = self._ffi_pipelines.get(dof)
        if pipeline is not None:
            return pipeline
        limits = self._get_limits(dof)
        pipeline = shield_ffi.ShieldPipeline(
            joint_names=limits["joint_names"],
            position_min=limits["position_min"],
            position_max=limits["position_max"],
            velocity_max=limits["velocity_max"],
            dt=self._dt,
            collision_epsilon=0.02,
        )
        self._ffi_pipelines[dof] = pipeline
        return pipeline

    def _python_fallback(self, req: EvalInput) -> dict[str, Any]:
        dof = len(req.action)
        limits = self._get_limits(dof)
        reasons: list[tuple[str, str, float]] = []

        # 1) velocity cap
        for i, v in enumerate(req.action):
            vmax = limits["velocity_max"][i]
            if abs(v) > vmax:
                detail = self._rules.render(
                    "PHY.VELOCITY_LIMIT",
                    f"joint={i} requested={v:.3f} exceeds limit={vmax:.3f} rad/s",
                    joint_name=f"j{i}",
                    requested=v,
                    limit=vmax,
                )
                reasons.append(("PHY.VELOCITY_LIMIT", detail, 0.6))

        # 2) one-step projected position limit
        projected = np.array(req.current_joints, dtype=np.float64) + self._dt * np.array(
            req.action, dtype=np.float64
        )
        lower = np.array(limits["position_min"], dtype=np.float64)
        upper = np.array(limits["position_max"], dtype=np.float64)
        for i in range(dof):
            if projected[i] < lower[i] or projected[i] > upper[i]:
                detail = self._rules.render(
                    "PHY.JOINT_LIMIT",
                    f"joint={i} projected={projected[i]:.3f} out_of [{lower[i]:.3f}, {upper[i]:.3f}]",
                    joint_name=f"j{i}",
                    value=float(projected[i]),
                    lower=float(lower[i]),
                    upper=float(upper[i]),
                    margin_rad=0.01,
                )
                reasons.append(("PHY.JOINT_LIMIT", detail, 1.0))

        # Decision policy: any triggered reason → BLOCK on the hot path; the
        # downstream arbiter (rule_engine + shadow prior) can downgrade to
        # PASS-with-clamp later if every reason is non-blocking.
        decision = "BLOCK" if reasons else "PASS"
        risk = max((r[2] for r in reasons), default=0.0)
        return {"decision": decision, "reasons": reasons, "risk": risk}

    def evaluate(self, req: EvalInput) -> dict[str, Any]:
        dof = len(req.action)
        current = _coerce_joints(list(req.current_joints), dof)
        req = EvalInput(
            robot_id=req.robot_id,
            action=list(req.action),
            t_ns=req.t_ns,
            sequence_id=req.sequence_id,
            current_joints=current,
        )

        t0 = time.perf_counter()
        ingest_ms = 0.0  # the API layer measures wire-time separately

        # Shadow prior (reference predictor) for latency/risk enrichment.
        t_shadow0 = time.perf_counter()
        shadow = self._get_shadow_predictor(dof).predict(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            action=req.action,
            language_task="runtime_eval",
            current_joints=req.current_joints,
        )
        shadow_ms = (time.perf_counter() - t_shadow0) * 1000.0

        # Main decision path
        ffi_pipeline = self._get_ffi_pipeline(dof)
        used_ffi = False
        if ffi_pipeline is not None:
            # Prefer the zero-copy numpy path: borrows `&[f32]` and `&[f64]`
            # straight from contiguous ndarrays instead of paying the Python
            # list → Rust Vec iteration on every call.
            evaluate_numpy = getattr(ffi_pipeline, "evaluate_numpy", None)
            if evaluate_numpy is not None:
                action_np = np.ascontiguousarray(req.action, dtype=np.float32)
                current_np = np.ascontiguousarray(req.current_joints, dtype=np.float64)
                ffi_decision = evaluate_numpy(
                    action_np,
                    current_np,
                    t_ns=req.t_ns,
                    sequence_id=req.sequence_id,
                )
            else:
                ffi_decision = ffi_pipeline.evaluate(
                    req.action,
                    req.current_joints,
                    t_ns=req.t_ns,
                    sequence_id=req.sequence_id,
                )
            raw_reasons = list(ffi_decision.reasons)
            reasons = [
                (str(oid), str(detail), float(score)) for oid, detail, score in raw_reasons
            ]
            decision = str(ffi_decision.decision)
            risk = max((score for _, _, score in reasons), default=0.0)
            latency = dict(ffi_decision.latency())
            used_ffi = True
        else:
            py = self._python_fallback(req)
            decision = py["decision"]
            reasons = py["reasons"]
            risk = py["risk"]
            latency = {
                "ingest_ms": ingest_ms,
                "urdf_fk_ms": None,
                "physics_ms": 0.2,
                "collision_ms": 0.2,
                "tf2_ms": None,
                "arbiter_ms": 0.2,
                "shadow_ms": None,
                "total_ms": 0.0,
            }

        # Inject shadow prior into final decision if needed.
        if shadow.hazard_score >= 0.5:
            existing_ids = {oid for oid, _, _ in reasons}
            for oid in shadow.triggered_ontology_ids:
                if oid not in existing_ids:
                    detail = self._rules.render(
                        oid,
                        f"shadow_path: hazard={shadow.hazard_score:.2f}",
                        joint_name="?",
                        value=float(shadow.hazard_score),
                        lower=-3.14,
                        upper=3.14,
                        margin_rad=0.01,
                    )
                    reasons.append((oid, detail, float(shadow.hazard_score)))
            if reasons:
                decision = "BLOCK"
                risk = max(risk, float(shadow.hazard_score))

        total_ms = (time.perf_counter() - t0) * 1000.0
        # Preserve FFI per-stage timings; only fill in shadow_ms (taken at API layer)
        # and total wall-clock so downstream charts reflect real cost.
        latency["shadow_ms"] = float(shadow_ms)
        if not used_ffi or not latency.get("total_ms"):
            latency["total_ms"] = float(total_ms)
        latency.setdefault("ingest_ms", float(ingest_ms))

        ontology_ids = [oid for oid, _, _ in reasons]
        ontology_details = {oid: detail for oid, detail, _ in reasons if detail}
        return {
            "robot_id": req.robot_id,
            "sequence_id": req.sequence_id,
            "ts_ns": req.t_ns,
            "decision": decision,
            "risk": float(risk),
            "ontology_ids": ontology_ids,
            "ontology_details": ontology_details,
            "latency": latency,
            "used_ffi": used_ffi,
        }
