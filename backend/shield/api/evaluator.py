"""/v1/evaluate guts.

Prefers shield_ffi.ShieldPipeline (Rust). Python fallback when shield_ffi
isn't built. Same reasons the Rust path emits.

ShadowSimPredictor fills shadow_ms and can add ontology ids. RuleRegistry
then stamps severity / block|clamp|warn / explanation from
dataset/ontology/rules_*.json so the UI matches the files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shield.api.kinematics import (
    collision_pairs,
    default_urdf_for_dof,
    fk_skeleton_for,
    forbidden_zone_hits,
    load_urdf_chain,
    obstacles_as_tuples,
    parse_obstacles,
    shadow_polyline,
    zones_for,
    UrdfChain,
)
from shield.api.physical_checks import clamp_joint_velocity, extra_physical_reasons
from shield.api.rule_engine import RuleRegistry
from shield.vfv.predictor import ShadowSimPredictor, UrdfShadowConfig
from shield.vfv.semantic import SemanticVFVPredictor

try:
    import shield_ffi  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — shield_ffi isn't built
    shield_ffi = None


REPO_ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_DIR = REPO_ROOT / "dataset" / "ontology"


@dataclass
class EvalInput:
    """One /v1/evaluate request. image/obstacles are optional."""

    robot_id: str
    action: list[float]
    t_ns: int
    sequence_id: int
    current_joints: list[float]
    language_task: str = ""
    scene_hints: list[str] = field(default_factory=list)
    image: np.ndarray | None = None
    obstacles: list[dict] | None = None
    prev_velocity: list[float] = field(default_factory=list)


def _coerce_joints(current_joints: list[float], dof: int) -> list[float]:
    """Pad or trim current_joints to the action's dof."""
    if not current_joints:
        return [0.0] * dof
    if len(current_joints) == dof:
        return current_joints
    if len(current_joints) > dof:
        return current_joints[:dof]
    return list(current_joints) + [0.0] * (dof - len(current_joints))


def _sem_template_kwargs(score: float) -> dict[str, Any]:
    return {
        "score": float(score),
        "object_label": "scene_object",
        "heat_label": "heat_source",
        "distance": 0.10,
        "region_name": "annotated_region",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "electrical_label": "electrical_panel",
        "perimeter_m": 0.5,
        "ee_velocity": 0.40,
        "velocity_threshold_ms": 0.3,
    }


class ShieldEvaluator:
    """What /v1/evaluate actually calls."""

    def __init__(
        self,
        dt: float = 0.01,
        rule_registry: RuleRegistry | None = None,
    ) -> None:
        self._dt = dt
        self._ffi_pipelines: dict[tuple[int, str], Any] = {}
        self._shadow_predictors: dict[int, ShadowSimPredictor] = {}
        self._vfv = SemanticVFVPredictor()
        self._rules = rule_registry or RuleRegistry.load(ONTOLOGY_DIR)
        self._urdf_chains: dict[int, UrdfChain | None] = {}

    @property
    def rules(self) -> RuleRegistry:
        """Loaded rules_*.json. Same object the API uses for explanations."""
        return self._rules

    def _get_limits(self, dof: int) -> dict[str, list[float]]:
        torque_max = [50.0] * dof
        acceleration_max = [10.0] * dof
        if dof == 7:
            # Franka wrist (joint 5, 0-based index 4): 20 Nm nominal.
            torque_max[4] = 20.0
        if dof >= 8:
            acceleration_max[-1] = 1.5
        return {
            "joint_names": [f"j{i}" for i in range(dof)],
            "position_min": [-3.14] * dof,
            "position_max": [3.14] * dof,
            "velocity_max": [1.0] * dof,
            "acceleration_max": acceleration_max,
            "torque_max": torque_max,
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

    def _get_urdf(self, dof: int) -> tuple[str, str, str] | None:
        spec = default_urdf_for_dof(dof)
        if spec is None:
            return None
        path, root, ee = spec
        return str(path), root, ee

    def _get_python_chain(self, dof: int) -> UrdfChain | None:
        if dof in self._urdf_chains:
            return self._urdf_chains[dof]
        spec = self._get_urdf(dof)
        chain: UrdfChain | None = None
        if spec is not None:
            try:
                chain = load_urdf_chain(*spec)
            except (OSError, ValueError, SyntaxError):
                chain = None
        self._urdf_chains[dof] = chain
        return chain

    def _get_ffi_pipeline(self, dof: int) -> Any | None:
        if shield_ffi is None:
            return None
        urdf = self._get_urdf(dof)
        key = (dof, urdf[0] if urdf else "")
        pipeline = self._ffi_pipelines.get(key)
        if pipeline is not None:
            return pipeline
        limits = self._get_limits(dof)
        kwargs: dict[str, Any] = {
            "joint_names": limits["joint_names"],
            "position_min": limits["position_min"],
            "position_max": limits["position_max"],
            "velocity_max": limits["velocity_max"],
            "dt": self._dt,
            "collision_epsilon": 0.02,
            "acceleration_max": limits["acceleration_max"],
            "torque_max": limits["torque_max"],
        }
        if urdf is not None:
            kwargs["urdf_path"] = urdf[0]
            kwargs["root_link"] = urdf[1]
            kwargs["ee_link"] = urdf[2]
        pipeline = shield_ffi.ShieldPipeline(**kwargs)
        self._ffi_pipelines[key] = pipeline
        return pipeline

    def _python_fallback(  # pylint: disable=too-many-locals
        self, req: EvalInput, obstacles: list[dict]
    ) -> dict[str, Any]:
        dof = len(req.action)
        limits = self._get_limits(dof)
        reasons: list[tuple[str, str, float]] = []

        # Velocity cap on the raw command.
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

        clamped = clamp_joint_velocity(
            list(req.action),
            limits["velocity_max"],
            prev_velocity=list(req.prev_velocity),
            acceleration_max=limits["acceleration_max"],
            dt=self._dt,
        )

        # One-step projected position vs joint limits.
        projected = np.array(req.current_joints, dtype=np.float64) + self._dt * np.array(
            clamped, dtype=np.float64
        )
        lower = np.array(limits["position_min"], dtype=np.float64)
        upper = np.array(limits["position_max"], dtype=np.float64)
        for i in range(dof):
            if projected[i] < lower[i] or projected[i] > upper[i]:
                detail = self._rules.render(
                    "PHY.JOINT_LIMIT",
                    (
                        f"joint={i} projected={projected[i]:.3f} "
                        f"out_of [{lower[i]:.3f}, {upper[i]:.3f}]"
                    ),
                    joint_name=f"j{i}",
                    value=float(projected[i]),
                    lower=float(lower[i]),
                    upper=float(upper[i]),
                    margin_rad=0.01,
                )
                reasons.append(("PHY.JOINT_LIMIT", detail, 1.0))

        # URDF (or EE) AABB vs scene obstacles.
        chain = self._get_python_chain(dof)
        q_proj = [float(v) for v in projected]
        for link, obstacle in collision_pairs(q_proj, obstacles, chain):
            detail = self._rules.render(
                "PHY.COLLISION",
                f"pair={link}:{obstacle}",
                link=link,
                obstacle=obstacle,
                min_distance=0.0,
            )
            reasons.append(("PHY.COLLISION", detail, 1.0))

        for zone in forbidden_zone_hits(q_proj, obstacles, chain):
            detail = self._rules.render(
                "PHY.FORBIDDEN_ZONE",
                f"ee in forbidden zone {zone}",
                region_name=zone,
                x=0.0,
                y=0.0,
                z=0.0,
            )
            reasons.append(("PHY.FORBIDDEN_ZONE", detail, 1.0))

        for oid, detail, score in extra_physical_reasons(
            list(clamped),
            list(req.current_joints),
            torque_max=limits["torque_max"],
            chain=chain,
            joint_names=limits["joint_names"],
        ):
            if any(r[0] == oid for r in reasons):
                continue
            rendered = self._rules.render(oid, detail)
            reasons.append((oid, rendered or detail, score))

        risk = max((r[2] for r in reasons), default=0.0)
        return {"reasons": reasons, "risk": risk}

    def evaluate(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self, req: EvalInput
    ) -> dict[str, Any]:
        """Rust pipeline if we have it, else Python. Always fills shadow + VFV."""
        dof = len(req.action)
        current = _coerce_joints(list(req.current_joints), dof)
        req = EvalInput(
            robot_id=req.robot_id,
            action=list(req.action),
            t_ns=req.t_ns,
            sequence_id=req.sequence_id,
            current_joints=current,
            language_task=req.language_task,
            scene_hints=list(req.scene_hints),
            image=req.image,
            obstacles=req.obstacles,
            prev_velocity=list(req.prev_velocity),
        )

        obstacles = parse_obstacles(req.obstacles)
        obstacle_tuples = obstacles_as_tuples(obstacles)

        t0 = time.perf_counter()
        ingest_ms = 0.0  # wire-time is measured in the API layer, not here

        # Python shadow prior — feeds shadow_ms and extra ontology ids.
        t_shadow0 = time.perf_counter()
        shadow = self._get_shadow_predictor(dof).predict(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            action=req.action,
            language_task=req.language_task or "runtime_eval",
            current_joints=req.current_joints,
        )
        shadow_ms = (time.perf_counter() - t_shadow0) * 1000.0

        vfv_image = req.image if req.image is not None else np.zeros((4, 4, 3), dtype=np.uint8)
        vfv = self._vfv.predict(
            image=vfv_image,
            action=req.action,
            language_task=req.language_task,
            current_joints=req.current_joints,
            scene_hints=req.scene_hints,
        )

        # Rust if we have it, else Python.
        ffi_pipeline = self._get_ffi_pipeline(dof)
        used_ffi = False
        if ffi_pipeline is not None:
            setter = getattr(ffi_pipeline, "set_obstacles", None)
            if setter is not None:
                try:
                    setter(obstacle_tuples)
                except TypeError:
                    # Older wheels don't take the ontology-tagged tuple. Send
                    # the plain box; zones may come back as PHY.COLLISION.
                    setter([row[:7] for row in obstacle_tuples])
            # evaluate_numpy borrows contiguous ndarrays — skip list→Vec every tick.
            evaluate_numpy = getattr(ffi_pipeline, "evaluate_numpy", None)
            if evaluate_numpy is not None:
                action_np = np.ascontiguousarray(req.action, dtype=np.float32)
                current_np = np.ascontiguousarray(req.current_joints, dtype=np.float64)
                kwargs_np: dict[str, Any] = {
                    "t_ns": req.t_ns,
                    "sequence_id": req.sequence_id,
                }
                if req.prev_velocity:
                    kwargs_np["prev_velocity"] = np.ascontiguousarray(
                        req.prev_velocity, dtype=np.float32
                    )
                try:
                    ffi_decision = evaluate_numpy(action_np, current_np, **kwargs_np)
                except TypeError:
                    ffi_decision = evaluate_numpy(
                        action_np,
                        current_np,
                        t_ns=req.t_ns,
                        sequence_id=req.sequence_id,
                    )
            else:
                eval_kwargs: dict[str, Any] = {
                    "t_ns": req.t_ns,
                    "sequence_id": req.sequence_id,
                }
                if req.prev_velocity:
                    eval_kwargs["prev_velocity"] = [float(v) for v in req.prev_velocity]
                try:
                    ffi_decision = ffi_pipeline.evaluate(
                        req.action,
                        req.current_joints,
                        **eval_kwargs,
                    )
                except TypeError:
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
            risk = max((score for _, _, score in reasons), default=0.0)
            latency = dict(ffi_decision.latency())
            used_ffi = True
        else:
            py = self._python_fallback(req, obstacles)
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

        # Fold shadow hits in if physics didn't already fire them.
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
                risk = max(risk, float(shadow.hazard_score))

        existing_ids = {oid for oid, _, _ in reasons}
        for oid in vfv.triggered_ontology_ids:
            if oid in existing_ids:
                continue
            score = float(vfv.scores.get(oid, vfv.hazard_score) or 0.7)
            detail = self._rules.render(
                oid,
                f"vfv: {oid} score={score:.2f}",
                **_sem_template_kwargs(score),
            )
            reasons.append((oid, detail, score))
            existing_ids.add(oid)
        if vfv.hazard_score:
            risk = max(risk, float(vfv.hazard_score))

        ontology_ids = [oid for oid, _, _ in reasons]
        decision = self._rules.decide(ontology_ids)
        if reasons:
            risk = max(risk, max(score for _, _, score in reasons))

        total_ms = (time.perf_counter() - t0) * 1000.0
        # Keep FFI stage timings. We only stamp shadow_ms (API-side) and total wall-clock.
        latency["shadow_ms"] = float(shadow_ms)
        if not used_ffi or not latency.get("total_ms"):
            latency["total_ms"] = float(total_ms)
        latency.setdefault("ingest_ms", float(ingest_ms))

        ontology_details = {oid: detail for oid, detail, _ in reasons if detail}

        projected = [
            float(j) + self._dt * float(a)
            for j, a in zip(req.current_joints, req.action)
        ]
        chain = self._get_python_chain(dof)
        skeleton: list[list[float]] = []
        if used_ffi and ffi_pipeline is not None:
            skel_fn = getattr(ffi_pipeline, "skeleton", None)
            if skel_fn is not None:
                try:
                    skeleton = [list(p) for p in skel_fn(list(req.current_joints))]
                except (TypeError, ValueError, AttributeError):
                    skeleton = []
        if not skeleton:
            skeleton = fk_skeleton_for(req.current_joints, chain)
        traj = shadow.trajectory or [list(req.current_joints), projected]
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
            "current_joints": [float(v) for v in req.current_joints],
            "projected_joints": projected,
            "skeleton": skeleton,
            "shadow_path": shadow_polyline(traj),
            "ee": skeleton[-1] if skeleton else [0.0, 0.0, 0.0],
            "zones": obstacles or zones_for(ontology_ids),
            "scene_rev": int(req.sequence_id),
            "vfv_backend": vfv.backend,
        }
