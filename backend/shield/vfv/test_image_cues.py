"""Image cues fire on a saturated heat-colored frame, not on the dummy 4×4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shield.vfv.image_cues import CueConfig, CueSpec, cue_config, image_cue_scores, load_cue_config
from shield.vfv.semantic import SemanticVFVPredictor

REPO = Path(__file__).resolve().parents[3]


def test_dummy_image_is_ignored() -> None:
    """4×4 black frames must not fire SEM.*."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    assert not image_cue_scores(img)


def test_red_frame_heatsource() -> None:
    """Saturated red frame → SEM.HEAT_SOURCE via the image prior."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 220
    img[..., 1] = 20
    img[..., 2] = 20
    scores = image_cue_scores(img)
    assert "SEM.HEAT_SOURCE" in scores
    heat_floor = cue_config().spec("SEM.HEAT_SOURCE").min_score
    assert scores["SEM.HEAT_SOURCE"] >= heat_floor
    vfv = SemanticVFVPredictor(clip_scorer=lambda _im: {})
    result = vfv.predict(img, action=[0.0] * 6, language_task="inspect")
    assert "SEM.HEAT_SOURCE" in result.triggered_ontology_ids
    assert result.backend == "image"


def test_yellow_frame_sharp() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 220
    img[..., 1] = 200
    img[..., 2] = 20
    scores = image_cue_scores(img)
    assert "SEM.SHARP_OBJECT" in scores
    assert scores["SEM.SHARP_OBJECT"] >= 0.6


def test_blue_frame_liquid() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 20
    img[..., 1] = 140
    img[..., 2] = 200
    scores = image_cue_scores(img)
    assert "SEM.LIQUID_ELECTRICAL" in scores
    assert scores["SEM.LIQUID_ELECTRICAL"] >= 0.65


def test_task_name_welding_does_not_fire_heat() -> None:
    """PASS rows use short task names; do not scan language_task for phrases."""
    vfv = SemanticVFVPredictor(clip_scorer=lambda _im: {})
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    result = vfv.predict(img, action=[0.2] * 6, language_task="welding")
    assert "SEM.HEAT_SOURCE" not in result.triggered_ontology_ids
    result_hint = vfv.predict(
        img,
        action=[0.2] * 6,
        language_task="welding",
        scene_hints=["hot plate in workspace"],
    )
    assert "SEM.HEAT_SOURCE" in result_hint.triggered_ontology_ids


def test_parked_arm_drops_human_proximity() -> None:
    """Ontology velocity_threshold_ms is 0.3 — parked commands must not warn."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 180
    img[..., 1] = 110
    img[..., 2] = 80
    vfv = SemanticVFVPredictor(clip_scorer=lambda _im: {})
    parked = vfv.predict(img, action=[0.01] * 6, language_task="inspect")
    moving = vfv.predict(img, action=[0.5] * 6, language_task="inspect")
    if "SEM.HUMAN_PROXIMITY" in moving.triggered_ontology_ids:
        assert "SEM.HUMAN_PROXIMITY" not in parked.triggered_ontology_ids


def test_cue_json_aligned_with_rules() -> None:
    cfg = load_cue_config(REPO / "dataset" / "ontology" / "vfv_cues.json")
    rules = json.loads(
        (REPO / "dataset" / "ontology" / "rules_semantic.json").read_text(encoding="utf-8")
    )
    min_scores = {
        r["rule_id"]: float(r["threshold"]["min_score"])
        for r in rules
        if isinstance(r.get("threshold"), dict) and "min_score" in r["threshold"]
    }
    for oid, floor in min_scores.items():
        spec = cfg.spec(oid)
        if spec is None:
            continue
        assert spec.min_score >= floor - 1e-9, oid


def test_custom_cue_config_raises_floor() -> None:
    cfg = CueConfig(
        min_side_px=32,
        cues={"SEM.HEAT_SOURCE": CueSpec(fraction=0.01, min_score=0.8)},
    )
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[..., 0] = 220
    img[..., 1] = 20
    img[..., 2] = 20
    scores = image_cue_scores(img, config=cfg)
    assert scores["SEM.HEAT_SOURCE"] >= 0.8
