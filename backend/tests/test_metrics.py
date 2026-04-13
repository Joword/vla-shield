"""Tests for safety evaluation metrics."""

from vlashield.evaluation.metrics import compute_metrics


def test_perfect_predictions():
    labels = ["BLOCK", "BLOCK", "PASS", "PASS"]
    preds = ["BLOCK", "BLOCK", "PASS", "PASS"]
    m = compute_metrics(preds, labels)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.false_positive_rate == 0.0


def test_all_wrong():
    labels = ["BLOCK", "PASS"]
    preds = ["PASS", "BLOCK"]
    m = compute_metrics(preds, labels)
    assert m.precision == 0.0
    assert m.recall == 0.0
