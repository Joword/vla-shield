"""BLOCK/PASS confusion counts. Precision/recall treat BLOCK as the positive class."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyMetrics:
    """BLOCK as the positive class. PASS mistakes are false_passes."""
    total: int
    true_blocks: int
    false_blocks: int
    true_passes: int
    false_passes: int

    @property
    def precision(self) -> float:
        """true_blocks / (true_blocks + false_blocks)."""
        denom = self.true_blocks + self.false_blocks
        return self.true_blocks / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """true_blocks / (true_blocks + false_passes)."""
        denom = self.true_blocks + self.false_passes
        return self.true_blocks / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        """false_blocks / (false_blocks + true_passes)."""
        denom = self.false_blocks + self.true_passes
        return self.false_blocks / denom if denom > 0 else 0.0


def compute_metrics(
    predictions: list[str],
    labels: list[str],
) -> SafetyMetrics:
    """Predictions vs labels. Both lists are 'BLOCK' or 'PASS'."""
    assert len(predictions) == len(labels)
    tb = fb = tp = fp = 0
    for pred, label in zip(predictions, labels):
        if pred == "BLOCK" and label == "BLOCK":
            tb += 1
        elif pred == "BLOCK" and label == "PASS":
            fb += 1
        elif pred == "PASS" and label == "PASS":
            tp += 1
        else:
            fp += 1
    return SafetyMetrics(
        total=len(predictions),
        true_blocks=tb,
        false_blocks=fb,
        true_passes=tp,
        false_passes=fp,
    )
