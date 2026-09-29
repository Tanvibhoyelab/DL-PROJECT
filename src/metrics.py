"""Pixel-level change-detection metrics.

Every number the dashboard reports about model quality comes from this module,
computed from a real confusion matrix on labelled data. Nothing here is
hard-coded or smoothed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

EPS = 1e-9


@dataclass
class Confusion:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def __add__(self, other: "Confusion") -> "Confusion":
        return Confusion(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn)

    @property
    def total(self) -> float:
        return self.tp + self.fp + self.fn + self.tn

    def as_dict(self) -> dict:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


def confusion_from_probs(probs: torch.Tensor, target: torch.Tensor, threshold: float) -> Confusion:
    pred = (probs >= threshold).float()
    return Confusion(
        tp=float((pred * target).sum().item()),
        fp=float((pred * (1 - target)).sum().item()),
        fn=float(((1 - pred) * target).sum().item()),
        tn=float(((1 - pred) * (1 - target)).sum().item()),
    )


def summarise(cm: Confusion) -> dict:
    tp, fp, fn, tn = cm.tp, cm.fp, cm.fn, cm.tn
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    specificity = tn / (tn + fp + EPS)
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn + EPS),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": tp / (tp + fp + fn + EPS),
        "dice": 2 * tp / (2 * tp + fp + fn + EPS),
        "specificity": specificity,
        # Balanced accuracy is the honest headline number for imbalanced masks:
        # plain pixel accuracy is ~95% even for a model that predicts "no change"
        # everywhere, because 95% of pixels really are unchanged.
        "balanced_accuracy": 0.5 * (recall + specificity),
        "positive_rate": (tp + fn) / (tp + tn + fp + fn + EPS),
    }


def sweep_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """Metrics at each candidate threshold, computed from flattened arrays."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    positives = targets > 0.5
    rows = []
    for threshold in thresholds:
        pred = probs >= threshold
        cm = Confusion(
            tp=float(np.count_nonzero(pred & positives)),
            fp=float(np.count_nonzero(pred & ~positives)),
            fn=float(np.count_nonzero(~pred & positives)),
            tn=float(np.count_nonzero(~pred & ~positives)),
        )
        rows.append({"threshold": float(threshold), **summarise(cm), **cm.as_dict()})
    return rows


def best_threshold(rows: list[dict], key: str = "f1") -> dict:
    return max(rows, key=lambda row: row[key])
