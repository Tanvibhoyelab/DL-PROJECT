"""Evaluate the trained checkpoint on the LEVIR-CD test split.

Usage:
    python -m src.evaluate --data data/LEVIR-CD
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from config import CONFUSION_MATRIX_PNG, TEST_METRICS_JSON  # noqa: E402
from src.dataset import LevirCDDataset  # noqa: E402
from src.inference import load_model  # noqa: E402
from src.train import batch_metrics, summarise  # noqa: E402


def plot_confusion(tp, fp, fn, tn):
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    normed = cm / max(cm.sum(), 1)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(normed, cmap="Blues")
    labels = ["No change", "Change"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (pixels)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}\n{normed[i, j] * 100:.2f}%",
                    ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/LEVIR-CD")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    loaded = load_model()
    dataset = LevirCDDataset(args.data, args.split, size=loaded.input_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating {len(dataset)} pairs on {loaded.device}")

    tp = fp = fn = tn = 0.0
    with torch.no_grad():
        for a, b, y in loader:
            a, b, y = a.to(loaded.device), b.to(loaded.device), y.to(loaded.device)
            probs = torch.sigmoid(loaded.model(a, b))
            m = batch_metrics(probs, y, args.threshold)
            tp, fp, fn, tn = tp + m[0], fp + m[1], fn + m[2], tn + m[3]

    metrics = summarise(tp, fp, fn, tn)
    metrics.update({
        "split": args.split, "threshold": args.threshold, "pairs": len(dataset),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "checkpoint": loaded.checkpoint_path,
    })
    TEST_METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    plot_confusion(tp, fp, fn, tn)

    for k in ("accuracy", "precision", "recall", "f1", "iou", "dice"):
        print(f"{k:>10}: {metrics[k]:.4f}")
    print(f"Saved -> {TEST_METRICS_JSON} and {CONFUSION_MATRIX_PNG}")


if __name__ == "__main__":
    main()
