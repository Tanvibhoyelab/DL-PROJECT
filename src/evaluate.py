"""Evaluate the trained checkpoint on the held-out LEVIR-CD test split.

Usage:
    python -m src.evaluate --data data/LEVIR-CD

The threshold defaults to the value tuned on the validation split and stored in
the checkpoint, so test numbers are produced without ever looking at test data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from config import CONFUSION_MATRIX_PNG, QUALITATIVE_PNG, TEST_METRICS_JSON  # noqa: E402
from src.dataset import LevirCDDataset  # noqa: E402
from src.inference import load_model  # noqa: E402
from src.metrics import Confusion, confusion_from_probs, summarise  # noqa: E402


def plot_confusion(cm: Confusion) -> None:
    matrix = np.array([[cm.tn, cm.fp], [cm.fn, cm.tp]], dtype=float)
    normed = matrix / max(matrix.sum(), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(normed, cmap="Blues")
    labels = ["No change", "Change"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (pixels)")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{int(matrix[i, j]):,}\n{normed[i, j] * 100:.2f}%",
                ha="center", va="center", fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=130)
    plt.close()


def plot_examples(dataset, loaded, threshold: float, count: int = 4) -> None:
    step = max(len(dataset) // max(count, 1), 1)
    picks = [i * step for i in range(count) if i * step < len(dataset)]
    fig, axes = plt.subplots(len(picks), 4, figsize=(11, 2.8 * len(picks)))
    axes = np.atleast_2d(axes)
    with torch.no_grad():
        for row, idx in enumerate(picks):
            a, b, y = dataset[idx]
            probs = torch.sigmoid(
                loaded.model(a.unsqueeze(0).to(loaded.device), b.unsqueeze(0).to(loaded.device))
            )[0, 0].cpu().numpy()
            for col, (image, title, kwargs) in enumerate(
                [
                    (a.permute(1, 2, 0).numpy(), "Before", {}),
                    (b.permute(1, 2, 0).numpy(), "After", {}),
                    (y[0].numpy(), "Ground truth", {"cmap": "gray"}),
                    ((probs >= threshold).astype(float), "Prediction", {"cmap": "gray"}),
                ]
            ):
                axes[row, col].imshow(image, **kwargs)
                axes[row, col].set_title(title, fontsize=9)
                axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(QUALITATIVE_PNG, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/LEVIR-CD")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="defaults to the threshold tuned on validation and stored in the checkpoint",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    loaded = load_model()
    threshold = float(args.threshold if args.threshold is not None else loaded.threshold)
    dataset = LevirCDDataset(args.data, args.split, size=loaded.input_size, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating {len(dataset)} pairs on {loaded.device} at threshold {threshold:.2f}")

    cm = Confusion()
    with torch.no_grad():
        for a, b, y in loader:
            a, b, y = a.to(loaded.device), b.to(loaded.device), y.to(loaded.device)
            probs = torch.sigmoid(loaded.model(a, b))
            cm = cm + confusion_from_probs(probs, y, threshold)

    metrics = summarise(cm)
    metrics.update(
        {
            "split": args.split,
            "threshold": threshold,
            "pairs": len(dataset),
            **{k: int(v) for k, v in cm.as_dict().items()},
            "checkpoint": loaded.checkpoint_path,
            "dataset": args.data,
            "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
    )
    TEST_METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm)
    plot_examples(dataset, loaded, threshold)

    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "iou", "dice"):
        print(f"{key:>18}: {metrics[key]:.4f}")
    print(
        f"\nNote: {metrics['positive_rate'] * 100:.2f}% of test pixels are labelled 'change', so "
        "plain pixel accuracy is high by default; F1/IoU are the meaningful numbers."
    )
    print(f"Saved -> {TEST_METRICS_JSON}, {CONFUSION_MATRIX_PNG}, {QUALITATIVE_PNG}")


if __name__ == "__main__":
    main()
