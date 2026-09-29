"""Training pipeline for the Siamese U-Net on LEVIR-CD.

Usage:
    python -m src.train --data data/LEVIR-CD --epochs 40 --batch-size 8

What this script does about the "low score" problem:

* the BCE positive weight is measured from the training labels, so the network
  cannot minimise the loss by predicting "no change" everywhere;
* model selection uses validation F1, not validation loss;
* after training, the decision threshold is tuned on the validation split and
  stored in the checkpoint, so inference does not have to guess it.

Every metric printed or saved is computed from real predictions on real
labelled data.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import CHECKPOINT_PATH, METRICS_CSV, OUTPUTS_DIR  # noqa: E402
from src.dataset import build_loaders, positive_fraction  # noqa: E402
from src.metrics import (  # noqa: E402
    Confusion,
    best_threshold,
    confusion_from_probs,
    summarise,
    sweep_thresholds,
)
from src.model import DiceBCELoss, build_model, count_parameters  # noqa: E402


def plot_curve(history, keys, title, ylabel, filename):
    plt.figure(figsize=(7, 4))
    for key, label in keys:
        plt.plot([h["epoch"] for h in history], [h[key] for h in history], label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / filename, dpi=130)
    plt.close()


@torch.no_grad()
def validate(model, loader, criterion, device, threshold: float = 0.5, collect: bool = False):
    model.eval()
    total_loss = 0.0
    cm = Confusion()
    probs_pool: list[np.ndarray] = []
    target_pool: list[np.ndarray] = []
    for a, b, y in loader:
        a, b, y = a.to(device), b.to(device), y.to(device)
        logits = model(a, b)
        total_loss += criterion(logits, y).item()
        probs = torch.sigmoid(logits)
        cm = cm + confusion_from_probs(probs, y, threshold)
        if collect:
            # Subsample pixels: the full validation split is billions of pixels,
            # and a fixed stride keeps threshold tuning unbiased but cheap.
            probs_pool.append(probs.detach().cpu().numpy().ravel()[::37])
            target_pool.append(y.detach().cpu().numpy().ravel()[::37])
    loss = total_loss / max(len(loader), 1)
    if collect:
        return loss, cm, np.concatenate(probs_pool), np.concatenate(target_pool)
    return loss, cm, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/LEVIR-CD", help="LEVIR-CD root folder")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8, help="early-stopping patience")
    ap.add_argument("--train-limit", type=int, default=None, help="use only N training pairs")
    ap.add_argument("--val-limit", type=int, default=None, help="use only N validation pairs")
    ap.add_argument("--max-pos-weight", type=float, default=12.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, train_set, val_set = build_loaders(
        args.data, args.size, args.batch_size, args.workers, args.train_limit, args.val_limit
    )
    print(f"Train pairs: {len(train_set)} | Val pairs: {len(val_set)}")

    pos_fraction = positive_fraction(train_set)
    pos_weight = min(args.max_pos_weight, (1 - pos_fraction) / max(pos_fraction, 1e-6))
    print(f"Changed pixels in training labels: {pos_fraction * 100:.2f}%  ->  BCE pos_weight={pos_weight:.2f}")

    model = build_model(base_channels=args.base).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    criterion = DiceBCELoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history: list[dict] = []
    best_f1 = -1.0
    bad_epochs = 0
    Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for a, b, y in train_loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a, b), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        val_loss, cm, _, _ = validate(model, val_loader, criterion, device)
        metrics = summarise(cm)
        scheduler.step(metrics["f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{k: v for k, v in metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"F1 {metrics['f1']:.4f} | IoU {metrics['iou']:.4f} | "
            f"recall {metrics['recall']:.4f} | {time.time() - t0:.0f}s"
        )

        if metrics["f1"] > best_f1 + 1e-4:
            best_f1, bad_epochs = metrics["f1"], 0
            save_checkpoint(model, args, epoch, metrics, val_loss, pos_weight, pos_fraction, 0.5)
            print(f"  saved checkpoint -> {CHECKPOINT_PATH}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs (best validation F1 {best_f1:.4f}).")
                break

        with open(METRICS_CSV, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    # ---- threshold tuning on the validation split, using the best weights ----
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, probs, targets = validate(model, val_loader, criterion, device, collect=True)
    sweep = sweep_thresholds(probs, targets)
    best = best_threshold(sweep, "f1")
    print(
        f"Tuned decision threshold on validation: {best['threshold']:.2f} "
        f"(F1 {best['f1']:.4f}, IoU {best['iou']:.4f}, precision {best['precision']:.4f}, "
        f"recall {best['recall']:.4f})"
    )

    _, cm_tuned, _, _ = validate(model, val_loader, criterion, device, threshold=best["threshold"])
    tuned_metrics = summarise(cm_tuned)
    save_checkpoint(
        model,
        args,
        checkpoint["epoch"],
        tuned_metrics,
        checkpoint.get("val_loss"),
        pos_weight,
        pos_fraction,
        best["threshold"],
    )
    with open(OUTPUTS_DIR / "threshold_sweep.json", "w") as handle:
        json.dump({"sweep": sweep, "selected": best}, handle, indent=2)

    if history:
        plot_curve(history, [("train_loss", "Train loss"), ("val_loss", "Validation loss")],
                   "Loss", "Loss", "training_loss.png")
        plot_curve(history, [("iou", "Validation IoU")], "IoU", "IoU", "iou_curve.png")
        plot_curve(history, [("f1", "Validation F1")], "F1 score", "F1", "f1_curve.png")
    print(f"Done. Metrics: {METRICS_CSV}  Plots: {OUTPUTS_DIR}")
    print("Now run the held-out test evaluation:  python -m src.evaluate --data " + args.data)


def save_checkpoint(model, args, epoch, metrics, val_loss, pos_weight, pos_fraction, threshold):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "architecture": "SiameseUNet",
                "in_channels": 3,
                "base_channels": args.base,
                "img_size": args.size,
            },
            "epoch": epoch,
            "val_metrics": metrics,
            "val_loss": val_loss,
            "threshold": threshold,
            "train_pos_fraction": pos_fraction,
            "pos_weight": pos_weight,
            "dataset": str(args.data),
        },
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()
