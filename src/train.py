"""Training pipeline for the Siamese U-Net on LEVIR-CD.

Usage:
    python -m src.train --data data/LEVIR-CD --epochs 40 --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from config import CHECKPOINT_PATH, METRICS_CSV, OUTPUTS_DIR  # noqa: E402
from src.dataset import build_loaders  # noqa: E402
from src.model import DiceBCELoss, build_model  # noqa: E402


def batch_metrics(probs, target, threshold: float = 0.5):
    pred = (probs >= threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    return tp, fp, fn, tn


def summarise(tp, fp, fn, tn):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "iou": iou, "dice": dice,
    }


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
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_loaders(args.data, args.size, args.batch_size, args.workers)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(base=args.base).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history, best_val, bad_epochs = [], float("inf"), 0
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

        model.eval()
        val_loss, tp = 0.0, 0.0
        fp = fn = tn = 0.0
        with torch.no_grad():
            for a, b, y in val_loader:
                a, b, y = a.to(device), b.to(device), y.to(device)
                logits = model(a, b)
                val_loss += criterion(logits, y).item()
                m = batch_metrics(torch.sigmoid(logits), y)
                tp, fp, fn, tn = tp + m[0], fp + m[1], fn + m[2], tn + m[3]
        val_loss /= max(len(val_loader), 1)
        metrics = summarise(tp, fp, fn, tn)
        scheduler.step(val_loss)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "lr": optimizer.param_groups[0]["lr"], **metrics}
        history.append(row)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"IoU {metrics['iou']:.4f} | F1 {metrics['f1']:.4f} | {time.time() - t0:.0f}s")

        if val_loss < best_val - 1e-4:
            best_val, bad_epochs = val_loss, 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "base": args.base,
                 "input_size": args.size, "val_metrics": metrics, "val_loss": val_loss,
                 "architecture": "SiameseUNet"},
                CHECKPOINT_PATH,
            )
            print(f"  saved checkpoint -> {CHECKPOINT_PATH}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

        with open(METRICS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    plot_curve(history, [("train_loss", "Train loss")], "Training loss", "Loss", "training_loss.png")
    plot_curve(history, [("val_loss", "Validation loss")], "Validation loss", "Loss", "validation_loss.png")
    plot_curve(history, [("iou", "Validation IoU")], "IoU", "IoU", "iou_curve.png")
    plot_curve(history, [("f1", "Validation F1")], "F1 score", "F1", "f1_curve.png")
    print(f"Done. Metrics: {METRICS_CSV}  Plots: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
