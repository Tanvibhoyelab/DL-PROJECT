# ============================================================
# Qualitative results figure — before / after / ground truth /
# predicted mask, side by side, for a handful of test examples.
# Produces figure2.png to drop into the paper in place of the
# "FIGURE 2 PLACEHOLDER" box.
#
# Assumes: model, test_loader, device already defined/loaded
# (same as evaluate_test_metrics.py).
# ============================================================

import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def make_figure2(model, test_loader, device, n_examples: int = 4,
                  threshold: float = 0.5, out_path: str = "figure2.png"):
    model.eval()

    shown = 0
    fig, axes = plt.subplots(n_examples, 4, figsize=(12, 3 * n_examples))
    col_titles = ["Before", "After", "Ground Truth", "Prediction"]

    for batch in test_loader:
        before_img, after_img, mask = batch  # ADAPT if your loader differs

        for i in range(before_img.size(0)):
            if shown >= n_examples:
                break

            b = before_img[i:i+1].to(device)
            a = after_img[i:i+1].to(device)
            gt = mask[i, 0].cpu().numpy() if mask.dim() == 4 else mask[i].cpu().numpy()

            # Only pick examples that actually contain change pixels,
            # so the figure is informative rather than all-black.
            if gt.sum() == 0:
                continue

            logits = model(b, a)
            pred = (torch.sigmoid(logits) > threshold).float()[0, 0].cpu().numpy()

            # Un-normalize for display if you normalized inputs earlier (ADAPT as needed).
            before_disp = before_img[i].permute(1, 2, 0).cpu().numpy()
            after_disp = after_img[i].permute(1, 2, 0).cpu().numpy()

            row = axes[shown]
            row[0].imshow(before_disp.clip(0, 1)); row[0].axis("off")
            row[1].imshow(after_disp.clip(0, 1));  row[1].axis("off")
            row[2].imshow(gt, cmap="gray");        row[2].axis("off")
            row[3].imshow(pred, cmap="gray");      row[3].axis("off")

            if shown == 0:
                for ax, title in zip(row, col_titles):
                    ax.set_title(title, fontsize=11)

            shown += 1

        if shown >= n_examples:
            break

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path} — insert this image in place of the Figure 2 placeholder.")

# Run it:
# make_figure2(model, test_loader, device, n_examples=4)