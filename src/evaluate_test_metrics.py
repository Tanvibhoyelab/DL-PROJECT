# ============================================================
# Test-set evaluation cell — computes Accuracy, Precision,
# Recall, F1-score and IoU for Table III.
#
# Assumes you already have, from earlier cells:
#   - model            : your trained SiameseUNet, on `device`
#   - test_loader       : DataLoader over the 2,048-pair LEVIR-CD test split
#   - best checkpoint already loaded into `model`
#
# Adjust the three lines marked "ADAPT" to match your actual
# variable / method names if they differ.
# ============================================================

import torch

@torch.no_grad()
def evaluate_test_set(model, test_loader, device, threshold: float = 0.5):
    model.eval()
    TP = TN = FP = FN = 0

    for batch in test_loader:
        # ADAPT: unpack however your dataset returns items.
        # Common pattern: (before_img, after_img, mask)
        before_img, after_img, mask = batch
        before_img = before_img.to(device)
        after_img = after_img.to(device)
        mask = mask.to(device).float()          # ground-truth, values in {0,1}

        # ADAPT: call your model the way it's actually defined.
        # Per the paper's Fig. 1, forward(before, after) -> single-channel logits
        logits = model(before_img, after_img)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()

        # Flatten to 1D so shape mismatches (e.g. extra channel dim) don't break the sums
        pred = pred.view(-1)
        mask = mask.view(-1)

        TP += torch.sum((pred == 1) & (mask == 1)).item()
        TN += torch.sum((pred == 0) & (mask == 0)).item()
        FP += torch.sum((pred == 1) & (mask == 0)).item()
        FN += torch.sum((pred == 0) & (mask == 1)).item()

    eps = 1e-8  # avoid div-by-zero if a batch is degenerate
    accuracy  = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = TP / (TP + FP + FN + eps)

    print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"IoU      : {iou:.4f}")

    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "iou": iou,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    }

# Run it:
# results = evaluate_test_set(model, test_loader, device)
# Copy the five printed numbers straight into Table III, replacing [INSERT].