"""Inference with the trained Siamese U-Net checkpoint.

The model is the only change detector used by default. A classical
pixel-difference detector is available as an explicit, clearly-labelled
alternative, but it is never substituted silently for the model: if the model
finds nothing, the dashboard says so instead of inventing regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from config import (
    CHECKPOINT_PATH,
    DEFAULT_THRESHOLD,
    DEVICE,
    IMAGE_SIZE,
    MIN_REGION_PIXELS,
)
from src.model import build_model, count_parameters
from src.utils import (
    clean_mask,
    compute_statistics,
    diff_change_mask,
    difference_image,
    extract_regions,
    mask_image,
    overlay_image,
)

METHOD_MODEL = "model"
METHOD_DIFFERENCE = "difference"


class ModelNotAvailable(RuntimeError):
    pass


@dataclass
class LoadedModel:
    model: torch.nn.Module
    device: torch.device
    parameters: int
    input_size: int
    trained_epochs: int
    checkpoint_path: str
    val_metrics: dict = field(default_factory=dict)
    threshold: float = DEFAULT_THRESHOLD
    base_channels: int = 32
    dataset: str = ""


_CACHE: LoadedModel | None = None


def checkpoint_exists(path: Path = CHECKPOINT_PATH) -> bool:
    return Path(path).is_file()


def _extract_state_dict(checkpoint: dict) -> dict:
    """Accept both the current and the older checkpoint layouts."""
    for key in ("model_state_dict", "model_state", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict) and state:
            return state
    # A bare state dict saved with torch.save(model.state_dict()).
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ModelNotAvailable(
        "The checkpoint contains no model weights (expected a 'model_state_dict' entry).\n"
        "Retrain with:  python -m src.train --data data/LEVIR-CD"
    )


def _infer_base_channels(state: dict, fallback: int) -> int:
    weight = state.get("e1.block.0.weight")
    if weight is not None and hasattr(weight, "shape"):
        return int(weight.shape[0])
    return fallback


def load_model(path: Path = CHECKPOINT_PATH, force: bool = False) -> LoadedModel:
    global _CACHE

    path = Path(path).resolve()
    if _CACHE is not None and not force and Path(_CACHE.checkpoint_path).resolve() == path:
        return _CACHE

    if not path.is_file():
        raise ModelNotAvailable(
            "No trained checkpoint was found.\n\n"
            f"Expected location:\n{path}\n\n"
            "Train one with:\n"
            "  python download_data.py\n"
            "  python -m src.train --data data/LEVIR-CD"
        )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ModelNotAvailable(f"Unable to read {path.name}.\n\nError: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise ModelNotAvailable("Invalid checkpoint format: expected a dictionary.")

    state = _extract_state_dict(checkpoint)
    saved_config = checkpoint.get("model_config")
    if not isinstance(saved_config, dict):
        # Older checkpoints stored these flat.
        saved_config = {
            "architecture": checkpoint.get("architecture", "SiameseUNet"),
            "in_channels": 3,
            "base_channels": checkpoint.get("base", 32),
            "img_size": checkpoint.get("input_size", IMAGE_SIZE),
        }

    architecture = saved_config.get("architecture", "SiameseUNet")
    if architecture != "SiameseUNet":
        raise ModelNotAvailable(
            f"Architecture mismatch: the checkpoint was trained as '{architecture}', "
            "this app builds a SiameseUNet."
        )

    in_channels = int(saved_config.get("in_channels", 3))
    base_channels = _infer_base_channels(state, int(saved_config.get("base_channels", 32)))
    input_size = int(saved_config.get("img_size", IMAGE_SIZE))

    model = build_model(in_channels=in_channels, base_channels=base_channels)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ModelNotAvailable(
            f"The checkpoint weights do not match the SiameseUNet definition.\n\n{exc}"
        ) from exc

    model.to(DEVICE).eval()

    _CACHE = LoadedModel(
        model=model,
        device=DEVICE,
        parameters=count_parameters(model),
        input_size=input_size,
        trained_epochs=int(checkpoint.get("epoch", 0)),
        checkpoint_path=str(path),
        val_metrics=checkpoint.get("val_metrics") or {},
        threshold=float(checkpoint.get("threshold", DEFAULT_THRESHOLD)),
        base_channels=base_channels,
        dataset=str(checkpoint.get("dataset", "")),
    )
    return _CACHE


def _to_tensor(rgb: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    image = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Training images are uint8 0-255; arrays that already arrive as 0-1 floats
    # must not be divided again, which would crush all signal to near zero.
    if float(image.max()) > 1.5:
        image /= 255.0
    image = np.clip(image, 0.0, 1.0)
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
    return tensor.to(device)


@torch.no_grad()
def predict_probability(
    before_rgb: np.ndarray, after_rgb: np.ndarray, loaded: LoadedModel
) -> np.ndarray:
    size = loaded.input_size
    before = _to_tensor(before_rgb, size, loaded.device)
    after = _to_tensor(after_rgb, size, loaded.device)
    try:
        logits = loaded.model(before, after)
        # The change map must be symmetric in time: a pair swap should not
        # change where the change is. Averaging both orders removes the small
        # asymmetry left by training and steadies real Sentinel-2 predictions.
        logits_swapped = loaded.model(after, before)
        probability = torch.sigmoid((logits + logits_swapped) / 2)[0, 0].cpu().numpy()
    except RuntimeError as exc:
        raise ModelNotAvailable(f"Model inference failed:\n{exc}") from exc

    h, w = before_rgb.shape[:2]
    return cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)


def _align(before_rgb: np.ndarray, after_rgb: np.ndarray):
    if before_rgb.shape[:2] == after_rgb.shape[:2]:
        return before_rgb, after_rgb
    h = min(before_rgb.shape[0], after_rgb.shape[0])
    w = min(before_rgb.shape[1], after_rgb.shape[1])
    return (
        cv2.resize(before_rgb, (w, h), interpolation=cv2.INTER_AREA),
        cv2.resize(after_rgb, (w, h), interpolation=cv2.INTER_AREA),
    )


def run_change_detection(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    bounds,
    threshold: float | None = None,
    min_pixels: int = MIN_REGION_PIXELS,
    loaded: LoadedModel | None = None,
    method: str = METHOD_MODEL,
) -> dict:
    """Detect change between two aligned RGB images.

    method="model"      -> trained Siamese U-Net (default).
    method="difference" -> classical pixel-difference detector, no model. Only
                           used when the user explicitly asks for it.
    """
    if before_rgb is None or after_rgb is None:
        raise ValueError("Both a before and an after image are required.")
    if before_rgb.ndim != 3 or after_rgb.ndim != 3:
        raise ValueError("Before/after images must be RGB arrays.")

    before_rgb, after_rgb = _align(before_rgb, after_rgb)

    if method == METHOD_DIFFERENCE:
        binary, confidence_map = diff_change_mask(before_rgb, after_rgb, min_pixels=min_pixels)
        probability = confidence_map
        raw_mask = binary.copy()
        model_info = {"architecture": "pixel difference (no neural network)", "parameters": 0}
        used_threshold = float("nan")
    else:
        loaded = loaded if loaded is not None else load_model()
        used_threshold = float(threshold if threshold is not None else loaded.threshold)
        probability = predict_probability(before_rgb, after_rgb, loaded)
        raw_mask = (probability >= used_threshold).astype(np.uint8)
        binary = clean_mask(raw_mask, min_pixels=min_pixels)
        confidence_map = probability
        model_info = {
            "architecture": "Siamese U-Net",
            "parameters": loaded.parameters,
            "device": str(loaded.device),
            "checkpoint": loaded.checkpoint_path,
            "trained_epochs": loaded.trained_epochs,
            "tuned_threshold": loaded.threshold,
            "val_metrics": loaded.val_metrics,
            "training_dataset": loaded.dataset,
        }

    regions = extract_regions(confidence_map, binary, bounds, before_rgb, after_rgb, min_pixels)
    statistics = compute_statistics(binary, confidence_map, regions, bounds)

    total_pixels = int(binary.size)
    diagnostics = {
        "method": method,
        "min_probability": float(np.min(probability)),
        "max_probability": float(np.max(probability)),
        "mean_probability": float(np.mean(probability)),
        "p95_probability": float(np.percentile(probability, 95)),
        "p99_probability": float(np.percentile(probability, 99)),
        "raw_changed_pixels": int(raw_mask.sum()),
        "cleaned_changed_pixels": int(binary.sum()),
        "total_pixels": total_pixels,
        "raw_change_percentage": 100.0 * float(raw_mask.sum()) / max(total_pixels, 1),
        "cleaned_change_percentage": 100.0 * float(binary.sum()) / max(total_pixels, 1),
    }

    return {
        "probability": probability,
        "raw_mask": raw_mask,
        "mask": binary,
        "raw_mask_image": mask_image(raw_mask),
        "mask_image": mask_image(binary),
        "difference": difference_image(before_rgb, after_rgb),
        "overlay": overlay_image(after_rgb, binary, regions),
        "before": before_rgb,
        "after": after_rgb,
        "regions": regions,
        "statistics": statistics,
        "diagnostics": diagnostics,
        "threshold": used_threshold,
        "method": method,
        "model": model_info,
    }
