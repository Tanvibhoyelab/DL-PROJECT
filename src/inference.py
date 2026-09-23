"""Inference using the trained change_detector.pth checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
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
    MODEL_CONFIG,
)

from src.model import (
    build_model,
    count_parameters,
)

from src.utils import (
    clean_mask,
    compute_statistics,
    diff_change_mask,
    difference_image,
    extract_regions,
    mask_image,
    overlay_image,
)


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

    val_metrics: dict


_CACHE: LoadedModel | None = None


def checkpoint_exists(
    path: Path = CHECKPOINT_PATH,
) -> bool:

    return Path(path).is_file()


def load_model(
    path: Path = CHECKPOINT_PATH,
    force: bool = False,
) -> LoadedModel:

    global _CACHE

    path = Path(path).resolve()

    if (
        _CACHE is not None
        and not force
        and Path(
            _CACHE.checkpoint_path
        ).resolve() == path
    ):
        return _CACHE

    if not path.is_file():

        raise ModelNotAvailable(
            "change_detector.pth was not found.\n\n"
            f"Expected location:\n{path}"
        )

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    try:

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as exc:

        raise ModelNotAvailable(
            "Unable to read change_detector.pth.\n\n"
            f"Error: {exc}"
        ) from exc

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise ModelNotAvailable(
            "Invalid checkpoint format."
        )

    # -----------------------------------------------------
    # Extract state dictionary
    # -----------------------------------------------------

    state = checkpoint.get(
        "model_state_dict"
    )

    if state is None:

        raise ModelNotAvailable(
            "change_detector.pth does not contain "
            "'model_state_dict'."
        )

    # -----------------------------------------------------
    # Read model configuration from checkpoint
    # -----------------------------------------------------

    saved_config = checkpoint.get(
        "model_config",
        {},
    )

    if not isinstance(
        saved_config,
        dict,
    ):

        raise ModelNotAvailable(
            "Invalid model_config in checkpoint."
        )

    architecture = saved_config.get(
        "architecture",
        "SiameseUNet",
    )

    in_channels = int(
        saved_config.get(
            "in_channels",
            3,
        )
    )

    base_channels = int(
        saved_config.get(
            "base_channels",
            32,
        )
    )

    input_size = int(
        saved_config.get(
            "img_size",
            IMAGE_SIZE,
        )
    )

    # -----------------------------------------------------
    # Validate checkpoint configuration
    # -----------------------------------------------------

    if architecture != "SiameseUNet":

        raise ModelNotAvailable(
            "Architecture mismatch.\n"
            f"Checkpoint: {architecture}\n"
            "Expected: SiameseUNet"
        )

    if in_channels != 3:

        raise ModelNotAvailable(
            "Input-channel mismatch.\n"
            f"Checkpoint: {in_channels}\n"
            "Expected: 3"
        )

    if base_channels != 32:

        raise ModelNotAvailable(
            "Base-channel mismatch.\n"
            f"Checkpoint: {base_channels}\n"
            "Expected: 32"
        )

    # -----------------------------------------------------
    # Build exact architecture
    # -----------------------------------------------------

    model = build_model(
        in_channels=in_channels,
        base_channels=base_channels,
    )

    # -----------------------------------------------------
    # Strict checkpoint loading
    # -----------------------------------------------------

    try:

        model.load_state_dict(
            state,
            strict=True,
        )

    except Exception as exc:

        raise ModelNotAvailable(
            "The model architecture does not match "
            "change_detector.pth.\n\n"
            f"{exc}"
        ) from exc

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    model.to(
        DEVICE
    )

    model.eval()

    # -----------------------------------------------------
    # Cache
    # -----------------------------------------------------

    _CACHE = LoadedModel(

        model=model,

        device=DEVICE,

        parameters=count_parameters(
            model
        ),

        input_size=input_size,

        trained_epochs=int(
            checkpoint.get(
                "epoch",
                0,
            )
        ),

        checkpoint_path=str(
            path
        ),

        val_metrics=checkpoint.get(
            "val_metrics",
            {},
        ),
    )

    return _CACHE


def _to_tensor(
    rgb: np.ndarray,
    size: int,
    device: torch.device,
) -> torch.Tensor:

    if rgb is None:

        raise ValueError(
            "Image is missing."
        )

    if rgb.ndim != 3:

        raise ValueError(
            "Image must have 3 dimensions."
        )

    if rgb.shape[2] != 3:

        raise ValueError(
            "Image must contain 3 RGB channels."
        )

    # Resize to checkpoint input size
    image = cv2.resize(
        rgb,
        (
            size,
            size,
        ),
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(
        np.float32
    )

    # -----------------------------------------------------
    # Normalize to 0-1.
    #
    # LEVIR-CD training images are uint8 0-255, but arrays
    # coming from Earth Engine / satellite.py can already be
    # float reflectance scaled to roughly 0-1 (or 0-1 after
    # earlier processing). Blindly dividing by 255 a second
    # time crushes every pixel to near-zero, which starves
    # the model of signal and makes it predict "no change"
    # everywhere regardless of the real input. Only divide
    # by 255 when the data actually looks like 0-255 range.
    # -----------------------------------------------------

    max_value = float(
        image.max()
    ) if image.size else 0.0

    if max_value > 1.5:

        image = image / 255.0

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    # HWC -> CHW
    image = np.ascontiguousarray(
        image.transpose(
            2,
            0,
            1,
        )
    )

    tensor = torch.from_numpy(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    return tensor.to(
        device
    )


@torch.no_grad()
def predict_probability(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    loaded: LoadedModel,
) -> np.ndarray:

    size = loaded.input_size

    before = _to_tensor(
        before_rgb,
        size,
        loaded.device,
    )

    after = _to_tensor(
        after_rgb,
        size,
        loaded.device,
    )

    try:

        logits = loaded.model(
            before,
            after,
        )

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    except RuntimeError as exc:

        raise ModelNotAvailable(
            f"Model inference failed:\n{exc}"
        ) from exc

    h, w = before_rgb.shape[:2]

    probability = cv2.resize(
        probability,
        (
            w,
            h,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    return probability


def run_change_detection(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    bounds,
    threshold: float = DEFAULT_THRESHOLD,
    min_pixels: int = MIN_REGION_PIXELS,
    loaded: LoadedModel | None = None,
) -> dict:

    # -----------------------------------------------------
    # Validate images
    # -----------------------------------------------------

    if before_rgb is None:

        raise ValueError(
            "Before image is missing."
        )

    if after_rgb is None:

        raise ValueError(
            "After image is missing."
        )

    if before_rgb.ndim != 3:

        raise ValueError(
            "Before image is invalid."
        )

    if after_rgb.ndim != 3:

        raise ValueError(
            "After image is invalid."
        )

    # -----------------------------------------------------
    # Keep dimensions compatible
    # -----------------------------------------------------

    if before_rgb.shape[:2] != after_rgb.shape[:2]:

        h = min(
            before_rgb.shape[0],
            after_rgb.shape[0],
        )

        w = min(
            before_rgb.shape[1],
            after_rgb.shape[1],
        )

        before_rgb = cv2.resize(
            before_rgb,
            (w, h),
            interpolation=cv2.INTER_AREA,
        )

        after_rgb = cv2.resize(
            after_rgb,
            (w, h),
            interpolation=cv2.INTER_AREA,
        )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    loaded = (
        loaded
        if loaded is not None
        else load_model()
    )

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    print(
        "[inference] before_rgb dtype="
        f"{before_rgb.dtype} min={before_rgb.min():.4f} "
        f"max={before_rgb.max():.4f} | after_rgb dtype="
        f"{after_rgb.dtype} min={after_rgb.min():.4f} "
        f"max={after_rgb.max():.4f}"
    )

    probability = predict_probability(
        before_rgb,
        after_rgb,
        loaded,
    )

    print(
        "[inference] probability min="
        f"{probability.min():.6f} max={probability.max():.6f} "
        f"mean={probability.mean():.6f}"
    )

    # -----------------------------------------------------
    # RAW MODEL MASK
    # -----------------------------------------------------

    raw_mask = (
        probability >= threshold
    ).astype(
        np.uint8
    )

    # IMPORTANT:
    # Count before cleanup.
    raw_changed_pixels = int(
        raw_mask.sum()
    )

    # -----------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------

    binary = clean_mask(
        raw_mask,
        min_pixels=min_pixels,
    )

    cleaned_changed_pixels = int(
        binary.sum()
    )

    total_pixels = int(
        binary.size
    )

    # -----------------------------------------------------
    # FALLBACK: classical pixel-difference detector
    #
    # At Sentinel-2 resolution the trained model can be
    # under-confident even when a real, visible change exists
    # (see RESOLUTION_NOTE). If the model's own mask is empty,
    # fall back to a percentile-based raw pixel-difference mask
    # so genuinely visible changes still surface — clearly
    # labelled as pixel-based, not model-based.
    # -----------------------------------------------------

    fallback_used = False
    confidence_map = probability

    if cleaned_changed_pixels == 0:
        fallback_binary, fallback_diff = diff_change_mask(
            before_rgb,
            after_rgb,
            min_pixels=min_pixels,
        )
        if int(fallback_binary.sum()) > 0:
            binary = fallback_binary
            cleaned_changed_pixels = int(binary.sum())
            confidence_map = fallback_diff
            fallback_used = True

    # -----------------------------------------------------
    # Regions
    # -----------------------------------------------------

    regions = extract_regions(
        confidence_map,
        binary,
        bounds,
        before_rgb,
        after_rgb,
        min_pixels,
    )

    if fallback_used:
        for region in regions:
            region.change_type = (
                "Possible change (pixel-difference fallback, "
                f"low model confidence): {region.change_type}"
            )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    statistics = compute_statistics(
        binary,
        probability,
        regions,
        bounds,
    )

    # -----------------------------------------------------
    # Percentages
    # -----------------------------------------------------

    raw_percentage = 0.0

    final_percentage = 0.0

    if total_pixels > 0:

        raw_percentage = (
            raw_changed_pixels
            / total_pixels
            * 100.0
        )

        final_percentage = (
            cleaned_changed_pixels
            / total_pixels
            * 100.0
        )

    # -----------------------------------------------------
    # Probability diagnostics
    # -----------------------------------------------------

    diagnostics = {

        "min_probability": float(
            np.min(
                probability
            )
        ),

        "max_probability": float(
            np.max(
                probability
            )
        ),

        "mean_probability": float(
            np.mean(
                probability
            )
        ),

        "p95_probability": float(
            np.percentile(
                probability,
                95,
            )
        ),

        "p99_probability": float(
            np.percentile(
                probability,
                99,
            )
        ),

        "raw_changed_pixels":
            raw_changed_pixels,

        "cleaned_changed_pixels":
            cleaned_changed_pixels,

        "total_pixels":
            total_pixels,

        "raw_change_percentage":
            raw_percentage,

        "cleaned_change_percentage":
            final_percentage,

        "fallback_used":
            fallback_used,
    }

    # -----------------------------------------------------
    # Return everything
    # -----------------------------------------------------

    return {

        "probability":
            probability,

        "raw_mask":
            raw_mask,

        "mask":
            binary,

        "raw_mask_image":
            mask_image(
                raw_mask
            ),

        "mask_image":
            mask_image(
                binary
            ),

        "difference":
            difference_image(
                before_rgb,
                after_rgb,
            ),

        "overlay":
            overlay_image(
                after_rgb,
                binary,
                regions,
            ),

        "regions":
            regions,

        "statistics":
            statistics,

        "diagnostics":
            diagnostics,

        "threshold":
            threshold,

        "model": {

            "architecture":
                "Siamese U-Net",

            "parameters":
                loaded.parameters,

            "device":
                str(
                    loaded.device
                ),

            "checkpoint":
                loaded.checkpoint_path,

            "trained_epochs":
                loaded.trained_epochs,
        },
    }