"""Region extraction, statistics, overlays and change interpretation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class ChangeRegion:
    change_id: int
    change_type: str
    pixels: int
    area_m2: float
    area_hectares: float
    confidence: float
    bbox: tuple[int, int, int, int]      # x, y, w, h in pixel space
    centroid_px: tuple[float, float]
    centroid_lat: float
    centroid_lon: float
    bbox_latlon: tuple[float, float, float, float]  # min_lat, min_lon, max_lat, max_lon

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Pixel <-> geographic mapping
# --------------------------------------------------------------------------
def pixel_size_m(bounds: tuple[float, float, float, float], width: int, height: int):
    """Return (metres per pixel in x, metres per pixel in y)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    mid_lat = (min_lat + max_lat) / 2
    width_m = (max_lon - min_lon) * 111_320 * math.cos(math.radians(mid_lat))
    height_m = (max_lat - min_lat) * 110_570
    return abs(width_m / max(width, 1)), abs(height_m / max(height, 1))


def pixel_to_latlon(x: float, y: float, bounds, width: int, height: int):
    min_lon, min_lat, max_lon, max_lat = bounds
    lon = min_lon + (x / max(width, 1)) * (max_lon - min_lon)
    lat = max_lat - (y / max(height, 1)) * (max_lat - min_lat)
    return lat, lon


# --------------------------------------------------------------------------
# Mask cleaning + region extraction
# --------------------------------------------------------------------------
def clean_mask(mask: np.ndarray, min_pixels: int = 20, kernel: int = 3) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            out[labels == i] = 1
    return out


def classify_change(area_m2: float, before_patch: np.ndarray, after_patch: np.ndarray) -> str:
    """Conservative, resolution-aware labelling.

    Sentinel-2 is ~10 m per pixel, so individual buildings cannot be resolved.
    Labels are therefore phrased as possibilities, not certainties.
    """
    def greenness(patch: np.ndarray) -> float:
        p = patch.astype(np.float32) + 1e-6
        return float((p[..., 1] / (p[..., 0] + p[..., 1] + p[..., 2])).mean())

    def blueness(patch: np.ndarray) -> float:
        p = patch.astype(np.float32) + 1e-6
        return float((p[..., 2] / (p[..., 0] + p[..., 1] + p[..., 2])).mean())

    def brightness(patch: np.ndarray) -> float:
        return float(patch.astype(np.float32).mean())

    if before_patch.size == 0 or after_patch.size == 0:
        return "Unclassified change"

    g_before, g_after = greenness(before_patch), greenness(after_patch)
    b_before, b_after = blueness(before_patch), blueness(after_patch)
    br_before, br_after = brightness(before_patch), brightness(after_patch)

    if b_after - b_before > 0.05:
        return "Water area increase"
    if b_before - b_after > 0.05:
        return "Water area decrease"
    if g_before - g_after > 0.03 and br_after > br_before:
        return ("Vegetation loss, possible construction activity"
                if area_m2 > 2000 else "Vegetation / land-cover change")
    if g_after - g_before > 0.03:
        return "Vegetation gain / re-greening"
    if br_after - br_before > 12:
        return ("Building / structural change (new bright surface)"
                if area_m2 > 2000 else "Small surface change")
    if br_before - br_after > 12:
        return ("Possible building removal or clearing"
                if area_m2 > 2000 else "Small surface change")
    if area_m2 > 10000:
        return "Large structural / land-use change"
    return "Land-cover change"


def extract_regions(
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    bounds,
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    min_pixels: int = 20,
) -> list[ChangeRegion]:
    h, w = binary_mask.shape
    px_x, px_y = pixel_size_m(bounds, w, h)
    pixel_area = px_x * px_y

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    regions: list[ChangeRegion] = []
    for i in range(1, n):
        pixels = int(stats[i, cv2.CC_STAT_AREA])
        if pixels < min_pixels:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        comp = labels == i
        confidence = float(prob_map[comp].mean())
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        lat, lon = pixel_to_latlon(cx, cy, bounds, w, h)
        min_lat, min_lon = pixel_to_latlon(x, y + bh, bounds, w, h)
        max_lat, max_lon = pixel_to_latlon(x + bw, y, bounds, w, h)

        area_m2 = pixels * pixel_area
        change_type = classify_change(
            area_m2, before_rgb[y:y + bh, x:x + bw], after_rgb[y:y + bh, x:x + bw]
        )
        regions.append(
            ChangeRegion(
                change_id=len(regions) + 1,
                change_type=change_type,
                pixels=pixels,
                area_m2=round(area_m2, 1),
                area_hectares=round(area_m2 / 10_000, 4),
                confidence=round(confidence, 4),
                bbox=(x, y, bw, bh),
                centroid_px=(round(cx, 1), round(cy, 1)),
                centroid_lat=round(lat, 6),
                centroid_lon=round(lon, 6),
                bbox_latlon=(round(min_lat, 6), round(min_lon, 6), round(max_lat, 6), round(max_lon, 6)),
            )
        )
    regions.sort(key=lambda r: r.pixels, reverse=True)
    for idx, r in enumerate(regions, start=1):
        r.change_id = idx
    return regions


def diff_change_mask(before: np.ndarray, after: np.ndarray, min_pixels: int = 20,
                      percentile: float = 97.0) -> tuple[np.ndarray, np.ndarray]:
    """Classical (non-model) fallback change detector.

    Used only when the trained model's confidence is too low to flag any
    region (a known limitation at Sentinel-2 resolution — see RESOLUTION_NOTE).
    Flags the most different pixels between before/after by raw intensity
    difference rather than a learned signal. This is intentionally a
    fallback, not a replacement: real dashboards should always be clear
    about which detector produced a given region.

    Returns (binary_mask, normalized_diff) where normalized_diff is 0-1
    and can be used the same way as the model's probability map (e.g. for
    per-region confidence in extract_regions).
    """
    diff = cv2.absdiff(after.astype(np.int16), before.astype(np.int16)).astype(np.uint8)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    normalized = gray.astype(np.float32) / 255.0

    cutoff = float(np.percentile(normalized, percentile))
    raw_mask = (normalized >= max(cutoff, 1e-6)).astype(np.uint8)
    binary = clean_mask(raw_mask, min_pixels=min_pixels)

    return binary, normalized


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def compute_statistics(binary_mask: np.ndarray, prob_map: np.ndarray, regions, bounds) -> dict:
    h, w = binary_mask.shape
    px_x, px_y = pixel_size_m(bounds, w, h)
    pixel_area = px_x * px_y
    total = h * w
    changed = int(binary_mask.sum())
    largest = max((r.area_hectares for r in regions), default=0.0)
    avg_conf = float(np.mean([r.confidence for r in regions])) if regions else 0.0
    return {
        "total_pixels": total,
        "changed_pixels": changed,
        "change_percentage": round(100 * changed / max(total, 1), 3),
        "regions": len(regions),
        "largest_region_hectares": round(largest, 4),
        "average_confidence": round(avg_conf, 4),
        "aoi_area_km2": round(total * pixel_area / 1_000_000, 4),
        "changed_area_m2": round(changed * pixel_area, 1),
        "changed_area_hectares": round(changed * pixel_area / 10_000, 4),
        "metres_per_pixel_x": round(px_x, 2),
        "metres_per_pixel_y": round(px_y, 2),
        "pixel_area_m2": round(pixel_area, 2),
    }


# --------------------------------------------------------------------------
# Visual products
# --------------------------------------------------------------------------
def difference_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(after.astype(np.int16), before.astype(np.int16)).astype(np.uint8)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)[..., ::-1]


def mask_image(binary_mask: np.ndarray) -> np.ndarray:
    return (binary_mask.astype(np.uint8) * 255)


def overlay_image(after: np.ndarray, binary_mask: np.ndarray, regions, high_conf: float = 0.75) -> np.ndarray:
    out = after.copy()
    fill = np.zeros_like(out)
    fill[binary_mask.astype(bool)] = (255, 90, 0)
    out = cv2.addWeighted(out, 1.0, fill, 0.45, 0)

    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (255, 40, 0), 1)

    for r in regions:
        x, y, w, h = r.bbox
        color = (255, 0, 0) if r.confidence >= high_conf else (255, 210, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, str(r.change_id), (x + 2, max(y - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# Plain-language summary
# --------------------------------------------------------------------------
def human_summary(location: str, before_date: str, after_date: str, regions, stats: dict) -> str:
    if not regions:
        return (
            f"Between {before_date} and {after_date}, the model found no significant changed "
            f"region inside the selected area at {location}. The scene looks essentially the "
            "same at Sentinel-2's 10 m resolution."
        )
    main = regions[0]
    return (
        f"Between {before_date} and {after_date}, the model detected "
        f"{len(regions)} significant changed region(s) within the selected AOI at {location}.\n\n"
        f"Main detected change: {main.change_type}\n"
        f"Confidence: {main.confidence * 100:.0f}%\n"
        f"Approximate changed area (all regions): {stats['changed_area_hectares']:.3f} hectares "
        f"({stats['change_percentage']:.2f}% of the AOI)\n"
        f"Average confidence across regions: {stats['average_confidence'] * 100:.0f}%"
    )


RESOLUTION_NOTE = (
    "Sentinel-2's finest bands are about 10 m per pixel, so one pixel covers roughly "
    "100 m². Individual houses cannot be resolved. Results are therefore described as "
    "'possible building/structural change' only when the changed patch is large enough "
    "to be meaningful at this scale. Detecting single buildings reliably would require "
    "higher-resolution imagery (for example 0.3–1 m commercial satellites or aerial data)."
)