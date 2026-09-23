"""Real Sentinel-2 retrieval through Google Earth Engine.

There is no synthetic-image fallback anywhere in this module. If real imagery
cannot be retrieved, a SatelliteError is raised with the exact reason so the
dashboard can show it to the user.
"""

from __future__ import annotations

import datetime as dt
import io
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests
from PIL import Image

from config import (
    DEFAULT_MAX_CLOUD,
    DEFAULT_SEARCH_WINDOW_DAYS,
    GEE_PRIVATE_KEY_FILE,
    GEE_PROJECT_ID,
    GEE_SERVICE_ACCOUNT,
    NATIVE_RESOLUTION_M,
    RGB_BANDS,
    S2_COLLECTION,
    THUMB_SIZE,
)


class SatelliteError(RuntimeError):
    """Any failure while retrieving real satellite imagery."""


_EE_READY = False


def init_earth_engine(project_id: str = "") -> str:
    """Initialise Earth Engine. Returns the project id actually used."""
    global _EE_READY
    project = project_id or GEE_PROJECT_ID
    if not project:
        raise SatelliteError(
            "No Google Earth Engine project id configured. Set GEE_PROJECT_ID in your "
            ".env file (see README, 'Google Earth Engine setup')."
        )
    try:
        import ee
    except ImportError as exc:  # pragma: no cover
        raise SatelliteError(
            "The 'earthengine-api' package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        if GEE_SERVICE_ACCOUNT and GEE_PRIVATE_KEY_FILE:
            credentials = ee.ServiceAccountCredentials(GEE_SERVICE_ACCOUNT, GEE_PRIVATE_KEY_FILE)
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project)
        # Force a real round-trip so authentication problems surface now.
        ee.Number(1).getInfo()
    except Exception as exc:
        raise SatelliteError(
            "Google Earth Engine authentication failed. Run 'earthengine authenticate' in "
            "your terminal, make sure the project is registered for Earth Engine, and check "
            f"your internet connection.\nOriginal error: {exc}"
        ) from exc
    _EE_READY = True
    return project


def is_ready() -> bool:
    return _EE_READY


# --------------------------------------------------------------------------
# Area of interest
# --------------------------------------------------------------------------
@dataclass
class AOI:
    kind: str                       # "radius" | "rectangle" | "polygon"
    lat: float
    lon: float
    radius_m: Optional[float] = None
    coords: Optional[list] = None   # [[lon, lat], ...] for rectangle/polygon
    label: str = ""

    def bounds(self) -> tuple[float, float, float, float]:
        """(min_lon, min_lat, max_lon, max_lat)"""
        if self.kind == "radius":
            d_lat = self.radius_m / 111_320.0
            d_lon = self.radius_m / (111_320.0 * max(math.cos(math.radians(self.lat)), 1e-6))
            return (self.lon - d_lon, self.lat - d_lat, self.lon + d_lon, self.lat + d_lat)
        lons = [c[0] for c in self.coords]
        lats = [c[1] for c in self.coords]
        return (min(lons), min(lats), max(lons), max(lats))

    def ee_geometry(self):
        import ee

        if self.kind == "radius":
            min_lon, min_lat, max_lon, max_lat = self.bounds()
            return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        return ee.Geometry.Polygon([self.coords])

    def area_km2(self) -> float:
        min_lon, min_lat, max_lon, max_lat = self.bounds()
        width_km = (max_lon - min_lon) * 111.32 * math.cos(math.radians(self.lat))
        height_km = (max_lat - min_lat) * 110.57
        return abs(width_km * height_km)

    def folium_bounds(self):
        min_lon, min_lat, max_lon, max_lat = self.bounds()
        return [[min_lat, min_lon], [max_lat, max_lon]]


def make_radius_aoi(lat: float, lon: float, radius_m: float, label: str = "") -> AOI:
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise SatelliteError("Latitude or longitude is out of range.")
    if radius_m <= 0 or radius_m > 20000:
        raise SatelliteError("AOI radius must be between 1 m and 20 km.")
    return AOI("radius", lat, lon, radius_m=radius_m, label=label)


def make_polygon_aoi(coords: list, label: str = "", kind: str = "polygon") -> AOI:
    if not coords or len(coords) < 3:
        raise SatelliteError("A drawn AOI needs at least three points.")
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    lat = sum(c[1] for c in coords) / len(coords)
    lon = sum(c[0] for c in coords) / len(coords)
    return AOI(kind, lat, lon, coords=coords, label=label)


# --------------------------------------------------------------------------
# Sentinel-2 scene search
# --------------------------------------------------------------------------
@dataclass
class Scene:
    rgb: np.ndarray                 # (H, W, 3) uint8, real Sentinel-2 imagery
    acquisition_date: str
    cloud_cover: float
    scene_id: str
    composite: bool = False
    n_images: int = 1
    resolution_m: int = NATIVE_RESOLUTION_M
    bands: str = "/".join(RGB_BANDS)
    meta: dict = field(default_factory=dict)


def _fetch_thumb(image, geometry, size: int) -> np.ndarray:
    url = image.getThumbURL(
        {
            "region": geometry,
            "dimensions": size,
            "format": "png",
            "crs": "EPSG:3857",
        }
    )
    try:
        response = requests.get(url, timeout=120)
    except requests.RequestException as exc:
        raise SatelliteError(f"Earth Engine image download timed out or failed: {exc}") from exc
    if response.status_code != 200:
        raise SatelliteError(
            f"Earth Engine refused the image download (HTTP {response.status_code}). "
            "The area may be too large — try a smaller AOI radius."
        )
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def fetch_scene(
    aoi: AOI,
    target_date: dt.date,
    window_days: int = DEFAULT_SEARCH_WINDOW_DAYS,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    size: int = THUMB_SIZE,
    allow_composite: bool = True,
) -> Scene:
    """Retrieve the best (least cloudy, closest in time) real Sentinel-2 scene."""
    if not _EE_READY:
        raise SatelliteError("Earth Engine is not connected yet. Connect it on the Home page.")
    if target_date > dt.date.today():
        raise SatelliteError("Dates in the future have no satellite imagery.")

    import ee

    geometry = aoi.ee_geometry()
    start = (target_date - dt.timedelta(days=window_days)).isoformat()
    end = min(target_date + dt.timedelta(days=window_days), dt.date.today()).isoformat()

    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )

    try:
        count = int(collection.size().getInfo())
    except Exception as exc:
        raise SatelliteError(f"Earth Engine query failed: {exc}") from exc

    if count == 0:
        raise SatelliteError(
            f"No Sentinel-2 image with cloud cover under {max_cloud:.0f}% was found for this "
            f"area between {start} and {end}. Widen the search window, allow more cloud, "
            "or pick a different date."
        )

    best = ee.Image(collection.sort("CLOUDY_PIXEL_PERCENTAGE").first())
    try:
        props = best.getInfo()["properties"]
    except Exception as exc:
        raise SatelliteError(f"Could not read Sentinel-2 scene metadata: {exc}") from exc

    cloud = float(props.get("CLOUDY_PIXEL_PERCENTAGE", 0.0))
    millis = props.get("system:time_start")
    acq = dt.datetime.utcfromtimestamp(millis / 1000).date().isoformat() if millis else start
    scene_id = props.get("PRODUCT_ID") or props.get("system:index", "unknown")

    use_composite = allow_composite and cloud > max(5.0, max_cloud * 0.5) and count > 1
    if use_composite:
        source = collection.median()
        composite_note = True
        n_images = count
    else:
        source = best
        composite_note = False
        n_images = 1

    visual = source.select(RGB_BANDS).visualize(min=0, max=3000, gamma=1.15).clip(geometry)
    rgb = _fetch_thumb(visual, geometry, size)

    if rgb.size == 0 or rgb.std() < 1e-3:
        raise SatelliteError(
            "Earth Engine returned an empty image for this AOI. The area may be outside "
            "Sentinel-2 coverage for these dates."
        )

    return Scene(
        rgb=rgb,
        acquisition_date=acq,
        cloud_cover=cloud,
        scene_id=str(scene_id),
        composite=composite_note,
        n_images=n_images,
        meta={
            "collection": S2_COLLECTION,
            "search_start": start,
            "search_end": end,
            "candidates": count,
            "max_cloud_filter": max_cloud,
        },
    )


def validate_dates(before: dt.date, after: dt.date) -> list[str]:
    """Return a list of human-readable warnings; raises on hard errors."""
    today = dt.date.today()
    if before >= after:
        raise SatelliteError("The 'before' date must be earlier than the 'after' date.")
    if after > today or before > today:
        raise SatelliteError("Dates cannot be in the future.")
    warnings: list[str] = []
    gap = (after - before).days
    if gap < 60:
        warnings.append(
            f"The two dates are only {gap} days apart. Real land-cover change is usually "
            "hard to see over such a short period."
        )
    if before < dt.date(2017, 3, 28):
        warnings.append(
            "Sentinel-2 Level-2A surface reflectance starts around March 2017. Earlier "
            "dates will usually return nothing."
        )
    return warnings
