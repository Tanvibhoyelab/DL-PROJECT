"""Real Sentinel-2 retrieval through Google Earth Engine.

There is no synthetic-image fallback anywhere in this module. If real imagery
cannot be retrieved, a SatelliteError is raised with the exact reason so the
dashboard can show it to the user.

Both dates are always rendered from the same rectangle, in the same projection,
at the same pixel dimensions and with the same reflectance stretch, so the
before/after pair is pixel-aligned and visually comparable by construction.
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
    MAX_AOI_RADIUS_KM,
    MIN_AOI_RADIUS_KM,
    NATIVE_RESOLUTION_M,
    RGB_BANDS,
    S2_COLLECTION,
    S2_START_DATE,
    S2_VIS_GAMMA,
    S2_VIS_MAX,
    S2_VIS_MIN,
    THUMB_SIZE,
)

# Scene Classification Layer classes that are cloud, shadow or cirrus.
SCL_CLOUD_CLASSES = [3, 8, 9, 10]


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
            ".env file or type the project id in the sidebar (see README, "
            "'Google Earth Engine setup')."
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
        """Always a rectangle: the before/after images must share one grid."""
        import ee

        min_lon, min_lat, max_lon, max_lat = self.bounds()
        return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], None, False)

    def extent_m(self) -> tuple[float, float]:
        min_lon, min_lat, max_lon, max_lat = self.bounds()
        mid_lat = (min_lat + max_lat) / 2
        width_m = (max_lon - min_lon) * 111_320 * math.cos(math.radians(mid_lat))
        height_m = (max_lat - min_lat) * 110_570
        return abs(width_m), abs(height_m)

    def area_km2(self) -> float:
        width_m, height_m = self.extent_m()
        return width_m * height_m / 1_000_000

    def pixel_dimensions(self, max_size: int = THUMB_SIZE) -> tuple[int, int]:
        """Output size in pixels, never finer than Sentinel-2's 10 m grid."""
        width_m, height_m = self.extent_m()
        longest_m = max(width_m, height_m, 1.0)
        size = min(max_size, max(64, int(round(longest_m / NATIVE_RESOLUTION_M))))
        if width_m >= height_m:
            width = size
            height = max(64, int(round(size * height_m / max(width_m, 1.0))))
        else:
            height = size
            width = max(64, int(round(size * width_m / max(height_m, 1.0))))
        return width, height

    def folium_bounds(self):
        min_lon, min_lat, max_lon, max_lat = self.bounds()
        return [[min_lat, min_lon], [max_lat, max_lon]]


def make_radius_aoi(lat: float, lon: float, radius_m: float, label: str = "") -> AOI:
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise SatelliteError("Latitude or longitude is out of range.")
    if not (MIN_AOI_RADIUS_KM * 1000 <= radius_m <= MAX_AOI_RADIUS_KM * 1000):
        raise SatelliteError(
            f"AOI radius must be between {MIN_AOI_RADIUS_KM:g} km and {MAX_AOI_RADIUS_KM:g} km."
        )
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
    acquisition_date: str           # ISO date of the scene (or of the newest scene in a composite)
    cloud_cover: float              # cloud cover inside the AOI, in percent
    scene_id: str
    composite: bool = False
    n_images: int = 1
    contributing_dates: list[str] = field(default_factory=list)
    resolution_m: float = NATIVE_RESOLUTION_M
    bands: str = "/".join(RGB_BANDS)
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.composite:
            return f"cloud-masked median of {self.n_images} scenes"
        return "single scene"


def _mask_clouds(image):
    """Mask cloud, cloud-shadow and cirrus pixels using the L2A SCL band."""
    import ee

    scl = image.select("SCL")
    mask = ee.Image.constant(1)
    for cls in SCL_CLOUD_CLASSES:
        mask = mask.And(scl.neq(cls))
    return image.updateMask(mask)


def _aoi_cloud_fraction(image, geometry):
    """Fraction of AOI pixels flagged as cloud/shadow/cirrus, as an ee.Number."""
    import ee

    scl = image.select("SCL")
    cloudy = ee.Image.constant(0)
    for cls in SCL_CLOUD_CLASSES:
        cloudy = cloudy.Or(scl.eq(cls))
    return (
        cloudy.rename("cloudy")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=60,
            maxPixels=1e9,
            bestEffort=True,
        )
        .get("cloudy")
    )


def _fetch_thumb(image, geometry, width: int, height: int) -> np.ndarray:
    url = image.getThumbURL(
        {
            "region": geometry,
            "dimensions": f"{width}x{height}",
            "format": "png",
            "crs": "EPSG:3857",
        }
    )
    try:
        response = requests.get(url, timeout=180)
    except requests.RequestException as exc:
        raise SatelliteError(f"Earth Engine image download timed out or failed: {exc}") from exc
    if response.status_code != 200:
        raise SatelliteError(
            f"Earth Engine refused the image download (HTTP {response.status_code}). "
            "The area may be too large — try a smaller AOI radius."
        )
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _visualize(image, geometry):
    return (
        image.select(RGB_BANDS)
        .visualize(min=S2_VIS_MIN, max=S2_VIS_MAX, gamma=S2_VIS_GAMMA)
        .clip(geometry)
    )


def fetch_scene(
    aoi: AOI,
    target_date: dt.date,
    window_days: int = DEFAULT_SEARCH_WINDOW_DAYS,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    size: int = THUMB_SIZE,
    mode: str = "best",
) -> Scene:
    """Retrieve real Sentinel-2 imagery for the AOI around ``target_date``.

    mode="best"      -> the single scene with the least cloud over the AOI.
    mode="composite" -> cloud-masked median of every qualifying scene in the
                        search window (cleaner, but blends several dates).
    """
    if not _EE_READY:
        raise SatelliteError("Earth Engine is not connected yet. Connect it in the sidebar.")
    if target_date > dt.date.today():
        raise SatelliteError("Dates in the future have no satellite imagery.")

    import ee

    geometry = aoi.ee_geometry()
    width, height = aoi.pixel_dimensions(size)
    start = max(
        target_date - dt.timedelta(days=window_days),
        dt.date.fromisoformat(S2_START_DATE),
    ).isoformat()
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
            f"No Sentinel-2 scene with tile cloud cover under {max_cloud:.0f}% was found for "
            f"this area between {start} and {end}. Widen the search window, allow more cloud, "
            "or pick a different date."
        )

    # Rank candidates by cloud cover measured inside the AOI itself, not over
    # the whole 110 km tile: a tile can be 40% cloudy while the AOI is clear.
    scored = collection.map(
        lambda img: img.set("aoi_cloud", _aoi_cloud_fraction(img, geometry))
    ).sort("aoi_cloud")

    try:
        candidates = scored.limit(40).getInfo()["features"]
    except Exception as exc:
        raise SatelliteError(f"Could not read Sentinel-2 scene metadata: {exc}") from exc

    def scene_date(props: dict) -> str:
        millis = props.get("system:time_start")
        if not millis:
            return start
        return dt.datetime.utcfromtimestamp(millis / 1000).date().isoformat()

    usable = [
        feature
        for feature in candidates
        if feature["properties"].get("aoi_cloud") is not None
        and float(feature["properties"]["aoi_cloud"]) * 100 <= max_cloud
    ]
    if not usable:
        best_available = min(
            (float(f["properties"].get("aoi_cloud") or 1.0) for f in candidates),
            default=1.0,
        )
        raise SatelliteError(
            f"Every Sentinel-2 scene between {start} and {end} is cloudy over this exact area "
            f"(best is {best_available * 100:.0f}% cloud inside the AOI, limit is "
            f"{max_cloud:.0f}%). Raise the cloud threshold, widen the search window, or pick "
            "another date."
        )

    if mode == "composite":
        ids = [feature["id"] for feature in usable]
        selected = ee.ImageCollection(
            [_mask_clouds(ee.Image(image_id)) for image_id in ids]
        )
        source = selected.median()
        dates = sorted({scene_date(feature["properties"]) for feature in usable})
        cloud = float(
            np.mean([float(f["properties"]["aoi_cloud"]) for f in usable]) * 100
        )
        acquisition_date = dates[-1]
        scene_id = f"median composite of {len(ids)} scenes"
        composite = True
        n_images = len(ids)
        contributing = dates
    else:
        # Closest to the requested date among the least-cloudy scenes, so the
        # image both matches the user's date and is actually usable.
        cloud_floor = float(usable[0]["properties"]["aoi_cloud"])
        near_clearest = [
            feature
            for feature in usable
            if float(feature["properties"]["aoi_cloud"]) <= cloud_floor + 0.05
        ]
        best = min(
            near_clearest,
            key=lambda f: abs(
                dt.date.fromisoformat(scene_date(f["properties"])) - target_date
            ),
        )
        props = best["properties"]
        source = ee.Image(best["id"])
        cloud = float(props["aoi_cloud"]) * 100
        acquisition_date = scene_date(props)
        scene_id = str(props.get("PRODUCT_ID") or props.get("system:index", "unknown"))
        composite = False
        n_images = 1
        contributing = [acquisition_date]

    rgb = _fetch_thumb(_visualize(source, geometry), geometry, width, height)

    if rgb.size == 0 or float(rgb.std()) < 1e-3:
        raise SatelliteError(
            "Earth Engine returned an empty image for this AOI. The area may be outside "
            "Sentinel-2 coverage for these dates."
        )

    width_m, height_m = aoi.extent_m()
    return Scene(
        rgb=rgb,
        acquisition_date=acquisition_date,
        cloud_cover=cloud,
        scene_id=scene_id,
        composite=composite,
        n_images=n_images,
        contributing_dates=contributing,
        resolution_m=round(width_m / max(rgb.shape[1], 1), 2),
        meta={
            "collection": S2_COLLECTION,
            "search_start": start,
            "search_end": end,
            "candidates": count,
            "max_cloud_filter": max_cloud,
            "requested_date": target_date.isoformat(),
            "pixel_grid": f"{rgb.shape[1]}x{rgb.shape[0]}",
        },
    )


def fetch_pair(
    aoi: AOI,
    before_date: dt.date,
    after_date: dt.date,
    window_days: int = DEFAULT_SEARCH_WINDOW_DAYS,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    size: int = THUMB_SIZE,
    mode: str = "best",
) -> tuple[Scene, Scene]:
    """Fetch both dates on one identical pixel grid."""
    before = fetch_scene(aoi, before_date, window_days, max_cloud, size, mode)
    after = fetch_scene(aoi, after_date, window_days, max_cloud, size, mode)
    if before.rgb.shape != after.rgb.shape:
        raise SatelliteError(
            "Earth Engine returned two differently sized images for the same area "
            f"({before.rgb.shape} vs {after.rgb.shape}). Try again or change the AOI."
        )
    return before, after


def validate_dates(before: dt.date, after: dt.date) -> list[str]:
    """Return a list of human-readable warnings; raises on hard errors."""
    today = dt.date.today()
    if before >= after:
        raise SatelliteError("The 'before' date must be earlier than the 'after' date.")
    if after > today or before > today:
        raise SatelliteError("Dates cannot be in the future.")
    if before < dt.date.fromisoformat(S2_START_DATE):
        raise SatelliteError(
            f"Sentinel-2 Level-2A surface reflectance starts on {S2_START_DATE}. "
            "Choose a later 'before' date."
        )
    warnings: list[str] = []
    gap = (after - before).days
    if gap < 60:
        warnings.append(
            f"The two dates are only {gap} days apart. Real land-cover change is usually "
            "hard to see over such a short period."
        )
    if abs(before.month - after.month) > 1 and gap < 330:
        warnings.append(
            "The two dates fall in different seasons. Seasonal vegetation differences can "
            "look like change — comparing the same month in different years is cleaner."
        )
    return warnings
