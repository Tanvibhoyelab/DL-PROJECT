"""Satellite change detection for any location in India.

Workflow: Location -> Dates -> Cloud threshold -> Run detection ->
Before/After -> Change overlay -> Metrics -> Downloads.

Everything shown comes from real Sentinel-2 imagery fetched through Google
Earth Engine and from a real PyTorch model. Nothing is simulated: when data or
a trained checkpoint is missing, the app says so instead of showing a number.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import india_locations as loc  # noqa: E402
from src import satellite as sat  # noqa: E402
from src.inference import (  # noqa: E402
    METHOD_DIFFERENCE,
    METHOD_MODEL,
    ModelNotAvailable,
    checkpoint_exists,
    load_model,
    run_change_detection,
)
from src.report import build_pdf  # noqa: E402
from src.utils import RESOLUTION_NOTE, human_summary  # noqa: E402

st.set_page_config(
    page_title="India Satellite Change Detection",
    page_icon="🛰️",
    layout="wide",
)

INDIA_CENTER = (22.5937, 78.9629)

STATE = st.session_state
STATE.setdefault("lat", None)
STATE.setdefault("lon", None)
STATE.setdefault("place", "")
STATE.setdefault("result", None)
STATE.setdefault("scenes", None)
STATE.setdefault("gee_project", config.GEE_PROJECT_ID)
STATE.setdefault("gee_error", "")
STATE.setdefault("city_options", {})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def png_bytes(array: np.ndarray) -> bytes:
    arr = array
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def set_point(lat: float, lon: float, label: str) -> None:
    STATE.lat = float(lat)
    STATE.lon = float(lon)
    STATE.place = label
    STATE.result = None
    STATE.scenes = None


def connect_gee(project: str) -> bool:
    try:
        sat.init_earth_engine(project)
        STATE.gee_error = ""
        return True
    except sat.SatelliteError as exc:
        STATE.gee_error = str(exc)
        return False


def test_metrics() -> dict | None:
    if config.TEST_METRICS_JSON.is_file():
        try:
            return json.loads(config.TEST_METRICS_JSON.read_text())
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")

    project = st.text_input(
        "Earth Engine project id",
        value=STATE.gee_project,
        help="A Google Cloud project registered for Earth Engine. Run "
        "`earthengine authenticate` once on this machine first.",
    )
    STATE.gee_project = project
    if st.button("Connect to Earth Engine", use_container_width=True):
        with st.spinner("Contacting Earth Engine..."):
            connect_gee(project)

    if sat.is_ready():
        st.success("Earth Engine connected")
    elif STATE.gee_error:
        st.error(STATE.gee_error)
    else:
        st.info("Not connected yet. Satellite imagery needs this connection.")

    st.divider()
    st.header("Detection settings")

    method_label = st.radio(
        "Detector",
        ["Trained Siamese U-Net", "Pixel difference (no model)"],
        help="The neural model is the real detector. The pixel-difference option is a "
        "simple baseline, useful to sanity-check imagery — it is not a trained model.",
    )
    method = METHOD_MODEL if method_label.startswith("Trained") else METHOD_DIFFERENCE

    model_ready = checkpoint_exists()
    tuned_threshold = float(config.DEFAULT_THRESHOLD)
    loaded = None
    if method == METHOD_MODEL and model_ready:
        try:
            loaded = load_model()
            tuned_threshold = float(loaded.threshold)
        except ModelNotAvailable as exc:
            st.error(str(exc))
            model_ready = False

    threshold = st.slider(
        "Change probability threshold",
        min_value=0.05,
        max_value=0.95,
        value=tuned_threshold,
        step=0.01,
        disabled=method == METHOD_DIFFERENCE,
        help="Defaults to the threshold tuned on the validation split and stored in the "
        "checkpoint.",
    )
    min_pixels = st.number_input(
        "Ignore regions smaller than (pixels)",
        min_value=1,
        max_value=500,
        value=int(config.MIN_REGION_PIXELS),
        step=1,
    )
    max_cloud = st.slider("Maximum cloud cover in the AOI (%)", 0, 80, int(config.DEFAULT_MAX_CLOUD))
    window_days = st.slider(
        "Search window around each date (± days)",
        7,
        120,
        int(config.DEFAULT_SEARCH_WINDOW_DAYS),
        help="Sentinel-2 revisits every ~5 days, but a clear scene on the exact date is "
        "not guaranteed. The app searches this window and reports the real date it used.",
    )
    scene_mode = st.radio(
        "Imagery",
        ["Single clearest scene", "Cloud-masked median composite"],
        help="A composite blends several dates but removes clouds; the single scene keeps "
        "one exact acquisition date.",
    )
    mode = "best" if scene_mode.startswith("Single") else "composite"


st.title("Satellite Change Detection — India")
st.caption(
    "Sentinel-2 Level-2A surface reflectance via Google Earth Engine + a Siamese U-Net "
    "trained on LEVIR-CD."
)

# --------------------------------------------------------------------------
# 1. Location
# --------------------------------------------------------------------------
st.subheader("1. Location")

tab_admin, tab_search, tab_map, tab_coords = st.tabs(
    ["State → District → City", "Search", "Pick on map", "Latitude / longitude"]
)

with tab_admin:
    col1, col2, col3 = st.columns(3)
    state_name = col1.selectbox("State / UT", loc.STATES, index=None, placeholder="Select a state")
    districts = loc.districts_of(state_name) if state_name else []
    district = col2.selectbox(
        "District", districts, index=None, placeholder="Select a district", disabled=not districts
    )

    city_key = f"{state_name}|{district}"
    cities = STATE.city_options.get(city_key, [])
    with col3:
        if state_name and district and not cities:
            if st.button("Load cities & towns", use_container_width=True):
                with st.spinner("Fetching places from OpenStreetMap..."):
                    try:
                        STATE.city_options[city_key] = loc.cities_of(state_name, district)
                        cities = STATE.city_options[city_key]
                    except loc.GeocodingError as exc:
                        st.warning(str(exc))
        city = st.selectbox(
            "City / town (optional)",
            [c["name"] for c in cities],
            index=None,
            placeholder="Whole district" if not cities else "Select a city",
            disabled=not cities,
        )

    if st.button("Use this location", disabled=not (state_name and district)):
        chosen = next((c for c in cities if c["name"] == city), None)
        if chosen:
            set_point(chosen["lat"], chosen["lon"], f"{city}, {district}, {state_name}")
        else:
            with st.spinner("Locating..."):
                try:
                    found = loc.resolve_location(state_name, district, city or "")
                    set_point(found["lat"], found["lon"], found["display_name"])
                except loc.GeocodingError as exc:
                    st.error(str(exc))

with tab_search:
    query = st.text_input("Search any place in India", placeholder="e.g. Hinjawadi Phase 2, Pune")
    if st.button("Search", disabled=not query):
        try:
            STATE.search_results = loc.search(query)
        except loc.GeocodingError as exc:
            STATE.search_results = []
            st.error(str(exc))
    results = STATE.get("search_results", [])
    if results:
        pick = st.radio(
            "Matches",
            list(range(len(results))),
            format_func=lambda i: results[i]["display_name"],
        )
        if st.button("Use selected match"):
            chosen = results[pick]
            set_point(chosen["lat"], chosen["lon"], chosen["display_name"])

with tab_map:
    st.caption("Click anywhere on the map to drop the area of interest.")
    try:
        import folium
        from streamlit_folium import st_folium

        center = (
            (STATE.lat, STATE.lon)
            if STATE.lat is not None
            else INDIA_CENTER
        )
        fmap = folium.Map(location=center, zoom_start=5 if STATE.lat is None else 12, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
        ).add_to(fmap)
        folium.TileLayer("OpenStreetMap", name="Map").add_to(fmap)
        folium.LayerControl().add_to(fmap)
        if STATE.lat is not None:
            folium.Marker([STATE.lat, STATE.lon], tooltip=STATE.place).add_to(fmap)
        clicked = st_folium(fmap, height=420, width=None, key="picker")
        if clicked and clicked.get("last_clicked"):
            lat = clicked["last_clicked"]["lat"]
            lon = clicked["last_clicked"]["lng"]
            if (STATE.lat, STATE.lon) != (lat, lon):
                set_point(lat, lon, loc.reverse(lat, lon))
                st.rerun()
    except ImportError:
        st.warning("Install folium and streamlit-folium to use the map picker.")

with tab_coords:
    col1, col2 = st.columns(2)
    manual_lat = col1.number_input("Latitude", value=float(STATE.lat or 20.5937), format="%.6f")
    manual_lon = col2.number_input("Longitude", value=float(STATE.lon or 78.9629), format="%.6f")
    if st.button("Use these coordinates"):
        set_point(manual_lat, manual_lon, f"{manual_lat:.5f}, {manual_lon:.5f}")

radius_km = st.slider(
    "Area of interest radius (km)",
    min_value=float(config.MIN_AOI_RADIUS_KM),
    max_value=float(config.MAX_AOI_RADIUS_KM),
    value=float(config.DEFAULT_AOI_RADIUS_KM),
    step=0.5,
)

if STATE.lat is None:
    st.info("Pick a location to continue.")
    st.stop()

aoi = sat.make_radius_aoi(STATE.lat, STATE.lon, radius_km * 1000, STATE.place)
col_a, col_b, col_c = st.columns(3)
col_a.metric("Latitude", f"{STATE.lat:.5f}")
col_b.metric("Longitude", f"{STATE.lon:.5f}")
col_c.metric("AOI area", f"{aoi.area_km2():.2f} km²")
st.write(f"**Selected:** {STATE.place}")

# --------------------------------------------------------------------------
# 2. Dates
# --------------------------------------------------------------------------
st.subheader("2. Dates")
today = dt.date.today()
col1, col2 = st.columns(2)
before_date = col1.date_input(
    "Before date",
    value=today - dt.timedelta(days=730),
    min_value=dt.date.fromisoformat(config.S2_START_DATE),
    max_value=today,
)
after_date = col2.date_input(
    "After date",
    value=today - dt.timedelta(days=30),
    min_value=dt.date.fromisoformat(config.S2_START_DATE),
    max_value=today,
)

try:
    for warning in sat.validate_dates(before_date, after_date):
        st.warning(warning)
    dates_ok = True
except sat.SatelliteError as exc:
    st.error(str(exc))
    dates_ok = False

# --------------------------------------------------------------------------
# 3. Run
# --------------------------------------------------------------------------
st.subheader("3. Run detection")
run_disabled = not (dates_ok and sat.is_ready()) or (method == METHOD_MODEL and not model_ready)
if method == METHOD_MODEL and not model_ready:
    st.error(
        "No trained checkpoint found at "
        f"`{config.CHECKPOINT_PATH}`. Train one with:\n\n"
        "```\npython download_data.py\npython -m src.train --data data/LEVIR-CD\n```\n"
        "Or switch the detector to 'Pixel difference' in the sidebar to inspect imagery only."
    )
if not sat.is_ready():
    st.warning("Connect to Earth Engine in the sidebar before running detection.")

if st.button("Run change detection", type="primary", disabled=run_disabled):
    try:
        with st.spinner("Fetching Sentinel-2 imagery..."):
            before_scene, after_scene = sat.fetch_pair(
                aoi, before_date, after_date, window_days, max_cloud, config.THUMB_SIZE, mode
            )
        with st.spinner("Detecting change..."):
            result = run_change_detection(
                before_scene.rgb,
                after_scene.rgb,
                aoi.bounds(),
                threshold=threshold,
                min_pixels=int(min_pixels),
                loaded=loaded,
                method=method,
            )
        STATE.scenes = (before_scene, after_scene)
        STATE.result = result
    except (sat.SatelliteError, ModelNotAvailable, ValueError) as exc:
        STATE.result = None
        st.error(str(exc))

if not STATE.result:
    st.stop()

before_scene, after_scene = STATE.scenes
result = STATE.result
stats = result["statistics"]

# --------------------------------------------------------------------------
# 4. Before / after
# --------------------------------------------------------------------------
st.subheader("4. Before and after")
col1, col2 = st.columns(2)
for column, scene, title in (
    (col1, before_scene, "Before"),
    (col2, after_scene, "After"),
):
    column.image(scene.rgb, use_container_width=True, caption=f"{title} — {scene.acquisition_date}")
    column.markdown(
        f"**Acquired:** {scene.acquisition_date}  \n"
        f"**Cloud cover in AOI:** {scene.cloud_cover:.1f}%  \n"
        f"**Source:** {scene.label}  \n"
        f"**Scene:** `{scene.scene_id}`  \n"
        f"**Grid:** {scene.meta['pixel_grid']} px at ~{scene.resolution_m:g} m/px"
    )
if before_scene.composite:
    st.caption(
        "Composites blend several acquisition dates: "
        f"before {', '.join(before_scene.contributing_dates)} | "
        f"after {', '.join(after_scene.contributing_dates)}"
    )

st.markdown("**Slider comparison**")
alpha = st.slider("Move towards the after image", 0.0, 1.0, 0.5, 0.02)
blended = (
    before_scene.rgb.astype(np.float32) * (1 - alpha) + after_scene.rgb.astype(np.float32) * alpha
).astype(np.uint8)
st.image(blended, use_container_width=True, caption=f"{(1 - alpha) * 100:.0f}% before / {alpha * 100:.0f}% after")

# --------------------------------------------------------------------------
# 5. Change overlay
# --------------------------------------------------------------------------
st.subheader("5. Where the change is")
col1, col2 = st.columns(2)
col1.image(result["mask_image"], use_container_width=True, caption="Change mask (white = changed)")
col2.image(result["overlay"], use_container_width=True, caption="Change overlay on the after image")
st.image(result["difference"], use_container_width=True, caption="Absolute difference (visual aid only)")

if not result["regions"]:
    st.info(
        "No changed region survived the threshold and the minimum-region filter. That is a real "
        "result, not an error: lower the threshold or enlarge the date gap to look for subtler "
        "change."
    )

st.markdown(
    human_summary(
        STATE.place,
        before_scene.acquisition_date,
        after_scene.acquisition_date,
        result["regions"],
        stats,
    )
)

# --------------------------------------------------------------------------
# 6. Metrics
# --------------------------------------------------------------------------
st.subheader("6. Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total area", f"{stats['aoi_area_km2']:.3f} km²")
col2.metric("Changed area", f"{stats['changed_area_km2']:.3f} km²")
col3.metric("Unchanged area", f"{stats['unchanged_area_km2']:.3f} km²")
col4.metric("Change", f"{stats['change_percentage']:.2f}%")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Changed pixels", f"{stats['changed_pixels']:,}")
col2.metric("Unchanged pixels", f"{stats['unchanged_pixels']:,}")
col3.metric("Regions", stats["regions"])
col4.metric("Ground sampling", f"{stats['metres_per_pixel_x']:g} m/px")

if result["regions"]:
    st.dataframe(
        pd.DataFrame([region.as_dict() for region in result["regions"]])[
            ["change_id", "change_type", "area_hectares", "confidence", "centroid_lat", "centroid_lon"]
        ],
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Model quality on labelled data"):
    metrics = test_metrics()
    checkpoint_metrics = (result["model"].get("val_metrics") or {}) if method == METHOD_MODEL else {}
    if method == METHOD_DIFFERENCE:
        st.warning(
            "The pixel-difference baseline is not a trained model, so accuracy, precision, "
            "recall, F1, IoU and Dice cannot be attributed to it here."
        )
    elif metrics:
        st.caption(
            f"Held-out {metrics['split']} split of {metrics['dataset']} — {metrics['pairs']} image "
            f"pairs, decision threshold {metrics['threshold']:.2f}, evaluated {metrics['evaluated_at']}."
        )
        cols = st.columns(4)
        for column, key in zip(cols, ("accuracy", "precision", "recall", "f1")):
            column.metric(key.capitalize(), f"{metrics[key] * 100:.2f}%")
        cols = st.columns(4)
        for column, key in zip(cols, ("iou", "dice", "balanced_accuracy", "specificity")):
            column.metric(key.replace("_", " ").capitalize(), f"{metrics[key] * 100:.2f}%")
        st.caption(
            f"Confusion matrix (pixels): TP {metrics['tp']:,} · FP {metrics['fp']:,} · "
            f"FN {metrics['fn']:,} · TN {metrics['tn']:,}. Only "
            f"{metrics['positive_rate'] * 100:.2f}% of pixels are truly changed, so plain pixel "
            "accuracy looks high for any model — read F1 and IoU instead."
        )
        if config.CONFUSION_MATRIX_PNG.is_file():
            st.image(str(config.CONFUSION_MATRIX_PNG), width=430)
        if config.QUALITATIVE_PNG.is_file():
            st.image(str(config.QUALITATIVE_PNG), caption="Test-set predictions vs ground truth")
    elif checkpoint_metrics:
        st.warning(
            "No test-set evaluation has been run yet. The numbers below are validation metrics "
            "stored in the checkpoint. Run `python -m src.evaluate --data data/LEVIR-CD` for "
            "held-out test metrics."
        )
        st.json({k: round(v, 4) for k, v in checkpoint_metrics.items() if isinstance(v, (int, float))})
    else:
        st.error(
            "No labelled evaluation data has been processed, so accuracy, precision, recall, F1, "
            "IoU and Dice cannot be reliably calculated for this checkpoint. Run:\n\n"
            "```\npython download_data.py\npython -m src.evaluate --data data/LEVIR-CD\n```"
        )
    st.caption(
        "These figures describe the model on LEVIR-CD (0.5 m aerial imagery). They do not "
        "transfer directly to the 10 m Sentinel-2 pairs analysed above — no labelled ground "
        "truth exists for this AOI, so the change percentage on this page cannot be scored."
    )

st.caption(RESOLUTION_NOTE)

# --------------------------------------------------------------------------
# 7. Downloads
# --------------------------------------------------------------------------
st.subheader("7. Downloads")
stamp = f"{STATE.place.split(',')[0].strip().replace(' ', '_') or 'aoi'}_{before_scene.acquisition_date}_{after_scene.acquisition_date}"
files = {
    "Before image": (png_bytes(before_scene.rgb), f"{stamp}_before.png"),
    "After image": (png_bytes(after_scene.rgb), f"{stamp}_after.png"),
    "Change mask": (png_bytes(result["mask_image"]), f"{stamp}_mask.png"),
    "Change overlay": (png_bytes(result["overlay"]), f"{stamp}_overlay.png"),
}
columns = st.columns(len(files))
for column, (label, (data, name)) in zip(columns, files.items()):
    column.download_button(label, data, file_name=name, mime="image/png", use_container_width=True)

col1, col2 = st.columns(2)
payload = {
    "location": STATE.place,
    "latitude": STATE.lat,
    "longitude": STATE.lon,
    "radius_km": radius_km,
    "before": {
        "requested": before_date.isoformat(),
        "acquired": before_scene.acquisition_date,
        "cloud_percent": before_scene.cloud_cover,
        "scene": before_scene.scene_id,
    },
    "after": {
        "requested": after_date.isoformat(),
        "acquired": after_scene.acquisition_date,
        "cloud_percent": after_scene.cloud_cover,
        "scene": after_scene.scene_id,
    },
    "detector": result["model"].get("architecture"),
    "threshold": result["threshold"],
    "statistics": stats,
    "regions": [region.as_dict() for region in result["regions"]],
}
col1.download_button(
    "Analysis JSON",
    json.dumps(payload, indent=2, default=str),
    file_name=f"{stamp}_analysis.json",
    mime="application/json",
    use_container_width=True,
)

if col2.button("Build PDF report", use_container_width=True):
    pdf = build_pdf(
        {
            "location": STATE.place,
            "coordinates": f"{STATE.lat:.5f}, {STATE.lon:.5f}",
            "aoi": f"{radius_km:g} km radius",
            "before_requested": before_date.isoformat(),
            "before_acquired": before_scene.acquisition_date,
            "before_cloud": before_scene.cloud_cover,
            "after_requested": after_date.isoformat(),
            "after_acquired": after_scene.acquisition_date,
            "after_cloud": after_scene.cloud_cover,
            "model": result["model"],
            "threshold": result["threshold"],
            "statistics": stats,
            "regions": result["regions"],
            "summary": human_summary(
                STATE.place,
                before_scene.acquisition_date,
                after_scene.acquisition_date,
                result["regions"],
                stats,
            ),
            "images": {
                "before": before_scene.rgb,
                "after": after_scene.rgb,
                "mask": result["mask_image"],
                "overlay": result["overlay"],
            },
        }
    )
    st.download_button(
        "Download PDF",
        pdf,
        file_name=f"{stamp}_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
