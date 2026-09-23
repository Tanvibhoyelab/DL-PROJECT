"""AI-Based Satellite Image Change Detection — Streamlit dashboard.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import calendar
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
from src.inference import ModelNotAvailable, checkpoint_exists, load_model, run_change_detection  # noqa: E402
from src.report import build_pdf  # noqa: E402
from src.utils import RESOLUTION_NOTE, human_summary  # noqa: E402

st.set_page_config(page_title="Satellite Change Detection — India", page_icon="🛰️", layout="wide")

PAGES = [
    "1. Home",
    "2. Location Selection",
    "3. Satellite Data",
    "4. Change Detection",
    "5. Results",
    "6. Visualization",
    "7. Model Architecture",
    "8. Dataset",
    "9. Download",
]

DEFAULTS = {
    "gee_ready": False,
    "gee_error": "",
    "gee_project": config.GEE_PROJECT_ID,
    "location": None,
    "aoi": None,
    "before_scene": None,
    "after_scene": None,
    "before_date": dt.date(2024, 1, 1),
    "after_date": dt.date(2025, 1, 1),
    "result": None,
    "fetch_error": "",
    "detect_error": "",
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def png_bytes(array: np.ndarray) -> bytes:
    arr = array
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def model_status() -> tuple[str, str]:
    """Returns (badge, detail)."""
    if not checkpoint_exists():
        return ("🔴 TRAINED MODEL NOT AVAILABLE",
                f"No checkpoint at {config.CHECKPOINT_PATH}. Train it first — see the Dataset page.")
    try:
        m = load_model()
    except ModelNotAvailable as exc:
        return "🔴 TRAINED MODEL NOT AVAILABLE", str(exc)
    return ("🟢 REAL DEEP LEARNING MODEL",
            f"Siamese U-Net · {m.parameters:,} parameters · {m.trained_epochs} epochs · {m.device}")


def data_status() -> tuple[str, str]:
    if st.session_state.gee_ready:
        return "🟢 REAL SATELLITE DATA", "Google Earth Engine connected (Sentinel-2 Level-2A)."
    return "🔴 SATELLITE DATA NOT CONNECTED", st.session_state.gee_error or "Earth Engine is not connected yet."


def sidebar():
    st.sidebar.title("🛰️ Change Detection")
    page = st.sidebar.radio("Pages", PAGES, label_format=None) if False else st.sidebar.radio("Pages", PAGES)
    st.sidebar.markdown("---")
    d_badge, d_detail = data_status()
    m_badge, m_detail = model_status()
    st.sidebar.markdown(f"**{d_badge}**")
    st.sidebar.caption(d_detail)
    st.sidebar.markdown(f"**{m_badge}**")
    st.sidebar.caption(m_detail)
    st.sidebar.markdown("---")
    if st.session_state.location:
        l = st.session_state.location
        st.sidebar.caption(f"📍 {l['display_name'][:70]}")
        st.sidebar.caption(f"{l['lat']:.4f}, {l['lon']:.4f}")
    if st.session_state.aoi:
        st.sidebar.caption(f"AOI: {st.session_state.aoi.area_km2():.3f} km²")
    return page


def require(condition: bool, message: str) -> bool:
    if not condition:
        st.warning(message)
    return condition


def build_map(center, zoom=14, aoi=None, regions=None):
    import folium

    m = folium.Map(location=center, zoom_start=zoom, control_scale=True, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite basemap", max_zoom=19,
    ).add_to(m)
    folium.Marker(center, tooltip="Selected location",
                  icon=folium.Icon(color="blue", icon="map-pin", prefix="fa")).add_to(m)

    fit_bounds = None
    if aoi is not None:
        aoi_bounds = aoi.folium_bounds()
        folium.Rectangle(aoi_bounds, color="#0077ff", weight=2,
                         fill=True, fill_opacity=0.08, tooltip="Analysis area (AOI)").add_to(m)
        fit_bounds = list(aoi_bounds)

    for r in regions or []:
        min_lat, min_lon, max_lat, max_lon = r.bbox_latlon
        color = "#e11d48" if r.confidence >= 0.75 else "#f59e0b"
        popup = folium.Popup(
            f"<b>Change #{r.change_id}</b><br>{r.change_type}<br>"
            f"Confidence: {r.confidence * 100:.0f}%<br>"
            f"Area: {r.area_hectares:.4f} ha<br>"
            f"{r.centroid_lat:.5f}, {r.centroid_lon:.5f}",
            max_width=260,
        )
        folium.Rectangle([[min_lat, min_lon], [max_lat, max_lon]], color=color,
                         weight=2, fill=True, fill_opacity=0.25, popup=popup).add_to(m)
        region_box = [[min_lat, min_lon], [max_lat, max_lon]]
        fit_bounds = region_box if fit_bounds is None else [
            [min(fit_bounds[0][0], region_box[0][0]), min(fit_bounds[0][1], region_box[0][1])],
            [max(fit_bounds[1][0], region_box[1][0]), max(fit_bounds[1][1], region_box[1][1])],
        ]

    folium.LayerControl(collapsed=True).add_to(m)

    # Auto-fit/zoom so the whole AOI (and any detected regions) is fully
    # visible, instead of relying on a fixed zoom level that can crop a
    # large or oddly-shaped AOI. A little padding keeps the AOI edge from
    # touching the map border exactly.
    if fit_bounds is not None:
        m.fit_bounds(fit_bounds, padding=(30, 30))

    return m


def show_map(m, height=520):
    from streamlit_folium import st_folium

    return st_folium(m, height=height, width=None, returned_objects=["last_active_drawing"])


def regions_dataframe() -> pd.DataFrame:
    res = st.session_state.result
    l = st.session_state.location
    rows = []
    for r in res["regions"]:
        rows.append({
            "location": l["display_name"],
            "latitude": l["lat"],
            "longitude": l["lon"],
            "before_date": st.session_state.before_scene.acquisition_date,
            "after_date": st.session_state.after_scene.acquisition_date,
            "change_id": r.change_id,
            "change_type": r.change_type,
            "area_hectares": r.area_hectares,
            "area_m2": r.area_m2,
            "confidence": r.confidence,
            "centroid_lat": r.centroid_lat,
            "centroid_lon": r.centroid_lon,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home():
    st.title("AI-Based Satellite Image Change Detection")
    st.subheader("Real-Time Geospatial Change Analysis using Sentinel-2 and Deep Learning")

    c1, c2 = st.columns(2)
    with c1:
        badge, detail = data_status()
        st.markdown(f"### {badge}")
        st.caption(detail)
    with c2:
        badge, detail = model_status()
        st.markdown(f"### {badge}")
        st.caption(detail)

    st.markdown("---")
    a, b, c = st.columns(3)
    a.markdown("**Input**\n\nTwo real Sentinel-2 satellite images of the same place, taken on two different dates.")
    b.markdown("**Processing**\n\nA Siamese CNN encoder reads both images with shared weights; a U-Net decoder turns the feature difference into a pixel-wise change probability.")
    c.markdown("**Output**\n\nA change mask, a list of meaningful changed regions with area and confidence, statistics, a map and a PDF report.")

    st.markdown("---")
    st.markdown("### Connect Google Earth Engine")
    st.caption("Real Sentinel-2 imagery is downloaded through Earth Engine. Nothing synthetic is ever generated.")
    project = st.text_input("Earth Engine project id", value=st.session_state.gee_project,
                            placeholder="my-ee-project", help="Set GEE_PROJECT_ID in .env to prefill this.")
    if st.button("Connect to Earth Engine", type="primary"):
        try:
            used = sat.init_earth_engine(project.strip())
            st.session_state.gee_ready = True
            st.session_state.gee_project = used
            st.session_state.gee_error = ""
            st.success(f"Connected to Earth Engine (project: {used}).")
        except sat.SatelliteError as exc:
            st.session_state.gee_ready = False
            st.session_state.gee_error = str(exc)
            st.error(str(exc))

    st.markdown("---")
    st.info(RESOLUTION_NOTE)
    st.markdown("**Workflow:** Location → AOI → dates → fetch real Sentinel-2 → detect changes → "
                "results, map, downloads and PDF report.")


def page_location():
    st.title("Location Selection")
    st.caption("Pick anywhere in India. The list is a shortcut — free-text search and manual coordinates cover the rest.")

    mode = st.radio("How do you want to choose the location?",
                    ["State → District → Area", "Search any place in India", "Enter latitude / longitude"],
                    horizontal=True)

    resolved = None
    if mode == "State → District → Area":
        c1, c2, c3 = st.columns(3)
        state = c1.selectbox("State / UT", loc.STATES, index=loc.STATES.index("Maharashtra"))
        districts = loc.districts_of(state)
        district = c2.selectbox("District / City", districts)
        areas = ["(whole district)"] + loc.areas_of(state, district)
        area = c3.selectbox("Area / Locality", areas)
        extra = st.text_input("Extra detail (optional)", placeholder="e.g. Bandra Kurla Complex")
        if st.button("Locate", type="primary"):
            with st.spinner("Geocoding…"):
                try:
                    resolved = loc.resolve_location(state, district, "" if area.startswith("(") else area, extra)
                except loc.GeocodingError as exc:
                    st.error(str(exc))

    elif mode == "Search any place in India":
        query = st.text_input("Search", placeholder="e.g. Hinjewadi Phase 2, Pune")
        if st.button("Search", type="primary") and query.strip():
            with st.spinner("Geocoding…"):
                try:
                    lat, lon, display = loc.geocode(query.strip() + ", India")
                    resolved = {"lat": lat, "lon": lon, "display_name": display, "query": query}
                except loc.GeocodingError as exc:
                    st.error(str(exc))

    else:
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=19.0596, format="%.6f", min_value=-90.0, max_value=90.0)
        lon = c2.number_input("Longitude", value=72.8295, format="%.6f", min_value=-180.0, max_value=180.0)
        label = st.text_input("Label (optional)", placeholder="Custom point")
        if st.button("Use these coordinates", type="primary"):
            resolved = {"lat": float(lat), "lon": float(lon),
                        "display_name": label or f"Manual point ({lat:.4f}, {lon:.4f})", "query": "manual"}

    if resolved:
        st.session_state.location = resolved
        st.session_state.aoi = None
        st.session_state.before_scene = None
        st.session_state.after_scene = None
        st.session_state.result = None
        st.success("Location set.")

    if not st.session_state.location:
        st.info("No location selected yet.")
        return

    l = st.session_state.location
    st.markdown("### Selected location")
    c1, c2, c3 = st.columns(3)
    c1.metric("Latitude", f"{l['lat']:.4f}")
    c2.metric("Longitude", f"{l['lon']:.4f}")
    c3.metric("Zoom", "14 (local)")
    st.write(f"**{l['display_name']}**")

    st.markdown("### Area of Interest (AOI)")
    aoi_mode = st.radio("AOI type", ["Radius around the point", "Draw a rectangle or polygon"], horizontal=True)

    if aoi_mode == "Radius around the point":
        radius = st.select_slider("Radius", options=[250, 500, 1000, 2000, 5000],
                                  value=1000, format_func=lambda v: f"{v} m" if v < 1000 else f"{v / 1000:g} km")
        try:
            st.session_state.aoi = sat.make_radius_aoi(l["lat"], l["lon"], float(radius), l["display_name"])
        except sat.SatelliteError as exc:
            st.error(str(exc))
    else:
        st.caption("Use the draw tools on the map, then press the button below the map.")

    aoi = st.session_state.aoi
    m = build_map([l["lat"], l["lon"]], zoom=14, aoi=aoi)
    if aoi_mode.startswith("Draw"):
        from folium.plugins import Draw

        Draw(export=False, draw_options={"polyline": False, "circle": False,
                                         "circlemarker": False, "marker": False}).add_to(m)
    state_map = show_map(m)

    if aoi_mode.startswith("Draw"):
        drawing = (state_map or {}).get("last_active_drawing")
        if drawing and st.button("Use the drawn shape as AOI", type="primary"):
            try:
                coords = drawing["geometry"]["coordinates"][0]
                st.session_state.aoi = sat.make_polygon_aoi([[c[0], c[1]] for c in coords], l["display_name"])
                st.success("AOI set from the drawn shape.")
                st.rerun()
            except (KeyError, IndexError, TypeError, sat.SatelliteError) as exc:
                st.error(f"Could not read the drawn shape: {exc}")

    if st.session_state.aoi:
        st.success(f"AOI area: **{st.session_state.aoi.area_km2():.3f} km²** "
                   f"({st.session_state.aoi.kind})")


_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def date_picker(label: str, key: str, default: dt.date,
                 min_date: dt.date, max_date: dt.date) -> dt.date:
    """Year / Month / Day dropdown picker.

    Streamlit's built-in st.date_input renders a calendar popover whose
    month/year dropdown can visually overlap the day grid, especially
    inside a half-width column — that overlap is what was blocking date
    selection on this page. Three plain selectboxes avoid that popover
    entirely, so there is nothing that can overlap.
    """

    st.caption(label)

    years = list(range(min_date.year, max_date.year + 1))
    yc, mc, dc = st.columns(3)

    year = yc.selectbox(
        "Year", years,
        index=years.index(min(max(default.year, min_date.year), max_date.year)),
        key=f"{key}_year",
    )

    # Restrict month choices to what's valid for the chosen year given
    # the overall min/max bounds.
    month_start = 1 if year > min_date.year else min_date.month
    month_end = 12 if year < max_date.year else max_date.month
    months = list(range(month_start, month_end + 1))
    default_month = min(max(default.month, month_start), month_end)

    month = mc.selectbox(
        "Month", months,
        index=months.index(default_month),
        format_func=lambda m: _MONTH_NAMES[m - 1],
        key=f"{key}_month",
    )

    # Clip day range to the real number of days in the chosen month/year,
    # and to the overall min/max bounds.
    days_in_month = calendar.monthrange(year, month)[1]
    day_start = min_date.day if (year == min_date.year and month == min_date.month) else 1
    day_end = max_date.day if (year == max_date.year and month == max_date.month) else days_in_month
    days = list(range(day_start, day_end + 1))
    default_day = min(max(default.day, day_start), day_end)

    day = dc.selectbox(
        "Day", days,
        index=days.index(default_day),
        key=f"{key}_day",
    )

    return dt.date(year, month, day)


def page_satellite():
    st.title("Satellite Data")
    if not require(st.session_state.location and st.session_state.aoi,
                   "Select a location and an AOI on the Location Selection page first."):
        return
    if not require(st.session_state.gee_ready,
                   "Earth Engine is not connected. Connect it on the Home page — no synthetic imagery is used."):
        if st.session_state.gee_error:
            st.error(st.session_state.gee_error)
        return

    c1, c2 = st.columns(2)
    with c1:
        before_date = date_picker("Before date", "before_date_picker",
                                   st.session_state.before_date,
                                   dt.date(2017, 3, 28), dt.date.today())
    with c2:
        after_date = date_picker("After date", "after_date_picker",
                                  st.session_state.after_date,
                                  dt.date(2017, 3, 28), dt.date.today())
    c3, c4, c5 = st.columns(3)
    max_cloud = c3.slider("Maximum cloud cover (%)", 0, 80, config.DEFAULT_MAX_CLOUD, 5)
    window = c4.slider("Search window (± days)", 5, 120, config.DEFAULT_SEARCH_WINDOW_DAYS, 5)
    size = c5.select_slider("Image size (px)", options=[256, 384, 512, 768], value=config.THUMB_SIZE)

    st.session_state.before_date, st.session_state.after_date = before_date, after_date

    try:
        warnings = sat.validate_dates(before_date, after_date)
    except sat.SatelliteError as exc:
        st.error(str(exc))
        return
    for w in warnings:
        st.warning(w)

    if st.button("Fetch Satellite Images", type="primary"):
        st.session_state.result = None
        with st.spinner("Searching Sentinel-2 archive and downloading real imagery…"):
            try:
                before = sat.fetch_scene(st.session_state.aoi, before_date, window, max_cloud, size)
                after = sat.fetch_scene(st.session_state.aoi, after_date, window, max_cloud, size)
                st.session_state.before_scene, st.session_state.after_scene = before, after
                st.session_state.fetch_error = ""
            except sat.SatelliteError as exc:
                st.session_state.before_scene = st.session_state.after_scene = None
                st.session_state.fetch_error = str(exc)

    if st.session_state.fetch_error:
        st.error(st.session_state.fetch_error)
        st.caption("No fake or synthetic image is shown in place of missing satellite data.")

    b, a = st.session_state.before_scene, st.session_state.after_scene
    if not (b and a):
        return

    st.success("Real Sentinel-2 imagery retrieved.")
    c1, c2 = st.columns(2)
    for col, scene, label in ((c1, b, "Before"), (c2, a, "After")):
        with col:
            st.markdown(f"#### {label} image")
            st.image(scene.rgb, use_container_width=True)
            st.markdown(
                f"- **Satellite:** Sentinel-2 (Level-2A)\n"
                f"- **Acquisition date:** {dt.date.fromisoformat(scene.acquisition_date):%d-%m-%Y}\n"
                f"- **Cloud cover:** {scene.cloud_cover:.1f}%\n"
                f"- **Resolution:** {scene.resolution_m} m\n"
                f"- **Bands:** {scene.bands}\n"
                f"- **Scene:** `{scene.scene_id[:44]}`\n"
                f"- **Composite:** {'median of ' + str(scene.n_images) + ' scenes' if scene.composite else 'single scene'}"
            )
    st.info(RESOLUTION_NOTE)


def page_detection():
    st.title("Change Detection")
    if not require(st.session_state.before_scene and st.session_state.after_scene,
                   "Fetch the before/after satellite images first (Satellite Data page)."):
        return

    badge, detail = model_status()
    st.markdown(f"### {badge}")
    st.caption(detail)
    if badge.startswith("🔴"):
        st.error("A trained checkpoint is required. Pixel differencing is **not** used as a stand-in "
                 "for a deep-learning prediction.")
        st.code("python -m src.train --data data/LEVIR-CD --epochs 40", language="bash")
        return

    c1, c2 = st.columns(2)
    threshold = c1.slider(
        "Change probability threshold",
        min_value=0.01,
        max_value=0.95,
        value=float(config.DEFAULT_THRESHOLD),
        step=0.01,
        help="Threshold applied to the trained Siamese U-Net probability map.",
    )
    min_px = c2.slider(
        "Minimum region size (pixels)",
        min_value=1,
        max_value=400,
        value=int(config.MIN_REGION_PIXELS),
        step=1,
    )

    if st.button("Detect Changes", type="primary"):
        with st.spinner("Running the Siamese U-Net…"):
            try:
                st.session_state.result = run_change_detection(
                    st.session_state.before_scene.rgb,
                    st.session_state.after_scene.rgb,
                    st.session_state.aoi.bounds(),
                    threshold=threshold,
                    min_pixels=min_px,
                )
                st.session_state.detect_error = ""
            except (ModelNotAvailable, Exception) as exc:  # noqa: BLE001
                st.session_state.result = None
                st.session_state.detect_error = f"{type(exc).__name__}: {exc}"

    if st.session_state.detect_error:
        st.error(st.session_state.detect_error)
        return

    res = st.session_state.result
    if not res:
        return

    st.success(f"{len(res['regions'])} meaningful changed region(s) detected.")

    # Diagnostics show the actual output of the trained model before/after cleanup.
    d = res.get("diagnostics", {})
    st.markdown("### Model prediction diagnostics")
    dc = st.columns(5)
    dc[0].metric("Min probability", f"{d.get('min_probability', 0.0) * 100:.2f}%")
    dc[1].metric("Max probability", f"{d.get('max_probability', 0.0) * 100:.2f}%")
    dc[2].metric("Average probability", f"{d.get('mean_probability', 0.0) * 100:.2f}%")
    dc[3].metric("Raw changed pixels", f"{d.get('raw_changed_pixels', 0):,}")
    dc[4].metric("Final changed pixels", f"{d.get('cleaned_changed_pixels', 0):,}")

    max_probability = float(d.get("max_probability", 0.0))
    if max_probability < threshold:
        st.warning(
            f"The model maximum probability is {max_probability * 100:.2f}%, "
            f"below the current threshold of {threshold * 100:.2f}%. "
            "Therefore the binary change mask is empty at this threshold."
        )
    elif d.get("raw_changed_pixels", 0) > 0 and d.get("cleaned_changed_pixels", 0) == 0:
        st.warning(
            "The model predicted changed pixels, but the minimum-region cleanup "
            "removed all of them. Reduce the minimum region size."
        )

    cols = st.columns(5)
    cols[0].image(st.session_state.before_scene.rgb, caption="Before", use_container_width=True)
    cols[1].image(st.session_state.after_scene.rgb, caption="After", use_container_width=True)
    cols[2].image(res["raw_mask_image"], caption="Raw model mask", use_container_width=True)
    cols[3].image(res["mask_image"], caption="Final change mask", use_container_width=True)
    cols[4].image(res["overlay"], caption="Change overlay", use_container_width=True)
    st.image(res["difference"], caption="Raw difference image (reference only, not the model output)", width=380)

    st.markdown("**Legend** — 🟥 red box: high-confidence change (≥75%) · 🟨 amber box: lower confidence · "
                "orange fill: predicted changed pixels · everything else: unchanged area.")


def page_results():
    st.title("Results")
    if not require(st.session_state.result, "Run the detection first (Change Detection page)."):
        return

    res = st.session_state.result
    l, aoi = st.session_state.location, st.session_state.aoi
    b, a = st.session_state.before_scene, st.session_state.after_scene
    s = res["statistics"]

    st.markdown("### A. Location summary")
    c = st.columns(3)
    c[0].write(f"**Location**\n\n{l['display_name']}")
    c[1].write(f"**Coordinates**\n\n{l['lat']:.5f}, {l['lon']:.5f}")
    aoi_desc = f"{aoi.radius_m:.0f} m radius" if aoi.kind == "radius" else f"drawn {aoi.kind}"
    c[2].write(f"**AOI**\n\n{aoi_desc} · {aoi.area_km2():.3f} km²")

    st.markdown("### B. Satellite information")
    c = st.columns(4)
    c[0].metric("Satellite", "Sentinel-2")
    c[1].metric("Before acquisition", f"{dt.date.fromisoformat(b.acquisition_date):%d %b %Y}", f"{b.cloud_cover:.1f}% cloud")
    c[2].metric("After acquisition", f"{dt.date.fromisoformat(a.acquisition_date):%d %b %Y}", f"{a.cloud_cover:.1f}% cloud")
    c[3].metric("Ground sampling", f"{s['metres_per_pixel_x']:.1f} m/px")

    st.markdown("### C. Change statistics")
    c = st.columns(4)
    c[0].metric("Changed area", f"{s['changed_area_hectares']:.3f} ha")
    c[1].metric("Change percentage", f"{s['change_percentage']:.2f}%")
    c[2].metric("Detected regions", s["regions"])
    c[3].metric("Average confidence", f"{s['average_confidence'] * 100:.0f}%")

    with st.expander("Advanced model diagnostics (for report / debugging)"):
        d = res.get("diagnostics", {})
        dc = st.columns(5)
        dc[0].metric("Threshold", f"{res.get('threshold', 0.0) * 100:.1f}%")
        dc[1].metric("Max probability", f"{d.get('max_probability', 0.0) * 100:.2f}%")
        dc[2].metric("Average probability", f"{d.get('mean_probability', 0.0) * 100:.2f}%")
        dc[3].metric("Raw changed pixels", f"{d.get('raw_changed_pixels', 0):,}")
        dc[4].metric("Final changed pixels", f"{d.get('cleaned_changed_pixels', 0):,}")
        st.caption(
            "Detector used: pixel-difference fallback (model confidence was below threshold)"
            if d.get("fallback_used") else "Detector used: trained Siamese U-Net model"
        )

    st.markdown("### D. Detected changes")
    if res.get("diagnostics", {}).get("fallback_used") and res["regions"]:
        st.warning(
            "The trained model's confidence was too low to flag any region here, so these "
            "results come from a classical pixel-difference fallback, not the neural network. "
            "Treat them as candidate changes to verify visually, not confirmed model output."
        )
    if res["regions"]:
        df = pd.DataFrame([{
            "ID": r.change_id, "Type": r.change_type,
            "Area (ha)": round(r.area_hectares, 4),
            "Confidence": f"{r.confidence * 100:.0f}%",
            "Latitude": r.centroid_lat, "Longitude": r.centroid_lon,
        } for r in res["regions"]])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No region passed the size and probability thresholds.")

    st.markdown("### E. Visual result")
    cols = st.columns(4)
    cols[0].image(b.rgb, caption="Before", use_container_width=True)
    cols[1].image(a.rgb, caption="After", use_container_width=True)
    cols[2].image(res["mask_image"], caption="Change mask", use_container_width=True)
    cols[3].image(res["overlay"], caption="Overlay", use_container_width=True)

    st.markdown("### F. Plain-language explanation")
    summary = human_summary(
        l["display_name"],
        f"{dt.date.fromisoformat(b.acquisition_date):%d %B %Y}",
        f"{dt.date.fromisoformat(a.acquisition_date):%d %B %Y}",
        res["regions"], s,
    )
    st.success(summary)

    st.markdown("### G. Map of detected changes")
    st.caption("Click a red or amber box to see the change id, type, confidence, area and coordinates.")
    show_map(build_map([l["lat"], l["lon"]], 14, aoi, res["regions"]), height=520)

    st.info(RESOLUTION_NOTE)


def page_visualization():
    st.title("Visualization")
    if not require(st.session_state.result, "Run the detection first."):
        return
    res = st.session_state.result
    before = st.session_state.before_scene.rgb
    after = st.session_state.after_scene.rgb

    st.markdown("### Side-by-side comparison")
    c1, c2 = st.columns(2)
    c1.image(before, caption=f"Before — {st.session_state.before_scene.acquisition_date}", use_container_width=True)
    c2.image(after, caption=f"After — {st.session_state.after_scene.acquisition_date}", use_container_width=True)

    st.markdown("### Blend slider (before ↔ after)")
    alpha = st.slider("Move towards the after image", 0.0, 1.0, 0.5, 0.02)
    blended = (before.astype(np.float32) * (1 - alpha) + after.astype(np.float32) * alpha).astype(np.uint8)
    st.image(blended, use_container_width=True)

    st.markdown("### Change overlay opacity")
    op = st.slider("Overlay opacity", 0.0, 1.0, 0.45, 0.05)
    mask = res["mask"].astype(bool)
    over = after.copy().astype(np.float32)
    over[mask] = over[mask] * (1 - op) + np.array([255, 90, 0], dtype=np.float32) * op
    st.image(over.astype(np.uint8), use_container_width=True)

    st.markdown("### Downloads")
    c = st.columns(4)
    c[0].download_button("Before image", png_bytes(before), "before.png", "image/png")
    c[1].download_button("After image", png_bytes(after), "after.png", "image/png")
    c[2].download_button("Change mask", png_bytes(res["mask_image"]), "change_mask.png", "image/png")
    c[3].download_button("Overlay", png_bytes(res["overlay"]), "overlay.png", "image/png")

    st.markdown("### Statistics from the predicted mask")
    st.json(res["statistics"])


def page_performance():
    st.title("Model Performance")
    st.caption("Every number here is read from files produced by training and evaluation. Nothing is invented.")

    if not checkpoint_exists():
        st.error("Model training has not been completed yet.")
        st.code("python -m src.train --data data/LEVIR-CD --epochs 40\n"
                "python -m src.evaluate --data data/LEVIR-CD", language="bash")
        return

    if config.TEST_METRICS_JSON.exists():
        metrics = json.loads(config.TEST_METRICS_JSON.read_text())
        st.markdown(f"### Test-set results ({metrics.get('pairs', '?')} image pairs)")
        c = st.columns(6)
        for col, key in zip(c, ["accuracy", "precision", "recall", "f1", "iou", "dice"]):
            col.metric(key.upper() if key == "iou" else key.title(), f"{metrics[key]:.4f}")
        st.caption(f"Threshold {metrics.get('threshold')} · checkpoint `{metrics.get('checkpoint')}`")
    else:
        st.warning("Test evaluation has not been run yet. Run: `python -m src.evaluate --data data/LEVIR-CD`")

    if config.CONFUSION_MATRIX_PNG.exists():
        st.markdown("### Confusion matrix")
        st.image(str(config.CONFUSION_MATRIX_PNG), width=420)

    if config.METRICS_CSV.exists():
        st.markdown("### Training history")
        df = pd.read_csv(config.METRICS_CSV)
        st.dataframe(df, use_container_width=True, hide_index=True)
        cols = st.columns(2)
        for col, name, title in (
            (cols[0], "training_loss.png", "Training loss"),
            (cols[1], "validation_loss.png", "Validation loss"),
            (cols[0], "iou_curve.png", "IoU curve"),
            (cols[1], "f1_curve.png", "F1 curve"),
        ):
            path = config.OUTPUTS_DIR / name
            if path.exists():
                col.image(str(path), caption=title, use_container_width=True)
    else:
        st.warning("No training history file yet (outputs/metrics.csv).")


def page_architecture():
    st.title("Model Architecture")
    st.code(
        "Before image                After image\n"
        "     |                            |\n"
        "     v                            v\n"
        "  Siamese Encoder  <shared weights>  Siamese Encoder\n"
        "     |                            |\n"
        "     +--------> Feature difference <--------+\n"
        "                        |\n"
        "                 U-Net Decoder (skip connections)\n"
        "                        |\n"
        "              Change probability map (0-1)\n"
        "                        |\n"
        "                   Thresholding\n"
        "                        |\n"
        "        Morphological cleaning + connected components\n"
        "                        |\n"
        "               Meaningful change regions",
        language="text",
    )
    st.markdown(
        "**Siamese encoder** — one small CNN reads both dates using the *same* weights, so the two "
        "images are described in the same language.\n\n"
        "**Feature difference** — the absolute difference of the two feature stacks. Things that stayed "
        "the same cancel out; things that changed stand out.\n\n"
        "**U-Net decoder** — upsamples the difference back to full resolution, reusing the fine detail "
        "from earlier layers through skip connections.\n\n"
        "**Threshold** — every pixel gets a probability between 0 and 1; above the threshold it counts as changed.\n\n"
        "**Cleaning + regions** — morphological opening/closing removes speckle, then connected components "
        "group the remaining pixels into regions with a bounding box, area, centroid and confidence."
    )
    if checkpoint_exists():
        try:
            m = load_model()
            c = st.columns(4)
            c[0].metric("Parameters", f"{m.parameters:,}")
            c[1].metric("Input size", f"{m.input_size}px")
            c[2].metric("Trained epochs", m.trained_epochs)
            c[3].metric("Device", str(m.device))
        except ModelNotAvailable as exc:
            st.error(str(exc))
    else:
        st.warning("No checkpoint loaded, so no live parameter count is shown.")
    st.markdown("**Loss:** BCE with positive-class weighting + Dice loss (change pixels are rare). "
                "**Optimiser:** AdamW with ReduceLROnPlateau and early stopping.")


def page_dataset():
    st.title("Dataset")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Training data — LEVIR-CD")
        st.markdown(
            "- Public building-change detection benchmark\n"
            "- 637 very-high-resolution image pairs (0.5 m per pixel), usually cut into 256×256 patches\n"
            "- Binary labels: changed building / no change\n"
            "- Used **only** to train and evaluate the model\n"
            "- These are **not** Indian satellite images — they come from Texas, USA"
        )
        st.code("data/LEVIR-CD/\n  train/A  train/B  train/label\n  val/A    val/B    val/label\n"
                "  test/A   test/B   test/label", language="text")
    with c2:
        st.markdown("### Real inference data — Sentinel-2")
        st.markdown(
            "- ESA Copernicus Sentinel-2, Level-2A surface reflectance\n"
            "- Free and open, ~5-day revisit over India\n"
            "- 10 m per pixel for B4/B3/B2 (true colour) and B8 (near infrared)\n"
            "- Retrieved live through Google Earth Engine for the AOI and dates you choose"
        )
        st.warning("The two are deliberately kept separate: the model learns change on LEVIR-CD and is "
                   "then applied to real Sentinel-2 imagery. Because Sentinel-2 is 20× coarser, results "
                   "are reported as area-level change, not individual buildings.")
    st.markdown("### Training commands")
    st.code("python -m src.train --data data/LEVIR-CD --epochs 40 --batch-size 8\n"
            "python -m src.evaluate --data data/LEVIR-CD", language="bash")
    st.info(RESOLUTION_NOTE)


def page_download():
    st.title("Download")
    if not require(st.session_state.result, "Run the detection first."):
        return
    res = st.session_state.result
    b, a = st.session_state.before_scene, st.session_state.after_scene
    l, aoi = st.session_state.location, st.session_state.aoi

    st.markdown("### Images")
    c = st.columns(4)
    c[0].download_button("Before image (PNG)", png_bytes(b.rgb), "before.png", "image/png")
    c[1].download_button("After image (PNG)", png_bytes(a.rgb), "after.png", "image/png")
    c[2].download_button("Change mask (PNG)", png_bytes(res["mask_image"]), "change_mask.png", "image/png")
    c[3].download_button("Overlay (PNG)", png_bytes(res["overlay"]), "overlay.png", "image/png")

    st.markdown("### Data")
    df = regions_dataframe()
    st.dataframe(df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("Detected changes (CSV)", df.to_csv(index=False).encode(),
                       "detected_changes.csv", "text/csv")
    c2.download_button("Statistics (JSON)", json.dumps(res["statistics"], indent=2).encode(),
                       "statistics.json", "application/json")

    st.markdown("### Analysis report")
    if st.button("Generate Analysis Report (PDF)", type="primary"):
        with st.spinner("Building the PDF…"):
            summary = human_summary(
                l["display_name"],
                f"{dt.date.fromisoformat(b.acquisition_date):%d %B %Y}",
                f"{dt.date.fromisoformat(a.acquisition_date):%d %B %Y}",
                res["regions"], res["statistics"],
            )
            aoi_desc = f"{aoi.radius_m:.0f} m radius" if aoi.kind == "radius" else f"drawn {aoi.kind}"
            pdf = build_pdf({
                "location": l["display_name"],
                "coordinates": f"{l['lat']:.5f}, {l['lon']:.5f}",
                "aoi": aoi_desc,
                "before_requested": st.session_state.before_date.strftime("%d %B %Y"),
                "after_requested": st.session_state.after_date.strftime("%d %B %Y"),
                "before_acquired": b.acquisition_date,
                "after_acquired": a.acquisition_date,
                "before_cloud": b.cloud_cover,
                "after_cloud": a.cloud_cover,
                "images": {"before": b.rgb, "after": a.rgb,
                           "mask": res["mask_image"], "overlay": res["overlay"]},
                "statistics": res["statistics"],
                "regions": res["regions"],
                "model": res["model"],
                "threshold": res["threshold"],
                "summary": summary,
            })
            name = f"change_report_{dt.datetime.now():%Y%m%d_%H%M%S}.pdf"
            (config.REPORTS_DIR / name).write_bytes(pdf)
            st.download_button("Download the PDF report", pdf, name, "application/pdf")
            st.success(f"Report also saved to reports/{name}")


# ---------------------------------------------------------------------------
PAGE_FUNCS = {
    PAGES[0]: page_home,
    PAGES[1]: page_location,
    PAGES[2]: page_satellite,
    PAGES[3]: page_detection,
    PAGES[4]: page_results,
    PAGES[5]: page_visualization,
    PAGES[6]: page_architecture,
    PAGES[7]: page_dataset,
    PAGES[8]: page_download,
}


def main():
    page = sidebar()
    try:
        PAGE_FUNCS[page]()
    except Exception as exc:  # never crash the whole dashboard
        st.error(f"Something went wrong on this page: {type(exc).__name__}: {exc}")
        st.caption("Fix the reported problem and try again — the rest of the dashboard is still usable.")


if __name__ == "__main__":
    main()