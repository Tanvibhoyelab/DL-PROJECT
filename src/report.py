"""PDF analysis report generation (ReportLab)."""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.utils import RESOLUTION_NOTE


def _img_flowable(array: np.ndarray, width_cm: float = 7.5) -> RLImage:
    arr = array
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    h, w = arr.shape[:2]
    width = width_cm * cm
    return RLImage(buf, width=width, height=width * h / w)


def build_pdf(context: dict) -> bytes:
    """context keys: location, coordinates, aoi, before/after dates + metadata,
    images, statistics, regions, model, summary."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=17, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=10)
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)

    story = [
        Paragraph("AI-Based Satellite Image Change Detection", h1),
        Paragraph("Real-Time Geospatial Change Analysis using Sentinel-2 and Deep Learning", body),
        Paragraph(f"Report generated {dt.datetime.now():%d %B %Y, %H:%M}", small),
        Spacer(1, 8),
    ]

    def kv_table(rows):
        t = Table(rows, colWidths=[5.5 * cm, 10.5 * cm])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#444444")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
        ]))
        return t

    story += [Paragraph("1. Location", h2), kv_table([
        ["Location", context["location"]],
        ["Coordinates", context["coordinates"]],
        ["Area of interest", context["aoi"]],
        ["AOI area", f"{context['statistics']['aoi_area_km2']} km²"],
    ])]

    story += [Paragraph("2. Satellite data", h2), kv_table([
        ["Satellite", "Sentinel-2 (Level-2A surface reflectance)"],
        ["Requested before date", context["before_requested"]],
        ["Actual before acquisition", context["before_acquired"]],
        ["Before cloud cover", f"{context['before_cloud']:.1f}%"],
        ["Requested after date", context["after_requested"]],
        ["Actual after acquisition", context["after_acquired"]],
        ["After cloud cover", f"{context['after_cloud']:.1f}%"],
        ["Bands", "B4 / B3 / B2 (true colour)"],
        ["Native resolution", "10 m per pixel"],
        ["Processed resolution", f"{context['statistics']['metres_per_pixel_x']} m per pixel"],
    ])]

    m = context["model"]
    story += [Paragraph("3. Detector", h2), kv_table([
        ["Method", m.get("architecture", "")],
        ["Trainable parameters", f"{m.get('parameters', 0):,}"],
        ["Checkpoint", str(m.get("checkpoint", "not applicable"))],
        ["Trained epochs", str(m.get("trained_epochs", "not applicable"))],
        ["Decision threshold", str(context["threshold"])],
        ["Compute device", str(m.get("device", "cpu"))],
    ])]

    s = context["statistics"]
    story += [Paragraph("4. Change statistics", h2), kv_table([
        ["Total pixels analysed", f"{s['total_pixels']:,}"],
        ["Changed pixels", f"{s['changed_pixels']:,}"],
        ["Unchanged pixels", f"{s['unchanged_pixels']:,}"],
        ["Change percentage", f"{s['change_percentage']}%"],
        ["Detected regions", str(s["regions"])],
        ["Total area", f"{s['aoi_area_km2']} km²"],
        ["Changed area", f"{s['changed_area_km2']} km² ({s['changed_area_hectares']} ha)"],
        ["Unchanged area", f"{s['unchanged_area_km2']} km² ({s['unchanged_area_hectares']} ha)"],
        ["Largest changed region", f"{s['largest_region_hectares']} hectares"],
        ["Average confidence", f"{s['average_confidence'] * 100:.1f}%"],
    ])]

    story += [Paragraph("5. Plain-language result", h2)]
    for line in context["summary"].split("\n"):
        if line.strip():
            story.append(Paragraph(line, body))

    story.append(PageBreak())
    story.append(Paragraph("6. Imagery", h2))
    pairs = [
        ("Before image", context["images"]["before"]),
        ("After image", context["images"]["after"]),
        ("Change mask", context["images"]["mask"]),
        ("Change overlay", context["images"]["overlay"]),
    ]
    grid, row = [], []
    for label, arr in pairs:
        row.append([Paragraph(label, small), _img_flowable(arr)])
        if len(row) == 2:
            grid.append([row[0][1], row[1][1]])
            grid.append([row[0][0], row[1][0]])
            row = []
    t = Table(grid, colWidths=[8 * cm, 8 * cm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.append(t)

    story.append(PageBreak())
    story.append(Paragraph("7. Detected changes", h2))
    if context["regions"]:
        data = [["ID", "Type", "Area (ha)", "Confidence", "Latitude", "Longitude"]]
        for r in context["regions"][:40]:
            data.append([str(r.change_id), r.change_type, f"{r.area_hectares:.4f}",
                         f"{r.confidence * 100:.0f}%", f"{r.centroid_lat:.5f}", f"{r.centroid_lon:.5f}"])
        table = Table(data, colWidths=[1.2 * cm, 6.3 * cm, 2.2 * cm, 2.3 * cm, 2.2 * cm, 2.2 * cm])
        table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No significant changed region was detected.", body))

    story += [Paragraph("8. Limitations", h2), Paragraph(RESOLUTION_NOTE, body),
              Paragraph(
                  "Change labels are derived from spectral behaviour inside each detected region "
                  "and from region size. They are indicative, not a verified land-use survey. "
                  "Residual cloud, haze, seasonal vegetation cycles and different sun angles can "
                  "also produce apparent change.", body)]

    doc.build(story)
    return buf.getvalue()
