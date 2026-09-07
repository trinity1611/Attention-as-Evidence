"""Downloadable artefacts for one analysed film: a JSON bundle and a PDF.

Both carry the model versions, the thresholds used, the disclaimer, and every
number shown in the UI, so a downloaded report can be audited without the app.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from src.config import CLASSES
from src.medgemma import DISCLAIMER
from src.palette import colour, pretty
from src.service import VERSION, Analysis
from src.viz import compose, probability_bar_png, to_png_bytes


# ------------------------------------------------------------------ JSON ---
def analysis_json(a: Analysis, report: dict | None) -> bytes:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "versions": VERSION,
        "image": {"name": a.image_name, "view": a.view, "known_nih_film": a.known_nih},
        "classifier": {
            "findings": [{"finding": c, "probability": round(float(a.probs[i]), 4),
                          "threshold": round(float(a.thresholds[i]), 4),
                          "flagged": c in a.flagged} for i, c in enumerate(CLASSES)],
            "flagged": a.flagged, "top3": a.top3,
            "note": "Probabilities are uncalibrated model scores, not clinical probabilities.",
        },
        "explanation": {
            "method": VERSION["cam"],
            "zones": a.zones,
            "cam_area_fraction_at_50pct": {k: round(v, 4) for k, v in a.cam_area.items()},
            "radiologist_boxes": a.boxes,
            "reliability_from_evaluation": a.reliability,
        },
        "ground_truth_labels": a.truth,
    }
    if report:
        payload["report"] = {
            "condition": report.get("condition"),
            "findings": report.get("findings"), "locations": report.get("locations"),
            "impression": report.get("impression"), "text": report.get("prose"),
            "scores": report.get("scores"), "parse_ok": report.get("parse_ok"),
        }
    return json.dumps(payload, indent=2).encode("utf-8")


# ------------------------------------------------------------------- PDF ---
def _rl_image(png_bytes: bytes, max_w: float, max_h: float) -> RLImage:
    im = Image.open(io.BytesIO(png_bytes))
    w, h = im.size
    scale = min(max_w / w, max_h / h)
    return RLImage(io.BytesIO(png_bytes), width=w * scale, height=h * scale)


def analysis_pdf(a: Analysis, report: dict | None, img_gray: Image.Image,
                 theme: str | None = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm,
                            title=f"Chest X-ray draft — {a.image_name}")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=17, textColor=colors.HexColor("#2E4B73"),
                        alignment=TA_CENTER, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#2E4B73"),
                        spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#5B6673"))
    warn = ParagraphStyle("warn", parent=small, textColor=colors.HexColor("#B3261E"), alignment=TA_CENTER)

    W = A4[0] - 32 * mm
    story = [Paragraph("Attention as Evidence — Chest Radiograph Draft", h1),
             Paragraph(f"Film: {a.image_name} &nbsp;·&nbsp; View: {a.view or 'unknown'} &nbsp;·&nbsp; "
                       f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", small),
             Paragraph(DISCLAIMER, warn), Spacer(1, 6)]

    # film + composite heatmap side by side
    overlay = compose(img_gray, [(a.cams[f], f) for f in a.flagged if f in a.cams], theme=theme,
                      box=tuple(a.boxes[0]["box"]) if a.boxes else None)
    film_png = to_png_bytes(img_gray.convert("RGB"), 700)
    over_png = to_png_bytes(overlay, 700)
    story.append(Table([[_rl_image(film_png, W / 2 - 4, 80 * mm), _rl_image(over_png, W / 2 - 4, 80 * mm)],
                        [Paragraph("Original film", small),
                         Paragraph("Flagged findings, LayerCAM (blocks 3+4); outline = region at 50 % of peak; "
                                   "x = peak" + ("; green box = radiologist" if a.boxes else ""), small)]],
                       colWidths=[W / 2, W / 2]))

    # classifier table
    story.append(Paragraph("1. Classifier findings", h2))
    rows = [["Finding", "Score", "Threshold", "Flagged", "Attention zone"]]
    for r in a.summary_rows():
        rows.append([pretty(r["finding"]), f"{r['probability']:.2f}", f"{r['threshold']:.2f}",
                     "yes" if r["flagged"] else "", r["zone"]])
    t = Table(rows, colWidths=[38 * mm, 16 * mm, 20 * mm, 16 * mm, W - 90 * mm], repeatRows=1)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF1F6")),
             ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8D5CE")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i, r in enumerate(a.summary_rows(), start=1):
        style.append(("TEXTCOLOR", (0, i), (0, i), colors.HexColor(colour(r["finding"]))))
        if r["flagged"]:
            style.append(("FONTNAME", (0, i), (-1, i), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    story += [t, Paragraph("Scores are uncalibrated model outputs (DenseNet-121 trained on 77,988 NIH films, "
                           "macro AUROC 0.812 on the official test split). Thresholds are per-finding "
                           "F1-optimal values fitted on held-out data.", small)]
    story.append(_rl_image(probability_bar_png(a.probs, a.flagged, CLASSES), W, 70 * mm))

    # explanation metrics
    story.append(Paragraph("2. Explanation quality", h2))
    if a.boxes:
        rows = [["Finding", "Radiologist box", "Peak inside box", "IoU", "Box coverage"]]
        for b in a.boxes:
            rows.append([pretty(b["finding"]), "yes", "HIT" if b["hit"] else "miss",
                         f"{b['iou']:.2f}", f"{b['box_coverage']:.2f}"])
        t = Table(rows, colWidths=[38 * mm, 30 * mm, 30 * mm, 20 * mm, 30 * mm])
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF1F6"))]))
        story += [Paragraph("This film carries NIH radiologist boxes, so localisation is measured directly.", small), t]
    if a.reliability:
        rows = [["Finding", "Evaluated boxes", "Peak-in-box rate", "Mean IoU"]]
        for f, r in a.reliability.items():
            if r:
                rows.append([pretty(f), str(r["n_boxes"]), f"{r['hits']}/{r['n_boxes']} ({r['hit_rate']:.0%})",
                             f"{r['mean_iou']:.2f}"])
        if len(rows) > 1:
            t = Table(rows, colWidths=[38 * mm, 30 * mm, 40 * mm, 25 * mm])
            t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF1F6"))]))
            story += [Paragraph("How often this model's heatmap for each finding landed inside the radiologist's "
                                "box in our evaluation (53 boxes; chance about 7 %). This is the reliability of the "
                                "explanation, not of the diagnosis.", small), t]

    # report
    if report:
        story += [PageBreak(), Paragraph("3. Draft report (MedGemma 4B, grounded prompt)", h2)]
        for line in (report.get("prose") or "").split("\n"):
            if line.strip():
                st = h2 if line.strip().isupper() and len(line) < 24 else body
                story.append(Paragraph(line.strip(), st))
        rows = [["Finding", "Model's call", "Classifier", "Location given"]]
        for c in CLASSES:
            rows.append([pretty(c), report["findings"].get(c, "uncertain"),
                         "flagged" if c in a.flagged else "", a.zones.get(c, "") if c in a.flagged else ""])
        t = Table(rows, colWidths=[38 * mm, 25 * mm, 22 * mm, W - 85 * mm], repeatRows=1)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF1F6"))]))
        story += [Spacer(1, 6), t]
        s = report.get("scores", {})
        story.append(Paragraph("Report scores", h2))
        lines = [f"Findings asserted: {s.get('findings_asserted')}",
                 f"Asserted and flagged by classifier: {s.get('asserted_and_flagged')}",
                 f"Asserted but not flagged: {', '.join(pretty(x) for x in s.get('asserted_not_flagged', [])) or 'none'}",
                 f"Flagged but not asserted: {', '.join(pretty(x) for x in s.get('flagged_not_asserted', [])) or 'none'}",
                 f"Agreement with classifier (Jaccard): {s.get('agreement_with_classifier_jaccard', 0):.2f}"]
        if s.get("localisation_agreement") is not None:
            lines.append(f"Location agreement with given zones: {s['localisation_agreement']:.0%}")
        if s.get("ground_truth_available"):
            lines += [f"NIH labels for this film: {', '.join(pretty(x) for x in s['true_findings']) or 'No finding'}",
                      f"Precision vs labels: {s['precision_vs_truth'] if s['precision_vs_truth'] is None else round(s['precision_vs_truth'], 2)}"
                      f" · Recall vs labels: {s['recall_vs_truth'] if s['recall_vs_truth'] is None else round(s['recall_vs_truth'], 2)}"]
        for l in lines:
            story.append(Paragraph(l, body))
        story.append(Paragraph("In our 600-report study the grounded prompt averaged 3.9 findings per report not in "
                               "the labels and missed 22 % of labelled findings. Read this draft accordingly.", small))

    story += [Spacer(1, 10), Paragraph(f"Models: {VERSION['classifier']} · {VERSION['cam']} · {VERSION['writer']}", small),
              Paragraph(DISCLAIMER, warn)]
    doc.build(story)
    return buf.getvalue()
