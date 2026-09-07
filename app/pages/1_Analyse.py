"""Analyse a chest X-ray: probabilities, per-finding heatmaps, evidence metrics,
MedGemma draft, JSON / PDF export."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (CFG, DISCLAIMER_SHORT, METRICS, chip, eyebrow, footer, inject_css,  # noqa: E402
                     plotly_layout, read_csv, service, status)
from src.config import CLASSES  # noqa: E402
from src.palette import FILM_TINTS, colour, pretty  # noqa: E402
from src.viz import compose  # noqa: E402

st.set_page_config(page_title="Analyse · Attention as Evidence", page_icon="🩻", layout="wide")
inject_css()
st.title("Analyse a chest X-ray")
st.markdown(f'<div class="warn">{DISCLAIMER_SHORT} Outputs are drafts for a qualified reader.</div>',
            unsafe_allow_html=True)

svc = service()
img_dir = Path(CFG.data.images_full)
DEMO_DIR = Path(__file__).resolve().parents[2] / "data" / "demo_films"   # 11 CC0 films shipped in the repo

# --------------------------------------------------------------- input ---
left, right = st.columns([1, 1.2], gap="large")
with left:
    st.subheader("1 · Film")
    up = st.file_uploader("Upload a frontal chest X-ray (PNG / JPG)", type=["png", "jpg", "jpeg"])
    if img_dir.exists():
        boxed = read_csv(METRICS / "gradcam_localization_block3+4_layercam_none.csv")
        boxed = boxed[boxed.model == "densenet121_full224"].image.drop_duplicates().tolist()
        sample = st.selectbox("…or pick a held-out NIH film with a radiologist box", ["—"] + boxed)
    else:
        img_dir = DEMO_DIR
        demo = sorted(p.name for p in DEMO_DIR.glob("*.png")) if DEMO_DIR.exists() else []
        sample = st.selectbox("…or pick one of the demo films shipped with the repository", ["—"] + demo)
        st.markdown('<div class="cap">The full NIH sample is not on this machine, so the picker shows the '
                    f'{len(demo)} held-out demo films from <code>data/demo_films</code> (all from the official '
                    'test split, never trained on). Upload any other frontal chest X-ray as well; known NIH '
                    'films are recognised by content. To get the full sample, see README → Data.</div>',
                    unsafe_allow_html=True)
    view = st.radio("View position", ["auto", "PA", "AP"], horizontal=True,
                    help="AP (bedside) films magnify the heart; the report writer is told which it is.")

    file_bytes, name = None, None
    if up is not None:
        file_bytes, name = up.getvalue(), up.name
    elif sample != "—":
        file_bytes, name = (img_dir / sample).read_bytes(), sample

if not file_bytes:
    with right:
        st.info("Upload a film or choose a sample to begin. Known NIH films are recognised by content "
                "and scored against their radiologist boxes and labels.")
    footer()
    st.stop()

key = hashlib.md5(file_bytes).hexdigest()
if st.session_state.get("key") != key:
    st.session_state.clear()
    st.session_state["key"] = key
    img = Image.open(__import__("io").BytesIO(file_bytes))
    with st.spinner("Classifying and computing heatmaps…"):
        st.session_state["img"] = img.convert("L")
        st.session_state["a"] = svc.analyse(img, file_bytes=file_bytes, name=name,
                                            view=None if view == "auto" else view)
img_gray, a = st.session_state["img"], st.session_state["a"]

with left:
    st.image(img_gray, caption=f"{a.image_name} · {a.view or 'view unknown'}"
             + (" · known NIH film" if a.known_nih else ""), width="stretch")
    if a.known_nih:
        st.markdown("**NIH labels:** " + (" ".join(chip(t) for t in a.truth) if a.truth else "No finding"),
                    unsafe_allow_html=True)

# ------------------------------------------------------- predictions ---
with right:
    st.subheader("2 · Classifier findings")
    order = np.argsort(a.probs)
    fig = go.Figure(go.Bar(
        x=a.probs[order], y=[pretty(CLASSES[i]) for i in order], orientation="h",
        marker_color=[colour(CLASSES[i]) for i in order],
        marker_line_color=["#FFFFFF" if CLASSES[i] in a.flagged else "rgba(0,0,0,0)" for i in order],
        marker_line_width=2,
        text=[f"{a.probs[i]:.2f}" + ("  ★" if CLASSES[i] in a.flagged else "") for i in order],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    for i in order:
        fig.add_shape(type="line", x0=a.thresholds[i], x1=a.thresholds[i],
                      y0=list(order).index(i) - 0.4, y1=list(order).index(i) + 0.4,
                      line=dict(color="#5B6673", width=1, dash="dot"))
    plotly_layout(fig, height=470, xaxis=dict(range=[0, 1.12], title="score"))
    fig.update_traces(textfont=dict(color="#E6ECF5"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('<div class="cap">★ above that finding\'s threshold (dotted tick). Scores are uncalibrated '
                'model outputs, not probabilities of disease. Model: DenseNet-121, 77,988 training films, '
                'AUROC 0.812 on the official NIH test split.</div>', unsafe_allow_html=True)
    st.markdown("**Flagged:** " + (" ".join(chip(f, float(a.probs[CLASSES.index(f)])) for f in a.flagged)
                                   if a.flagged else "nothing crossed its threshold"), unsafe_allow_html=True)

# --------------------------------------------------------- heatmaps ---
st.subheader("3 · Where the model looked")
h1, h2 = st.columns([1.15, 1], gap="large")
with h2:
    avail = list(a.cams.keys())
    default = a.flagged or a.top3[:1]
    chosen = st.multiselect("Findings to overlay", avail, default=default, format_func=pretty)
    theme = st.select_slider("Film tint", options=list(FILM_TINTS), value="sepia")
    alpha = st.slider("Heatmap opacity", 0.2, 1.0, 0.75, 0.05)
    show_box = st.toggle("Show radiologist box", value=bool(a.boxes), disabled=not a.boxes)
    st.markdown(" ".join(chip(f) for f in chosen), unsafe_allow_html=True)
    st.markdown('<div class="cap">Each finding has its own colour. Fill = LayerCAM intensity; outline = region at '
                '50 % of the peak (the region IoU is scored on); × = peak. Method: LayerCAM on fused dense '
                'blocks 3+4, chosen from a 111-configuration sweep.</div>', unsafe_allow_html=True)

    st.markdown("**Localisation**")
    rows = []
    for f in chosen:
        r = {"finding": pretty(f), "attention zone": a.zones.get(f, ""),
             "region (% of image)": f"{100 * a.cam_area.get(f, 0):.1f}"}
        rel = a.reliability.get(f)
        r["peak-in-box rate (our eval)"] = (f"{rel['hits']}/{rel['n_boxes']} ({rel['hit_rate']:.0%})"
                                            if rel else "not evaluated (no boxes for this finding)")
        rows.append(r)
    st.dataframe(rows, hide_index=True, width="stretch")
    if a.boxes:
        st.markdown("**Against this film's radiologist boxes**")
        st.dataframe([{"finding": pretty(b["finding"]), "peak in box": "HIT ✓" if b["hit"] else "miss ✗",
                       "IoU": round(b["iou"], 3), "box coverage": round(b["box_coverage"], 2)}
                      for b in a.boxes], hide_index=True, width="stretch")
    else:
        st.markdown('<div class="cap">Hit / miss and IoU need a radiologist\'s box; this film has none, so the '
                    'per-finding rate from our 53-box evaluation is shown instead.</div>', unsafe_allow_html=True)

with h1:
    layers = [(a.cams[f], f) for f in chosen if f in a.cams]
    box = tuple(a.boxes[0]["box"]) if (a.boxes and show_box) else None
    overlay = compose(img_gray, layers, theme=None if theme == "grayscale" else theme,
                      alpha_max=alpha, box=box)
    st.image(overlay, width="stretch",
             caption="Overlay of the selected findings" + (" · green box = radiologist" if box else ""))
    if len(chosen) > 1:
        cols = st.columns(min(3, len(chosen)))
        for k, f in enumerate(chosen):
            with cols[k % len(cols)]:
                st.image(compose(img_gray, [(a.cams[f], f)], theme=None if theme == "grayscale" else theme,
                                 alpha_max=alpha, box=box if any(b["finding"] == f for b in a.boxes) else None),
                         caption=pretty(f), width="stretch")

# ----------------------------------------------------------- report ---
st.subheader("4 · Draft report (MedGemma 4B, grounded)")
st.markdown('<div class="cap">The writer receives the film, the 14 scores and the attention zone for each '
            'flagged finding — the grounded condition from our study, which cut unsupported assertions by '
            '1.07 per report versus scores alone. About 15 s on this machine via Ollama.</div>',
            unsafe_allow_html=True)
if st.button("Generate draft report", type="primary"):
    with st.spinner("MedGemma is writing…"):
        tmp = Path(tempfile.gettempdir()) / f"cxr_{key}.png"
        img_gray.save(tmp)
        try:
            st.session_state["report"] = svc.report(tmp, a, condition="C")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not reach MedGemma through Ollama: {e}. Is `ollama serve` running?")

rep = st.session_state.get("report")
if rep:
    r1, r2 = st.columns([1.2, 1], gap="large")
    with r1:
        st.markdown(rep["prose"].replace("\n", "  \n"))
        st.markdown(f'<div class="warn">{DISCLAIMER_SHORT}</div>', unsafe_allow_html=True)
    with r2:
        st.markdown("**Model's call per finding**")
        st.dataframe([{"finding": pretty(c), "call": rep["findings"].get(c, "uncertain"),
                       "classifier": "flagged" if c in a.flagged else "",
                       "location given": a.zones.get(c, "") if c in a.flagged else ""} for c in CLASSES],
                     hide_index=True, width="stretch", height=380)
        s = rep["scores"]
        st.markdown("**Report scores**")
        lines = [f"Findings asserted: **{s['findings_asserted']}**",
                 f"Asserted and flagged by the classifier: **{s['asserted_and_flagged']}**",
                 "Asserted but not flagged: " + (", ".join(pretty(x) for x in s["asserted_not_flagged"]) or "none"),
                 "Flagged but not asserted: " + (", ".join(pretty(x) for x in s["flagged_not_asserted"]) or "none"),
                 f"Agreement with classifier (Jaccard): **{s['agreement_with_classifier_jaccard']:.2f}**"]
        if s.get("localisation_agreement") is not None:
            lines.append(f"Location agreement with the zones given: **{s['localisation_agreement']:.0%}**")
        if s.get("ground_truth_available"):
            p_, r_ = s["precision_vs_truth"], s["recall_vs_truth"]
            lines.append(f"Against NIH labels — precision **{'—' if p_ is None else f'{p_:.2f}'}**, "
                         f"recall **{'—' if r_ is None else f'{r_:.2f}'}**"
                         + (f"; false: {', '.join(pretty(x) for x in s['false_vs_truth'])}" if s["false_vs_truth"] else ""))
        lines.append(f"Generated in {s['generation_seconds']} s · JSON parsed: {'yes' if s['parse_ok'] else 'no'}")
        st.markdown("\n\n".join("- " + l for l in lines))

# ---------------------------------------------------------- downloads ---
st.subheader("5 · Download")
from src.export import analysis_json, analysis_pdf  # noqa: E402
d1, d2, d3 = st.columns(3)
stem = Path(a.image_name).stem
with d1:
    st.download_button("JSON — probabilities, zones, metrics, report, scores",
                       data=analysis_json(a, rep), file_name=f"{stem}_analysis.json", mime="application/json")
with d2:
    with st.spinner("Rendering PDF…"):
        pdf = analysis_pdf(a, rep, img_gray, theme=None if theme == "grayscale" else theme)
    st.download_button("PDF — film, heatmaps, findings, report, scores",
                       data=pdf, file_name=f"{stem}_report.pdf", mime="application/pdf")
with d3:
    from src.viz import to_png_bytes  # noqa: E402
    st.download_button("PNG — current heatmap overlay", data=to_png_bytes(overlay),
                       file_name=f"{stem}_heatmap.png", mime="image/png")

footer()
