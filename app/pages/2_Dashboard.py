"""Dashboard: the deployed classifier's final metrics, the adopted explanation
method's results, and how the heatmaps relate to MedGemma's reports.
Read live from outputs/. No model comparisons here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (CFG, METRICS, PLOTS, card, chip, eyebrow, footer, inject_css,  # noqa: E402
                     plotly_layout, read_csv, show_plot, status)
from src.config import CLASSES  # noqa: E402
from src.palette import colour, pretty  # noqa: E402

st.set_page_config(page_title="Dashboard · Attention as Evidence", page_icon="📊", layout="wide")
inject_css()

FULL, CAM = "densenet121_full224", "block3+4_layercam_none"
fm = json.loads((METRICS / "final_model_summary.json").read_text())
pc = read_csv(METRICS / "final_model_per_class.csv")
gsum = read_csv(METRICS / f"gradcam_summary_{CAM}.csv")
gsum = gsum[gsum.model == FULL].sort_values("n", ascending=False)
gloc = read_csv(METRICS / f"gradcam_localization_{CAM}.csv")
gloc = gloc[gloc.model == FULL]
study = read_csv(METRICS / "report_study_summary.csv", index_col=0)
sweep = read_csv(METRICS / "gradcam_sweep.csv")
adopted = sweep[(sweep.layer == "block3+4") & (sweep.method == "layercam") &
                (sweep.smoothing == "none") & (sweep.res == 224)].iloc[0]

st.title("Dashboard — the deployed system")
st.markdown('<div class="cap">Numbers are read from <code>outputs/metrics</code> at page load. '
            'Every number below is held-out: the classifier on NIH\'s official test split, the '
            'heatmaps on radiologist boxes, the reports on films the classifier never trained on.</div>',
            unsafe_allow_html=True)

# ================================================================ classifier ===
st.header("1 · Classification model")
st.markdown(f"**DenseNet-121**, checkpoint `{fm['checkpoint']}` — the model behind the Analyse page.  \n"
            f"Trained on {fm['trained_on']}.  \nEvaluated on {fm['evaluated_on']}.")

m, mi, fl = fm["macro"], fm["micro"], fm["film_level"]
k = st.columns(5)
for col, (title, val, sub) in zip(k, [
    ("AUROC", f"{m['auroc']:.3f}", "macro over 14 findings · 25,596 films"),
    ("Accuracy", f"{m['accuracy']:.1%}", "macro, per-finding · see note"),
    ("Precision", f"{m['precision']:.3f}", f"macro · micro {mi['precision']:.3f}"),
    ("Recall", f"{m['recall']:.3f}", f"macro · micro {mi['recall']:.3f}"),
    ("F1", f"{m['f1']:.3f}", f"macro · micro {mi['f1']:.3f}"),
]):
    with col:
        card(title, "", kpi=val, sub=sub)

st.markdown(f'<div class="warn"><b>Read accuracy with care.</b> Findings occur in 0.3–18 % of films, so a '
            f'model that never reports a finding would already score ~90 % accuracy. AUROC, AUPRC and F1 '
            f'are the informative numbers. Macro AUPRC: <b>{m["auprc"]:.3f}</b>. At the per-finding '
            f'thresholds, {fl["abnormal_films_with_at_least_one_correct_finding"]:.0%} of abnormal films get '
            f'at least one correct finding and {fl["normal_films_with_no_finding_reported"]:.0%} of normal '
            f'films are left clean; {fl["exact_label_set_match"]:.0%} of films get every one of the 14 '
            f'labels right.</div>', unsafe_allow_html=True)

st.markdown("#### Per finding")
show = pc.copy()
show["finding"] = show.finding.map(pretty)
show = show[["finding", "n_pos_test", "auroc", "auprc", "accuracy", "precision", "recall", "f1", "threshold"]]
show.columns = ["finding", "positives (test)", "AUROC", "AUPRC", "accuracy", "precision", "recall", "F1", "threshold"]
st.dataframe(show.sort_values("AUROC", ascending=False).round(3), hide_index=True, width="stretch",
             height=530,
             column_config={"AUROC": st.column_config.ProgressColumn(format="%.3f", min_value=0.5, max_value=1.0),
                            "F1": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0)})

c1, c2 = st.columns(2)
with c1:
    order = pc.sort_values("auroc").finding.tolist()
    fig = go.Figure()
    for metric, name, op in (("auroc", "AUROC", 1.0), ("f1", "F1", 0.55)):
        fig.add_bar(y=[pretty(f) for f in order], x=[float(pc.set_index("finding").loc[f, metric]) for f in order],
                    orientation="h", name=name, marker_color=[colour(f) for f in order], opacity=op,
                    hovertemplate="%{y} " + name + ": %{x:.3f}<extra></extra>")
    plotly_layout(fig, height=480, barmode="group", xaxis=dict(range=[0, 1], title="score"),
                  legend=dict(orientation="h", y=1.08, font=dict(color="#E6ECF5")))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown('<div class="cap">AUROC (solid) and F1 (faded) per finding, in each finding\'s colour.</div>',
                unsafe_allow_html=True)
with c2:
    show_plot("full_model_training.png", "Training run: validation AUROC per epoch (still rising at epoch 8)")
    st.markdown("**Design.** 14 independent sigmoids (multi-label), weighted binary cross-entropy with "
                "per-finding positive weights capped at 20, label smoothing 0.05, two-stage fine-tuning "
                "from ImageNet weights, horizontal-flip test-time augmentation. Probabilities are "
                "uncalibrated; thresholds are per finding.")

# =========================================================== explainability ===
st.header("2 · Explainability — the adopted heatmap method")
hits = int(gloc.pointing_hit.sum())
st.markdown(f"**LayerCAM on fused dense blocks 3 + 4**, no smoothing, 224 px — the method behind every "
            f"heatmap in the Analyse page. Chosen from a 111-configuration sweep scored on the 53 "
            f"radiologist-drawn boxes that fall on our films (all in the official test split).")
k = st.columns(4)
with k[0]:
    card("Peak inside the box", "pointing game; chance ≈ 7 %", kpi=f"{hits}/{len(gloc)}",
         sub=f"{hits / len(gloc):.1%} of radiologist boxes")
with k[1]:
    card("IoU", "heatmap region at 50 % of peak vs the box", kpi=f"{gloc.iou.mean():.3f}", sub="mean over 53 boxes")
with k[2]:
    card("Box coverage", "share of the box inside the heatmap region", kpi=f"{gloc.box_coverage.mean():.2f}",
         sub="mean over 53 boxes")
with k[3]:
    card("Sweep rank", "of 111 configurations, by hits then IoU",
         kpi=f"#{int(sweep.reset_index().index[(sweep.layer == 'block3+4') & (sweep.method == 'layercam') & (sweep.smoothing == 'none') & (sweep.res == 224)][0]) + 1}",
         sub="best IoU of all 111; one forward pass")

st.markdown("#### Per finding")
g = gsum.copy()
st.dataframe(pd.DataFrame({
    "finding": [pretty(f) for f in g.finding], "radiologist boxes": g.n.astype(int),
    "peak in box": [f"{int(round(r * n))} / {int(n)}" for r, n in zip(g.pointing_hit_rate, g.n)],
    "hit rate": g.pointing_hit_rate, "mean IoU": g.mean_iou.round(3),
    "box coverage": g.mean_box_coverage.round(2), "mean score on those films": g.mean_prob.round(2)}),
    hide_index=True, width="stretch",
    column_config={"hit rate": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1)})
st.markdown('<div class="cap">Cardiomegaly and Infiltration localise reliably; Nodule does not — the box is '
            'usually inside the heatmap region (high coverage) but the peak is elsewhere: right neighbourhood, '
            'no precision. 53 boxes is a small set; the figure is indicative.</div>', unsafe_allow_html=True)

e1, e2 = st.columns(2)
with e1:
    show_plot("00023325_019.png", "Cardiomegaly — HIT, IoU 0.80. Green box = radiologist, × = heatmap peak.",
              sub=f"gradcam/bbox/{CAM}/{FULL}")
with e2:
    show_plot("00013951_001.png", "Nodule — miss. The region covers the right lobe; the peak touches the tiny box "
                                  "but does not fall inside it.", sub=f"gradcam/bbox/{CAM}/{FULL}")
with st.expander("All 53 boxes on one sheet"):
    show_plot(f"{CAM}_{FULL}.png", "Every radiologist box: green = HIT, red = miss.", sub="gradcam/sheets")

# ================================================= heatmaps -> MedGemma ===
st.header("3 · Heatmaps → report: how the explanation feeds MedGemma")
C = study.loc["C"]
dB = C["unsupported_by_classifier_per_report"] - study.loc["B", "unsupported_by_classifier_per_report"]
dF = C["false_findings_vs_truth_per_report"] - study.loc["B", "false_findings_vs_truth_per_report"]
st.markdown(f"Each flagged finding's heatmap is reduced to a radiographic zone (\"left lower zone, "
            f"costophrenic region\") and handed to MedGemma 4B together with the film and the 14 scores — "
            f"the grounded prompt used by the Analyse page. Measured on 200 held-out films "
            f"({int(C['n_reports'])} reports).")
k = st.columns(4)
with k[0]:
    card("Location agreement", "report's stated location shares side and level with the heatmap zone",
         kpi=f"{C['localisation_agreement']:.0%}", sub="largely compliance: the model repeats the zone given")
with k[1]:
    card("Effect of supplying the zone", "unsupported findings per report, vs the same prompt without zones",
         kpi=f"{dB:+.2f}", sub="95 % CI excludes zero (paired, n = 200)")
with k[2]:
    card("Effect on truth", "false findings per report vs NIH labels, vs the prompt without zones",
         kpi=f"{dF:+.2f}", sub="the heatmap moves the report toward the labels, not only toward the classifier")
with k[3]:
    card("Flagged findings omitted", "share of classifier-flagged findings the grounded report leaves out",
         kpi=f"{C['omission_rate_vs_classifier']:.1%}", sub="the report follows the evidence it is given")

st.markdown("#### The grounded report, per film")
st.dataframe(pd.DataFrame([
    ["Findings asserted per report", f"{C['mean_present_per_report']:.2f}"],
    ["Asserted but not flagged by the classifier", f"{C['unsupported_by_classifier_per_report']:.2f}"],
    ["…of which actually in the NIH labels", f"{C['share_of_unsupported_that_were_true']:.0%}"],
    ["False findings vs NIH labels", f"{C['false_findings_vs_truth_per_report']:.2f}"],
    ["Recall vs NIH labels", f"{C['recall_vs_truth']:.2f}"],
    ["Normal films left clean", f"{C['normal_films_kept_clean']:.0%}"],
    ["Seconds per report (RTX 4060, Ollama)", f"{C['mean_seconds']:.0f}"],
], columns=["metric", "value"]), hide_index=True, width="stretch")
st.markdown('<div class="warn">The grounded draft still asserts about four findings per report that are not '
            'in the labels and misses about a quarter of those that are. It is a draft for a qualified '
            'reader, never a report.</div>', unsafe_allow_html=True)

ex_dir = Path(CFG.dirs.root) / "reports" / "examples"
exs = sorted(ex_dir.glob("*_C.txt"))
if exs:
    with st.expander("Example grounded drafts from the study"):
        for f in exs[:3]:
            st.code(f.read_text(encoding="utf-8")[:2000], language=None)

footer()
