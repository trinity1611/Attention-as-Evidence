"""Landing page: animated hero on real model output, then the project itself --
problem, pipeline, numbers, dataset, findings, study, how to use, limits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from _common import (CFG, DIST, METRICS, PLOTS, card, chip, eyebrow, footer, inject_css,
                     read_csv, status)
from src.config import CLASSES
from src.palette import UI, colour, pretty

st.set_page_config(page_title="Attention as Evidence", page_icon="🫁", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

FULL = "densenet121_full224"
fm = json.loads((METRICS / "final_model_summary.json").read_text())
pc = read_csv(METRICS / "final_model_per_class.csv").set_index("finding")
counts = read_csv(DIST / "class_counts.csv", index_col=0)["positives"]
folds = read_csv(Path(CFG.dirs.root) / "folds.csv")
loc = read_csv(METRICS / f"gradcam_localization_block3+4_layercam_none.csv")
loc = loc[loc.model == FULL]
gsum = read_csv(METRICS / f"gradcam_summary_block3+4_layercam_none.csv")
gsum = gsum[gsum.model == FULL].set_index("finding")
study = read_csv(METRICS / "report_study_summary.csv", index_col=0)
shared = read_csv(METRICS / "comparison_shared_test_macro.csv").set_index("model")


# ------------------------------------------------------------------ hero ---
def hero_html(assets: dict) -> str:
    films = json.dumps(assets["films"])
    return f"""
<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root {{ --ink:{UI['ink']}; --muted:{UI['muted']}; --accent:{UI['accent']}; --line:{UI['line']}; --card:{UI['card']}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:transparent; font-family:Inter,-apple-system,Segoe UI,Roboto,sans-serif; color:var(--ink); }}
  .wrap {{ display:grid; grid-template-columns: 1fr 1.05fr; gap:32px; align-items:center; padding:8px 4px 4px; }}
  .stage {{ position:relative; aspect-ratio:1/1; border-radius:20px; overflow:hidden; background:#06101C;
            border:1px solid var(--line); box-shadow: 0 0 0 1px rgba(79,209,255,.10), 0 0 60px rgba(79,209,255,.16), 0 30px 60px rgba(0,0,0,.5); }}
  .stage::after {{ content:""; position:absolute; inset:0; pointer-events:none; border-radius:20px;
                   background: linear-gradient(180deg, rgba(255,255,255,.04), transparent 30%, transparent 70%, rgba(0,0,0,.25)); }}
  .film, .layer {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }}
  .film {{ animation: breathe 6s ease-in-out infinite; opacity:0; transition:opacity .9s; filter: saturate(.85) contrast(1.05); }}
  .film.on {{ opacity:1; }}
  .layer {{ opacity:0; transition: opacity 1.1s ease; }}
  .layer.on {{ opacity:1; }}
  .grid {{ position:absolute; inset:0; pointer-events:none;
           background-image: linear-gradient(rgba(79,209,255,.06) 1px, transparent 1px), linear-gradient(90deg, rgba(79,209,255,.06) 1px, transparent 1px);
           background-size: 10% 10%; }}
  .beam {{ position:absolute; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,var(--accent),transparent);
           box-shadow:0 0 26px 8px rgba(79,209,255,.45); top:-4px; opacity:0; }}
  .beam.run {{ animation: sweep 3.2s cubic-bezier(.4,0,.2,1) 1 forwards; opacity:1; }}
  @keyframes sweep {{ from {{ top:-2%; }} to {{ top:102%; }} }}
  @keyframes breathe {{ 0%,100% {{ transform:scale(1); }} 50% {{ transform:scale(1.018); }} }}
  .hud {{ position:absolute; left:14px; top:12px; font-family:'JetBrains Mono',monospace; font-size:.72rem; color:rgba(207,239,255,.85);
          letter-spacing:.06em; text-shadow:0 0 8px rgba(79,209,255,.5); }}
  .hud b {{ color:var(--accent); }}
  .tag {{ position:absolute; transform:translate(-50%,-140%); padding:.3em .75em; border-radius:999px;
          font-size:.8rem; font-weight:600; color:#E6ECF5; background:rgba(11,18,32,.88);
          border:1.5px solid; white-space:nowrap; opacity:0; transition:opacity .6s, transform .6s;
          box-shadow:0 6px 18px rgba(0,0,0,.45); backdrop-filter: blur(6px); }}
  .tag.on {{ opacity:1; transform:translate(-50%,-170%); }}
  .tag b {{ font-family:'JetBrains Mono',monospace; font-weight:600; }}
  .side .eyebrow {{ color:var(--accent); font-size:.74rem; letter-spacing:.14em; text-transform:uppercase; font-weight:600; }}
  .side h1 {{ font-size:2.55rem; line-height:1.06; margin:.2rem 0 .6rem; letter-spacing:-.025em;
              background: linear-gradient(90deg, #FFFFFF, #BFE7FF); -webkit-background-clip:text; background-clip:text; color:transparent; }}
  .side p.lead {{ color:var(--muted); font-size:1.04rem; line-height:1.55; margin:.2rem 0 1rem; }}
  .meta {{ font-size:.84rem; color:var(--muted); font-family:'JetBrains Mono',monospace; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.55rem; }}
  .chip {{ padding:.22em .7em; border-radius:999px; font-size:.8rem; border:1.5px solid; background:rgba(255,255,255,.03); color:var(--ink); }}
  .chip b {{ font-family:'JetBrains Mono',monospace; }}
  .impr {{ margin-top:1rem; min-height:5em; font-size:.98rem; color:var(--ink); border-left:3px solid var(--accent);
           padding:.55rem .95rem; background:rgba(79,209,255,.05); border-radius:0 12px 12px 0; }}
  .impr .lbl {{ font-size:.7rem; letter-spacing:.12em; color:var(--muted); text-transform:uppercase; display:block; margin-bottom:.3rem; }}
  .caret {{ display:inline-block; width:.55em; height:1em; background:var(--accent); vertical-align:-.15em; margin-left:2px; animation: blink 1s steps(2) infinite; }}
  @keyframes blink {{ to {{ visibility:hidden; }} }}
  .disc {{ font-size:.74rem; color:var(--muted); margin-top:.9rem; }}
  @media (prefers-reduced-motion: reduce) {{ .film, .beam.run {{ animation:none; }} .beam {{ display:none; }} }}
  @media (max-width: 900px) {{ .wrap {{ grid-template-columns:1fr; }} }}
</style></head><body>
<div class="wrap">
  <div class="stage" id="stage">
    <img class="film" id="film" alt="chest X-ray">
    <div class="grid"></div>
    <div id="layers"></div>
    <div class="beam" id="beam"></div>
    <div id="tags"></div>
    <div class="hud" id="hud">DENSENET-121 · LAYERCAM 3+4 · <b>LIVE</b></div>
  </div>
  <div class="side">
    <div class="eyebrow">Capstone · explainable chest radiography</div>
    <h1>Attention as Evidence</h1>
    <p class="lead">A chest X-ray classifier that shows <em>where</em> it looked, measures whether that
       location is right against radiologists' boxes, and hands the evidence to a medical language model
       to draft the report — with every step scored, not assumed.</p>
    <div class="meta" id="meta"></div>
    <div class="chips" id="chips"></div>
    <div class="impr"><span class="lbl">Draft impression · MedGemma 4B, grounded prompt</span><span id="impr"></span><span class="caret" id="caret"></span></div>
    <div class="disc">Everything animated here is real output of the pipeline on held-out films. Research use only; not for diagnosis.</div>
  </div>
</div>
<script>
const FILMS = {films};
const stage = document.getElementById('stage'), film = document.getElementById('film');
const layersEl = document.getElementById('layers'), tagsEl = document.getElementById('tags');
const beam = document.getElementById('beam'), impr = document.getElementById('impr'), caret = document.getElementById('caret');
const meta = document.getElementById('meta'), chips = document.getElementById('chips'), hud = document.getElementById('hud');
let i = 0, timers = [];
const wait = (ms) => new Promise(r => timers.push(setTimeout(r, ms)));
function clear() {{ timers.forEach(clearTimeout); timers = []; }}
function countUp(el, target, ms) {{
  const t0 = performance.now();
  (function step(now) {{ const k = Math.min(1, (now - t0) / ms); el.textContent = (target * k).toFixed(2);
     if (k < 1) requestAnimationFrame(step); }})(t0);
}}
async function typewrite(text) {{
  impr.textContent = ''; caret.style.display = 'inline-block';
  for (let k = 0; k < text.length; k++) {{ impr.textContent += text[k]; await wait(text.length > 160 ? 12 : 22); }}
  caret.style.display = 'none';
}}
async function play(f) {{
  clear();
  layersEl.innerHTML = ''; tagsEl.innerHTML = ''; impr.textContent = ''; chips.innerHTML = '';
  film.classList.remove('on'); beam.classList.remove('run');
  film.src = 'data:image/jpeg;base64,' + f.base_jpg;
  hud.innerHTML = `DENSENET-121 · LAYERCAM 3+4 · <b>${{f.name.replace('.png','')}}</b> · ${{f.view || ''}}`;
  meta.innerHTML = `NIH labels: ${{f.truth.length ? f.truth.join(', ') : 'No finding'}}`;
  f.top.forEach(t => {{ const c = document.createElement('span'); c.className = 'chip';
     c.style.borderColor = t.colour + '88'; c.innerHTML = `${{t.label}} <b>${{t.prob.toFixed(2)}}</b>`; chips.appendChild(c); }});
  await wait(150); film.classList.add('on'); await wait(900);
  beam.classList.add('run');
  const S = {assets["size"]};
  const sorted = [...f.layers].sort((a, b) => a.peak[1] - b.peak[1]);
  for (const L of sorted) {{
    const delay = Math.max(0, 3200 * (L.peak[1] / S) - 200);
    setTimeout(() => {{
      const img = document.createElement('img'); img.className = 'layer'; img.src = 'data:image/png;base64,' + L.png;
      layersEl.appendChild(img); requestAnimationFrame(() => img.classList.add('on'));
      const tag = document.createElement('div'); tag.className = 'tag'; tag.style.borderColor = L.colour;
      tag.style.left = (L.peak[0] / S * 100) + '%'; tag.style.top = (L.peak[1] / S * 100) + '%';
      tag.innerHTML = `<span style="display:inline-block;width:.6em;height:.6em;border-radius:50%;background:${{L.colour}};box-shadow:0 0 10px ${{L.colour}};margin-right:.45em"></span>${{L.label}} <b>0.00</b>`;
      tagsEl.appendChild(tag); requestAnimationFrame(() => tag.classList.add('on'));
      countUp(tag.querySelector('b'), L.prob, 900);
    }}, delay);
  }}
  await wait(3500);
  if (!f.layers.length) {{
    const tag = document.createElement('div'); tag.className = 'tag'; tag.style.borderColor = '#3DDC97';
    tag.style.left = '50%'; tag.style.top = '50%'; tag.textContent = 'No finding crossed its threshold';
    tagsEl.appendChild(tag); requestAnimationFrame(() => tag.classList.add('on'));
  }}
  await typewrite(f.impression || (f.layers.length ? 'Findings flagged; see the classifier panel for probabilities.' : 'No acute cardiopulmonary abnormality flagged by the classifier.'));
  await wait(3800);
  i = (i + 1) % FILMS.length; play(FILMS[i]);
}}
if (FILMS.length) play(FILMS[0]);
</script></body></html>"""


assets_path = Path(CFG.dirs.root) / "hero" / "hero_assets.json"
if assets_path.exists():
    html = hero_html(json.loads(assets_path.read_text()))
    if hasattr(st, "iframe"):
        st.iframe(html, height=590)
    else:
        components.html(html, height=590)
else:
    st.title("Attention as Evidence")
    st.caption("Run `python scripts/make_hero_assets.py` to generate the animated hero.")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.page_link("pages/1_Analyse.py", label="Analyse a chest X-ray →", icon="🩻")
with c2:
    st.page_link("pages/2_Dashboard.py", label="Dashboard: final metrics →", icon="📊")

# ------------------------------------------------------------ the problem ---
st.markdown("")
eyebrow("Why this exists")
st.markdown("## A probability is not evidence, and a number is not a report")
p1, p2, p3 = st.columns(3)
with p1:
    card("Radiologists are outnumbered", "Chest X-rays are the most common imaging exam in the world. "
         "Queues are measured in days; fatigue costs accuracy. An AI pre-read could order the worklist and "
         "hand the reader a draft.")
with p2:
    card("Opacity blocks trust", "\"Pneumothorax, 0.87\" gives a clinician nothing to check. Without "
         "<em>where</em> the conclusion came from — and how often that location is right — the number "
         "cannot be acted on, so it is ignored.")
with p3:
    card("Labels are noisy and findings are rare", "NIH labels are text-mined at ~90 % accuracy; most "
         "findings occur in 2–4 % of films. Plain accuracy is meaningless here, so every metric on this site "
         "is built for that regime.")

# ---------------------------------------------------------------- pipeline ---
st.markdown("")
eyebrow("How it works")
st.markdown("## Three coupled components, each measured")
s1, s2, s3 = st.columns(3)
with s1:
    card("① Classify", f"DenseNet-121 fine-tuned on 77,988 NIH films (Kaggle T4, one GPU-hour). Fourteen "
         f"independent sigmoids — a film can carry several findings — weighted BCE, per-finding thresholds, "
         f"flip test-time augmentation.", kpi=f"{fm['macro']['auroc']:.3f}",
         sub="macro AUROC · official test split, 25,596 films")
with s2:
    hits = int(loc.pointing_hit.sum())
    card("② Explain", "LayerCAM on fused dense blocks 3+4 — chosen from a 111-configuration sweep scored "
         "against radiologist boxes with a pre-registered rule. Each heatmap's peak is tested against the "
         "box: hit or miss, IoU, coverage.", kpi=f"{hits}/{len(loc)}",
         sub=f"heatmap peaks inside the radiologist's box · chance ≈ 7 %")
with s3:
    d = study.loc["C", "unsupported_by_classifier_per_report"] - study.loc["B", "unsupported_by_classifier_per_report"]
    card("③ Report", "MedGemma 4B (Google) via Ollama drafts Findings / Impression / Recommendation from "
         "the film, the 14 scores and the heatmap's anatomical zone. A 600-report study measured what that "
         "grounding does.", kpi=f"{d:+.2f}",
         sub="unsupported findings per report when the zone is supplied · CI excludes 0")
st.image(str(PLOTS / "diagram_cls_pipeline.png"),
         caption="Two training regimes, one evaluation rule: no image is ever scored by a model that trained on it.")

# ---------------------------------------------------------- by the numbers ---
st.markdown("")
eyebrow("By the numbers")
st.markdown("## Held-out, every one of them")
m, fl = fm["macro"], fm["film_level"]
k = st.columns(6)
tiles = [
    ("AUROC", f"{m['auroc']:.3f}", "macro, 14 findings"),
    ("AUPRC", f"{m['auprc']:.3f}", "≈ 6× lift over prevalence"),
    ("F1", f"{m['f1']:.3f}", f"macro · micro {fm['micro']['f1']:.3f}"),
    ("Abnormal films caught", f"{fl['abnormal_films_with_at_least_one_correct_finding']:.0%}", "≥ 1 correct finding"),
    ("Heatmap IoU", f"{loc.iou.mean():.2f}", "region vs radiologist box"),
    ("Reports parsed", "100 %", "600 / 600 structured replies"),
]
for col, (t, v, s) in zip(k, tiles):
    with col:
        card(t, "", kpi=v, sub=s)

# ------------------------------------------------------------------ dataset ---
st.markdown("")
eyebrow("Data")
st.markdown("## NIH ChestX-ray14")
d1, d2 = st.columns([1.1, 1])
with d1:
    st.markdown(f"""
- **112,120** frontal films from **30,805** patients, released by the US National Institutes of Health (CC0).
  Full set used only on Kaggle for the deployed model; the **5,606-film / 4,230-patient** official 5 % sample
  was used for all local experiments.
- **14 findings**, labels mined from radiology reports by NLP — the authors estimate **> 90 % accuracy**,
  so roughly one label in ten is wrong. **984 radiologist-drawn boxes** for eight findings; 53 fall on our films.
- **Splits by patient, never by image.** Many patients contribute several films; a random image split would
  let the model learn the person instead of the disease. `prepare_data.py` asserts zero patients span folds.
- **1,293 films** sit in both our sample and NIH's official test list — the common ground on which every
  model was compared fairly.
""")
    st.markdown(" ".join(chip(c) for c in CLASSES), unsafe_allow_html=True)
with d2:
    st.image(str(DIST / "class_distribution.png"), caption="Positive films per finding in the sample. 54 % carry no finding.")

# ---------------------------------------------------------- per-finding ---
st.markdown("")
eyebrow("Per finding")
st.markdown("## How well each finding is classified and localised")
rows = []
for c in CLASSES:
    r = pc.loc[c]
    g = gsum.loc[c] if c in gsum.index else None
    rows.append({
        "finding": pretty(c), "test positives": int(r.n_pos_test), "AUROC": float(r.auroc),
        "F1": float(r.f1), "recall": float(r.recall), "precision": float(r.precision),
        "heatmap hits": (f"{int(round(g.pointing_hit_rate * g.n))} / {int(g.n)}" if g is not None else "no boxes"),
        "verdict": ("strong" if r.auroc >= 0.85 else "usable" if r.auroc >= 0.75 else "weak"),
    })
df = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
st.dataframe(df, hide_index=True, width="stretch", height=530,
             column_config={"AUROC": st.column_config.ProgressColumn(format="%.3f", min_value=0.5, max_value=1.0),
                            "F1": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                            "recall": st.column_config.NumberColumn(format="%.2f"),
                            "precision": st.column_config.NumberColumn(format="%.2f")})
st.markdown('<div class="cap">Strong ≥ 0.85 AUROC · usable ≥ 0.75 · weak below. Heatmap hits are on the 53 '
            'radiologist boxes (8 findings have boxes). Cardiomegaly and Emphysema are the model\'s best; '
            'Infiltration and Pneumonia its weakest — and Nodule, at millimetre scale, is never localised.</div>',
            unsafe_allow_html=True)

# ---------------------------------------------------------------- study ---
st.markdown("")
eyebrow("The research question")
st.markdown("## Does telling the writer where the classifier looked make a better report?")
q1, q2 = st.columns([1.15, 1])
with q1:
    A, B, C = study.loc["A"], study.loc["B"], study.loc["C"]
    st.markdown(f"""
The same 200 held-out films were written up three times by MedGemma with three briefings — **image only**,
**+ the 14 probabilities**, **+ the heatmap zone per flagged finding** — and every reply was scored against
both the classifier and the NIH labels.

| per report | image only | + probabilities | **+ heatmap zone** |
|---|--:|--:|--:|
| findings asserted | {A.mean_present_per_report:.1f} | {B.mean_present_per_report:.1f} | **{C.mean_present_per_report:.1f}** |
| unsupported by classifier | {A.unsupported_by_classifier_per_report:.2f} | {B.unsupported_by_classifier_per_report:.2f} | **{C.unsupported_by_classifier_per_report:.2f}** |
| false vs NIH labels | {A.false_findings_vs_truth_per_report:.2f} | {B.false_findings_vs_truth_per_report:.2f} | **{C.false_findings_vs_truth_per_report:.2f}** |
| recall vs NIH labels | {A.recall_vs_truth:.2f} | {B.recall_vs_truth:.2f} | **{C.recall_vs_truth:.2f}** |
| normal films left clean | {A.normal_films_kept_clean:.0%} | {B.normal_films_kept_clean:.0%} | **{C.normal_films_kept_clean:.0%}** |
""")
with q2:
    card("What we found",
         f"Handing the model a probability list without location makes it assert "
         f"<b>{B.mean_present_per_report / A.mean_present_per_report:.1f}×</b> more findings than from the image "
         f"and leave no normal film clean. Adding the heatmap zone removes about <b>one unsupported and one "
         f"false finding per report</b> (95 % CI excludes zero) — the heatmap moves the report toward the "
         f"labels, not merely toward the classifier. Only ~{C.share_of_unsupported_that_were_true:.0%} of what "
         f"MedGemma adds beyond the classifier is real. No condition produces a draft fit to go out unread.")
    st.image(str(PLOTS / "report_study.png"), caption="The three headline measures by condition")

# --------------------------------------------------------------- how to use ---
st.markdown("")
eyebrow("Using the app")
st.markdown("## From film to draft in about a minute")
u = st.columns(4)
for col, (n, t, b) in zip(u, [
    ("1", "Upload or pick a film", "PNG / JPG, or one of the 50 held-out NIH films with radiologist boxes. Known films are recognised by content and scored against their boxes and labels."),
    ("2", "Read the probabilities", "Fourteen bars in fixed finding colours. A ★ means the score crossed that finding's own threshold — thresholds differ per finding because the scores are uncalibrated."),
    ("3", "Inspect the heatmaps", "Overlay any findings; each has its colour, a contour at 50 % of peak, and its anatomical zone. Hit / miss / IoU where a box exists, the evaluated reliability rate otherwise."),
    ("4", "Draft and export", "MedGemma writes under the grounded prompt in ~15 s. Read the scores beside it. Download JSON, PDF or PNG — every file carries the disclaimer."),
]):
    with col:
        card(f"{n} · {t}", b)

# --------------------------------------------------------------- limits ---
st.markdown("")
eyebrow("Read before trusting anything above")
st.markdown("## Limits, stated plainly")
l1, l2 = st.columns(2)
with l1:
    st.markdown(f"""
- {status('data', 'warn')} About one label in ten is wrong; it caps every model on this dataset at ~0.85 AUROC.
- {status('scale', 'warn')} Nodules are millimetres; at 224 px a few pixels. Nodule AUROC {pc.loc['Nodule','auroc']:.2f}, 0 of 3 boxes localised.
- {status('calibration', 'bad')} Class-weighted training inflates scores; 0.88 is not an 88 % chance. Thresholds are per finding for that reason.
- {status('detection', 'warn')} At those thresholds, {fl['abnormal_films_with_at_least_one_correct_finding']:.0%} of abnormal films get a correct finding; {fl['normal_films_with_no_finding_reported']:.0%} of normal films are left clean.
""", unsafe_allow_html=True)
with l2:
    st.markdown(f"""
- {status('xai', 'warn')} 111 configurations ranked on 53 boxes is optimistic; confirmation on all 984 is pending.
- {status('reports', 'bad')} The grounded draft still asserts ~{study.loc['C','false_findings_vs_truth_per_report']:.0f} findings per report not in the labels and misses a quarter of those that are. Its 93 % location agreement is compliance, not verification.
- {status('scope', 'info')} One institution, no external validation, no radiologist has reviewed these outputs.
- {status('status', 'bad')} Not a medical device. A drafting and triage aid for a qualified reader.
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ stack ---
st.markdown("")
eyebrow("Under the hood")
st.markdown(" ".join(f'<span class="chip" style="border-color:{UI["line"]}"><span class="dot" style="background:{UI["accent"]};color:{UI["accent"]}"></span>{t}</span>'
                     for t in ["PyTorch 2.6 · CUDA", "DenseNet-121 (ImageNet init)", "RAD-DINO (frozen probe, study)",
                               "pytorch-grad-cam · LayerCAM", "MedGemma 4B-it · Ollama Q4_K_M", "Kaggle T4 (full-data run)",
                               "RTX 4060 laptop (everything else)", "Streamlit · Plotly · ReportLab",
                               "5-fold patient-grouped CV · paired bootstrap"]),
            unsafe_allow_html=True)

footer()
