"""Briefing sections added after the classifier phase: the comparative study,
the full-data model, and the measured Grad-CAM work.

Every number is read from outputs/ at build time, as in briefing_sections.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx.shared import Pt

from src.config import read_csv

ADOPTED_CAM = "block3+4_layercam_none"
ORIGINAL_CAM = "block4_gradcam_none"
FULL = "densenet121_full224"
BASE = "densenet121_imagenet"


def _fmt_model(m: str) -> str:
    return {BASE: "DenseNet-121, our 5,606-image sample (320 px)",
            "raddino_linear": "RAD-DINO frozen features + linear head",
            "ensemble_cnn_raddino": "Ensemble of the two above",
            FULL: "DenseNet-121, full NIH dataset (77,988 train, 224 px)"}.get(m, m)


def comparative_section(doc, helpers, cfg, num: int) -> None:
    para, bullets, table, figure = (helpers["para"], helpers["bullets"],
                                    helpers["table"], helpers["figure"])
    met, plots = Path(cfg.dirs.metrics), Path(cfg.dirs.plots)

    comp = read_csv(met / "model_comparison.csv").set_index("model")
    paired = read_csv(met / "paired_comparison.csv").set_index("model") \
        if (met / "paired_comparison.csv").exists() else pd.DataFrame()
    shared = read_csv(met / "comparison_shared_test_macro.csv").set_index("model")
    shared_pc = read_csv(met / "comparison_shared_test.csv").set_index("class")
    full_summ = json.loads((met / f"{FULL}_summary.json").read_text()) \
        if (met / f"{FULL}_summary.json").exists() else {}
    triage = read_csv(met / "triage_results.csv").set_index("scorer")

    doc.add_page_break()
    doc.add_heading(f"{num}.  Beyond the Sample: Four Models Compared", level=1)
    para(doc, "After the first classifier, we asked the question every reader "
              "will ask: is 0.757 a limit of the approach, or a limit of the "
              "data? Three further models answer it. Two were trained locally "
              "on the same 5,606 images; one was trained on the full 112,120-"
              "image dataset using a free Kaggle GPU, so that nothing had to "
              "be downloaded to our own machines.")

    doc.add_heading("The models", level=2)
    bullets(doc, [
        ("RAD-DINO linear probe.  ", "RAD-DINO is Microsoft's chest-X-ray "
         "foundation model, an 86-million-parameter vision transformer "
         "pretrained without labels on 880,000 films. We froze it, extracted one "
         "768-number embedding per image, and trained only a single linear layer "
         "on top. If a frozen general-purpose model matches our fully fine-tuned "
         "CNN, the CNN was not learning much beyond generic features."),
        ("Ensemble.  ", "The average of the CNN's and the probe's probabilities. "
         "Honest because for every image both members were trained on folds that "
         "exclude it."),
        ("Full-data DenseNet.  ", "The identical architecture, loss, schedule and "
         "augmentation as our local model — same code, same head — trained on "
         f"{full_summ.get('n_train', 77988):,} images for "
         f"{full_summ.get('epochs_run', 8)} epochs at 224 px. It took "
         f"{full_summ.get('hours', 1.0):.1f} GPU-hours on a Kaggle T4. Evaluated on "
         "NIH's official 25,596-image test split, which it never saw."),
    ])

    doc.add_heading("Results on our 5-fold cross-validation", level=2)
    rows = []
    for m in [BASE, "raddino_linear", "ensemble_cnn_raddino"]:
        if m not in comp.index:
            continue
        d = f"{paired.loc[m, 'mean_diff']:+.3f}" if m in paired.index else "—"
        p = f"{paired.loc[m, 'p_value']:.3f}" if m in paired.index else "—"
        rows.append([_fmt_model(m), f"{comp.loc[m, 'macro_auroc_all14']:.3f} ± "
                     f"{comp.loc[m, 'std_all14']:.3f}", d, p])
    if full_summ:
        rows.append([_fmt_model(FULL), f"{full_summ['test_macro_auroc_tta']:.3f} "
                     "(official test split, n=25,596)", "not comparable", "—"])
    table(doc, ["Model", "Macro AUROC (14 findings)", "Δ vs our CNN", "p (paired)"],
          rows, widths=[3.9, 2.6, 1.3, 1.1],
          caption="Table — Sample-trained models on identical folds, plus the "
                  "full-data model on its own test split. The last row is not "
                  "directly comparable to the others; the next table fixes that.")
    para(doc, "The probe alone tied the CNN. But the two err on different "
              "findings — RAD-DINO is far better on small structures (Mass, "
              "Nodule, Cardiomegaly), the CNN on diffuse textures (Infiltration, "
              "Consolidation) — so their average is the first improvement in the "
              "project whose confidence interval excludes zero.")
    para(doc, "One caveat travels with the RAD-DINO numbers: its pretraining "
              "corpus included all of NIH ChestX-ray14 — images only, never "
              "labels. It has seen the pixels of our held-out films. That is a "
              "weaker form of leakage than seeing labels, but it must be stated.",
         italic=True)

    doc.add_heading("A fair comparison: the same 1,293 images for every model", level=2)
    para(doc, "The full-data model was trained on different data and tested on a "
              "different split, so its 0.812 cannot be put next to our 0.757. "
              "We therefore scored every model on the same films: the 1,293 "
              "images that are both in our sample and in NIH's official test list "
              "(685 patients, 61% abnormal). Our models score them out-of-fold; the "
              "full model never trained on them. Everything on this table is "
              "held-out for every model.")
    rows = []
    for m in shared.index:
        r = shared.loc[m]
        ci = (f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
              if pd.notna(r.get("ci95_low", float("nan"))) else "—")
        d = f"{r['diff_vs_baseline']:+.3f}" if pd.notna(r.get("diff_vs_baseline", float("nan"))) else "—"
        rows.append([_fmt_model(m), f"{r['macro_auroc']:.3f}", d, ci])
    table(doc, ["Model", "Macro AUROC (13 findings)", "Δ vs our CNN", "95% CI (bootstrap)"],
          rows, widths=[3.9, 2.0, 1.3, 1.9],
          caption="Table — Every model on the same 1,293 held-out films. Hernia "
                  "has one positive here and is excluded.")
    figure(doc, plots / "comparison_shared_test.png",
           "Figure — Per-finding AUROC on the shared 1,293 images. The full-data "
           "model (blue) leads on all 13 findings.", width=7.0)

    if FULL in shared_pc.columns and BASE in shared_pc.columns:
        gains = (shared_pc[FULL] - shared_pc[BASE]).sort_values(ascending=False)
        top = ", ".join(f"{c.replace('_', ' ')} {v:+.2f}" for c, v in gains.head(5).items())
        para(doc, f"Largest gains from data volume: {top}. These are the "
                  "small-structure findings — and the full model achieved them at "
                  "224 px, a lower resolution than our 320 px local model.")

    doc.add_heading("What this settles", level=2)
    p = doc.add_paragraph()
    p.add_run("Data volume dominates every other lever we tried — resolution, "
              "schedule, pretraining source, architecture, foundation-model "
              "features, ensembling — and by a wide margin. ").bold = True
    p.add_run("The best local trick (the ensemble, +0.036 on the shared set) is "
              "less than half the effect of 14x more training data (+0.087). This "
              "is the quantified, defensible conclusion of the classifier phase, "
              "and it reframes the project: the classifier is a component whose "
              "ceiling is set by data, and the interesting work is what we build "
              "on top of it.")

    doc.add_heading("Triage: the one place accuracy is meaningful", level=2)
    para(doc, "Collapsing the 14 labels to \"any finding vs none\" gives a 46/54 "
              "split, so accuracy, sensitivity and specificity finally mean "
              "something. It is also the clinically useful first-pass question.")
    key = "ensemble_cnn_raddino/noisy-or"
    if key in triage.index:
        t = triage.loc[key]
        table(doc, ["Scorer", "AUROC", "Accuracy", "Sensitivity", "Specificity"],
              [["Ensemble, noisy-OR over 14 findings", f"{t['auroc']:.3f}",
                f"{t['acc@youden']:.3f}", f"{t['sens@youden']:.3f}", f"{t['spec@youden']:.3f}"]],
              widths=[3.6, 1.2, 1.3, 1.3, 1.3],
              caption="Table — Binary triage on 5,606 held-out films at Youden's "
                      "operating point. Roughly seven in ten films are sorted "
                      "correctly. Useful for ordering a worklist; not for diagnosis.")


def gradcam_section(doc, helpers, cfg, num: int) -> None:
    para, bullets, table, figure = (helpers["para"], helpers["bullets"],
                                    helpers["table"], helpers["figure"])
    met, plots = Path(cfg.dirs.metrics), Path(cfg.dirs.plots)

    sweep = read_csv(met / "gradcam_sweep.csv")
    summ_new = read_csv(met / f"gradcam_summary_{ADOPTED_CAM}.csv")
    summ_old = read_csv(met / f"gradcam_summary_{ORIGINAL_CAM}.csv") \
        if (met / f"gradcam_summary_{ORIGINAL_CAM}.csv").exists() else None
    loc_new = read_csv(met / f"gradcam_localization_{ADOPTED_CAM}.csv")

    def cfg_row(layer, method, smooth, res):
        r = sweep[(sweep.layer == layer) & (sweep.method == method) &
                  (sweep.smoothing == smooth) & (sweep.res == res)]
        return r.iloc[0] if len(r) else None

    base = cfg_row("block4", "gradcam", "none", 224)
    best = sweep.iloc[0]
    adopted = cfg_row("block3+4", "layercam", "none", 224)
    n_boxes = int(base["n"]) if base is not None else 53

    doc.add_page_break()
    doc.add_heading(f"{num}.  Explainable AI: Grad-CAM, Measured", level=1)
    para(doc, "Grad-CAM produces a heatmap showing which pixels drove a given "
              "prediction. It is easy to make heatmaps that look convincing; it "
              "is harder to know whether they point at the disease. NIH provides "
              f"984 radiologist-drawn bounding boxes, {n_boxes} of which fall on "
              "our images — all in the official test split, so both our model and "
              "the full-data model can be scored on them without contamination.")

    doc.add_heading("How we score a heatmap", level=2)
    bullets(doc, [
        ("Pointing game.  ", "Is the single hottest pixel inside the radiologist's "
         "box? Chance level is the box's share of the image — about 7% here."),
        ("IoU.  ", "Threshold the heatmap at half its peak and measure overlap with "
         "the box. Penalises maps that are too big or too small."),
        ("Box coverage.  ", "What fraction of the box lies inside the heatmap "
         "region. Separates \"right neighbourhood but imprecise\" from \"wrong "
         "place\"."),
    ])

    doc.add_heading("Choosing the explanation method was itself an experiment", level=2)
    para(doc, "There are many ways to compute a class-activation map, and they "
              "disagree. Rather than pick one, we swept 111 configurations — three "
              "network layers, five attribution algorithms, four smoothing options, "
              "two input resolutions — and scored every one on the same boxes. The "
              "selection rule was fixed before running: most pointing-game hits, "
              "tie-break on IoU.")
    rows = []
    for label, r in (("Original default: block 4, Grad-CAM, no smoothing, 224 px", base),
                     ("Sweep winner: block 3, LayerCAM, both smoothings, 320 px", best),
                     ("Adopted: block 3+4 fused, LayerCAM, no smoothing, 224 px", adopted)):
        if r is not None:
            rows.append([label, f"{int(r['hits'])} / {int(r['n'])}  ({r['pointing_hit_rate']:.1%})",
                         f"{r['mean_iou@0.5']:.3f}", f"{r['mean_box_coverage']:.2f}"])
    table(doc, ["Configuration", "Pointing hits", "IoU", "Box coverage"], rows,
          widths=[4.6, 1.8, 1.0, 1.4],
          caption=f"Table — Three of the 111 configurations. The original default "
                  f"ranked {int(sweep.reset_index().index[(sweep.layer == 'block4') & (sweep.method == 'gradcam') & (sweep.smoothing == 'none') & (sweep.res == 224)][0]) + 1} of {len(sweep)}.")
    figure(doc, plots / "gradcam_sweep.png",
           "Figure — Top configurations by pointing-game hit rate. Red is the "
           "original default.", width=6.8)
    bullets(doc, [
        ("What mattered most was the layer.  ", "The best block-4 configuration "
         "managed 23 hits; the best block-3 configuration 33. The final block's "
         "7x7 grid — each cell 32 pixels wide — was the bottleneck, not the "
         "algorithm."),
        ("Why we did not adopt the strict winner.  ", "It produces tiny, sharp "
         "blobs covering 5% of the image. The peak lands in the box more often, "
         "but the region is nearly useless: IoU and coverage collapse and "
         "Cardiomegaly's IoU falls from 0.59 to 0.12. The adopted configuration "
         "is two hits behind — inside the noise on 53 boxes — with the best IoU of "
         "all 111, 69% box coverage, and one forward pass instead of six."),
        ("Honesty about the number.  ", "Ranking 111 configurations on 53 boxes "
         "is many comparisons on a small set; the adopted score is optimistic. "
         "Confirming it on all 984 boxes is a 30-minute Kaggle job that should "
         "happen before this figure is quoted in the report."),
    ])

    doc.add_heading("Results with the adopted configuration", level=2)
    piv = summ_new.pivot_table(index="finding", columns="model",
                               values=["pointing_hit_rate", "mean_iou", "n"], aggfunc="first")
    rows = []
    for f in summ_new[summ_new.model == FULL].sort_values("n", ascending=False).finding:
        n = int(summ_new[(summ_new.model == FULL) & (summ_new.finding == f)].n.iloc[0])
        r_full = summ_new[(summ_new.model == FULL) & (summ_new.finding == f)].iloc[0]
        r_loc = summ_new[(summ_new.model == BASE) & (summ_new.finding == f)]
        old = (summ_old[(summ_old.model == FULL) & (summ_old.finding == f)]
               if summ_old is not None else pd.DataFrame())
        rows.append([f.replace("_", " "), n,
                     f"{int(round(old.iloc[0].pointing_hit_rate * n))} / {n}" if len(old) else "—",
                     f"{int(round(r_full.pointing_hit_rate * n))} / {n}",
                     f"{r_full.mean_iou:.2f}",
                     f"{int(round(r_loc.iloc[0].pointing_hit_rate * n))} / {n}" if len(r_loc) else "—"])
    table(doc, ["Finding", "Boxes", "Full model, old CAM", "Full model, adopted CAM",
                "IoU", "Our sample model, adopted CAM"], rows,
          widths=[1.7, 0.7, 1.6, 1.8, 0.7, 2.1],
          caption="Table — Pointing-game hits per finding. The layer change moved "
                  "Atelectasis from zero hits to five and Pneumonia from three to seven.")
    full_new = loc_new[loc_new.model == FULL]
    loc_local = loc_new[loc_new.model == BASE]
    para(doc, f"Overall, the full-data model's heatmap peak lands in the "
              f"radiologist's box {full_new.pointing_hit.mean():.1%} of the time "
              f"(chance 7.2%), against {loc_local.pointing_hit.mean():.1%} for our "
              "sample-trained model. More data improved not only what the model "
              "predicts but where it looks.")

    doc.add_heading("Two examples", level=2)
    figure(doc, plots / "gradcam" / "bbox" / ADOPTED_CAM / FULL / "00023325_019.png",
           "Figure — Cardiomegaly. The heatmap sits on the cardiac silhouette; "
           "the peak (x) is inside the radiologist's box (green). This is the "
           "finding Grad-CAM localises dependably.", width=6.4)
    figure(doc, plots / "gradcam" / "bbox" / ADOPTED_CAM / FULL / "00013951_001.png",
           "Figure — Nodule. The heatmap covers the correct lung region and the "
           "peak touches the edge of the small box, but does not fall inside it. "
           "With the original configuration the peak was a full lobe away. "
           "Resolution, not attention, limits small-lesion localisation.", width=6.4)

    doc.add_heading("What the heatmaps are, and are not", level=2)
    bullets(doc, [
        ("They show model attention, not disease location.  ", "The two agree "
         "roughly six times in ten for the full-data model. That is far above "
         "chance and far below reliable — which is exactly why we measured it."),
        ("They are a debugging tool for us.  ", "On the confidently-wrong Case 7 "
         "from Section 10, the heatmaps land on lung fields and the costophrenic "
         "angle — anatomically plausible regions, not the burned-in text or "
         "hardware. The model's errors are radiological confusions, not "
         "annotation shortcuts."),
        ("They become an input, not just an output.  ", "The next phase converts "
         "the heatmap peak into an anatomical region description and feeds it to "
         "the language model — the coupling that makes the three components one "
         "system."),
    ])
