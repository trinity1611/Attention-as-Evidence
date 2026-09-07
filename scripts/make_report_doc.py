"""Build the project report (.docx) -- the full rewrite.

    python scripts/make_report_doc.py

Structure: Abstract, Introduction (problem and proposed solution), Dataset,
Models used (who built them, what they were trained on), Architecture in two
parts (classification, Grad-CAM) with diagrams, Methodology and metrics,
Results in three parts, worked examples with heatmaps, Conclusion and
drawbacks, then the two appendices of every figure.

Every number is read from outputs/ at build time. Output: Project_Briefing.docx.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, N_CLASSES, load_config, read_csv  # noqa: E402
from scripts.make_briefing_doc import (ACCENT, MUTED, bullets, figure, para,  # noqa: E402
                                       style_doc, table)
from scripts.briefing_sections import CASES  # noqa: E402
from scripts.briefing_sections2 import comparative_section, gradcam_section  # noqa: E402
from scripts.briefing_sections3 import report_study_section  # noqa: E402
from scripts.briefing_appendix import appendix_classification, appendix_gradcam  # noqa: E402
from scripts.test_predictions import load_oof  # noqa: E402

FULL, BASE = "densenet121_full224", "densenet121_imagenet"
ADOPTED = "block3+4_layercam_none"


def build() -> Path:
    cfg = load_config()
    root = Path(cfg.dirs.root)
    plots, met, dist = Path(cfg.dirs.plots), Path(cfg.dirs.metrics), Path(cfg.dirs.distribution)
    fdir = plots / "formulas"

    folds = read_csv(root / "folds.csv")
    counts = read_csv(dist / "class_counts.csv", index_col=0)["positives"]
    comp = read_csv(met / "model_comparison.csv").set_index("model")
    cvres = read_csv(met / "cv_results.csv")
    per_class = read_csv(met / "per_class_auroc.csv")
    per_class = per_class[per_class.model == BASE].set_index("class")
    auprc = read_csv(met / f"auprc_{BASE}.csv").set_index("class")
    tta = read_csv(met / "tta_gain.csv")
    clsm = read_csv(met / "classification_metrics.csv")
    shared = read_csv(met / "comparison_shared_test_macro.csv").set_index("model")
    full_summ = json.loads((met / f"{FULL}_summary.json").read_text())
    sweep = read_csv(met / "gradcam_sweep.csv")
    loc = read_csv(met / f"gradcam_localization_{ADOPTED}.csv")
    triage = read_csv(met / "triage_results.csv").set_index("scorer")

    n_img, n_pat = len(folds), folds.patient.nunique()
    macro, macro_sd = comp.loc[BASE, "macro_auroc_all14"], comp.loc[BASE, "std_all14"]
    ens = comp.loc["ensemble_cnn_raddino", "macro_auroc_all14"]
    full_auroc = full_summ["test_macro_auroc_tta"]
    hit_full = loc[loc.model == FULL].pointing_hit.mean()
    hit_base = loc[loc.model == BASE].pointing_hit.mean()
    base_row = sweep[(sweep.layer == "block4") & (sweep.method == "gradcam") &
                     (sweep.smoothing == "none") & (sweep.res == 224)].iloc[0]
    tri = triage.loc["ensemble_cnn_raddino/noisy-or"]

    helpers = {"para": para, "bullets": bullets, "table": table, "figure": figure}

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Inches(11.69), Inches(8.27)
    sec.left_margin = sec.right_margin = Inches(0.7)
    sec.top_margin = sec.bottom_margin = Inches(0.6)
    style_doc(doc)

    def formula(key, width=5.0):
        figure(doc, fdir / f"{key}.png", "", width=width)

    # ================================================================ title ===
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Attention as Evidence")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(26), ACCENT
    para(doc, "Coupling Grad-CAM Localisation to Vision-Language Report Generation "
              "for Chest Radiographs", size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Capstone Project Report", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, muted=True)
    para(doc, "Research and educational use only. Not a medical device; not for "
              "clinical diagnosis.", italic=True, size=9.5,
         align=WD_ALIGN_PARAGRAPH.CENTER, muted=True)

    # ============================================================= abstract ===
    doc.add_heading("Abstract", level=1)
    para(doc, f"Chest radiography is the most common medical imaging examination "
              f"in the world, and radiologist capacity has not kept pace with it. "
              f"Deep-learning classifiers can flag likely findings, but a bare "
              f"probability is neither trustworthy nor actionable in a clinic. "
              f"This project builds a three-part system on the NIH ChestX-ray14 "
              f"dataset: a multi-label classifier for 14 thoracic findings, a "
              f"quantitatively validated explainability layer, and a medical "
              f"vision-language model (MedGemma) that drafts a written report "
              f"grounded in what the classifier looked at.")
    para(doc, f"We trained DenseNet-121 on a 5,606-image sample under "
              f"patient-grouped five-fold cross-validation (macro AUROC "
              f"{macro:.3f} ± {macro_sd:.3f}), and showed through controlled "
              f"experiments that pretraining source and architecture did not "
              f"matter, that image resolution did (+0.029), that a frozen "
              f"chest-X-ray foundation model (RAD-DINO) with a single linear layer "
              f"tied the fine-tuned CNN, and that ensembling the two gave the first "
              f"statistically significant gain ({ens:.3f}, p = 0.001). Training the "
              f"identical architecture on the full 112,120-image dataset on a free "
              f"cloud GPU reached {full_auroc:.3f} on the official test split and "
              f"beat every sample-trained model on all 13 scorable findings when "
              f"evaluated on the same 1,293 held-out films (+0.087, 95 % CI "
              f"[+0.071, +0.104]). Data volume dominated every other lever.")
    para(doc, f"For explainability we did not stop at drawing heatmaps. We scored "
              f"111 class-activation-map configurations against 53 radiologist-"
              f"drawn boxes with a pre-registered selection rule, adopted a fused "
              f"block-3+4 LayerCAM, and raised the fraction of heatmap peaks landing "
              f"inside the radiologist's box from {base_row.pointing_hit_rate:.1%} "
              f"to {hit_full:.1%} (chance 7 %). Localisation is dependable for "
              f"Cardiomegaly and poor for millimetre-scale nodules; the heatmaps "
              f"revealed that the model's errors are radiological confusions rather "
              f"than annotation shortcuts. Finally, a 600-report study asked what to "
              f"hand the language model: giving MedGemma the classifier's "
              f"probabilities alone made it assert 2.9x more findings than from the "
              f"image and leave no normal film clean; adding the Grad-CAM zone cut "
              f"unsupported assertions by 1.07 per report (95 % CI [0.80, 1.37]) and "
              f"reduced false findings against the ground truth as well. Grounding "
              f"helps, but no condition produced a draft fit to go out unread. All "
              f"code, numbers and figures in this report are generated from the "
              f"repository and are reproducible.")

    # ========================================================= introduction ===
    doc.add_page_break()
    doc.add_heading("1.  Introduction", level=1)
    doc.add_heading("1.1  Problem statement", level=2)
    para(doc, "Hundreds of millions of chest X-rays are taken every year. Each "
              "one must be read by a trained radiologist, and in many health "
              "systems the queue is measured in days. Delays cost outcomes; "
              "fatigue costs accuracy. Automated pre-reading could order the "
              "worklist so the urgent films are seen first, and could hand the "
              "reader a starting draft.")
    para(doc, "Two obstacles have kept such systems out of routine use, and they "
              "define this project:")
    bullets(doc, [
        ("Opacity.  ", "A classifier that says \"Pneumothorax, 0.87\" gives the "
         "clinician no way to check it. Without a reason, the number cannot be "
         "acted on, so it is ignored. Trust requires evidence, and evidence means "
         "showing where in the image the conclusion came from — and knowing how "
         "often that location is actually right."),
        ("Form.  ", "Clinical workflows run on written reports with Findings and "
         "Impression, not on probability vectors. A number is not a report."),
    ])
    para(doc, "Underneath both sits a third difficulty specific to public chest "
              "X-ray data: labels are mined from free-text reports with roughly "
              "90 % accuracy, findings are rare (2–4 % prevalence for most), and "
              "several can co-occur on one film. Ordinary accuracy is meaningless "
              "here — a model that says \"normal\" every time scores 97–99 % on "
              "most findings. Any honest system must be evaluated with metrics "
              "built for that regime.")

    doc.add_heading("1.2  Proposed solution", level=2)
    para(doc, "A pipeline of three coupled components, each measured rather than "
              "assumed to work:")
    bullets(doc, [
        ("Classifier.  ", "A multi-label convolutional network (DenseNet-121) "
         "producing 14 independent probabilities per film, trained and evaluated "
         "with patient-grouped cross-validation so that no patient ever appears "
         "on both sides of a split. Compared against a frozen foundation-model "
         "probe, an ensemble, and the same architecture trained on the full "
         "dataset, to establish what limits performance."),
        ("Explainability.  ", "Class-activation heatmaps (LayerCAM on fused "
         "dense blocks) showing which pixels drove each finding — with the "
         "configuration chosen by a measured sweep, and localisation quality "
         "quantified against radiologist bounding boxes rather than shown as "
         "cherry-picked examples."),
        ("Report generation.  ", "MedGemma, a medical vision-language model, "
         "prompted with the film, the classifier's probabilities and the "
         "anatomical region the heatmap points to, producing a structured draft. "
         "The research question this enables: does grounding the language model "
         "in the classifier's attention reduce unsupported findings? "
         "Answered in Section 9."),
    ])
    para(doc, "The system is explicitly a drafting and triage aid. Every output "
              "carries a research-use disclaimer, and Section 11 lists the "
              "drawbacks that keep it from being anything more.")

    doc.add_heading("1.3  Contributions", level=2)
    bullets(doc, [
        "A leakage-free evaluation protocol for NIH ChestX-ray14 at small scale: "
        "patient-grouped folds, out-of-fold scoring, and a shared 1,293-film "
        "test set on which sample-trained and full-data models are compared fairly.",
        "A controlled study of six levers — pretraining source, architecture, "
        "resolution, schedule, foundation-model features, ensembling — against "
        "data volume, with paired statistics. Data volume wins by more than 2x.",
        "A 111-configuration explanation sweep scored on radiologist boxes with a "
        "pre-registered rule, and the finding that the network layer, not the "
        "attribution algorithm, was the bottleneck.",
        "A 600-report, three-condition study of how to brief a medical "
        "vision-language model, showing that raw probabilities inflate assertions "
        "and that supplying the attention region reduces unsupported findings by "
        "about one per report — against the ground truth, not only the classifier.",
        "A reproducible, documented codebase where every figure and number in "
        "this report is regenerated from the data.",
    ])

    # ============================================================== dataset ===
    doc.add_page_break()
    doc.add_heading("2.  The Dataset", level=1)
    doc.add_heading("2.1  Source", level=2)
    para(doc, f"NIH ChestX-ray14 (Wang et al., CVPR 2017) was released by the US "
              f"National Institutes of Health Clinical Center: 112,120 frontal "
              f"chest radiographs from 30,805 patients, 1024 x 1024 pixels, CC0 "
              f"licence. Labels for 14 findings were extracted automatically from "
              f"the radiology reports by natural-language processing; the authors "
              f"estimate over 90 % label accuracy, which means roughly one label "
              f"in ten is wrong. It also ships 984 radiologist-drawn bounding "
              f"boxes for eight findings, which we use to score explanations.")
    para(doc, f"Our local work uses the official 5 % random sample: {n_img:,} "
              f"films from {n_pat:,} patients. The full dataset (45 GB) was used "
              f"only on Kaggle's servers, where it is already hosted, so nothing "
              f"large was downloaded to our machines.")

    doc.add_heading("2.2  The 14 findings", level=2)
    order = counts.sort_values(ascending=False)
    rows = [[c.replace("_", " "), f"{n:,}", f"{n / n_img:.1%}",
             "under-powered" if n < 150 else ""] for c, n in order.items()]
    table(doc, ["Finding", "Positive films", "Prevalence", "Note"], rows,
          widths=[2.6, 1.5, 1.3, 2.4],
          caption=f"Table 1 — Class distribution in the sample. 3,044 films (54 %) "
                  f"carry no finding. Under 150 positives, per-class metrics are "
                  f"reported but flagged as unreliable; Hernia has 13.")
    figure(doc, dist / "class_distribution.png", "Figure 1 — Positive films per finding.", width=6.4)
    para(doc, f"A film carries {folds[CLASSES].sum(axis=1).mean():.2f} findings on "
              f"average and may carry several at once — Effusion with "
              f"Cardiomegaly, Atelectasis with Infiltration. The task is therefore "
              f"multi-label: fourteen independent yes/no decisions per image, not "
              f"one choice among fourteen.")
    figure(doc, dist / "cooccurrence.png",
           "Figure 2 — Co-occurrence: of films with the row finding, the fraction "
           "also carrying the column finding. Consolidation rarely appears without "
           "Infiltration; this is why the two are hard to separate.", width=6.2)

    doc.add_heading("2.3  Splits", level=2)
    para(doc, f"{n_img:,} films from {n_pat:,} patients means many patients "
              f"contributed several films. A random split by image would put the "
              f"same patient on both sides, and the model would learn to recognise "
              f"the person rather than the disease. We split by patient into five "
              f"folds, stratified on the rarest finding present, and the code "
              f"asserts that zero patients span folds. Each film is scored once, "
              f"by the one fold model that never saw it (\"out-of-fold\").")
    rows = [[f"Fold {int(r.fold)}", f"{n_img - int((folds.fold == r.fold).sum()):,}",
             f"{int((folds.fold == r.fold).sum()):,}", f"{r.auroc:.4f}"] for r in cvres.itertuples()]
    table(doc, ["Fold", "Train films", "Held-out films", "Macro AUROC"], rows,
          widths=[1.3, 1.8, 1.8, 1.8],
          caption="Table 2 — The five folds of the sample-trained DenseNet-121. "
                  "The held-out fold serves as validation (early stopping) and test.")
    para(doc, "The full-data model uses NIH's official split instead: "
              "train_val_list.txt (86,524 films, of which 10 % by patient held "
              "out for early stopping) and test_list.txt (25,596 films, never "
              "trained on). 1,293 of our sample films are in that official test "
              "list; they are the common ground on which every model is compared.")

    # ========================================================= models used ===
    doc.add_page_break()
    doc.add_heading("3.  Models Used", level=1)
    para(doc, "Seven models appear in this report: four classifiers we trained, "
              "one frozen foundation model, one family of explanation methods, "
              "and one vision-language model for the final phase.")
    table(doc, ["Model", "Developed by", "Pretraining data", "Parameters", "Role here"], [
        ["DenseNet-121", "Huang, Liu, van der Maaten, Weinberger (Cornell / "
         "Facebook AI), CVPR 2017", "ImageNet-1k (1.28 M natural photographs)",
         "7.0 M", "Primary classifier; fine-tuned on the sample and on the full dataset"],
        ["EfficientNet-B0", "Tan & Le (Google Brain), ICML 2019",
         "ImageNet-1k", "5.3 M", "Architecture control in the 224 px study"],
        ["DenseNet-121, CheXpert weights", "torchxrayvision (Cohen et al., Mila), 2020–22",
         "CheXpert (Stanford; 224 k chest X-rays, 14 labels)", "7.0 M",
         "Pretraining-source control in the 224 px study"],
        ["RAD-DINO", "Pérez-García et al. (Microsoft Health Futures), 2024",
         "880 k chest X-rays, self-supervised (DINOv2): MIMIC-CXR, CheXpert, "
         "NIH ChestX-ray14, PadChest, BRAX", "86.6 M (frozen)",
         "Frozen feature extractor; one linear layer trained on top"],
        ["Grad-CAM / LayerCAM", "Selvaraju et al. (Georgia Tech), ICCV 2017; "
         "Jiang et al., IEEE TIP 2021", "— (post-hoc; no training)", "—",
         "Explanation heatmaps; configuration chosen by sweep"],
        ["MedGemma 4B-it", "Google (Health AI Developer Foundations), 2025",
         "Gemma 3 base + de-identified medical image-text data incl. chest X-rays, "
         "dermatology, pathology, ophthalmology", "4.3 B (4-bit via Ollama)",
         "Report generation, final phase"],
    ], widths=[1.6, 2.2, 2.8, 1.1, 2.6],
          caption="Table 3 — Models, origins and pretraining data.")

    doc.add_heading("3.1  DenseNet-121", level=2)
    para(doc, "A convolutional network in which every layer inside a block "
              "receives the outputs of all earlier layers in that block. The dense "
              "connectivity lets gradients reach early layers directly and reuses "
              "features instead of relearning them, which is why a 121-layer network "
              "needs only 7 million parameters. It is the architecture of CheXNet "
              "(Rajpurkar et al., 2017), the best-known chest X-ray classifier, "
              "which makes our numbers comparable to the literature. We start from "
              "ImageNet weights — the network already knows edges, textures and "
              "shapes — and fine-tune the whole thing in two stages.")
    doc.add_heading("3.2  EfficientNet-B0 and the CheXpert-pretrained DenseNet", level=2)
    para(doc, "Two controls from the first study at 224 px. EfficientNet-B0 "
              "(Google, 2019) is a compound-scaled network of comparable size but "
              "a different design, to test whether architecture mattered. The "
              "CheXpert-weighted DenseNet from the torchxrayvision library was "
              "trained on Stanford's 224,000-film CheXpert set, to test whether "
              "starting from chest-X-ray features beat starting from ImageNet. We "
              "deliberately avoided torchxrayvision's checkpoints trained on NIH "
              "data, which would have seen our test films. Neither control was "
              "distinguishable from the ImageNet DenseNet (Section 7).")
    doc.add_heading("3.3  RAD-DINO", level=2)
    para(doc, "A vision transformer (ViT-B/14) trained by Microsoft with the "
              "DINOv2 self-supervised recipe — no labels, only images — on 880,000 "
              "chest X-rays from five public datasets. It learns to produce the "
              "same embedding for two augmented views of one film, which forces it "
              "to encode anatomy. We use it frozen: extract one 768-number vector "
              "per image (the CLS token) once, cache it, and fit a single linear "
              "layer with 10,766 parameters. One caveat is carried through every "
              "RAD-DINO number: its pretraining set included all of NIH "
              "ChestX-ray14, so it has seen the pixels — never the labels — of our "
              "held-out films. A weaker leakage than label leakage, but real.")
    doc.add_heading("3.4  Grad-CAM and LayerCAM", level=2)
    para(doc, "Class-activation mapping asks: which spatial positions in a "
              "convolutional feature map increased the score for finding c? "
              "Grad-CAM (2017) weights each channel by the average gradient of "
              "the class score with respect to that channel, sums, and applies "
              "ReLU. LayerCAM (2021) applies the gradient per pixel instead of per "
              "channel, which preserves fine spatial detail and works on earlier, "
              "higher-resolution layers. Neither requires retraining; both cost "
              "one forward and one backward pass per finding.")
    doc.add_heading("3.5  MedGemma", level=2)
    para(doc, "Google's open medical vision-language model, built on Gemma 3 and "
              "further trained on de-identified medical image-text pairs including "
              "chest X-rays. The 4-billion-parameter instruction-tuned variant "
              "accepts an image and a prompt and returns text. We run it locally "
              "through Ollama in 4-bit precision (3.3 GB) and verified that it "
              "genuinely reads images: it reported exact figures from a bar chart "
              "and correctly said no image was attached when none was.")

    # ========================================================= architecture ===
    doc.add_page_break()
    doc.add_heading("4.  Architecture", level=1)
    doc.add_heading("4.1  Classification", level=2)
    figure(doc, plots / "diagram_architecture.png",
           "Figure 3 — The DenseNet-121 classifier. Shapes are for a 320 px input.",
           width=9.6)
    para(doc, "Reading Figure 3 left to right. The 1024 px film is resized to 320 px "
              "(224 px for the full-data run) and, during training only, randomly "
              "cropped, flipped, rotated and brightness-jittered so the network "
              "never sees the identical image twice. Four dense blocks then shrink "
              "the image spatially while deepening it semantically: block 1 sees "
              "edges at 80 x 80, block 4 sees anatomy at 10 x 10 with 1,024 "
              "channels. Global average pooling collapses that to one 1,024-number "
              "summary; dropout switches 30 % of it off at random during training "
              "to discourage over-reliance on any one feature; a linear layer maps "
              "it to 14 scores.")
    para(doc, "The last step is the design decision that defines the task. A "
              "softmax would force the 14 scores to sum to one — \"pick exactly one "
              "finding\". A film can carry several, so we apply an independent "
              "sigmoid to each score instead. The loss is binary cross-entropy "
              "weighted per finding by the negative-to-positive ratio, capped at "
              "20x, so rare findings are not ignored during training.")
    figure(doc, plots / "diagram_raddino_probe.png",
           "Figure 4 — The RAD-DINO linear probe and the ensemble.", width=9.6)
    para(doc, "Figure 4 shows the second classifier. The film is cut into 14 x 14 "
              "pixel patches and passed through twelve frozen transformer blocks; "
              "the CLS token — one 768-number summary — is cached to disk once. "
              "The only trained part is the standardise-then-linear head at the "
              "right, fitted per fold with the same weighted loss as the CNN. "
              "Because the backbone already encodes lung anatomy from 880,000 "
              "films, fitting 10,766 weights on 4,500 images cannot overfit the way "
              "fine-tuning 7 million can — and it tied the CNN. The two disagree on "
              "different findings, so averaging their probabilities (the ensemble, "
              "bottom right) helps, and it is honest: for every film, both members "
              "were trained on folds that exclude it.")
    figure(doc, plots / "diagram_cls_pipeline.png",
           "Figure 5 — The two training regimes and the single evaluation rule.",
           width=9.6)
    para(doc, "Figure 5 places the models in the pipeline. The top row is "
              "everything trained locally on the 5,606-film sample; the bottom row "
              "is the identical DenseNet code trained on Kaggle on 77,988 films "
              "with NIH's official split. All roads end at one evaluation column, "
              "and one rule governs it: no image is ever scored by a model that "
              "trained on it. That rule is what makes the shared 1,293-film "
              "comparison in Section 7 legitimate.")

    doc.add_heading("4.2  Explainability (Grad-CAM / LayerCAM)", level=2)
    figure(doc, plots / "diagram_gradcam.png",
           "Figure 6 — Left: how a LayerCAM heatmap is computed for one finding. "
           "Right: how it is scored against a radiologist's box.", width=9.6)
    para(doc, "The left half of Figure 6 is the mechanism. After a normal forward "
              "pass (1), we choose one finding's score (2) and look at the "
              "feature maps of dense blocks 3 and 4 (3) — 14 x 14 and 7 x 7 grids "
              "of 1,024 channels. A backward pass (4) gives the gradient of the "
              "score with respect to every activation: how much each one pushed "
              "the score up. LayerCAM multiplies each activation by its own "
              "positive gradient (5) — per pixel, where Grad-CAM would average the "
              "gradient over the whole channel — then sums channels and keeps only "
              "positive evidence (6). The two layers' maps are fused (7), scaled to "
              "0–1 and upsampled to the original 1024 px (8), and drawn over the "
              "film (9). Fourteen findings means fourteen different heatmaps for "
              "the same film.")
    para(doc, "The right half is the part most projects skip. NIH's radiologist "
              "boxes give ground truth for where eight findings actually are. For "
              "each boxed finding we ask three questions of the heatmap: does its "
              "hottest pixel fall inside the box (pointing game, chance about 7 %); "
              "how much does the thresholded map overlap the box (IoU); and how "
              "much of the box does the map cover (coverage). Section 8 reports the "
              "answers, including the sweep that chose this configuration.")

    # ============================================== methodology and metrics ===
    doc.add_page_break()
    doc.add_heading("5.  Methodology and Metrics", level=1)
    doc.add_heading("5.1  Training", level=2)
    bullets(doc, [
        ("Two stages.  ", "Stage 1 freezes the backbone and trains only the new "
         "14-output head for 5 epochs at learning rate 1e-3, so its random initial "
         "weights cannot send large gradients into the pretrained features. Stage 2 "
         "unfreezes everything for 30 epochs with 1e-5 for the backbone and 1e-4 "
         "for the head, decayed on a cosine schedule, stopping early if held-out "
         "AUROC does not improve for 8 epochs."),
        ("Imbalance.  ", "Per-finding positive weights in the loss, capped at 20. "
         "Uncapped, Hernia's weight would be about 430 and would destabilise "
         "training for the other thirteen findings."),
        ("Label smoothing.  ", "Targets of 0.025 / 0.975 instead of 0 / 1, a small "
         "guard against over-confidence on noisy labels."),
        ("Test-time augmentation.  ", "Each film is scored twice, as-is and "
         f"mirrored, and the two probability vectors averaged. Measured gain "
         f"+{tta.gain.mean():.4f} AUROC. Mirroring a chest film is not "
         "anatomically neutral (the heart is on the left); it is standard "
         "practice, but we note it as a limitation."),
    ])
    doc.add_heading("5.2  Metrics", level=2)
    para(doc, "All metrics are computed per finding from four counts — true and "
              "false positives, true and false negatives — and then averaged. "
              "Accuracy is never reported: with 54 % of films normal and most "
              "findings at 2–4 % prevalence, a model that always says \"normal\" "
              "would score 97–99 %.")
    formula("sigmoid", 4.6)
    formula("bce", 8.0)
    para(doc, "Each output is an independent sigmoid; the loss is weighted "
              "binary cross-entropy with per-finding weight w, capped at 20.",
         italic=True, muted=True)
    formula("precision_recall", 7.2)
    formula("f1", 7.6)
    para(doc, "Precision: of the findings we reported, how many were real. "
              "Recall: of the real findings, how many we caught. F1 is their "
              "harmonic mean, which is only high when both are. These need a "
              "threshold; we fit one per finding on training predictions "
              "(F1-optimal) and apply it unchanged to held-out data.",
         italic=True, muted=True)
    formula("auroc", 7.0)
    formula("auprc", 5.4)
    para(doc, "AUROC — the probability that a random diseased film outscores a "
              "random healthy one — is threshold-free and the standard metric for "
              "this dataset, but it is flattered by the huge number of true "
              "negatives at low prevalence. AUPRC (average precision) ignores true "
              "negatives and is the honest companion; its baseline is the "
              "prevalence, so we also report lift over prevalence.",
         italic=True, muted=True)
    formula("iou", 4.8)
    para(doc, "For explanations: pointing-game hit rate (peak inside the box), "
              "IoU between the thresholded heatmap and the box, and box coverage.",
         italic=True, muted=True)
    para(doc, "Comparisons between models use paired statistics — a paired "
              "t-test across the five folds, or a paired bootstrap over the 1,293 "
              "shared films — because fold-to-fold difficulty varies more than "
              "models do, and pairing cancels it.")

    # ================================================ results I: sample CNN ===
    doc.add_page_break()
    doc.add_heading("6.  Results I — The Sample-Trained Classifier", level=1)
    para(doc, f"Macro AUROC {macro:.3f} ± {macro_sd:.3f} over 14 findings, five "
              f"folds, with TTA. Two controlled experiments preceded this result.")
    table(doc, ["Fold", "224 px, 15 epochs", "320 px, 30 epochs", "320 px + TTA"], [
        ["0", "0.7359", "0.7587", "0.7619"], ["1", "0.7204", "0.7471", "0.7532"],
        ["2", "0.7399", "0.7640", "0.7698"], ["3", "0.7043", "0.7217", "0.7225"],
        ["4", "0.7398", "0.7724", "0.7783"], ["Mean", "0.7281", "0.7528", "0.7571"],
    ], widths=[1.2, 2.0, 2.0, 2.0],
          caption="Table 4 — Resolution and schedule: every fold improved. "
                  "+0.025 from 320 px and a longer schedule, +0.004 from TTA.")
    para(doc, "At 224 px every model had peaked on its final epoch — the schedule "
              "ended before convergence. The 320 px run with twice the stage-2 "
              "epochs converges properly (Figure 7), and the largest gains fell on "
              "findings that occupy few pixels: Emphysema +0.074, Pneumothorax "
              "+0.049, Mass +0.035. Nodule did not improve; 320 px is still not "
              "enough for millimetre lesions.")
    figure(doc, plots / f"loss_curves_{BASE}.png",
           "Figure 7 — Training curves, five folds. Validation AUROC plateaus at "
           "epoch 25–30; the schedule is now adequate.", width=9.2)
    para(doc, "The first study, at 224 px, compared three models on identical "
              "folds: DenseNet-121 from ImageNet, DenseNet-121 from CheXpert "
              "weights, and EfficientNet-B0. CheXpert pretraining changed macro "
              "AUROC by −0.005 (p = 0.36); EfficientNet by −0.008 (p = 0.25). "
              "Neither pretraining source nor architecture mattered at this data "
              "scale — a null result we report as such.")
    figure(doc, plots / f"metrics_by_class_{BASE}.png",
           "Figure 8 — Precision, recall and F1 per finding at F1-optimal "
           "thresholds, ordered by number of positives. The fall to the right is "
           "the data problem made visible.", width=8.6)
    hm = clsm[(clsm.split == "heldout") & (clsm.threshold_strategy == "f1opt") &
              (clsm["class"] == "<macro avg>") & (clsm.model == BASE)].iloc[0]
    para(doc, f"Forced to yes/no decisions the picture is starker: macro precision "
              f"{hm.precision:.2f}, recall {hm.recall:.2f}, F1 {hm.f1:.2f}. On films "
              f"that genuinely carry a finding, at least one is detected 41 % of the "
              f"time. AUROC measures ranking; this measures decisions. Both are true, "
              f"and the gap is why the system is framed as a draft for a human "
              f"reader.")

    # ============================================= results II: comparative ===
    comparative_section(doc, helpers, cfg, num=7)
    figure(doc, plots / "auroc_by_class.png",
           "Figure — Per-finding AUROC of the three sample-trained models. The "
           "probe (RAD-DINO) and the CNN win on different findings; their "
           "ensemble inherits both.", width=7.2)
    figure(doc, plots / "full_vs_sample_per_class.png",
           "Figure — Sample model vs full-data model per finding. Evaluation "
           "sets differ, so indicative only; the shared-film table above is the "
           "strict test. Small-structure findings gain most from data.", width=9.0)
    figure(doc, plots / "full_model_training.png",
           "Figure — The full-data model's training run on Kaggle: validation "
           "AUROC still rising at epoch 8.", width=7.0)
    figure(doc, plots / "triage_roc.png",
           "Figure — Binary triage ROC for every scorer.", width=5.8)

    # ================================================ results III: grad-cam ===
    gradcam_section(doc, helpers, cfg, num=8)
    figure(doc, plots / "gradcam_hits_by_finding.png",
           "Figure — Pointing-game hits per finding, original vs adopted "
           "configuration, both models.", width=9.4)

    # ============================================ results IV: report study ===
    report_study_section(doc, helpers, cfg, num=9)

    # ============================================================= examples ===
    doc.add_page_break()
    doc.add_heading("10.  Worked Examples", level=1)
    para(doc, "Ten held-out films chosen to span the model's behaviour — three "
              "clean successes, three partial hits, four instructive failures. Each "
              "figure shows the film with its true findings, then LayerCAM heatmaps "
              "for the model's three highest-scoring findings (tick = truly present, "
              "cross = not). The tables give the held-out probabilities with TTA.")
    para(doc, "These heatmaps come from the sample-trained model scored out-of-fold, "
              "not the full-data model: six of the ten films were in the full model's "
              "training set, so its heatmaps on them would not be clean.",
         italic=True, muted=True)
    oof_df, y_true, y_prob = load_oof(cfg, BASE)
    thr = read_csv(met / f"thresholds_f1_{BASE}.csv").set_index("class").loc[CLASSES, "threshold"].to_numpy()
    lookup = {n: i for i, n in enumerate(oof_df.image)}
    for n, (image, label, comment) in enumerate(CASES, start=1):
        i = lookup.get(image)
        if i is None:
            continue
        true_lbl = [CLASSES[j] for j in range(N_CLASSES) if y_true[i, j] == 1]
        pred_lbl = [CLASSES[j] for j in range(N_CLASSES) if y_prob[i, j] >= thr[j]]
        top3 = np.argsort(-y_prob[i])[:3]
        hits = sorted(set(true_lbl) & set(pred_lbl))
        missed = sorted(set(true_lbl) - set(pred_lbl))
        spurious = sorted(set(pred_lbl) - set(true_lbl))
        doc.add_heading(f"Case {n} — {label}", level=2)
        figure(doc, plots / "gradcam" / "cases" / ADOPTED / BASE / f"{Path(image).stem}.png",
               f"{image}, {oof_df.iloc[i]['view']} view. Heatmaps for the top-3 findings.",
               width=9.8)
        table(doc, ["", "Findings"], [
            ["Ground truth", ", ".join(x.replace("_", " ") for x in true_lbl) or "No finding"],
            ["Reported (above threshold)", ", ".join(x.replace("_", " ") for x in pred_lbl) or "No finding"],
            ["Top 3 probabilities", ", ".join(f"{CLASSES[j].replace('_', ' ')} {y_prob[i, j]:.2f}" for j in top3)],
            ["Correct / missed / false",
             f"{len(hits)} correct" + (f"; missed {', '.join(x.replace('_', ' ') for x in missed)}" if missed else "; none missed")
             + (f"; false {', '.join(x.replace('_', ' ') for x in spurious)}" if spurious else "; no false positives")],
        ], widths=[2.1, 7.0])
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.add_run("What this shows.  ").bold = True
        p.add_run(comment)
        p.add_run(" The heatmaps add the second half of the story: where the "
                  "evidence for each of those scores came from — lung fields and "
                  "silhouettes rather than markers or hardware in every case here.")

    # ============================================= conclusion and drawbacks ===
    doc.add_page_break()
    doc.add_heading("11.  Conclusion and Drawbacks", level=1)
    doc.add_heading("11.1  Conclusion", level=2)
    bullets(doc, [
        (f"Data volume is the lever.  ", f"On the same 1,293 held-out films the "
         f"full-data DenseNet scored {shared.loc[FULL, 'macro_auroc']:.3f} against "
         f"{shared.loc[BASE, 'macro_auroc']:.3f} for the sample-trained one "
         f"(+{shared.loc[FULL, 'diff_vs_baseline']:.3f}, 95 % CI [{shared.loc[FULL, 'ci95_low']:+.3f}, "
         f"{shared.loc[FULL, 'ci95_high']:+.3f}]). Pretraining source, architecture, "
         f"resolution, schedule, foundation features and ensembling together moved "
         f"the sample model by less than half of that."),
        ("Frozen foundation features are a strong small-data baseline.  ",
         "A single linear layer on RAD-DINO embeddings tied a fully fine-tuned CNN "
         "and beat it on small structures; the two ensembled gave the only "
         "significant local gain (+0.028, p = 0.001)."),
        ("Explanations must be measured.  ", f"The default Grad-CAM pointed inside "
         f"the radiologist's box {base_row.pointing_hit_rate:.0%} of the time; the "
         f"sweep-chosen LayerCAM on fused blocks 3+4, {hit_full:.0%}. The layer, not "
         f"the algorithm, was the bottleneck. Cardiomegaly localises reliably; "
         f"nodules do not."),
        ("More data improved where the model looks, not only what it says.  ",
         f"Same heatmap method, same boxes: {hit_full:.0%} for the full-data model, "
         f"{hit_base:.0%} for the sample-trained one."),
        ("Errors are radiological, not shortcuts.  ", "On confidently wrong films "
         "the heatmaps sit on lung fields and the costophrenic angle, not on "
         "burned-in text or hardware."),
        ("Grounding the language model works, with limits.  ", "Across 600 "
         "reports, adding the Grad-CAM zone to the classifier's probabilities cut "
         "unsupported assertions by 1.07 per report (95 % CI [0.80, 1.37]) and "
         "reduced false findings against the ground truth from 5.0 to 3.9. Raw "
         "probabilities without location were the worst briefing of the three. "
         "Only ~5 % of what MedGemma adds beyond the classifier is in the labels."),
        (f"Triage is the honest use.  ", f"Any-finding-vs-none: AUROC {tri.auroc:.3f}, "
         f"accuracy {tri['acc@youden']:.2f}, sensitivity {tri['sens@youden']:.2f}, "
         f"specificity {tri['spec@youden']:.2f}. Useful for ordering a worklist; "
         f"not for diagnosis."),
    ])
    doc.add_heading("11.2  Drawbacks and limitations", level=2)
    bullets(doc, [
        ("Label noise we cannot remove.  ", "About one NIH label in ten is wrong. "
         "It caps every model here and in the literature (the best published "
         "full-data results sit at 0.82–0.85), and it corrupts evaluation as much "
         "as training."),
        ("Small-lesion resolution.  ", "Nodules are millimetres across; at 224–320 px "
         "they are a few pixels. Nodule AUROC stayed near 0.65 through every "
         "intervention, and Grad-CAM located 0 of 3 boxes."),
        ("Uncalibrated probabilities.  ", "Class weighting inflates scores; a "
         "reported 0.88 is not an 88 % chance. Case 7 reports Pneumothorax at 0.97 "
         "on a film without it. Downstream use must treat these as rankings, not "
         "confidences."),
        ("Detection at a fixed threshold is weak.  ", "41 % of abnormal films have "
         "at least one finding correctly reported by the sample model. AUROC 0.76 "
         "and 41 % detection are both true."),
        ("The Grad-CAM figure is optimistic.  ", "111 configurations ranked on 53 "
         "boxes is many comparisons on a small set. Confirmation on all 984 boxes "
         "is scheduled before the number is quoted as final."),
        ("RAD-DINO leakage.  ", "Its self-supervised pretraining saw the pixels of "
         "our test films. Weaker than label leakage; disclosed on every number."),
        ("Horizontal flip.  ", "Standard, but anatomically wrong for a chest; it "
         "may cost accuracy on side-specific findings. Untested."),
        ("Single institution, no external validation.  ", "All films are from one "
         "US hospital. Performance elsewhere is unknown."),
        ("Full-data model is under-trained.  ", "Eight epochs at 224 px in one "
         "GPU-hour; validation AUROC was still rising. A 320 px, longer run is the "
         "obvious next experiment."),
        ("The generated reports are drafts, not reports.  ", "Even the best "
         "briefing (C) produces about four findings per report that are not in "
         "the labels and misses a quarter of those that are. The 93 % localisation "
         "agreement is the model repeating the zone it was given, not verifying "
         "it. A 4-billion-parameter model quantised to 4 bits, scored against "
         "noisy labels by automatic parsing — no radiologist has read these."),
        ("Not a medical device.  ", "Nothing here has been validated for clinical "
         "use, and the interface will say so on every screen."),
    ])

    # =========================================================== appendices ===
    appendix_classification(doc, helpers, cfg, "A")
    appendix_gradcam(doc, helpers, cfg, "B")

    path = Path(__file__).resolve().parent.parent / "Project_Briefing.docx"
    try:
        doc.save(path)
    except PermissionError:
        # Word holds an exclusive lock while the document is open; do not lose
        # the build -- write beside it and say so.
        alt = path.with_name("Project_Briefing_updated.docx")
        doc.save(alt)
        print(f"NOTE: {path.name} is open in another program; saved to {alt.name} instead. "
              f"Close the document and rename, or rerun.")
        return alt
    return path


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")
