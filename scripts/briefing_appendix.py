"""Appendices for the briefing: every performance graph, every Grad-CAM result.

The narrative sections show the figures that carry the argument. These two
appendices hold everything else so a reader can check any claim without
opening the repository. Figures already embedded in the narrative are listed
with a pointer rather than repeated.
"""

from __future__ import annotations

from pathlib import Path

FULL, BASE = "densenet121_full224", "densenet121_imagenet"
ADOPTED, ORIGINAL = "block3+4_layercam_none", "block4_gradcam_none"


def appendix_classification(doc, helpers, cfg, letter: str) -> None:
    para, bullets, figure = helpers["para"], helpers["bullets"], helpers["figure"]
    plots = Path(cfg.dirs.plots)
    r224 = plots / "res224_study"

    doc.add_page_break()
    doc.add_heading(f"Appendix {letter} — Performance Graphs: All Classification Models", level=1)
    para(doc, "Every evaluation figure produced for the classifier phase. All "
              "held-out figures use pooled out-of-fold predictions with flip TTA "
              "unless the caption says otherwise. Figures already shown in the "
              "narrative are not repeated: the sample model's training curves, "
              "precision-recall curves and per-finding precision/recall/F1 are in "
              "Section 6; the shared-image comparison is in Section 7.",
         italic=True, muted=True)

    doc.add_heading(f"{letter}.1  Ranking quality — ROC and AUROC", level=2)
    figure(doc, plots / "auroc_by_class.png",
           "Per-finding AUROC for the three sample-trained models (mean over "
           "5 folds). Warning marks: under 150 positives, underpowered. The "
           "probe and the CNN win on different findings, which is why their "
           "ensemble helps.", width=7.2)
    figure(doc, plots / f"roc_{BASE}.png",
           "ROC curves per finding — sample-trained DenseNet-121 (320 px).", width=6.2)
    figure(doc, plots / "roc_raddino_linear.png",
           "ROC curves per finding — RAD-DINO frozen features + linear head.", width=6.2)
    figure(doc, plots / "roc_ensemble_cnn_raddino.png",
           "ROC curves per finding — ensemble of the two. Note the lift on Mass "
           "and Nodule relative to the CNN alone.", width=6.2)

    doc.add_heading(f"{letter}.2  Precision-recall and per-finding metrics — RAD-DINO probe", level=2)
    figure(doc, plots / "pr_curves_raddino_linear.png",
           "Precision-recall per finding — RAD-DINO probe. Macro AUPRC 0.176 vs "
           "the CNN's 0.144: the probe is the better ranker at the high-precision "
           "end even though the two tie on AUROC.", width=6.2)
    figure(doc, plots / "metrics_by_class_raddino_linear.png",
           "Precision / recall / F1 per finding — RAD-DINO probe, held-out. "
           "Thresholds default to 0.5 for this model (its F1-optimal thresholds "
           "were not fitted, since the confusion script recomputes training "
           "predictions through the CNN checkpoint loader).", width=8.4)
    figure(doc, plots / "loss_curves_raddino_linear.png",
           "Training curves — RAD-DINO linear probe, full-batch, early-stopped on "
           "held-out AUROC. Converges in 55–80 epochs of a few milliseconds each.",
           width=9.0)

    doc.add_heading(f"{letter}.3  Calibration", level=2)
    para(doc, "Class-weighted training deliberately breaks calibration: a "
              "predicted 0.8 does not mean an 80% chance. These curves quantify "
              "the gap, which matters because the language model in Phase B will "
              "read these numbers as confidences.")
    figure(doc, plots / f"calibration_{BASE}.png",
           "Reliability curve — sample-trained CNN. Points far below the "
           "diagonal: the model is over-confident throughout.", width=5.4)
    figure(doc, plots / "calibration_raddino_linear.png",
           "Reliability curve — RAD-DINO probe. Same pattern; same cause.", width=5.4)

    doc.add_heading(f"{letter}.4  Confusion matrices — sample-trained CNN", level=2)
    para(doc, "Fourteen 2x2 matrices, one per finding, row-normalised. Two "
              "operating points are shown: F1-optimal thresholds (fitted on "
              "training predictions, applied unchanged to held-out) and Youden's J "
              "(the screening operating point, high recall, many false alarms).")
    figure(doc, plots / f"cm_perclass_{BASE}_heldout_f1opt.png",
           "Held-out, F1-optimal thresholds. Hernia: 0 of 13 detected.", width=7.4)
    figure(doc, plots / f"cm_perclass_{BASE}_heldout_youden.png",
           "Held-out, Youden thresholds. Recall rises to ~0.7 per finding; "
           "precision collapses to ~0.1.", width=7.4)
    figure(doc, plots / f"cm_perclass_{BASE}_train_f1opt.png",
           "Training split, F1-optimal thresholds — for the train/held-out gap. "
           "Macro F1 0.28 here vs 0.17 held-out: the overfitting that AUROC hid.",
           width=7.4)
    figure(doc, plots / f"cm_labels_{BASE}_heldout_f1opt.png",
           "Label-conflation matrix, held-out: of films truly positive for the "
           "row finding, the fraction also reported as the column finding. "
           "Consolidation films fire Infiltration; Pleural Thickening films fire "
           "Effusion — the radiologically plausible confusions.", width=6.6)

    doc.add_heading(f"{letter}.5  Binary triage", level=2)
    figure(doc, plots / "triage_roc.png",
           "Any-finding-vs-none ROC for every scorer. The dedicated RAD-DINO "
           "binary head and the ensemble noisy-OR are the best at ~0.76; all "
           "sit within a narrow band, so the ceiling is the label noise in "
           "'No Finding', not the scorer.", width=6.0)

    doc.add_heading(f"{letter}.6  The full-data model", level=2)
    figure(doc, plots / "full_model_training.png",
           "Training curve on Kaggle: one head-only epoch, then seven epochs of "
           "full fine-tuning. Validation AUROC climbs monotonically to 0.827 on "
           "8,536 patient-disjoint validation films; the model was still "
           "improving when the epoch budget ended.", width=7.2)
    figure(doc, plots / "full_vs_sample_per_class.png",
           "Per-finding AUROC, sample model (5-fold) vs full-data model (official "
           "test). Different evaluation sets, so indicative only — Section 7 has "
           "the strict same-images comparison. Every finding improves; the "
           "small-structure findings improve most.", width=9.0)

    doc.add_heading(f"{letter}.7  The earlier 224 px study — pretraining and architecture", level=2)
    para(doc, "Three models at 224 px, 15 stage-2 epochs, all statistically "
              "indistinguishable (CheXpert vs ImageNet p = 0.36). Kept because a "
              "null result is a result; these models were later superseded by the "
              "320 px run and their checkpoints deleted.")
    figure(doc, r224 / "auroc_by_class.png",
           "Per-finding AUROC — DenseNet-121 (ImageNet), DenseNet-121 (CheXpert "
           "weights), EfficientNet-B0, all at 224 px.", width=7.2)
    figure(doc, r224 / f"roc_{BASE}.png", "ROC — DenseNet-121 ImageNet, 224 px.", width=5.6)
    figure(doc, r224 / "roc_densenet121_chexpert.png", "ROC — DenseNet-121 CheXpert-pretrained, 224 px.", width=5.6)
    figure(doc, r224 / "roc_efficientnet_b0.png", "ROC — EfficientNet-B0, 224 px.", width=5.6)
    figure(doc, r224 / "cm_perclass_densenet121_chexpert_heldout_f1opt.png",
           "Confusion matrices — CheXpert-pretrained DenseNet, held-out, F1-optimal.", width=7.0)
    figure(doc, r224 / "cm_perclass_efficientnet_b0_heldout_f1opt.png",
           "Confusion matrices — EfficientNet-B0, held-out, F1-optimal.", width=7.0)


def appendix_gradcam(doc, helpers, cfg, letter: str) -> None:
    para, bullets, figure = helpers["para"], helpers["bullets"], helpers["figure"]
    plots = Path(cfg.dirs.plots)
    sheets = plots / "gradcam" / "sheets"
    cases = plots / "gradcam" / "cases"

    doc.add_page_break()
    doc.add_heading(f"Appendix {letter} — Grad-CAM: Every Result", level=1)
    para(doc, "Section 8 gives the method, the sweep and two examples. This "
              "appendix shows every one of the 53 radiologist-boxed findings "
              "under the adopted configuration for both models, the same boxes "
              "under the original configuration for the full-data model, and "
              "the ten worked-example films with their top-3 heatmaps.",
         italic=True, muted=True)

    doc.add_heading(f"{letter}.1  Localisation per finding", level=2)
    figure(doc, plots / "gradcam_hits_by_finding.png",
           "Pointing-game hits per finding, original vs adopted CAM configuration, "
           "for the full-data model (left) and the sample-trained model (right). "
           "The layer change helps most on Atelectasis, Pneumonia and Effusion.",
           width=9.6)

    doc.add_heading(f"{letter}.2  All 53 boxes — full-data model, adopted configuration", level=2)
    para(doc, "One tile per radiologist box. Green rectangle: the radiologist's "
              "box. White x: the heatmap peak. Title green = hit (peak inside the "
              "box), red = miss. p is the model's probability for that finding.")
    figure(doc, sheets / f"{ADOPTED}_{FULL}.png",
           "Full-data DenseNet-121, LayerCAM on fused blocks 3+4: 31 of 53 hits. "
           "Cardiomegaly and Infiltration are consistently localised; Nodule and "
           "Effusion are the systematic misses.", width=10.2)

    doc.add_heading(f"{letter}.3  All 53 boxes — sample-trained model, adopted configuration", level=2)
    figure(doc, sheets / f"{ADOPTED}_{BASE}.png",
           "Sample-trained DenseNet-121 (320 px), same configuration: 16 of 53 "
           "hits (30.2%). Same boxes, same method — the difference is training "
           "data.", width=10.2)

    doc.add_heading(f"{letter}.4  All 53 boxes — full-data model, original configuration", level=2)
    figure(doc, sheets / f"{ORIGINAL}_{FULL}.png",
           "The same 53 boxes under the original block-4 Grad-CAM default: 19 of "
           "53 hits. Compare tile by tile with B.2 — the maps are blurrier and "
           "the peaks drift off small boxes.", width=10.2)

    doc.add_heading(f"{letter}.5  The ten worked examples — top-3 heatmaps", level=2)
    para(doc, "The same ten films as Section 10, now with a heatmap for each of the "
              "model's three highest-scoring findings. Tick = the finding is truly "
              "present; cross = it is not. Adopted CAM configuration.")
    try:
        from scripts.briefing_sections import CASES
    except ImportError:
        CASES = []
    para(doc, "Section 10 shows these films with the sample-trained model's heatmaps "
              "(the clean, out-of-fold view). Here the full-data model's heatmaps "
              "are shown for comparison; note that six of the ten films were in "
              "its training set, so its scores on them are not held-out.",
         italic=True, muted=True)
    for n, (image, label, _) in enumerate(CASES, start=1):
        f = cases / ADOPTED / FULL / f"{Path(image).stem}.png"
        figure(doc, f, f"Case {n} — {label}. Full-data model.", width=10.2)
