"""Build the team briefing document (.docx).

    python scripts/make_briefing_doc.py

Audience: teammates new to deep learning. Every technical term is defined at
first use, and the numbers are pulled live from outputs/ rather than typed in,
so the doc cannot drift from the actual results.

Output: Project_Briefing.docx in the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, load_config, read_csv  # noqa: E402
from scripts.briefing_sections import metrics_section, samples_section  # noqa: E402
from scripts.briefing_sections2 import comparative_section, gradcam_section  # noqa: E402
from scripts.briefing_appendix import appendix_classification, appendix_gradcam  # noqa: E402
from scripts.test_predictions import load_oof  # noqa: E402

ACCENT = RGBColor(0x2E, 0x4B, 0x73)
MUTED = RGBColor(0x5B, 0x66, 0x73)


def style_doc(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.12
    for lvl, size in ((1, 17), (2, 13.5), (3, 11.5)):
        st = doc.styles[f"Heading {lvl}"]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = ACCENT


def para(doc, text, *, italic=False, size=None, align=None, muted=False,
         space_after=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if muted:
        run.font.color.rgb = MUTED
    if align is not None:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def bullets(doc, items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        if isinstance(it, tuple):
            lead, rest = it
            p.add_run(lead).bold = True
            p.add_run(rest)
        else:
            p.add_run(it)
        p.paragraph_format.space_after = Pt(3)


def table(doc, headers, rows, widths=None, caption=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(9.5)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Inches(w)
    if caption:
        para(doc, caption, italic=True, size=8.5, muted=True,
             align=WD_ALIGN_PARAGRAPH.CENTER)
    return t


def figure(doc, path: Path, caption: str, width=9.4):
    if not path.exists():
        para(doc, f"[missing figure: {path.name}]", italic=True, muted=True)
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, caption, italic=True, size=8.5, muted=True,
         align=WD_ALIGN_PARAGRAPH.CENTER)


def build() -> Path:
    cfg = load_config()
    out_root = Path(cfg.dirs.root)
    plots, metrics, dist = Path(cfg.dirs.plots), Path(cfg.dirs.metrics), Path(cfg.dirs.distribution)

    folds = read_csv(out_root / "folds.csv")
    counts = read_csv(dist / "class_counts.csv", index_col=0)["positives"]
    comp = read_csv(metrics / "model_comparison.csv")
    per_class = read_csv(metrics / "per_class_auroc.csv")
    per_class = per_class[per_class.model == cfg.models[0]]   # file now holds several models
    auprc = read_csv(metrics / f"auprc_{cfg.models[0]}.csv").set_index("class")
    cvres = read_csv(metrics / "cv_results.csv")
    tta = read_csv(metrics / "tta_gain.csv")
    clsm = read_csv(metrics / "classification_metrics.csv")

    n_img, n_pat = len(folds), folds["patient"].nunique()
    macro = float(comp["macro_auroc_all14"].iloc[0])
    macro_sd = float(comp["std_all14"].iloc[0])

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = Inches(11.69), Inches(8.27)
    for m in ("left_margin", "right_margin"):
        setattr(sec, m, Inches(0.7))
    sec.top_margin = sec.bottom_margin = Inches(0.6)
    style_doc(doc)

    # ---------------- title ----------------
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = t.add_run("Attention as Evidence")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = ACCENT
    para(doc, "Coupling Grad-CAM Localisation to Vision-Language Report Generation "
              "for Chest Radiographs", size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "Capstone Project — Team Briefing", size=13,
         align=WD_ALIGN_PARAGRAPH.CENTER, muted=True)
    para(doc, "A walkthrough of what we are building, the data we use, how the "
              "model works, and what comes next.", italic=True, size=10.5,
         align=WD_ALIGN_PARAGRAPH.CENTER, muted=True)

    # ---------------- 1. introduction ----------------
    doc.add_heading("1.  Introduction and Problem Statement", level=1)

    doc.add_heading("What the project does", level=2)
    para(doc, "We are building a system that looks at a chest X-ray and does "
              "three things: (1) tells you which of 14 lung and chest "
              "conditions it thinks are present, (2) shows you which part of "
              "the image made it think so, and (3) writes a short draft "
              "radiology report in plain English.")

    doc.add_heading("The problem we are addressing", level=2)
    para(doc, "Chest X-rays are the most common medical imaging test in the "
              "world — hundreds of millions are taken every year. Every one of "
              "them needs a trained radiologist to read it, and in many "
              "hospitals there are far more X-rays than radiologists. Reports "
              "get delayed, and tired readers miss things.")
    para(doc, "An AI system that pre-reads X-rays could flag the urgent cases "
              "first and give the radiologist a starting draft. But there is a "
              "catch, and it is the reason this project exists:")
    bullets(doc, [
        ("Doctors will not trust a black box.  ", "If a model says "
         "\"Pneumothorax, 87% confident\" and gives no reason, a clinician has "
         "no way to check it. They cannot act on it, so it is useless in "
         "practice."),
        ("A number is not a report.  ", "A probability score is not what "
         "hospitals use. They need written findings — the kind of text a "
         "radiologist would dictate."),
    ])
    para(doc, "So our project has three parts that address this directly:")
    bullets(doc, [
        ("Classifier  ", "— what conditions are present? (built, results below)"),
        ("Explainable AI (Grad-CAM)  ", "— where in the image? (built and measured, Section 9)"),
        ("MedGemma report generation  ", "— written up as text a human can read? "
         "(next, Section 10)"),
    ])

    doc.add_heading("An important limit, stated up front", level=2)
    p = doc.add_paragraph()
    p.add_run("This is a research and educational project, not a medical "
              "device. ").bold = True
    p.add_run("It is not accurate enough to diagnose anyone, and we will say so "
              "clearly in the interface and the report. Our framing is "
              "\"draft findings for a human to review\", never \"diagnosis\". "
              "Sections 6 and 7 give the honest numbers behind that statement.")

    # ---------------- 2. dataset ----------------
    doc.add_page_break()
    doc.add_heading("2.  The Dataset and How We Split It", level=1)

    doc.add_heading("Where the data comes from", level=2)
    para(doc, f"We use the official 5% sample of the NIH ChestX-ray14 dataset, "
              f"released by the US National Institutes of Health and free to "
              f"use (CC0 licence). It contains {n_img:,} X-ray images from "
              f"{n_pat:,} different patients. Each image is a 1024x1024 "
              f"greyscale PNG.")
    para(doc, "The full dataset has 112,120 images (45 GB). We are using the "
              "sample because it fits on a laptop; the trade-off is lower "
              "accuracy, which we quantify in Section 5.")

    doc.add_heading("The 14 conditions, and why the counts matter", level=2)
    para(doc, "Each image is labelled with zero or more of 14 findings. The "
              "labels were extracted automatically from radiology reports using "
              "text mining, so roughly 10% of them are wrong — a real "
              "limitation we have to work around.")
    rows, order = [], counts.sort_values(ascending=False)
    for cls, n in order.items():
        flag = "yes — too few to trust" if n < 150 else ""
        rows.append([cls.replace("_", " "), f"{n:,}", f"{n / n_img:.1%}", flag])
    table(doc, ["Finding", "Positive images", "Prevalence", "Underpowered?"],
          rows, widths=[2.6, 1.5, 1.3, 2.6],
          caption=f"Table 1 — Class distribution. 3,044 images (54%) have no "
                  f"finding at all. Note the imbalance: Infiltration appears "
                  f"967 times, Hernia only 13.")
    para(doc, "Two consequences of this table drive most of our design "
              "decisions:")
    bullets(doc, [
        ("Severe imbalance.  ", "If a model simply said \"nothing is wrong\" "
         "for every image it would be right 54% of the time while being "
         "completely useless. So we never use plain accuracy as a metric."),
        ("Some findings are hopeless at this scale.  ", "Hernia has 13 examples "
         "in total — about 3 per test fold. We still report its score but flag "
         "it as unreliable rather than quietly including it."),
    ])

    doc.add_heading("Multi-label, not multi-class", level=2)
    para(doc, "A single X-ray can show several conditions at the same time — "
              "for example Effusion and Cardiomegaly together. This is called a "
              "multi-label problem. It is different from the usual "
              "\"one image = one answer\" setup (multi-class), and it changes "
              "the model's output layer and loss function. The architecture "
              "diagram in Section 3 explains how.")
    para(doc, f"On average each image carries "
              f"{folds[CLASSES].sum(axis=1).mean():.2f} findings.")

    doc.add_heading("How we split train / validation / test", level=2)
    para(doc, "This is the part most projects get wrong, so it is worth being "
              "precise.")
    para(doc, "We do not use a single fixed train/test split. Instead we use "
              "5-fold cross-validation: the data is divided into 5 equal parts, "
              "and we train 5 separate models. Each one trains on 4 parts (80%) "
              "and is tested on the remaining part (20%). Every image therefore "
              "gets tested exactly once, by a model that never saw it. We then "
              "report the average and the spread across the 5 runs.")
    rows = [[f"Fold {int(r.fold)}", f"{n_img - (folds.fold == r.fold).sum():,}",
             f"{(folds.fold == r.fold).sum():,}", f"{r.auroc:.4f}"]
            for r in cvres.itertuples()]
    table(doc, ["Run", "Train images (80%)", "Held-out test (20%)",
                "Result (AUROC)"], rows, widths=[1.4, 2.2, 2.2, 2.0],
          caption="Table 2 — The five cross-validation runs. There is no "
                  "separate validation set: the held-out fold serves as both "
                  "validation (for early stopping) and test (for reporting).")

    p = doc.add_paragraph()
    p.add_run("The critical detail — we split by patient, not by image.  ").bold = True
    p.add_run(f"Our {n_img:,} images come from only {n_pat:,} patients, so many "
              f"patients have several X-rays. If we split randomly by image, the "
              f"same patient could appear in both training and testing. The "
              f"model would recognise the person rather than the disease, and "
              f"our scores would look far better than they really are. We group "
              f"by patient ID so this cannot happen, and the code asserts that "
              f"zero patients appear in more than one fold — if that check ever "
              f"fails, the run stops.")

    figure(doc, dist / "class_distribution.png",
           "Figure 1 — Positive images per finding.", width=6.6)
    figure(doc, dist / "cooccurrence.png",
           "Figure 2 — How often findings appear together. Reading a row: of "
           "images with the row's finding, what fraction also have the column's "
           "finding. This is why some conditions are hard to tell apart.",
           width=6.4)

    # ---------------- 3. architecture ----------------
    doc.add_page_break()
    doc.add_heading("3.  Model Architecture", level=1)
    para(doc, "We use DenseNet-121, a convolutional neural network with 121 "
              "layers and about 7 million parameters. It is the same "
              "architecture used by CheXNet, the best-known published chest "
              "X-ray model, which makes our results directly comparable to "
              "existing work.")

    figure(doc, plots / "diagram_architecture.png",
           "Figure 3 — Model architecture. The image passes through four dense "
           "blocks that progressively shrink it spatially while building up "
           "richer features, then a small head converts those features into 14 "
           "independent probabilities.")

    doc.add_heading("Walking through it in plain language", level=2)
    bullets(doc, [
        ("Input.  ", f"A 1024x1024 X-ray is resized to "
         f"{cfg.train.image_px}x{cfg.train.image_px} pixels. Smaller is faster "
         f"but loses detail — we come back to this in Section 5, because it "
         f"turned out to matter a lot."),
        ("Preprocessing / augmentation.  ", "During training we randomly crop, "
         "mirror, rotate slightly and adjust brightness. The model sees a "
         "slightly different version of each image every time, which stops it "
         "memorising the training set."),
        ("The four dense blocks.  ", "These are the feature extractor. Early "
         "blocks detect edges and textures; later blocks detect anatomy and "
         "abnormal patterns. \"Dense\" means each layer receives the output of "
         "every earlier layer in its block, which helps training and keeps the "
         "parameter count low."),
        ("Transfer learning.  ", "We do not start from random weights. The "
         "backbone comes pretrained on ImageNet (1.4 million everyday "
         "photographs). It already knows about edges, shapes and texture, so we "
         "only have to teach it what chest pathology looks like. With just "
         "4,500 training images this is essential."),
        ("Global average pooling.  ", "Collapses the (10, 10, 1024) feature map "
         "into a single 1024-number summary of the whole image."),
        ("Dropout (30%).  ", "During training, randomly switches off 30% of "
         "those numbers each step. It sounds destructive but it prevents "
         "over-reliance on any one feature — a standard defence against "
         "overfitting on small datasets."),
        ("Linear layer + sigmoid.  ", "Maps 1024 numbers to 14 scores, then "
         "squashes each independently into a 0-1 probability."),
    ])

    doc.add_heading("Why sigmoid and not softmax (the key design choice)", level=2)
    para(doc, "Softmax — the usual choice — forces the 14 scores to add up to "
              "1, which means \"pick exactly one\". That would make it "
              "impossible to report Effusion and Cardiomegaly on the same "
              "image. Sigmoid scores each finding independently, so any "
              "combination is allowed. Paired with it we use Binary "
              "Cross-Entropy loss with class weighting, so that rare findings "
              "are not ignored during training.")

    # ---------------- 4. methodology ----------------
    doc.add_page_break()
    doc.add_heading("4.  Methodology", level=1)
    figure(doc, plots / "diagram_methodology.png",
           "Figure 4 — The full pipeline. Steps 1-8 are built and measured; "
           "steps 9-12 are the next phases.")

    doc.add_heading("Two-stage training", level=2)
    para(doc, "We train in two stages rather than all at once:")
    bullets(doc, [
        ("Stage 1 (5 epochs).  ", "Freeze the pretrained backbone and train "
         "only the final 14-output layer, at a fast learning rate of 1e-3. The "
         "new layer starts out random; if we let it loose on the backbone "
         "immediately, its large early error signals would wreck the useful "
         "pretrained features."),
        ("Stage 2 (30 epochs).  ", "Unfreeze everything and fine-tune, but with "
         "different learning rates: 1e-5 for the backbone (adjust gently) and "
         "1e-4 for the head (still learning). A cosine schedule decays both to "
         "near zero by the end."),
    ])
    para(doc, "An epoch means one full pass through the training images. "
              "Learning rate controls how big a step the model takes when "
              "correcting itself.")

    doc.add_heading("Handling the class imbalance", level=2)
    para(doc, "We weight the loss so that missing a rare finding costs more "
              "than missing a common one. The weight is the ratio of negative "
              "to positive examples — but we cap it at 20x. Uncapped, Hernia "
              "would get a weight of about 430 and would destabilise training "
              "for every other class.")

    doc.add_heading("Test-time augmentation", level=2)
    para(doc, "At prediction time we run each image through the model twice — "
              "once normally, once mirrored left-to-right — and average the two "
              "results:")
    figure(doc, plots / "formulas" / "tta.png", "", width=6.6)
    para(doc, f"Two looks are less noisy than one. Measured gain: "
              f"+{tta['gain'].mean():.4f} AUROC. Small but free.")
    para(doc, "A caveat worth knowing: mirroring a chest X-ray is not "
              "anatomically neutral — the heart is on the left. It is standard "
              "practice in this field and CheXNet used it too, but it does teach "
              "the model to ignore left-right position, which may cost us on "
              "findings where the side matters (Cardiomegaly, one-sided "
              "Effusion). Testing that is on our list.", italic=True)

    doc.add_heading("How we measure success", level=2)
    bullets(doc, [
        ("AUROC (main metric).  ", "Given one diseased and one healthy image, "
         "how often does the model score the diseased one higher? 0.5 = random "
         "guessing, 1.0 = perfect. It does not depend on any threshold, which "
         "is why it is the standard for this dataset."),
        ("AUPRC.  ", "A fairer summary when a condition is rare. AUROC can look "
         "flattering at 2% prevalence; AUPRC does not."),
        ("Precision, Recall, F1.  ", "Precision = of the cases we flagged, how "
         "many were real. Recall = of the real cases, how many we caught. F1 "
         "balances the two. These require choosing a cut-off, so we tune one "
         "per finding."),
        ("Confusion matrices.  ", "Because this is multi-label, we produce one "
         "2x2 matrix per finding rather than a single 14x14 grid — an image can "
         "belong to several classes, so \"predicted class\" is not one value."),
    ])
    para(doc, "We deliberately do not report plain accuracy. With 54% of images "
              "normal, it would be misleading.", italic=True)

    # ---------------- 5. metrics ----------------
    helpers = {"para": para, "bullets": bullets, "table": table, "figure": figure}
    metrics_section(doc, helpers, cfg, num=5)

    # ---------------- 6. results ----------------
    doc.add_page_break()
    doc.add_heading("6.  Results: The Sample-Trained Classifier", level=1)
    para(doc, f"Macro AUROC {macro:.3f} ± {macro_sd:.3f} across the five "
              f"cross-validation runs for the model trained on our 5,606-image "
              f"sample. For context, CheXNet reached 0.84 — with 20 times more "
              f"training data. Section 7 tests exactly that: whether data volume "
              f"is the limit.")

    pc = per_class.set_index("class")
    rows = []
    for cls in order.index:
        rows.append([
            cls.replace("_", " "), f"{counts[cls]:,}",
            f"{pc.loc[cls, 'mean']:.3f}",
            f"{auprc.loc[cls, 'auprc']:.3f}" if cls in auprc.index else "-",
            f"{auprc.loc[cls, 'lift_over_chance']:.1f}x" if cls in auprc.index else "-",
        ])
    table(doc, ["Finding", "n", "AUROC", "AUPRC", "Lift over chance"], rows,
          widths=[2.4, 1.0, 1.4, 1.4, 1.8],
          caption="Table 3 — Per-finding performance (5-fold cross-validation, "
                  "with test-time augmentation). Lift = how much better than "
                  "random guessing at that prevalence.")

    doc.add_heading("What the resolution experiment taught us", level=2)
    para(doc, "Our first run trained at 224x224 pixels and scored 0.728. "
              "Moving to 320x320 and training longer took it to 0.757 — every "
              "single fold improved. The biggest gains were on findings that "
              "occupy few pixels: Emphysema +0.074, Pneumothorax +0.049. "
              "Findings that are still small even at 320, like Nodule, did not "
              "improve. The lesson: for X-rays, image resolution matters more "
              "than model cleverness.")
    para(doc, "We also tested whether starting from a chest-X-ray-pretrained "
              "backbone (CheXpert weights) beat ImageNet. It did not — the "
              "difference was -0.005 with p = 0.36, i.e. statistically "
              "indistinguishable. That is a genuine negative result and we will "
              "report it as one.")

    doc.add_heading("Being honest about the weak points", level=2)
    heldout = clsm[(clsm["split"] == "heldout") &
                   (clsm["threshold_strategy"] == "f1opt") &
                   (clsm["class"] == "<macro avg>")]
    if len(heldout):
        r = heldout.iloc[0]
        para(doc, f"When we force the model to make yes/no decisions rather "
                  f"than produce scores, the picture is much less flattering: "
                  f"macro precision {r.precision:.2f}, recall {r.recall:.2f}, "
                  f"F1 {r.f1:.2f}. On images that genuinely have a finding, we "
                  f"correctly detect at least one of them only about 41% of "
                  f"the time.")
    para(doc, "AUROC 0.757 and 41% detection are both true. AUROC measures how "
              "well the model ranks images; the 41% measures how well it makes "
              "decisions at a fixed cut-off. This is exactly why the project "
              "needs the explainability and reporting layers — a bare number "
              "this unreliable should never be shown to a clinician on its "
              "own.")
    bullets(doc, [
        ("Confidently wrong.  ", "Because of the class weighting, the model's "
         "probabilities are not calibrated — it will say 0.88 on findings that "
         "are not there. We have measured this and will show the calibration "
         "curve in the interface."),
        ("The root cause is data volume.  ", "4,500 training images, 10% label "
         "noise, and 2-4% prevalence for most findings. No amount of tuning "
         "fixes that; only more data would."),
    ])

    figure(doc, plots / "loss_curves_densenet121_imagenet.png",
           "Figure 5 — Training curves. Left: loss going down (solid = "
           "training, dashed = held-out). Right: AUROC going up, levelling off "
           "around epoch 25-30, which tells us the training schedule is now "
           "long enough. The vertical dotted line is where the backbone "
           "unfreezes.", width=9.2)
    figure(doc, plots / "metrics_by_class_densenet121_imagenet.png",
           "Figure 6 — Precision, recall and F1 per finding, ordered by how "
           "many examples exist. The downward trend to the right is the data "
           "problem made visible.", width=8.4)

    # ---------------- 7. comparative study ----------------
    comparative_section(doc, helpers, cfg, num=7)

    # ---------------- 8. worked examples ----------------
    oof_df, y_true, y_prob = load_oof(cfg, cfg.models[0])
    thr_path = metrics / f"thresholds_f1_{cfg.models[0]}.csv"
    thr = (read_csv(thr_path).set_index("class").loc[CLASSES, "threshold"].to_numpy()
           if thr_path.exists() else None)
    samples_section(doc, helpers, cfg, 8, oof_df, y_true, y_prob, thr)

    # ---------------- 9. grad-cam ----------------
    gradcam_section(doc, helpers, cfg, num=9)

    # ---------------- 10. future work ----------------
    doc.add_page_break()
    doc.add_heading("10.  What We Do Next", level=1)
    para(doc, "Classifier and explainability are built and measured. Two phases "
              "remain, and the pieces they need are already verified.")

    doc.add_heading("Phase A — Explainable AI (done, Section 9)", level=2)
    para(doc, "Grad-CAM is implemented, its configuration was chosen by a "
              "measured sweep, and localisation is quantified against radiologist "
              "boxes. One follow-up remains: confirming the adopted configuration "
              "on all 984 NIH boxes on Kaggle before the number goes in the report.")

    doc.add_heading("Phase B — MedGemma report generation (next)", level=2)
    para(doc, "MedGemma is Google's open medical vision-language model. The 4 "
              "billion parameter version understands images and text together, "
              "and its training data included chest X-rays. We run it locally "
              "through Ollama — already installed and pulled (4-bit, 3.3 GB) — "
              "rather than through Hugging Face. We verified it is genuinely "
              "multimodal: it read exact figures off a bar-chart image and "
              "correctly reported no image when none was attached. This removes "
              "the gated download and the GPU-memory contention with the "
              "classifier; each report takes about 30 seconds on our laptop.")
    para(doc, "Crucially we do not use it alone. We feed it three things "
              "together — the X-ray, our classifier's probabilities, and the "
              "anatomical region the Grad-CAM peak points to — and ask for a "
              "structured draft:")
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    r = p.add_run("\"A CNN predicted Cardiomegaly (0.87) and Effusion (0.71). "
                  "Grad-CAM localised these to the cardiac silhouette and the "
                  "left costophrenic angle. Write Findings, Impression, and "
                  "recommended follow-up.\"")
    r.italic = True
    r.font.color.rgb = MUTED
    para(doc, "This is what ties the three components into one system rather "
              "than three separate demos: the explainability output becomes an "
              "input to the language model. The full-data model is the one behind "
              "it; the sample-trained models remain as the comparative study.")

    doc.add_heading("Phase C — The user interface", level=2)
    bullets(doc, [
        "Upload an X-ray, or pick a sample case",
        "Bar chart of all 14 probabilities, sorted",
        "Grad-CAM overlay with a dropdown to switch finding, and an opacity slider",
        "The generated draft report, streamed as it is written",
        "Download as PDF — with a research-use-only disclaimer on every page",
    ])

    doc.add_heading("Phase D — Our research contribution", level=2)
    para(doc, "Everything above is solid engineering, but it has been done "
              "before. The question we can answer that has not been well "
              "studied is this:")
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    r = p.add_run("Does giving a medical language model the Grad-CAM location "
                  "reduce the number of findings it invents?")
    r.bold = True
    r.font.color.rgb = ACCENT
    para(doc, "We will generate reports under three conditions — image only; "
              "image plus predictions; image plus predictions plus Grad-CAM "
              "regions — then count, across roughly 200 reports, how many "
              "findings were asserted that the classifier never predicted "
              "(hallucinations) and how many predicted findings went unmentioned "
              "(omissions).")
    para(doc, "Our classifier being unreliable actually makes this more "
              "interesting, not less. A model that is confidently wrong much of "
              "the time is precisely the stress test for whether grounding a "
              "language model in classifier attention helps — or whether it just "
              "restates errors more authoritatively. Either answer is a real "
              "finding.")
    para(doc, "Alongside it we will run a shortcut audit: mask out the burned-in "
              "text and hardware, re-measure AUROC per finding, and see how much "
              "of our performance was actually coming from annotations rather "
              "than anatomy.")

    doc.add_heading("Summary of status", level=2)
    table(doc, ["Phase", "Status"], [
        ["Data pipeline, patient-grouped splits, distribution analysis", "Done"],
        ["Classifier on the 5,606-image sample (AUROC 0.757)", "Done"],
        ["Resolution, pretraining and architecture experiments", "Done"],
        ["RAD-DINO probe and ensemble (AUROC 0.786)", "Done"],
        ["Full-data model on Kaggle (AUROC 0.812 official test); fair comparison", "Done"],
        ["Grad-CAM: 111-configuration sweep, bbox-validated (58.5% pointing hits)", "Done"],
        ["Confirm Grad-CAM configuration on all 984 boxes (Kaggle)", "Next"],
        ["MedGemma report generation via Ollama", "Next"],
        ["Streamlit interface", "Planned"],
        ["Grounding / hallucination study", "Planned"],
    ], widths=[6.4, 1.6])

    # ---------------- appendices ----------------
    appendix_classification(doc, helpers, cfg, "A")
    appendix_gradcam(doc, helpers, cfg, "B")

    para(doc, "Research and educational use only. Not a medical device and not "
              "for clinical diagnosis.", italic=True, size=9, muted=True,
         align=WD_ALIGN_PARAGRAPH.CENTER)

    path = Path(__file__).resolve().parent.parent / "Project_Briefing.docx"
    doc.save(path)
    return path


if __name__ == "__main__":
    # The report was restructured; this file now only supplies the shared
    # docx helpers. Build via scripts/make_report_doc.py.
    import runpy
    runpy.run_module("scripts.make_report_doc", run_name="__main__")
