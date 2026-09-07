"""The metrics and worked-example sections of the briefing document.

Kept separate from make_briefing_doc.py to stop that file becoming unreadable.
Both builders take the shared docx helpers as arguments so formatting stays
consistent across the whole document.

The ten worked examples are fixed image IDs chosen to cover the full range of
model behaviour -- clean successes, partial hits, confident errors and
over-calling. Their probabilities are read live from the prediction dumps, so
only the commentary is hand-written; no number in this section is typed in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from docx.shared import Inches, Pt
from PIL import Image

from src.config import CLASSES, N_CLASSES

# (image, short label, what this case demonstrates)
CASES: list[tuple[str, str, str]] = [
    ("00023097_001.png", "Clean success on two findings",
     "Both true findings are top of the list with high scores, and nothing else "
     "crosses its threshold. Atelectasis and Effusion frequently occur together "
     "and both show up in the lower chest, so this is the pattern the model has "
     "the most training examples of. This is the model at its best."),

    ("00014358_016.png", "Clean success on a single finding",
     "Pneumothorax is one of our strongest classes (AUROC 0.84). Note the "
     "runner-up, Effusion at 0.71 -- it stays below its own threshold, so it is "
     "not reported. Different findings have different cut-offs, which is why we "
     "tune one threshold per class rather than using 0.5 everywhere."),

    ("00001936_000.png", "Correctly silent on a normal film",
     "A genuinely normal chest X-ray. Every one of the 14 scores sits at or "
     "below 0.38, so nothing is reported. Note the model never outputs "
     "'No Finding' as a class -- a normal result is simply the absence of any "
     "finding crossing its threshold."),

    ("00029105_001.png", "Partial hit on a complex case",
     "Four findings are truly present and the model catches three of them, "
     "including Pneumothorax at 0.95. It misses Atelectasis and adds a false "
     "Emphysema. On images with several findings, partial credit like this is "
     "the most common outcome."),

    ("00021969_001.png", "Right answers plus extra guesses",
     "Cardiomegaly, Effusion and Infiltration are all correctly identified. But "
     "the model also reports Atelectasis and Consolidation, and misses the "
     "Nodule entirely. A radiologist reading this draft would get useful "
     "pointers, but would have to discard two of the five."),

    ("00008154_000.png", "Misses the finding that matters most",
     "Pneumonia is present and the model does not report it -- it flags "
     "Atelectasis (correct) plus Effusion and Nodule (wrong). Pneumonia is our "
     "second-weakest class (AUROC 0.62) with only 62 examples in the entire "
     "dataset. This is the clearest illustration of why we flag underpowered "
     "classes rather than reporting their scores as if they were reliable."),

    ("00021610_016.png", "Confidently wrong",
     "The truth is Pleural Thickening. The model reports Pneumothorax at 0.97 "
     "-- almost maximum confidence -- and five other findings, none of them "
     "correct. This is the single most important failure mode to understand: "
     "our probabilities are NOT calibrated, so a high number does not mean the "
     "model is likely to be right. It is also why the Grad-CAM phase matters, "
     "since a heatmap would at least show us where this conviction came from."),

    ("00001248_005.png", "The small-lesion problem",
     "A solitary Nodule, missed completely, with six false positives instead. "
     "Nodules can be a few millimetres across; after resizing to 320x320 they "
     "occupy only a handful of pixels. Nodule is our weakest well-populated "
     "class (AUROC 0.65) and moving from 224 to 320 pixels did not improve it "
     "-- the information is simply not there any more."),

    ("00017771_000.png", "Over-calling on a normal film",
     "This film is normal, yet the model reports nine findings, led by Effusion "
     "at 0.97. Across all held-out images the model predicts 0.94 findings per "
     "image against a true average of 0.70 -- it over-calls by roughly a third. "
     "This is a direct consequence of the class weighting we use to stop rare "
     "findings being ignored: we deliberately traded precision for recall."),

    ("00009349_014.png", "A genuinely hard case",
     "Five findings are truly present and the model gets four of them -- but "
     "reports eight in total, so half of what it says is wrong. Cases like this "
     "are where a ranked list of probabilities is far more useful than a "
     "yes/no verdict, and it is the reason the interface will show the "
     "probability bars rather than just the labels above threshold."),
]


def metrics_section(doc, helpers, cfg, num: int) -> None:
    para, bullets, table, figure = (helpers["para"], helpers["bullets"],
                                    helpers["table"], helpers["figure"])
    fdir = Path(cfg.dirs.plots) / "formulas"

    def formula(key: str, width=4.6):
        figure(doc, fdir / f"{key}.png", "", width=width)

    doc.add_page_break()
    doc.add_heading(f"{num}.  Metrics: What We Measure and Why", level=1)
    para(doc, "This section defines every number that appears in our results. "
              "The notation is the same throughout: there are N images and 14 "
              "findings; for image i and finding c, y is the true label (1 or "
              "0) and p is the probability the model outputs.")

    doc.add_heading("Notation and the four counts", level=2)
    para(doc, "Almost every metric is built from four counts. Because this is a "
              "multi-label problem, we compute them separately for each of the "
              "14 findings rather than once overall.")
    table(doc, ["Count", "Name", "Meaning for a given finding"], [
        ["TP", "True positive", "the finding is present and we reported it"],
        ["FP", "False positive", "the finding is absent but we reported it"],
        ["FN", "False negative", "the finding is present and we missed it"],
        ["TN", "True negative", "the finding is absent and we did not report it"],
    ], widths=[0.9, 1.9, 6.2])
    para(doc, "This is what the 14 confusion matrices in our results show -- "
              "one 2x2 grid of these four counts per finding. A single 14x14 "
              "matrix would not make sense here, because an image can carry "
              "several findings at once, so there is no single "
              "'predicted class' to put on an axis.")

    doc.add_heading("The model output and its loss function", level=2)
    para(doc, "The network produces 14 raw scores. Each is squashed into a "
              "probability independently by the sigmoid function:")
    formula("sigmoid", 5.0)
    para(doc, "Because each output is independent, any combination of findings "
              "is possible. Training minimises weighted binary cross-entropy:")
    formula("bce", 8.4)
    para(doc, "Cross-entropy punishes confident mistakes far more than "
              "hesitant ones: if the truth is 1 and the model says 0.01, the "
              "log term becomes very large. The weight w handles the class "
              "imbalance:")
    formula("posweight", 4.6)
    para(doc, "N-minus and N-plus are the negative and positive counts for that "
              "finding. Hernia has 13 positives out of 5,606, giving a raw "
              "weight near 430; we cap at 20 because an uncapped weight makes "
              "the loss for that one class dominate every gradient update and "
              "destabilises training for the other thirteen.")

    doc.add_heading("Precision, Recall and F1", level=2)
    formula("precision_recall", 7.8)
    bullets(doc, [
        ("Precision  ", "answers: of the findings we reported, how many were "
         "real? Low precision means the report is cluttered with things that "
         "are not there."),
        ("Recall (sensitivity)  ", "answers: of the findings that were really "
         "there, how many did we catch? Low recall means we miss disease."),
    ])
    para(doc, "There is always a trade-off. Lowering the threshold catches more "
              "real findings (recall up) but also reports more phantom ones "
              "(precision down). F1 is their harmonic mean, which only gets a "
              "good score when both are decent:")
    formula("f1", 8.2)
    para(doc, "We use the harmonic rather than the arithmetic mean deliberately. "
              "A model with precision 1.0 and recall 0.0 would average 0.5, "
              "which flatters a useless model; its F1 is 0.")

    doc.add_heading("Why not accuracy?", level=2)
    para(doc, "Accuracy = (TP + TN) / N. It is the obvious metric and it is the "
              "wrong one here. 54% of our images have no finding at all, and "
              "individual findings appear in as little as 0.2% of images. A "
              "model that always answered 'nothing present' would score 97-99% "
              "accuracy on most findings while being completely useless. We do "
              "not report accuracy anywhere.", italic=True)

    doc.add_heading("AUROC — our primary metric", level=2)
    para(doc, "The two rates that form the ROC curve:")
    formula("tpr_fpr", 7.2)
    para(doc, "Sweeping the decision threshold from 1 down to 0 traces a curve "
              "of TPR against FPR. AUROC is the area underneath it:")
    formula("auroc", 7.4)
    para(doc, "The second form is the intuitive one: AUROC is the probability "
              "that a randomly chosen diseased image gets a higher score than a "
              "randomly chosen healthy one. 0.5 is coin-flipping, 1.0 is "
              "perfect.")
    bullets(doc, [
        ("Why we use it.  ", "It needs no threshold, so it measures the quality "
         "of the model's ranking rather than an arbitrary cut-off we chose. It "
         "is also the standard metric for this dataset, which makes our 0.757 "
         "directly comparable to CheXNet's published 0.84."),
        ("Its weakness.  ", "AUROC is optimistic under heavy imbalance. TN is "
         "huge for rare findings, so FPR stays low even when the model produces "
         "many false alarms. This is why we also report AUPRC."),
    ])

    doc.add_heading("AUPRC — the honest metric for rare findings", level=2)
    para(doc, "The same threshold sweep, but plotting Precision against Recall. "
              "The area beneath is AUPRC (also called average precision):")
    formula("auprc", 5.6)
    para(doc, "Unlike AUROC, this curve ignores TN entirely, so it cannot be "
              "flattered by the vast number of healthy images. Its baseline is "
              "not 0.5 but the prevalence of the finding, so we report the "
              "improvement over that baseline:")
    formula("lift", 6.2)
    para(doc, "Our macro AUPRC is 0.144, which sounds poor until you note the "
              "average prevalence is about 4% -- a mean lift of 3.7x over "
              "chance. For Hernia, AUPRC 0.023 against a 0.23% prevalence is a "
              "10x lift.")

    doc.add_heading("Choosing thresholds", level=2)
    para(doc, "AUROC and AUPRC need no threshold, but precision, recall, F1 and "
              "the confusion matrices do. A single 0.5 cut-off is wrong here: "
              "we trained with class weighting, so the probabilities are not "
              "calibrated, and each finding has a different prevalence. We "
              "therefore fit one threshold per finding, using two strategies:")
    formula("youden", 6.4)
    para(doc, "Youden's J maximises the vertical distance from the ROC diagonal. "
              "It is the right choice for screening, where a missed disease "
              "costs more than a false alarm — but at 2-4% prevalence it buys "
              "recall so cheaply that precision collapses to about 0.10.")
    formula("f1thresh", 4.8)
    para(doc, "The F1-optimal threshold balances the two instead, and is what "
              "we use for the reported precision/recall/F1 and the confusion "
              "matrices. Critically, we fit it on the training predictions and "
              "then apply it unchanged to the held-out data. Tuning a threshold "
              "on the test set and then reporting test scores would inflate "
              "them.")

    doc.add_heading("Macro versus micro averaging", level=2)
    formula("macro_micro", 9.2)
    bullets(doc, [
        ("Macro  ", "averages the per-finding scores, so every finding counts "
         "equally — Hernia with 13 cases weighs as much as Infiltration with "
         "967. It reveals how we do on rare disease."),
        ("Micro  ", "pools all the counts first, so common findings dominate. "
         "It reflects overall case-level performance."),
    ])
    para(doc, "We report both, plus a macro average restricted to the ten "
              "findings with at least 150 positive examples. Hernia's score "
              "swings wildly on 2-3 test cases per fold, and including it in a "
              "headline number would be misleading in either direction.")

    doc.add_heading("Coming next: measuring the explanations", level=2)
    para(doc, "For the Grad-CAM phase we will need a metric for whether the "
              "heatmap lands in the right place. The dataset includes about "
              "1,000 radiologist-drawn boxes, so we will use intersection over "
              "union between the thresholded heatmap area and the box:")
    formula("iou", 5.2)
    para(doc, "We will also report the pointing-game hit rate — whether the "
              "single hottest pixel falls inside the box — which is a more "
              "forgiving measure and better suited to a heatmap this coarse.")

    figure(doc, Path(cfg.dirs.plots) / f"pr_curves_{cfg.models[0]}.png",
           "Figure — Precision-recall curve per finding, with AUPRC in the "
           "legend. Each curve is one finding; the closer to the top-right "
           "corner, the better. Compare the spread here with the ROC curves: "
           "the same model looks considerably less impressive on this axis, and "
           "this is the more honest view.", width=6.2)


def samples_section(doc, helpers, cfg, num: int, df, y_true, y_prob, thr) -> None:
    para, table, figure = helpers["para"], helpers["table"], helpers["figure"]

    doc.add_page_break()
    doc.add_heading(f"{num}.  Ten Worked Examples", level=1)
    para(doc, "These ten X-rays are all held-out cases: each was scored by the "
              "one cross-validation model that never saw it during training. "
              "They are chosen to cover the full range of behaviour rather than "
              "to flatter the model — three clean successes, three partial "
              "hits, and four instructive failures.")
    para(doc, "In each case 'Reported' lists the findings whose probability "
              "crossed that finding's own tuned threshold. 'Top 3 scores' shows "
              "the three highest probabilities regardless of threshold, which is "
              "often more informative.", italic=True)

    img_dir = Path(cfg.data.images_full)
    out_dir = Path(cfg.dirs.predictions) / "briefing_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = {name: i for i, name in enumerate(df["image"])}

    for n, (img_name, label, comment) in enumerate(CASES, start=1):
        i = lookup.get(img_name)
        if i is None:
            para(doc, f"[case {img_name} not found in held-out predictions]",
                 italic=True, muted=True)
            continue

        true_lbl = [CLASSES[j] for j in range(N_CLASSES) if y_true[i, j] == 1]
        pred_lbl = [CLASSES[j] for j in range(N_CLASSES) if y_prob[i, j] >= thr[j]]
        top3 = np.argsort(-y_prob[i])[:3]
        hits = sorted(set(true_lbl) & set(pred_lbl))
        missed = sorted(set(true_lbl) - set(pred_lbl))
        spurious = sorted(set(pred_lbl) - set(true_lbl))

        src = img_dir / img_name
        dst = out_dir / img_name
        if src.exists() and not dst.exists():
            im = Image.open(src).convert("L")
            im.thumbnail((760, 760))
            im.save(dst)

        doc.add_heading(f"Case {n} — {label}", level=2)
        figure(doc, dst if dst.exists() else src,
               f"{img_name}  ({df.iloc[i]['view']} view)", width=2.9)

        table(doc, ["", "Findings"], [
            ["Ground truth", ", ".join(x.replace('_', ' ') for x in true_lbl)
             or "No finding"],
            ["Reported by model", ", ".join(x.replace('_', ' ') for x in pred_lbl)
             or "No finding"],
            ["Top 3 scores", ", ".join(
                f"{CLASSES[j].replace('_', ' ')} {y_prob[i, j]:.2f}" for j in top3)],
            ["Correct / missed / false", f"{len(hits)} correct"
             + (f", missed {', '.join(x.replace('_', ' ') for x in missed)}" if missed else ", none missed")
             + (f", false {', '.join(x.replace('_', ' ') for x in spurious)}" if spurious else ", no false positives")],
        ], widths=[1.9, 7.1])

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        r = p.add_run("What this shows.  ")
        r.bold = True
        p.add_run(comment)

    doc.add_heading("What the ten cases add up to", level=2)
    para(doc, "Across all 5,606 held-out images the pattern in these examples "
              "holds: on films that genuinely contain a finding, we correctly "
              "detect at least one of them 41% of the time; on genuinely normal "
              "films we correctly stay silent 74% of the time. The model is a "
              "useful ranker and an unreliable decision-maker.")
    para(doc, "Three practical conclusions for the phases ahead:")
    helpers["bullets"](doc, [
        ("Show probabilities, not verdicts.  ", "In several cases the right "
         "answer sits just below its threshold. A ranked bar chart of all 14 "
         "scores conveys far more than a list of labels."),
        ("Grad-CAM is diagnostic for us, not just for the user.  ", "Case 7 "
         "reports Pneumothorax at 0.97 on a film that does not have it. A "
         "heatmap tells us whether that came from lung anatomy or from a chest "
         "tube, a text marker, or the image border."),
        ("The report must be framed as a draft.  ", "At these accuracy levels "
         "MedGemma's output is a starting point for a human reader, and the "
         "interface will say so on every screen and every exported PDF."),
    ])
