"""Draw the architecture and methodology diagrams used in the briefing doc.

    python scripts/make_diagrams.py

Written for an audience new to deep learning, so every box carries both the
plain-language role and the technical detail (tensor shapes, layer names).

Outputs to outputs/plots/:
    diagram_architecture.png
    diagram_methodology.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, read_csv# noqa: E402

INK = "#1c2430"
MUTED = "#5b6673"
PALETTE = {
    "input": ("#e8f0f8", "#3b6ea5"),
    "prep": ("#e6f2ec", "#3f8f68"),
    "model": ("#e9ecf7", "#4a5aa8"),
    "head": ("#fdf0e3", "#d1873c"),
    "out": ("#f6e9ef", "#a84a6b"),
    "future": ("#f2f0f7", "#7a6fa8"),
}


def box(ax, x, y, w, h, title, sub="", kind="model", fs=10, sfs=8.2, dashed=False):
    face, edge = PALETTE[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.7, edgecolor=edge, facecolor=face,
        linestyle="--" if dashed else "-", zorder=2))
    if not sub:
        ax.text(x + w / 2, y + h * 0.5, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=INK, zorder=3)
        return
    # Split the box by line count so a 5-line body cannot run into the heading.
    n_title = title.count("\n") + 1
    n_sub = sub.count("\n") + 1
    title_share = n_title / (n_title + n_sub * 0.95)
    ax.text(x + w / 2, y + h * (1 - title_share * 0.62), title, ha="center",
            va="center", fontsize=fs, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h * (1 - title_share) * 0.52, sub, ha="center",
            va="center", fontsize=sfs, color=MUTED, zorder=3, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, label="", colour=None, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
        linewidth=1.6, color=colour or "#8a929e", zorder=1,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.018, label, ha="center",
                va="bottom", fontsize=7.6, color=MUTED, zorder=3)


def canvas(w, h, title, subtitle):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.975, title, ha="center", va="top", fontsize=15,
            fontweight="bold", color=INK)
    ax.text(0.5, 0.928, subtitle, ha="center", va="top", fontsize=9.5,
            color=MUTED)
    return fig, ax


def architecture(path: Path, px: int) -> None:
    fig, ax = canvas(13.2, 7.2,
                     "Model Architecture — DenseNet-121 Multi-Label Classifier",
                     "One image in, fourteen independent probabilities out. "
                     "Numbers in brackets are tensor shapes.")

    y, h = 0.60, 0.20
    box(ax, 0.025, y, 0.14, h, "Chest X-ray",
        f"grayscale PNG\n1024 x 1024\n→ resized {px} x {px}", "input")
    box(ax, 0.195, y, 0.145, h, "Preprocess",
        "random crop, flip\nrotate, brightness\nImageNet normalise", "prep")

    # DenseNet blocks
    bx, bw, gap = 0.375, 0.098, 0.014
    blocks = [("Dense\nBlock 1", "6 layers\n(80, 80)"),
              ("Dense\nBlock 2", "12 layers\n(40, 40)"),
              ("Dense\nBlock 3", "24 layers\n(20, 20)"),
              ("Dense\nBlock 4", "16 layers\n(10, 10, 1024)")]
    for i, (t, s) in enumerate(blocks):
        box(ax, bx + i * (bw + gap), y, bw, h, t, s, "model", fs=9.3, sfs=7.6)
        if i:
            arrow(ax, bx + i * (bw + gap) - gap, y + h / 2,
                  bx + i * (bw + gap), y + h / 2)
    ax.add_patch(FancyBboxPatch(
        (bx - 0.012, y - 0.028), 4 * bw + 3 * gap + 0.024, h + 0.056,
        boxstyle="round,pad=0.006,rounding_size=0.02", linewidth=1.2,
        edgecolor="#4a5aa8", facecolor="none", linestyle=":", zorder=0))
    ax.text(bx + (4 * bw + 3 * gap) / 2, y - 0.075,
            "Pretrained backbone (ImageNet) — frozen in stage 1, fine-tuned in stage 2",
            ha="center", fontsize=8.4, color="#4a5aa8", style="italic")

    arrow(ax, 0.165, y + h / 2, 0.195, y + h / 2)
    arrow(ax, 0.34, y + h / 2, 0.375, y + h / 2)

    # Head
    hy = 0.285
    box(ax, 0.375, hy, 0.145, 0.16, "Global Avg Pool",
        "(10,10,1024)\n→ (1024,)", "head", fs=9.6)
    box(ax, 0.545, hy, 0.115, 0.16, "Dropout", "p = 0.3", "head", fs=9.6)
    box(ax, 0.685, hy, 0.145, 0.16, "Linear", "1024 → 14", "head", fs=9.6)
    box(ax, 0.855, hy, 0.12, 0.16, "Sigmoid",
        "14 independent\nprobabilities", "out", fs=9.6)
    arrow(ax, 0.447, y - 0.032, 0.447, hy + 0.16, rad=0.0)
    arrow(ax, 0.52, hy + 0.08, 0.545, hy + 0.08)
    arrow(ax, 0.66, hy + 0.08, 0.685, hy + 0.08)
    arrow(ax, 0.83, hy + 0.08, 0.855, hy + 0.08)

    # Output examples
    box(ax, 0.63, 0.045, 0.345, 0.175, "Output example",
        "Effusion        0.82\nCardiomegaly    0.71\nAtelectasis     0.34\n"
        "... 11 more, each scored independently", "out", fs=9.6, sfs=8.4)
    arrow(ax, 0.915, hy, 0.915, 0.22)

    # Why sigmoid, not softmax
    box(ax, 0.025, 0.045, 0.56, 0.175, "Why 14 sigmoids and not softmax?",
        "An X-ray can show several findings at once — Effusion AND Cardiomegaly.\n"
        "Softmax forces the 14 scores to sum to 1 (pick exactly one class).\n"
        "Sigmoid scores each finding on its own, so any combination is possible.\n"
        "This makes it MULTI-LABEL, not multi-class.", "prep", fs=10, sfs=8.6)

    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def methodology(path: Path, px: int, n_img: int, n_pat: int) -> None:
    fig, ax = canvas(13.2, 8.4, "Project Methodology — End-to-End Pipeline",
                     "Solid boxes = built and measured.  Purple dashed = planned next phases.")

    row = [0.755, 0.545, 0.335]
    w, h = 0.205, 0.135

    # Row 1 - data
    box(ax, 0.03, row[0], w, h, "1. Data",
        f"NIH ChestX-ray14 sample\n{n_img:,} images, {n_pat:,} patients\n14 findings, CC0", "input")
    box(ax, 0.275, row[0], w, h, "2. Preprocess",
        f"resize to 352 px cache\ntrain at {px} px\naugment: crop/flip/rotate", "prep")
    box(ax, 0.52, row[0], w, h, "3. Patient-grouped\n5-fold split",
        "same patient never in\ntrain and test together\n(asserted in code)", "prep")
    box(ax, 0.765, row[0], w, h, "4. Class imbalance",
        "BCE loss with pos_weight\ncapped at 20x\n(Hernia: 13 positives)", "prep")
    for x in (0.235, 0.48, 0.725):
        arrow(ax, x, row[0] + h / 2, x + 0.04, row[0] + h / 2)

    # Row 2 - training
    box(ax, 0.03, row[1], w, h, "5. Stage 1 training",
        "backbone frozen\ntrain the 14-output head\n5 epochs, lr 1e-3", "model")
    box(ax, 0.275, row[1], w, h, "6. Stage 2 training",
        "unfreeze everything\nlr 1e-5 body / 1e-4 head\n30 epochs, cosine decay", "model")
    box(ax, 0.52, row[1], w, h, "7. Test-time\naugmentation",
        "average normal + flipped\nprediction\n(+0.004 AUROC)", "model")
    box(ax, 0.765, row[1], w, h, "8. Evaluation",
        "AUROC, AUPRC, P/R/F1\nconfusion matrices\nper-class thresholds", "head")
    arrow(ax, 0.885, row[0], 0.885, row[0] - 0.03)
    arrow(ax, 0.885, row[0] - 0.03, 0.132, row[1] + h, rad=-0.06)
    for x in (0.235, 0.48, 0.725):
        arrow(ax, x, row[1] + h / 2, x + 0.04, row[1] + h / 2)

    # Result callout
    box(ax, 0.03, row[2], 0.45, h, "Result so far",
        "Macro AUROC 0.757 ± 0.022 (5-fold CV)\n"
        "Best findings: Emphysema 0.84, Pneumothorax 0.84, Effusion 0.82\n"
        "Weakest: Nodule 0.65, Pneumonia 0.62 (too few pixels / too few cases)",
        "out", fs=10.5, sfs=8.6)
    arrow(ax, 0.885, row[1], 0.885, row[1] - 0.03)
    arrow(ax, 0.885, row[1] - 0.03, 0.255, row[2] + h, rad=-0.06)

    # Row 3 - future
    fy, fw = 0.09, 0.215
    box(ax, 0.03, fy, fw, 0.185, "9. Explainable AI\n(Grad-CAM)",
        "Heatmap over the X-ray showing\nWHICH PIXELS drove each finding.\n\n"
        "Validated against radiologist\nbounding boxes (IoU, hit rate)\n"
        "— measured, not just shown.", "future", fs=10.5, sfs=8.4, dashed=True)
    box(ax, 0.285, fy, fw, 0.185, "10. MedGemma\nreport generation",
        "Google's medical vision-language\nmodel (4B, 4-bit on 8 GB GPU).\n\n"
        "Fed the image + our predictions\n+ Grad-CAM regions → drafts a\n"
        "Findings / Impression report.", "future", fs=10.5, sfs=8.4, dashed=True)
    box(ax, 0.54, fy, fw, 0.185, "11. Streamlit UI",
        "Upload an X-ray → see:\n• probability bars (14 findings)\n"
        "• Grad-CAM overlay per finding\n• the generated draft report\n"
        "• download as PDF", "future", fs=10.5, sfs=8.4, dashed=True)
    box(ax, 0.795, fy, fw, 0.185, "12. Novelty study",
        "Does giving the language model\nthe Grad-CAM location reduce\n"
        "hallucinated findings?\n\nMeasured as hallucination and\n"
        "omission rate over 200 reports.", "future", fs=10.5, sfs=8.4, dashed=True)
    for x in (0.245, 0.5, 0.755):
        arrow(ax, x, fy + 0.0925, x + 0.04, fy + 0.0925,
              colour="#7a6fa8", dashed=True)
    arrow(ax, 0.255, row[2], 0.137, fy + 0.185, colour="#7a6fa8", dashed=True)

    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    plots = Path(cfg.dirs.plots)
    import pandas as pd
    folds = read_csv(Path(cfg.dirs.root) / "folds.csv")

    architecture(plots / "diagram_architecture.png", cfg.train.image_px)
    methodology(plots / "diagram_methodology.png", cfg.train.image_px,
                len(folds), folds["patient"].nunique())
    print(f"wrote diagram_architecture.png and diagram_methodology.png -> {plots}")


if __name__ == "__main__":
    main()
