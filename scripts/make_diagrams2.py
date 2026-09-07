"""Additional architecture diagrams for the report.

    python scripts/make_diagrams2.py

Reuses the drawing helpers and palette from make_diagrams.py so every diagram
in the document looks like it came from the same hand.

Outputs to outputs/plots/:
    diagram_raddino_probe.png    frozen RAD-DINO features + linear head + ensemble
    diagram_cls_pipeline.png     both training regimes and the shared evaluation rule
    diagram_gradcam.png          how a LayerCAM map is computed and how it is scored
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.make_diagrams import MUTED, arrow, box, canvas  # noqa: E402
from src.config import load_config  # noqa: E402


def raddino_probe(path: Path) -> None:
    fig, ax = canvas(13.2, 6.6,
                     "Second Classifier: Frozen RAD-DINO Features + Linear Probe, and the Ensemble",
                     "Nothing inside the dotted box is trained. Only the 768 -> 14 linear layer "
                     "learns from our data (10,766 parameters).")
    y, h = 0.60, 0.20
    box(ax, 0.02, y, 0.115, h, "Chest X-ray", "1024 x 1024\n-> 518 x 518", "input", fs=9.6, sfs=7.6)
    box(ax, 0.16, y, 0.105, h, "Patchify", "14 x 14 pixel\npatches -> 1,369\ntokens + CLS",
        "prep", fs=9.6, sfs=7.6)
    bx, bw, gap = 0.29, 0.046, 0.006
    labels = ["Attn +\nMLP 1", "2", "3", "...", "11", "12"]
    for i, lbl in enumerate(labels):
        box(ax, bx + i * (bw + gap), y, bw, h, lbl, "", "model", fs=8.6)
        if i:
            arrow(ax, bx + i * (bw + gap) - gap, y + h / 2, bx + i * (bw + gap), y + h / 2)
    span = 6 * bw + 5 * gap
    ax.add_patch(FancyBboxPatch((0.15, y - 0.03), (bx - 0.15) + span + 0.012, h + 0.06,
                                boxstyle="round,pad=0.006,rounding_size=0.02", linewidth=1.3,
                                edgecolor="#4a5aa8", facecolor="none", linestyle=":", zorder=0))
    ax.text(0.15 + ((bx - 0.15) + span) / 2, y - 0.078,
            "RAD-DINO (Microsoft, 2024): ViT-B/14, 86 M parameters, FROZEN. Self-supervised on "
            "880 k chest X-rays\n(MIMIC-CXR, CheXpert, NIH, PadChest, BRAX); no labels ever seen.",
            ha="center", va="top", fontsize=8.2, color="#4a5aa8", style="italic")
    arrow(ax, 0.135, y + h / 2, 0.16, y + h / 2)
    arrow(ax, 0.265, y + h / 2, 0.29, y + h / 2)

    xa = bx + span + 0.012
    box(ax, xa + 0.014, y, 0.105, h, "CLS token",
        "one 768-d vector\nper image\n(cached to disk once)", "head", fs=9.6, sfs=7.6)
    arrow(ax, xa, y + h / 2, xa + 0.014, y + h / 2)
    box(ax, xa + 0.133, y, 0.105, h, "Standardise",
        "per-feature\nmean / std from\ntraining fold", "head", fs=9.6, sfs=7.6)
    arrow(ax, xa + 0.119, y + h / 2, xa + 0.133, y + h / 2)
    box(ax, xa + 0.252, y, 0.11, h, "Linear 768->14\n+ sigmoid",
        "weighted BCE,\nsame loss as CNN,\nearly-stopped", "out", fs=9.4, sfs=7.6)
    arrow(ax, xa + 0.238, y + h / 2, xa + 0.252, y + h / 2)

    by, bh = 0.08, 0.22
    box(ax, 0.025, by, 0.30, bh, "Feature-level test-time augmentation",
        "The flipped image is embedded too. Training uses both\n"
        "embeddings (doubling the data); at test time the two\n"
        "probability vectors are averaged.", "prep", fs=10, sfs=8.4)
    box(ax, 0.355, by, 0.30, bh, "Why it works with 5,606 images",
        "The backbone already encodes lung anatomy from 880 k films.\n"
        "Fitting 10,766 weights cannot overfit the way fine-tuning\n"
        "7 M can. It tied the fully fine-tuned CNN (0.755 vs 0.757).", "model", fs=10, sfs=8.4)
    box(ax, 0.685, by, 0.29, bh, "Ensemble = mean(CNN, probe)",
        "The two err on different findings: the probe wins on\n"
        "Mass / Nodule / Cardiomegaly, the CNN on Infiltration /\n"
        "Consolidation. Averaging gives +0.028 AUROC, p = 0.001.", "out", fs=10, sfs=8.4)
    arrow(ax, xa + 0.307, y, 0.83, by + bh, colour="#a84a6b")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def cls_pipeline(path: Path) -> None:
    fig, ax = canvas(13.2, 7.4,
                     "Classification Pipeline: Two Training Regimes, One Evaluation Protocol",
                     "Top: everything trained on our laptop from the 5,606-image sample. "
                     "Bottom: the same architecture trained on 112,120 images on a free Kaggle GPU.")
    w, h = 0.19, 0.15
    box(ax, 0.02, 0.58, 0.16, 0.30, "NIH ChestX-ray14",
        "112,120 films\n30,805 patients\n14 findings; labels\ntext-mined (~90 % acc.)",
        "input", fs=10.5, sfs=8.2)
    ty = 0.68
    box(ax, 0.22, ty, w, h, "5 % sample", "5,606 films\n4,230 patients", "input")
    box(ax, 0.44, ty, w, h, "Patient-grouped\n5-fold CV", "each film held out\nexactly once", "prep")
    box(ax, 0.66, ty, w, h, "Fine-tune @ 320 px", "DenseNet-121, 35 epochs\n5 folds x 25 min, RTX 4060", "model")
    box(ax, 0.66, ty - 0.20, w, h, "RAD-DINO probe", "frozen ViT-B/14\n+ linear layer", "model")
    box(ax, 0.66, ty - 0.40, w, h, "Ensemble", "mean of the two,\nout-of-fold", "model")
    for x in (0.41, 0.63):
        arrow(ax, x, ty + h / 2, x + 0.03, ty + h / 2)
    arrow(ax, 0.18, 0.78, 0.22, ty + h / 2)
    arrow(ax, 0.63, ty + h / 2, 0.66, ty - 0.20 + h / 2, rad=0.15)
    arrow(ax, 0.63, ty + h / 2, 0.66, ty - 0.40 + h / 2, rad=0.25)

    by = 0.08
    box(ax, 0.22, by, w, h, "Full dataset", "77,988 train, 8,536 val\n25,596 official test", "input")
    box(ax, 0.44, by, w, h, "Official NIH split", "test_list.txt never trained;\nval split by patient", "prep")
    box(ax, 0.66, by, w, h, "Fine-tune @ 224 px", "identical code and loss\n8 epochs, 1 h, Kaggle T4", "model")
    for x in (0.41, 0.63):
        arrow(ax, x, by + h / 2, x + 0.03, by + h / 2)
    arrow(ax, 0.18, 0.66, 0.22, by + h / 2)

    ex = 0.875
    box(ax, ex - 0.005, 0.40, 0.105, 0.40, "Evaluate",
        "hflip TTA\n\nAUROC, AUPRC\nP / R / F1\nconfusion\ntriage\n\nsame 1,293\nfilms for\nevery model",
        "out", fs=10, sfs=7.8)
    for yy in (ty + h / 2, ty - 0.20 + h / 2, ty - 0.40 + h / 2, by + h / 2):
        arrow(ax, 0.85, yy, ex - 0.005, 0.60, rad=-0.1 if yy > 0.5 else 0.15)
    ax.text(0.5, 0.01, "Rule throughout: no image is ever scored by a model that trained on it. "
            "Out-of-fold for the sample models; official test split for the full model.",
            ha="center", fontsize=8.6, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def gradcam_mechanism(path: Path) -> None:
    fig, ax = canvas(13.2, 7.8,
                     "Explainable AI: How a LayerCAM Heatmap Is Made, and How We Score It",
                     "Left: the mechanism (no retraining; one forward and one backward pass per finding). "
                     "Right: the evaluation against radiologist boxes.")
    w, h = 0.165, 0.135
    steps = [
        (0.02, 0.72, "1. Forward pass", "X-ray -> trained CNN\n-> 14 probabilities", "input"),
        (0.215, 0.72, "2. Pick a finding", "e.g. Cardiomegaly\nscore y_c", "prep"),
        (0.41, 0.72, "3. Feature maps A", "block 3: 14 x 14 x 1024\nblock 4:  7 x 7 x 1024", "model"),
        (0.02, 0.50, "4. Backward pass", "dy_c / dA: how much\neach activation\nmoved the score", "model"),
        (0.215, 0.50, "5. LayerCAM weights", "ReLU(dy_c/dA) * A,\nelement-wise per pixel\n(Grad-CAM averages instead)", "model"),
        (0.41, 0.50, "6. Sum channels\n+ ReLU", "one 2-D map per layer;\nkeep only positive\nevidence", "head"),
        (0.02, 0.28, "7. Fuse layers", "block 3 (fine) +\nblock 4 (semantic),\nboth upsampled", "head"),
        (0.215, 0.28, "8. Normalise\n+ upsample", "0 ... 1, bilinear\nto 1024 x 1024", "head"),
        (0.41, 0.28, "9. Overlay", "jet colour map at\n45 % opacity on the\noriginal film", "out"),
    ]
    for x, y, t, sub, kind in steps:
        box(ax, x, y, w, h, t, sub, kind, fs=9.8, sfs=7.8)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]:
        xa, ya = steps[a][0] + w, steps[a][1] + h / 2
        xb, yb = steps[b][0], steps[b][1] + h / 2
        if abs(ya - yb) < 1e-6:
            arrow(ax, xa, ya, xb, yb)
        else:
            arrow(ax, steps[a][0] + w / 2, steps[a][1], steps[b][0] + w / 2, steps[b][1] + h)
    ax.text(0.30, 0.20, "Why block 3+4 and LayerCAM: chosen from a 111-configuration sweep scored on "
            "radiologist boxes.\nThe 7 x 7 grid of block 4 alone (32-pixel cells) was the bottleneck; "
            "LayerCAM's per-pixel weighting keeps fine detail.",
            ha="center", va="top", fontsize=8.2, color=MUTED, style="italic")

    ex, ew = 0.655, 0.32
    box(ax, ex, 0.72, ew, 0.135, "Radiologist boxes (NIH)",
        "984 hand-drawn boxes, 8 findings.\n53 fall on our films, all in the official test split.",
        "input", fs=10, sfs=8.2)
    box(ax, ex, 0.545, ew, 0.135, "Pointing game",
        "Is the hottest pixel inside the box?\nChance = box area / image area, about 7 %.",
        "prep", fs=10, sfs=8.2)
    box(ax, ex, 0.37, ew, 0.135, "IoU at 50 % of peak",
        "Threshold the map, overlap with the box.\nPenalises maps that are too big or too small.",
        "prep", fs=10, sfs=8.2)
    box(ax, ex, 0.195, ew, 0.135, "Box coverage",
        "Share of the box inside the map region.\nSeparates 'right area, imprecise' from 'wrong place'.",
        "prep", fs=10, sfs=8.2)
    for yy in (0.72, 0.545, 0.37):
        arrow(ax, ex + ew / 2, yy, ex + ew / 2, yy - 0.04)
    box(ax, ex, 0.03, ew, 0.125, "Result (full-data model, adopted config)",
        "31 / 53 hits (58 %), IoU 0.26, coverage 0.69\nCardiomegaly 9/10, Atelectasis 5/11, Nodule 0/3",
        "out", fs=10, sfs=8.2)
    arrow(ax, ex + ew / 2, 0.195, ex + ew / 2, 0.155)
    arrow(ax, 0.41 + w, 0.28 + h / 2, ex, 0.545 + 0.06, rad=-0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    plots = Path(cfg.dirs.plots)
    raddino_probe(plots / "diagram_raddino_probe.png")
    cls_pipeline(plots / "diagram_cls_pipeline.png")
    gradcam_mechanism(plots / "diagram_gradcam.png")
    print(f"wrote diagram_raddino_probe, diagram_cls_pipeline, diagram_gradcam -> {plots}")


if __name__ == "__main__":
    main()
