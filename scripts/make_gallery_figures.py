"""Figures that exist only for the briefing appendices.

    python scripts/make_gallery_figures.py

  plots/full_model_training.png        full-data model: train loss + val AUROC per epoch
  plots/full_vs_sample_per_class.png   per-class AUROC, full model (official test) vs
                                       sample model (5-fold), side by side
  plots/gradcam_hits_by_finding.png    pointing-game hits per finding, old vs adopted
                                       CAM configuration, both models
  plots/gradcam/sheets/<config>_<model>.png
                                       every radiologist box as one thumbnail:
                                       X-ray + heatmap + box + peak, hit/miss coloured
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, load_config, read_csv  # noqa: E402
from src.gradcam import BBOX_TO_CLASS, CamEngine, cam_tag, localisation_metrics  # noqa: E402

FULL, BASE = "densenet121_full224", "densenet121_imagenet"
CONFIGS = {
    "adopted": {"method": "layercam", "layers": "block3+4", "aug_smooth": False, "eigen_smooth": False},
    "original": {"method": "gradcam", "layers": "block4", "aug_smooth": False, "eigen_smooth": False},
}


def full_model_training(met: Path, plots: Path) -> None:
    h = read_csv(met / f"{FULL}_history.csv")
    fig, ax1 = plt.subplots(figsize=(8, 4.6))
    ax1.plot(h.epoch, h.train_loss, "o-", color="#3b6ea5", lw=1.8, label="train loss")
    ax1.set_xlabel("epoch (1 = head only, 2-8 = full fine-tune)")
    ax1.set_ylabel("BCE loss", color="#3b6ea5")
    ax1.axvline(1.5, color="#999", lw=0.8, ls=":")
    ax2 = ax1.twinx()
    ax2.plot(h.epoch, h.val_auroc, "s-", color="#d1873c", lw=1.8, label="val macro AUROC")
    ax2.set_ylabel("macro AUROC (8,536 val films)", color="#d1873c")
    for e, a in zip(h.epoch, h.val_auroc):
        ax2.annotate(f"{a:.3f}", (e, a), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=7.5, color="#d1873c")
    ax1.set_title(f"Full-data DenseNet-121 (224 px) — {int(h.epoch.max())} epochs, "
                  f"{h.minutes.sum() / 60:.1f} GPU-hours on a Kaggle T4")
    fig.tight_layout()
    fig.savefig(plots / "full_model_training.png", dpi=130)
    plt.close(fig)


def full_vs_sample_per_class(met: Path, plots: Path) -> None:
    summ = json.loads((met / f"{FULL}_summary.json").read_text())
    full = pd.Series(summ["test_per_class_auroc"])
    pc = read_csv(met / "per_class_auroc.csv")
    sample = pc[pc.model == BASE].set_index("class")["mean"]
    counts = read_csv(Path(met).parent / "distribution" / "class_counts.csv", index_col=0)["positives"]
    order = counts.sort_values(ascending=False).index
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.38
    ax.bar(x - w / 2, [sample.get(c, np.nan) for c in order], w, color="#3b6ea5",
           label="sample model — 5,606 images, 5-fold CV")
    ax.bar(x + w / 2, [full.get(c, np.nan) for c in order], w, color="#4c9a6a",
           label="full-data model — 77,988 images, official test split")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xticks(x, [c.replace("_", " ") for c in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-finding AUROC: the same architecture with 14x more training data\n"
                 "(different evaluation sets — see the shared-image comparison for the strict test)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "full_vs_sample_per_class.png", dpi=130)
    plt.close(fig)


def gradcam_hits_by_finding(met: Path, plots: Path) -> None:
    frames = []
    for label, tag in (("original: block4 Grad-CAM", "block4_gradcam_none"),
                       ("adopted: block3+4 LayerCAM", "block3+4_layercam_none")):
        f = met / f"gradcam_localization_{tag}.csv"
        if f.exists():
            d = read_csv(f)
            d["config"] = label
            frames.append(d)
    d = pd.concat(frames)
    d["hits"] = d.pointing_hit.astype(int)
    g = d.groupby(["model", "config", "finding"]).agg(n=("hits", "size"), hits=("hits", "sum")).reset_index()
    order = g.groupby("finding").n.first().sort_values(ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, model, title in zip(axes, (FULL, BASE),
                                ("Full-data model", "Sample-trained model (320 px)")):
        sub = g[g.model == model]
        x = np.arange(len(order))
        w = 0.38
        for k, (cfg_lbl, colour) in enumerate((("original: block4 Grad-CAM", "#b3261e"),
                                               ("adopted: block3+4 LayerCAM", "#3b6ea5"))):
            s = sub[sub.config == cfg_lbl].set_index("finding").reindex(order)
            ax.bar(x + (k - 0.5) * w, s.hits, w, color=colour, label=cfg_lbl)
            for xi, (h, n) in enumerate(zip(s.hits, s.n)):
                if pd.notna(h):
                    ax.text(xi + (k - 0.5) * w, h + 0.1, f"{int(h)}/{int(n)}", ha="center",
                            fontsize=7.5)
        ax.set_xticks(x, [c.replace("_", " ") for c in order], rotation=45, ha="right", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("pointing-game hits (peak inside radiologist box)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Grad-CAM localisation per finding — 53 radiologist boxes", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(plots / "gradcam_hits_by_finding.png", dpi=130)
    plt.close(fig)


def contact_sheets(cfg, plots: Path) -> list[Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = read_csv(Path(cfg.dirs.root) / "folds.csv").set_index("image")
    bb = pd.read_csv(Path(cfg.data.root) / "meta" / "BBox_List_2017.csv").iloc[:, :6]
    bb.columns = ["image", "finding", "x", "y", "w", "h"]
    bb["finding"] = bb["finding"].map(lambda f: BBOX_TO_CLASS.get(f, f))
    bb = bb[bb.image.isin(folds.index)].reset_index(drop=True)
    img_dir = Path(cfg.data.images_full)
    out_dir = plots / "gradcam" / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for cfg_name, cam_cfg in CONFIGS.items():
        eng = CamEngine(cfg, device, cam_cfg)
        tag = cam_tag(cam_cfg)
        for model in (FULL, BASE):
            if cfg_name == "original" and model == BASE:
                continue   # three sheets are enough; the fourth adds little
            tiles = []
            for image, grp in bb.groupby("image", sort=False):
                img = Image.open(img_dir / image).convert("L")
                fold = int(folds.loc[image, "fold"])
                idx = [CLASSES.index(f) for f in grp.finding]
                probs, cams = eng.run(model, fold, img, idx)
                for (_, r), cam, c in zip(grp.iterrows(), cams, idx):
                    m = localisation_metrics(cam, (r.x, r.y, r.w, r.h))
                    tiles.append((img, cam, (r.x, r.y, r.w, r.h), r.finding,
                                  float(probs[c]), m["pointing_hit"], m["iou"]))
            eng.release()

            ncols = 7
            nrows = int(np.ceil(len(tiles) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 2.85 * nrows))
            for ax, (img, cam, box, finding, p, hit, iou) in zip(axes.flat, tiles):
                ax.imshow(img, cmap="gray")
                ax.imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
                x, y, w, h = box
                ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, lw=1.6, ec="#00ff88"))
                py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
                ax.plot(px, py, marker="x", ms=8, mew=2, color="white")
                ax.set_title(f"{finding.replace('_', ' ')}  p={p:.2f}\n"
                             f"{'HIT' if hit else 'miss'}  IoU {iou:.2f}",
                             fontsize=7.5, color="#1a7f37" if hit else "#b3261e")
                ax.axis("off")
            for ax in axes.flat[len(tiles):]:
                ax.axis("off")
            hits = sum(t[5] for t in tiles)
            fig.suptitle(f"{model}  |  {tag}  |  {hits}/{len(tiles)} pointing-game hits\n"
                         "green box = radiologist, x = heatmap peak", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.965))
            out = out_dir / f"{tag}_{model}.png"
            fig.savefig(out, dpi=95)
            plt.close(fig)
            written.append(out)
            print(f"  sheet {out.name}: {hits}/{len(tiles)} hits")
    return written


def main() -> None:
    cfg = load_config()
    met, plots = Path(cfg.dirs.metrics), Path(cfg.dirs.plots)
    full_model_training(met, plots)
    full_vs_sample_per_class(met, plots)
    gradcam_hits_by_finding(met, plots)
    print("summary figures written")
    contact_sheets(cfg, plots)
    print("done")


if __name__ == "__main__":
    main()
