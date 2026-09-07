"""Grad-CAM heatmaps, and how often they land where the radiologist drew the box.

    python -m src.gradcam                     # bbox evaluation + case overlays
    python -m src.gradcam --model densenet121_full224 --cases-only

Grad-CAM shows where the model LOOKED, not where the disease IS. The two agree
only when the model has learned the right features, so we measure the
agreement instead of assuming it. NIH ships 984 radiologist-drawn boxes; 53 of
them fall on images in our sample, all in the official test split, so both the
full-data model and the out-of-fold local model score them cleanly.

Two localisation metrics per (image, finding) pair:
  * pointing game  -- is the hottest CAM pixel inside the box? Chance level is
                      the box's share of the image area, reported alongside.
  * IoU            -- CAM thresholded at 50% of its max vs the box
plus the fraction of total CAM mass that falls inside the box.

Outputs:
  outputs/metrics/gradcam_localization.csv      one row per (image, finding, model)
  outputs/metrics/gradcam_summary.csv           per finding per model
  outputs/plots/gradcam/bbox/<model>/*.png      overlay + box, hit/miss in title
  outputs/plots/gradcam/cases/<model>/*.png     the ten briefing cases, top-3 findings
"""

from __future__ import annotations

import argparse
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
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, LayerCAM, XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, N_CLASSES, load_config, read_csv  # noqa: E402
from src.dataset import build_transforms  # noqa: E402
from src.model import build_model  # noqa: E402

BBOX_TO_CLASS = {"Infiltrate": "Infiltration"}   # NIH's bbox file uses the older name
CAM_THRESHOLD = 0.5                               # fraction of max for the IoU mask
ORIG_PX = 1024

METHODS = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "hirescam": HiResCAM,
           "layercam": LayerCAM, "xgradcam": XGradCAM}

# Default explanation configuration, chosen from the 111-configuration sweep
# (src.gradcam_sweep, outputs/metrics/gradcam_sweep.csv). The sweep's strict
# winner by pointing-game hits was block3 | layercam | both | 320px (33/53), but
# it produces tiny sharp blobs (IoU 0.17, 5% of the image) and costs 6 forward
# passes per map. This runner-up is 2 hits behind -- inside the noise on n=53 --
# while having the best IoU of all 111 (0.26 vs the old default's 0.23), 69% box
# coverage, and a single forward pass. Better on every metric than the previous
# default (block4 | gradcam | none | 224: 19/53, rank 81 of 111).
DEFAULT_CAM = {"method": "layercam", "layers": "block3+4", "aug_smooth": False,
               "eigen_smooth": False}


def cam_tag(cfg_cam: dict) -> str:
    sm = {(False, False): "none", (True, False): "aug", (False, True): "eigen",
          (True, True): "both"}[(cfg_cam["aug_smooth"], cfg_cam["eigen_smooth"])]
    return f"{cfg_cam['layers']}_{cfg_cam['method']}_{sm}"


def target_layers(model, layers: str):
    f = model.features
    return {"block4": [f.denseblock4], "block3": [f.denseblock3],
            "block3+4": [f.denseblock3, f.denseblock4]}[layers]


# ------------------------------------------------------------- model loading ---
def load_model(cfg, name: str, fold: int | None, device):
    """Return (model, spec, px). Full-data model is fold-agnostic; local models
    need the fold whose held-out set contains the image, or they are contaminated."""
    if name.startswith("densenet121_full"):
        ckpt = Path(cfg.paths.checkpoints) / f"{name}.pt"
        arch, px = "densenet121_imagenet", int(name.replace("densenet121_full", ""))
    else:
        assert fold is not None, "local models need a fold"
        ckpt = Path(cfg.paths.checkpoints) / f"{name}_fold{fold}.pt"
        arch, px = name, cfg.train.image_px
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model, spec = build_model(arch, N_CLASSES, cfg.train.dropout)
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), spec, px


class CamEngine:
    """One Grad-CAM instance per loaded model; caches by (model name, fold)."""

    def __init__(self, cfg, device, cam_cfg: dict | None = None):
        self.cfg, self.device, self._cache = cfg, device, {}
        self.cam_cfg = {**DEFAULT_CAM, **(cam_cfg or {})}

    def get(self, name: str, fold: int | None):
        key = (name, fold if not name.startswith("densenet121_full") else None)
        if key not in self._cache:
            model, spec, px = load_model(self.cfg, name, fold, self.device)
            cam = METHODS[self.cam_cfg["method"]](
                model=model, target_layers=target_layers(model, self.cam_cfg["layers"]))
            tf = build_transforms(px, spec, train=False)
            self._cache[key] = (model, cam, tf, px)
        return self._cache[key]

    def release(self):
        """Drop the forward/backward hooks -- they leak otherwise."""
        for _, cam, _, _ in self._cache.values():
            cam.activations_and_grads.release()
        self._cache.clear()

    def run(self, name: str, fold: int | None, img: Image.Image,
            class_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (probs[14], cams[len(class_idx), ORIG_PX, ORIG_PX] in 0..1)."""
        model, cam, tf, px = self.get(name, fold)
        x = tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()
        cams = []
        for c in class_idx:
            g = cam(input_tensor=x, targets=[ClassifierOutputTarget(c)],
                    aug_smooth=self.cam_cfg["aug_smooth"],
                    eigen_smooth=self.cam_cfg["eigen_smooth"])[0]        # (px, px)
            g = np.nan_to_num(g)
            g = np.array(Image.fromarray((g * 255).astype(np.uint8))
                         .resize((ORIG_PX, ORIG_PX), Image.BILINEAR)) / 255.0
            cams.append(g)
        return probs, np.stack(cams)


# ---------------------------------------------------------------- metrics ---
def localisation_metrics(cam: np.ndarray, box: tuple[float, float, float, float]) -> dict:
    x, y, w, h = box
    H, W = cam.shape
    x0, y0 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
    x1, y1 = int(min(W, np.ceil(x + w))), int(min(H, np.ceil(y + h)))
    box_mask = np.zeros_like(cam, dtype=bool)
    box_mask[y0:y1, x0:x1] = True

    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    hit = bool(box_mask[peak])

    cam_mask = cam >= CAM_THRESHOLD * cam.max()
    inter = np.logical_and(cam_mask, box_mask).sum()
    union = np.logical_or(cam_mask, box_mask).sum()
    total = cam.sum()
    return {"pointing_hit": hit, "iou": inter / max(union, 1),
            # containment: share of the box that lies inside the CAM region. A tiny
            # nodule box fully inside a large CAM blob scores ~1 here but ~0 IoU --
            # the "right neighbourhood, no precision" case.
            "box_coverage": float(inter / max(box_mask.sum(), 1)),
            "cam_mass_in_box": float(cam[box_mask].sum() / total) if total > 0 else 0.0,
            "box_area_frac": float(box_mask.mean()),        # chance level for pointing
            "cam_area_frac": float(cam_mask.mean())}


# ----------------------------------------------------------------- plotting ---
def overlay(ax, img: Image.Image, cam: np.ndarray, title: str, box=None, alpha=0.45):
    ax.imshow(img, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    if box is not None:
        x, y, w, h = box
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, lw=2.2, ec="#00ff88"))
        py, px = np.unravel_index(int(np.argmax(cam)), cam.shape)
        ax.plot(px, py, marker="x", ms=11, mew=2.5, color="white")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def bbox_figure(img, cams, rows, model, out: Path):
    n = len(rows)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 4.6))
    axes = np.atleast_1d(axes)
    axes[0].imshow(img, cmap="gray"); axes[0].axis("off")
    axes[0].set_title(rows[0]["image"], fontsize=9)
    for ax, cam, r in zip(axes[1:], cams, rows):
        overlay(ax, img, cam, f"{r['finding']}  p={r['prob']:.2f}\n"
                f"{'HIT' if r['pointing_hit'] else 'miss'}  IoU {r['iou']:.2f}  "
                f"mass-in-box {r['cam_mass_in_box']:.2f}", box=r["box"])
    fig.suptitle(f"{model} — green box: radiologist; x: CAM peak", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def case_figure(img, probs, cams, class_idx, true_lbls, name, model, out: Path):
    fig, axes = plt.subplots(1, len(class_idx) + 1, figsize=(4.2 * (len(class_idx) + 1), 4.6))
    axes[0].imshow(img, cmap="gray"); axes[0].axis("off")
    axes[0].set_title(f"{name}\ntrue: {', '.join(true_lbls) or 'No finding'}", fontsize=9)
    for ax, cam, c in zip(axes[1:], cams, class_idx):
        mark = "✓" if CLASSES[c] in true_lbls else "✗"
        overlay(ax, img, cam, f"{mark} {CLASSES[c]}  p={probs[c]:.2f}")
    fig.suptitle(f"{model} — top-3 findings, Grad-CAM on final dense block", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=110)
    plt.close(fig)


# -------------------------------------------------------------------- main ---
def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["densenet121_full224", "densenet121_imagenet"])
    ap.add_argument("--cases-only", action="store_true")
    ap.add_argument("--bbox-only", action="store_true")
    ap.add_argument("--method", default=DEFAULT_CAM["method"], choices=list(METHODS))
    ap.add_argument("--layers", default=DEFAULT_CAM["layers"],
                    choices=["block4", "block3", "block3+4"])
    ap.add_argument("--smooth", default="none", choices=["none", "aug", "eigen", "both"])
    args = ap.parse_args()

    cam_cfg = {"method": args.method, "layers": args.layers,
               "aug_smooth": args.smooth in ("aug", "both"),
               "eigen_smooth": args.smooth in ("eigen", "both")}
    tag = cam_tag(cam_cfg)
    print(f"CAM configuration: {tag}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eng = CamEngine(cfg, device, cam_cfg)
    folds = read_csv(Path(cfg.dirs.root) / "folds.csv").set_index("image")
    img_dir = Path(cfg.data.images_full)
    plot_root = Path(cfg.dirs.plots) / "gradcam"

    # ---- 1. bounding-box localisation ------------------------------------
    if not args.cases_only:
        bb = pd.read_csv(Path(cfg.data.root) / "meta" / "BBox_List_2017.csv")
        bb = bb.iloc[:, :6]
        bb.columns = ["image", "finding", "x", "y", "w", "h"]
        bb["finding"] = bb["finding"].map(lambda f: BBOX_TO_CLASS.get(f, f))
        bb = bb[bb.image.isin(folds.index)].reset_index(drop=True)
        print(f"bbox rows on sample images: {len(bb)} "
              f"({bb.image.nunique()} images, {bb.finding.nunique()} findings)")

        rows = []
        for model in args.models:
            out_dir = plot_root / "bbox" / tag / model
            out_dir.mkdir(parents=True, exist_ok=True)
            for image, grp in bb.groupby("image", sort=False):
                img = Image.open(img_dir / image).convert("L")
                fold = int(folds.loc[image, "fold"])
                cls_idx = [CLASSES.index(f) for f in grp.finding]
                probs, cams = eng.run(model, fold, img, cls_idx)
                fig_rows = []
                for (_, r), cam, c in zip(grp.iterrows(), cams, cls_idx):
                    box = (r.x, r.y, r.w, r.h)
                    m = localisation_metrics(cam, box)
                    rec = {"model": model, "image": image, "finding": r.finding,
                           "fold": fold, "prob": float(probs[c]), **m}
                    rows.append(rec)
                    fig_rows.append({**rec, "box": box})
                bbox_figure(img, cams, fig_rows, model, out_dir / f"{Path(image).stem}.png")
            done = [r for r in rows if r["model"] == model]
            print(f"  {model}: pointing-game {np.mean([r['pointing_hit'] for r in done]):.1%} "
                  f"(chance {np.mean([r['box_area_frac'] for r in done]):.1%}), "
                  f"mean IoU {np.mean([r['iou'] for r in done]):.3f}")

        df = pd.DataFrame(rows)
        df["cam_config"] = tag
        met = Path(cfg.dirs.metrics)
        df.round(4).to_csv(met / f"gradcam_localization_{tag}.csv", index=False)
        summ = (df.groupby(["model", "finding"])
                  .agg(n=("iou", "size"), pointing_hit_rate=("pointing_hit", "mean"),
                       chance_rate=("box_area_frac", "mean"), mean_iou=("iou", "mean"),
                       mean_cam_mass_in_box=("cam_mass_in_box", "mean"),
                       mean_box_coverage=("box_coverage", "mean"),
                       mean_prob=("prob", "mean"))
                  .reset_index().round(3))
        summ["cam_config"] = tag
        summ.to_csv(met / f"gradcam_summary_{tag}.csv", index=False)
        with pd.option_context("display.width", 200):
            print("\n=== localisation by finding ===")
            print(summ.to_string(index=False))

    # ---- 2. the ten briefing cases, top-3 predicted findings each --------
    if not args.bbox_only:
        try:
            from scripts.briefing_sections import CASES
        except ImportError:
            CASES = []
        for model in args.models:
            out_dir = plot_root / "cases" / tag / model
            out_dir.mkdir(parents=True, exist_ok=True)
            for image, _, _ in CASES:
                if image not in folds.index:
                    continue
                img = Image.open(img_dir / image).convert("L")
                fold = int(folds.loc[image, "fold"])
                model_, _, tf, _ = eng.get(model, fold)
                with torch.no_grad():
                    p = torch.sigmoid(model_(tf(img).unsqueeze(0).to(device))).squeeze(0).cpu().numpy()
                top3 = [int(i) for i in np.argsort(-p)[:3]]
                probs, cams = eng.run(model, fold, img, top3)
                true = [c for c in CLASSES if folds.loc[image, c] == 1]
                case_figure(img, probs, cams, top3, true, image, model,
                            out_dir / f"{Path(image).stem}.png")
            print(f"  {model}: {len(CASES)} case figures -> {out_dir}")

    print(f"\nwrote figures under {plot_root}")


if __name__ == "__main__":
    main()
