"""Which explanation configuration actually points at the lesion? Measure it.

    python -m src.gradcam_sweep
    python -m src.gradcam_sweep --model densenet121_imagenet --quick

Sweeps every combination of
    layer      : block4 (7x7 @224), block3 (14x14 @224), block3+4 fused
    method     : Grad-CAM, Grad-CAM++, HiResCAM, LayerCAM, XGrad-CAM
    smoothing  : none, aug (flip/shift averaging), eigen (1st principal comp.), both
    resolution : 224 (training res), 320
over the 53 radiologist boxes and scores each with the same metrics as
src.gradcam. Selection criterion, fixed BEFORE running:
    primary   = pointing-game hit rate
    tie-break = mean IoU at the 0.5 threshold
IoU at 0.3 and box coverage are recorded for context, not used to choose.

120 configurations on 53 boxes is a lot of comparisons for a small set; the
winner's number will be optimistic. Treat the ranking as a shortlist to confirm
on the full 984 boxes (Kaggle), not as a final figure.

Writes:
    outputs/metrics/gradcam_sweep.csv          one row per configuration
    outputs/metrics/gradcam_sweep_best.csv     per-finding breakdown of the winner
    outputs/plots/gradcam_sweep.png            top configurations vs the baseline
    outputs/plots/gradcam/best/<model>/*.png   overlays for the winning configuration
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, LayerCAM, XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, load_config, read_csv  # noqa: E402
from src.dataset import build_transforms  # noqa: E402
from src.gradcam import BBOX_TO_CLASS, ORIG_PX, bbox_figure, load_model  # noqa: E402

METHODS = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "hirescam": HiResCAM,
           "layercam": LayerCAM, "xgradcam": XGradCAM}
SMOOTHING = {"none": (False, False), "aug": (True, False),
             "eigen": (False, True), "both": (True, True)}
RESOLUTIONS = (224, 320)


def layers_for(model, name: str):
    f = model.features
    return {"block4": [f.denseblock4], "block3": [f.denseblock3],
            "block3+4": [f.denseblock3, f.denseblock4]}[name]


def metrics(cam: np.ndarray, box) -> dict:
    x, y, w, h = box
    H, W = cam.shape
    x0, y0 = int(max(0, np.floor(x))), int(max(0, np.floor(y)))
    x1, y1 = int(min(W, np.ceil(x + w))), int(min(H, np.ceil(y + h)))
    bm = np.zeros_like(cam, dtype=bool)
    bm[y0:y1, x0:x1] = True
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    out = {"pointing_hit": bool(bm[peak]), "box_area_frac": float(bm.mean()),
           "cam_mass_in_box": float(cam[bm].sum() / cam.sum()) if cam.sum() > 0 else 0.0}
    for thr in (0.5, 0.3):
        cm = cam >= thr * cam.max()
        inter = np.logical_and(cm, bm).sum()
        out[f"iou@{thr}"] = inter / max(np.logical_or(cm, bm).sum(), 1)
        if thr == 0.5:
            out["box_coverage"] = float(inter / max(bm.sum(), 1))
            out["cam_area_frac"] = float(cm.mean())
    return out


def upscale(g: np.ndarray) -> np.ndarray:
    g = np.nan_to_num(g)
    return np.array(Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
                    .resize((ORIG_PX, ORIG_PX), Image.BILINEAR)) / 255.0


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="densenet121_full224")
    ap.add_argument("--quick", action="store_true", help="block4 x 224 only (smoke test)")
    ap.add_argument("--finalize", action="store_true",
                    help="skip any configurations not yet in the partial CSV; just summarise")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = read_csv(Path(cfg.dirs.root) / "folds.csv").set_index("image")
    img_dir = Path(cfg.data.images_full)
    met_dir, plot_dir = Path(cfg.dirs.metrics), Path(cfg.dirs.plots)

    bb = pd.read_csv(Path(cfg.data.root) / "meta" / "BBox_List_2017.csv").iloc[:, :6]
    bb.columns = ["image", "finding", "x", "y", "w", "h"]
    bb["finding"] = bb["finding"].map(lambda f: BBOX_TO_CLASS.get(f, f))
    bb = bb[bb.image.isin(folds.index)].reset_index(drop=True)
    images = {n: Image.open(img_dir / n).convert("L") for n in bb.image.unique()}
    print(f"{len(bb)} boxes on {len(images)} images; model {args.model}")

    # models per resolution (one load each; full-data model is fold-agnostic)
    is_full = args.model.startswith("densenet121_full")
    models = {}
    for px in RESOLUTIONS:
        if is_full:
            m, spec, _ = load_model(cfg, args.model, None, device)
            models[px] = {None: (m, build_transforms(px, spec, train=False))}
        else:
            models[px] = {}
            for k in sorted(folds.loc[bb.image.unique(), "fold"].unique()):
                m, spec, _ = load_model(cfg, args.model, int(k), device)
                models[px][int(k)] = (m, build_transforms(px, spec, train=False))

    grid = list(itertools.product(["block4", "block3", "block3+4"], METHODS, SMOOTHING, RESOLUTIONS))
    if args.quick:
        grid = [g for g in grid if g[0] == "block4" and g[3] == 224]
    print(f"{len(grid)} configurations")

    rows, per_box = [], {}
    partial = met_dir / f"gradcam_sweep{'_quick' if args.quick else ''}_partial.csv"
    box_dir = met_dir / "gradcam_sweep_boxes"
    box_dir.mkdir(exist_ok=True)
    done = set()
    if partial.exists():
        prev = read_csv(partial)
        rows = prev.to_dict("records")
        done = {(r["layer"], r["method"], r["smoothing"], int(r["res"])) for r in rows}
        for r in rows:  # reload per-box tables for the final per-finding breakdown
            cid = f"{r['layer']}|{r['method']}|{r['smoothing']}|{int(r['res'])}"
            f = box_dir / (cid.replace("|", "_").replace("+", "p") + ".csv")
            if f.exists():
                per_box[cid] = read_csv(f)
        print(f"resuming: {len(done)} configurations already done")
    if args.finalize:
        skipped = [g for g in grid if g not in done]
        grid = [g for g in grid if g in done]
        print(f"finalize: summarising {len(grid)} configurations; "
              f"{len(skipped)} not run: {[' | '.join(map(str, g)) for g in skipped]}")
    t0 = time.time()
    for i, (layer, method, smooth, px) in enumerate(grid, 1):
        if (layer, method, smooth, px) in done:
            continue
        aug, eig = SMOOTHING[smooth]
        cams_cache = {}
        recs = []
        for image, grp in bb.groupby("image", sort=False):
            fold = None if is_full else int(folds.loc[image, "fold"])
            model, tf = models[px][fold]
            key = (id(model), layer, method)
            if key not in cams_cache:
                cams_cache[key] = METHODS[method](model=model, target_layers=layers_for(model, layer))
            cam = cams_cache[key]
            x = tf(images[image]).unsqueeze(0).to(device)
            for _, r in grp.iterrows():
                c = CLASSES.index(r.finding)
                g = cam(input_tensor=x, targets=[ClassifierOutputTarget(c)],
                        aug_smooth=aug, eigen_smooth=eig)[0]
                m = metrics(upscale(g), (r.x, r.y, r.w, r.h))
                recs.append({"image": image, "finding": r.finding, **m})
        # pytorch-grad-cam registers forward/backward hooks on the model; without
        # release() they persist and every later forward pass accumulates
        # activations for all of them. That leak is what killed the first run.
        for c_ in cams_cache.values():
            c_.activations_and_grads.release()
        cams_cache.clear()
        torch.cuda.empty_cache()

        df = pd.DataFrame(recs)
        cfg_id = f"{layer}|{method}|{smooth}|{px}"
        per_box[cfg_id] = df
        rows.append({"layer": layer, "method": method, "smoothing": smooth, "res": px,
                     "pointing_hit_rate": df.pointing_hit.mean(),
                     "hits": int(df.pointing_hit.sum()), "n": len(df),
                     "mean_iou@0.5": df["iou@0.5"].mean(), "mean_iou@0.3": df["iou@0.3"].mean(),
                     "mean_box_coverage": df.box_coverage.mean(),
                     "mean_cam_mass_in_box": df.cam_mass_in_box.mean(),
                     "mean_cam_area_frac": df.cam_area_frac.mean()})
        pd.DataFrame(rows).to_csv(partial, index=False)
        df.to_csv(box_dir / (cfg_id.replace("|", "_").replace("+", "p") + ".csv"), index=False)
        if i % 10 == 0 or i == len(grid):
            best = max(rows, key=lambda r: (r["pointing_hit_rate"], r["mean_iou@0.5"]))
            print(f"  {i}/{len(grid)}  ({(time.time() - t0) / 60:.1f} min)  "
                  f"best so far: {best['layer']}|{best['method']}|{best['smoothing']}|{best['res']} "
                  f"-> {best['hits']}/{best['n']} hits, IoU {best['mean_iou@0.5']:.3f}", flush=True)

    res = pd.DataFrame(rows).sort_values(["pointing_hit_rate", "mean_iou@0.5"],
                                         ascending=False).reset_index(drop=True)
    res["model"] = args.model
    res.round(4).to_csv(met_dir / f"gradcam_sweep{'_quick' if args.quick else ''}.csv", index=False)

    base = res[(res.layer == "block4") & (res.method == "gradcam") &
               (res.smoothing == "none") & (res.res == 224)].iloc[0]
    best = res.iloc[0]
    with pd.option_context("display.width", 220):
        print("\n=== top 10 configurations (primary: pointing hits; tie-break IoU@0.5) ===")
        print(res.head(10)[["layer", "method", "smoothing", "res", "hits", "n", "pointing_hit_rate",
                            "mean_iou@0.5", "mean_iou@0.3", "mean_box_coverage"]].to_string(index=False))
        print(f"\nbaseline (block4|gradcam|none|224): {int(base.hits)}/{int(base.n)} hits, "
              f"IoU {base['mean_iou@0.5']:.3f}")
        print(f"winner   ({best.layer}|{best.method}|{best.smoothing}|{best.res}): "
              f"{int(best.hits)}/{int(best.n)} hits, IoU {best['mean_iou@0.5']:.3f}")

    # per-finding breakdown, winner vs baseline
    bid = f"{best.layer}|{best.method}|{best.smoothing}|{best.res}"
    wb, bs = per_box[bid], per_box["block4|gradcam|none|224"]
    brk = (wb.groupby("finding").agg(n=("pointing_hit", "size"), hits_best=("pointing_hit", "sum"),
                                     iou_best=("iou@0.5", "mean"))
             .join(bs.groupby("finding").agg(hits_base=("pointing_hit", "sum"), iou_base=("iou@0.5", "mean")))
             .sort_values("n", ascending=False).round(3))
    brk.to_csv(met_dir / "gradcam_sweep_best.csv")
    print("\n=== winner vs baseline, per finding ===")
    print(brk.to_string())

    # plot: top 15 + baseline
    top = res.head(15).copy()
    if not ((top.layer == "block4") & (top.method == "gradcam") & (top.smoothing == "none") & (top.res == 224)).any():
        top = pd.concat([top, base.to_frame().T])
    top["label"] = top.apply(lambda r: f"{r.layer} | {r.method} | {r.smoothing} | {r.res}px", axis=1)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colours = ["#b3261e" if (r.layer == "block4" and r.method == "gradcam" and r.smoothing == "none" and r.res == 224)
               else "#3b6ea5" for _, r in top.iterrows()]
    ax.barh(top.label[::-1], top.pointing_hit_rate[::-1], color=colours[::-1])
    ax.axvline(res["n"].iloc[0] and bs.box_area_frac.mean(), color="k", ls="--", lw=0.8)
    ax.text(bs.box_area_frac.mean() + 0.005, 0.2, "chance", fontsize=8)
    ax.set_xlabel("pointing-game hit rate (53 radiologist boxes)")
    ax.set_title(f"Grad-CAM configuration sweep — {args.model}\n"
                 "red = current default; higher is better; ranking is optimistic on n=53")
    ax.set_xlim(0, max(0.6, top.pointing_hit_rate.max() + 0.05))
    fig.tight_layout()
    fig.savefig(plot_dir / "gradcam_sweep.png", dpi=130)

    # overlays for the winner
    out_dir = plot_dir / "gradcam" / "best" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    aug, eig = SMOOTHING[best.smoothing]
    for image, grp in bb.groupby("image", sort=False):
        fold = None if is_full else int(folds.loc[image, "fold"])
        model, tf = models[int(best.res)][fold]
        cam = METHODS[best.method](model=model, target_layers=layers_for(model, best.layer))
        x = tf(images[image]).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy()
        cams, fig_rows = [], []
        for _, r in grp.iterrows():
            c = CLASSES.index(r.finding)
            g = upscale(cam(input_tensor=x, targets=[ClassifierOutputTarget(c)],
                            aug_smooth=aug, eigen_smooth=eig)[0])
            m = metrics(g, (r.x, r.y, r.w, r.h))
            cams.append(g)
            fig_rows.append({"image": image, "finding": r.finding, "prob": float(probs[c]),
                             "pointing_hit": m["pointing_hit"], "iou": m["iou@0.5"],
                             "cam_mass_in_box": m["cam_mass_in_box"], "box": (r.x, r.y, r.w, r.h)})
        bbox_figure(images[image], np.stack(cams), fig_rows, f"{args.model} [{bid}]",
                    out_dir / f"{Path(image).stem}.png")
        cam.activations_and_grads.release()
    print(f"\nwinner overlays -> {out_dir}\n({(time.time() - t0) / 60:.1f} min total)")


if __name__ == "__main__":
    main()
