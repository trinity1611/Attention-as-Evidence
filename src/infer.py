"""Test-time augmentation, fold ensembling, and actual-vs-predicted samples.

    python -m src.infer                 # TTA held-out preds + prediction grids
    python -m src.infer --no-grids

Three things live here:

1. TTA. Held-out predictions are re-computed averaging the original and
   horizontally flipped view. This stays honest -- each fold's own model scores
   only its own held-out fold -- and writes <model>_fold<k>_val_tta.npz, which
   src.evaluate/src.confusion pick up via --suffix _tta.

2. predict_ensemble(). Averages all five fold models for a NEW image. This is
   the deployment path (Grad-CAM, MedGemma, the UI) and deliberately is NOT
   scored on this dataset: every one of the 5,606 images was in the training
   split of four folds, so an ensemble score here would be contaminated. The
   ensemble is an inference artefact, not a reportable result.

3. Actual-vs-predicted contact sheets, best and worst cases by per-image error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, N_CLASSES, load_config, read_csv# noqa: E402
from src.dataset import CXRDataset, build_transforms  # noqa: E402
from src.model import build_model  # noqa: E402


def load_fold_model(cfg, model_name: str, fold: int, device):
    ckpt = Path(cfg.paths.checkpoints) / f"{model_name}_fold{fold}.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model, spec = build_model(model_name, N_CLASSES, cfg.train.dropout)
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), spec


@torch.no_grad()
def predict_split(cfg, model, spec, df: pd.DataFrame, device, tta: bool):
    ds = CXRDataset(df, Path(cfg.data.images_small),
                    build_transforms(cfg.train.image_px, spec, train=False))
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                    num_workers=cfg.train.num_workers, pin_memory=True)
    probs, trues = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
            if tta:
                # Chest X-rays tolerate left-right mirroring; averaging the two
                # views reduces variance without changing the anatomy seen.
                p = (p + torch.sigmoid(model(torch.flip(x, dims=[3])).float())) / 2
        probs.append(p.cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(probs)


def macro_auroc(y_true, y_prob) -> float:
    vals = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(N_CLASSES)
            if 0 < y_true[:, i].sum() < len(y_true)]
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def predict_ensemble(cfg, model_name: str, image_paths: list[Path], device,
                     tta: bool = True) -> np.ndarray:
    """Mean probability over all fold models. Deployment inference only."""
    models = []
    for fold in range(cfg.split.n_folds):
        try:
            models.append(load_fold_model(cfg, model_name, fold, device))
        except FileNotFoundError:
            continue
    if not models:
        raise FileNotFoundError(f"no checkpoints for {model_name}")

    out = np.zeros((len(image_paths), N_CLASSES), dtype=np.float32)
    for model, spec in models:
        tf = build_transforms(cfg.train.image_px, spec, train=False)
        batch = torch.stack([tf(Image.open(p).convert("L")) for p in image_paths])
        batch = batch.to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            p = torch.sigmoid(model(batch).float())
            if tta:
                p = (p + torch.sigmoid(model(torch.flip(batch, dims=[3])).float())) / 2
        out += p.cpu().numpy()
        del model
    torch.cuda.empty_cache()
    return out / len(models)


def prediction_grid(df: pd.DataFrame, y_true, y_prob, image_dir: Path,
                    idx: np.ndarray, title: str, path: Path, thr: np.ndarray) -> None:
    """Contact sheet: image, true findings, and the model's calls."""
    n = len(idx)
    ncols, nrows = 4, int(np.ceil(n / 4))
    # Generous row height + hspace: each caption is up to 5 lines and would
    # otherwise print over the image in the row above.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 5.9 * nrows))
    fig.subplots_adjust(hspace=0.42)
    axes = np.atleast_2d(axes)
    for ax, i in zip(axes.flat, idx):
        name = Path(df.iloc[i]["image"]).with_suffix(".jpg").name
        ax.imshow(Image.open(image_dir / name), cmap="gray")
        ax.axis("off")

        true_set = {CLASSES[j] for j in range(N_CLASSES) if y_true[i, j] == 1}
        pred_set = {CLASSES[j] for j in range(N_CLASSES) if y_prob[i, j] >= thr[j]}
        top = np.argsort(-y_prob[i])[:3]

        true_txt = ", ".join(sorted(true_set)) or "No Finding"
        pred_txt = "\n".join(
            f"{'✓' if (CLASSES[j] in true_set) else '✗'} {CLASSES[j]} "
            f"{y_prob[i, j]:.2f}{' *' if y_prob[i, j] >= thr[j] else ''}"
            for j in top)
        hit = true_set & pred_set
        colour = "#1a7f37" if hit else ("#b3261e" if true_set or pred_set else "#444")
        ax.set_title(f"true: {true_txt}\n\n{pred_txt}", fontsize=8,
                     color=colour, loc="left")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{title}\n(top-3 probabilities; * = above per-class threshold, "
                 "✓ = in true labels)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.0)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=cfg.models)
    ap.add_argument("--no-grids", action="store_true")
    ap.add_argument("--n-samples", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_dir, metrics_dir = Path(cfg.dirs.predictions), Path(cfg.dirs.metrics)
    folds = read_csv(Path(cfg.dirs.root) / "folds.csv")
    rows = []

    for name in args.models:
        print(f"\n=== {name} ===")
        parts = []
        for fold in range(cfg.split.n_folds):
            try:
                model, spec = load_fold_model(cfg, name, fold, device)
            except FileNotFoundError:
                print(f"  fold {fold}: no checkpoint, skipped")
                continue
            va = folds[folds.fold == fold]
            yt, yp_plain = predict_split(cfg, model, spec, va, device, tta=False)
            _, yp_tta = predict_split(cfg, model, spec, va, device, tta=True)
            del model
            torch.cuda.empty_cache()

            a_plain, a_tta = macro_auroc(yt, yp_plain), macro_auroc(yt, yp_tta)
            print(f"  fold {fold}: plain {a_plain:.4f}  tta {a_tta:.4f} "
                  f"({a_tta - a_plain:+.4f})")
            rows.append({"model": name, "fold": fold, "auroc_plain": a_plain,
                         "auroc_tta": a_tta, "gain": a_tta - a_plain})
            np.savez(pred_dir / f"{name}_fold{fold}_val_tta.npz",
                     y_true=yt, y_prob=yp_tta)
            parts.append((va, yt, yp_tta))

        if not parts:
            continue
        df = pd.concat([p[0] for p in parts]).reset_index(drop=True)
        y_true = np.concatenate([p[1] for p in parts])
        y_prob = np.concatenate([p[2] for p in parts])

        if not args.no_grids:
            thr_path = metrics_dir / f"thresholds_f1_{name}.csv"
            thr = (read_csv(thr_path).set_index("class").loc[CLASSES, "threshold"].to_numpy()
                   if thr_path.exists() else np.full(N_CLASSES, 0.5))
            # Rank by mean absolute error over the 14 labels; only score images
            # that actually carry a finding, so "worst" is not all-negative cases.
            err = np.abs(y_prob - y_true).mean(axis=1)
            has_finding = y_true.sum(axis=1) > 0
            order = np.argsort(err)
            best = [i for i in order if has_finding[i]][: args.n_samples]
            worst = [i for i in order[::-1] if has_finding[i]][: args.n_samples]
            img_dir = Path(cfg.data.images_small)
            prediction_grid(df, y_true, y_prob, img_dir, np.array(best),
                            f"Best held-out predictions — {name}",
                            pred_dir / f"samples_best_{name}.png", thr)
            prediction_grid(df, y_true, y_prob, img_dir, np.array(worst),
                            f"Worst held-out predictions — {name}",
                            pred_dir / f"samples_worst_{name}.png", thr)
            print(f"  wrote prediction grids -> {pred_dir}")

    if rows:
        tta = pd.DataFrame(rows).round(4)
        tta.to_csv(metrics_dir / "tta_gain.csv", index=False)
        print("\n=== TTA gain (macro AUROC) ===")
        print(tta.groupby("model")[["auroc_plain", "auroc_tta", "gain"]]
              .mean().round(4).to_string())


if __name__ == "__main__":
    main()
