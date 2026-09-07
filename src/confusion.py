"""Confusion matrices and precision/recall/F1 for train and held-out data.

    python -m src.confusion
    python -m src.confusion --models densenet121_imagenet

Multi-label means there is no single 14x14 confusion matrix: an image can carry
several findings at once, so "predicted class" is not one value. We report:

  * one 2x2 matrix per finding (the correct decomposition)
  * a label-confusion matrix -- for images truly positive for i, how often is j
    also predicted -- which shows which findings get conflated
  * precision / recall / F1 per class, plus micro, macro and weighted averages

Held-out predictions are pooled out-of-fold: every image is scored by the one
fold model that never trained on it. Train predictions are recomputed here with
evaluation transforms (no augmentation) from the saved checkpoints.

Thresholds come from thresholds_<model>.csv (Youden's J, fitted on validation)
and the SAME thresholds are applied to the training set -- refitting per split
would make the two sets incomparable.

Note train metrics are optimistic: each fold checkpoint was selected on its own
validation fold, so the gap to held-out is the quantity of interest, not the
train numbers themselves.
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
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             multilabel_confusion_matrix)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, N_CLASSES, load_config, read_csv# noqa: E402
from src.dataset import CXRDataset, build_transforms  # noqa: E402
from src.model import build_model  # noqa: E402


@torch.no_grad()
def predict_train_split(cfg, model_name: str, fold: int, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the fold's checkpoint over its own training images, eval transforms."""
    ckpt = Path(cfg.paths.checkpoints) / f"{model_name}_fold{fold}.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)

    model, spec = build_model(model_name, N_CLASSES, cfg.train.dropout)
    model.load_state_dict(state["state_dict"])
    model.to(device).eval()

    df = read_csv(Path(cfg.dirs.root) / "folds.csv")
    tr = df[df.fold != fold]
    ds = CXRDataset(tr, Path(cfg.data.images_small),
                    build_transforms(cfg.train.image_px, spec, train=False))
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                    num_workers=cfg.train.num_workers, pin_memory=True)

    probs, trues = [], []
    for x, y in dl:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(out.float()).cpu().numpy())
        trues.append(y.numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(trues), np.concatenate(probs)


def f1_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-class threshold maximising F1, swept on the TRAINING predictions.

    Youden's J is the right operating point for ROC/screening but a poor one
    for F1: at 2-4% prevalence it buys recall with a flood of false positives.
    Fitting on train and applying to held-out keeps the held-out numbers honest
    -- tuning thresholds on the test split would inflate them.
    """
    from sklearn.metrics import precision_recall_curve

    thr = np.full(N_CLASSES, 0.5)
    for i in range(N_CLASSES):
        col = y_true[:, i]
        if not (0 < col.sum() < len(col)):
            continue
        p, r, t = precision_recall_curve(col, y_prob[:, i])
        f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
        # precision_recall_curve returns one more point than thresholds
        best = int(np.argmax(f1[:-1])) if len(t) else 0
        if len(t):
            thr[i] = float(t[best])
    return thr


def metrics_table(y_true: np.ndarray, y_pred: np.ndarray, split: str,
                  model: str) -> pd.DataFrame:
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(N_CLASSES))
    rows = [{"model": model, "split": split, "class": CLASSES[i],
             "precision": p[i], "recall": r[i], "f1": f1[i], "support": int(sup[i])}
            for i in range(N_CLASSES)]
    for avg in ("micro", "macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0)
        rows.append({"model": model, "split": split, "class": f"<{avg} avg>",
                     "precision": pa, "recall": ra, "f1": fa,
                     "support": int(y_true.sum())})
    return pd.DataFrame(rows).round(4)


def plot_per_class_cm(y_true, y_pred, title: str, path: Path) -> None:
    """4x4 grid of per-finding 2x2 confusion matrices, row-normalised."""
    cms = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))
    for i, ax in enumerate(axes.flat):
        if i >= N_CLASSES:
            ax.axis("off")
            continue
        cm = cms[i]
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]:,}\n{norm[r, c]:.0%}", ha="center",
                        va="center", fontsize=9,
                        color="white" if norm[r, c] > 0.5 else "#222")
        ax.set_title(f"{CLASSES[i]}  (n={int(y_true[:, i].sum())})", fontsize=10)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=8)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_label_confusion(y_true, y_pred, title: str, path: Path) -> None:
    """Row i = images truly positive for i; column j = fraction also predicted j.

    Not a true confusion matrix (labels are not mutually exclusive), but it is
    what reveals conflation -- e.g. Consolidation cases firing Infiltration.
    """
    m = np.zeros((N_CLASSES, N_CLASSES))
    for i in range(N_CLASSES):
        mask = y_true[:, i] == 1
        if mask.sum():
            m[i] = y_pred[mask].mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(m, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES), CLASSES, rotation=90, fontsize=8)
    ax.set_yticks(range(N_CLASSES), CLASSES, fontsize=8)
    ax.set_xlabel("predicted positive")
    ax.set_ylabel("true positive for")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if m[i, j] >= 0.30:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if m[i, j] < 0.7 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction predicted positive")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=cfg.models)
    ap.add_argument("--suffix", default="", help="_tta to use TTA predictions")
    args = ap.parse_args()

    out = Path(cfg.dirs.metrics)
    pred_dir = Path(cfg.dirs.predictions)
    plot_dir = Path(cfg.dirs.plots)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics = []

    for name in args.models:
        thr_path = out / f"thresholds_{name}.csv"
        if not thr_path.exists():
            print(f"skip {name}: run src.evaluate first")
            continue
        thr = read_csv(thr_path).set_index("class").loc[CLASSES, "threshold"].to_numpy()
        print(f"\n=== {name} ===")
        print("  thresholds:", np.round(thr, 3))

        # Held-out: pool the out-of-fold predictions saved during training.
        va_true, va_prob = [], []
        for f in range(cfg.split.n_folds):
            d = np.load(pred_dir / f"{name}_fold{f}_val{args.suffix}.npz")
            va_true.append(d["y_true"])
            va_prob.append(d["y_prob"])
        va_true, va_prob = np.concatenate(va_true), np.concatenate(va_prob)

        tr_true, tr_prob = [], []
        for f in range(cfg.split.n_folds):
            t, p = predict_train_split(cfg, name, f, device)
            tr_true.append(t)
            tr_prob.append(p)
            print(f"  train fold {f}: {len(t)} images")
        tr_true, tr_prob = np.concatenate(tr_true), np.concatenate(tr_prob)

        thr_f1 = f1_optimal_thresholds(tr_true, tr_prob)
        pd.DataFrame({"class": CLASSES, "threshold": thr_f1}).to_csv(
            out / f"thresholds_f1_{name}.csv", index=False)

        for strategy, cut in (("youden", thr), ("f1opt", thr_f1)):
            for split, yt, yp in (("train", tr_true, tr_prob),
                                  ("heldout", va_true, va_prob)):
                pred = (yp >= cut).astype(int)
                m = metrics_table(yt, pred, split, name)
                m.insert(2, "threshold_strategy", strategy)
                all_metrics.append(m)
                plot_per_class_cm(
                    yt, pred, f"Per-finding confusion — {name} ({split}, {strategy})",
                    plot_dir / f"cm_perclass_{name}_{split}_{strategy}.png")
                plot_label_confusion(
                    yt, pred, f"Label conflation — {name} ({split}, {strategy})",
                    plot_dir / f"cm_labels_{name}_{split}_{strategy}.png")
                macro = m[m["class"] == "<macro avg>"].iloc[0]
                micro = m[m["class"] == "<micro avg>"].iloc[0]
                print(f"  {strategy:<7} {split:<8} macro P/R/F1 "
                      f"{macro.precision:.3f}/{macro.recall:.3f}/{macro.f1:.3f}"
                      f"   micro F1 {micro.f1:.3f}")

    if all_metrics:
        df = pd.concat(all_metrics)
        df.to_csv(out / "classification_metrics.csv", index=False)
        print(f"\nwrote {out / 'classification_metrics.csv'} and confusion figures")


if __name__ == "__main__":
    main()
