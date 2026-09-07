"""Training curves and metric visualisations.

    python -m src.plots
    python -m src.plots --suffix _tta

Reads the per-fold history CSVs and prediction dumps and writes to
outputs/plots/:

    loss_curves_<model>.png      train/val loss + val AUROC per epoch, per fold
    pr_curves_<model>.png        precision-recall per finding (with AUPRC)
    metrics_by_class_<model>.png precision / recall / F1 bars per finding
    calibration_<model>.png      reliability curve

AUPRC is included because it is the metric that actually matters at 2-4%
prevalence: AUROC is optimistic under heavy imbalance, and F1 at one threshold
throws away the rest of the curve.
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
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, N_CLASSES, load_config, read_csv# noqa: E402


def plot_loss_curves(hist_paths: list[Path], model: str, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap = plt.get_cmap("tab10")
    for k, hp in enumerate(sorted(hist_paths)):
        h = read_csv(hp)
        # Normalise headers: one run's CSV came back with padded column names,
        # so don't trust them to be clean.
        h.columns = [c.strip() for c in h.columns]
        missing = {"train_loss", "val_loss", "val_auroc"} - set(h.columns)
        if missing:
            print(f"  ! {hp.name} missing {sorted(missing)}, skipped")
            continue
        x = np.arange(1, len(h) + 1)
        c = cmap(k)
        ax1.plot(x, h["train_loss"], color=c, lw=1.5, label=f"fold {k} train")
        ax1.plot(x, h["val_loss"], color=c, lw=1.5, ls="--", label=f"fold {k} val")
        ax2.plot(x, h["val_auroc"], color=c, lw=1.6, label=f"fold {k}")
        # Stage boundary: where the backbone unfreezes.
        if (h["stage"] == 2).any():
            b = int((h["stage"] == 1).sum())
            for ax in (ax1, ax2):
                ax.axvline(b + 0.5, color="#999", lw=0.8, ls=":", zorder=0)

    ax1.set_xlabel("epoch (stage 1 then stage 2)")
    ax1.set_ylabel("BCE loss")
    ax1.set_title("Loss — solid train, dashed val\n(dotted line = backbone unfrozen)")
    ax1.legend(fontsize=7, ncol=2)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("macro AUROC (held-out fold)")
    ax2.set_title("Validation AUROC")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Training curves — {model}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr_curves(y_true, y_prob, model: str, path: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab20")
    rows = []
    for i, cls in enumerate(CLASSES):
        col = y_true[:, i]
        if not (0 < col.sum() < len(col)):
            continue
        p, r, _ = precision_recall_curve(col, y_prob[:, i])
        ap = average_precision_score(col, y_prob[:, i])
        prev = col.mean()
        rows.append({"class": cls, "auprc": ap, "prevalence": prev,
                     "lift_over_chance": ap / prev, "n_pos": int(col.sum())})
        ax.plot(r, p, lw=1.6, color=cmap(i % 20), label=f"{cls} ({ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall by finding — {model}\n"
                 "(AUPRC in legend; chance = class prevalence)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows).round(4)


def plot_metrics_by_class(y_true, y_prob, thr, model: str, path: Path) -> None:
    pred = (y_prob >= thr).astype(int)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, pred, average=None, zero_division=0, labels=range(N_CLASSES))
    order = np.argsort(-sup)
    x = np.arange(N_CLASSES)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.27
    ax.bar(x - w, p[order], w, label="precision", color="#3b6ea5")
    ax.bar(x, r[order], w, label="recall", color="#e08a3c")
    ax.bar(x + w, f1[order], w, label="F1", color="#4c9a6a")
    ax.set_xticks(x, [f"{CLASSES[i]}\n(n={sup[i]})" for i in order],
                  rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Precision / Recall / F1 by finding — {model}\n"
                 "(held-out, per-class F1-optimal thresholds; sorted by support)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_calibration(y_true, y_prob, model: str, path: Path) -> None:
    """Reliability curve pooled over classes.

    Training with pos_weight deliberately breaks calibration -- probabilities
    come out far above true frequency. Worth showing before MedGemma consumes
    these numbers as if they were confidences.
    """
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(y_prob.ravel(), bins) - 1
    obs, exp, cnt = [], [], []
    flat_true = y_true.ravel()
    for b in range(10):
        m = idx == b
        if m.sum() > 30:
            obs.append(flat_true[m].mean())
            exp.append(y_prob.ravel()[m].mean())
            cnt.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(exp, obs, "o-", color="#3b6ea5", lw=1.8, label="model")
    for e, o, c in zip(exp, obs, cnt):
        ax.annotate(f"{c:,}", (e, o), textcoords="offset points",
                    xytext=(5, -9), fontsize=7, color="#666")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title(f"Calibration — {model}\n(labels = samples per bin)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=cfg.models)
    ap.add_argument("--suffix", default="", help="_tta to use TTA predictions")
    args = ap.parse_args()

    plot_dir, metrics_dir = Path(cfg.dirs.plots), Path(cfg.dirs.metrics)
    pred_dir = Path(cfg.dirs.predictions)

    for name in args.models:
        hists = list(metrics_dir.glob(f"{name}_fold*_history.csv"))
        if hists:
            plot_loss_curves(hists, name, plot_dir / f"loss_curves_{name}.png")
            print(f"{name}: loss curves from {len(hists)} folds")

        files = sorted(pred_dir.glob(f"{name}_fold*_val{args.suffix}.npz"))
        if not files:
            print(f"{name}: no predictions{args.suffix or ''} found")
            continue
        y_true = np.concatenate([np.load(f)["y_true"] for f in files])
        y_prob = np.concatenate([np.load(f)["y_prob"] for f in files])

        auprc = plot_pr_curves(y_true, y_prob, name, plot_dir / f"pr_curves_{name}.png")
        auprc.to_csv(metrics_dir / f"auprc_{name}.csv", index=False)

        thr_path = metrics_dir / f"thresholds_f1_{name}.csv"
        thr = (read_csv(thr_path).set_index("class").loc[CLASSES, "threshold"].to_numpy()
               if thr_path.exists() else np.full(N_CLASSES, 0.5))
        plot_metrics_by_class(y_true, y_prob, thr, name,
                              plot_dir / f"metrics_by_class_{name}.png")
        plot_calibration(y_true, y_prob, name, plot_dir / f"calibration_{name}.png")

        print(f"  macro AUPRC {auprc['auprc'].mean():.4f} "
              f"(mean lift over prevalence {auprc['lift_over_chance'].mean():.1f}x)")

    print(f"\nwrote figures to {plot_dir}")


if __name__ == "__main__":
    main()
