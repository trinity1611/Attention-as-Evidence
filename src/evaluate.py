"""Aggregate cross-validation results into report-ready tables and figures.

    python -m src.evaluate

Reads the per-fold prediction dumps written by train.py and produces:
    outputs/per_class_auroc.csv     mean +- std per class per model
    outputs/model_comparison.csv    macro AUROC per model
    outputs/thresholds.csv          per-class operating points (Youden's J)
    outputs/roc_curves.png
    outputs/auroc_by_class.png

Two macro averages are reported: over all 14 classes, and over the 10 with
enough positives to be meaningful. Hernia (13 positives in the whole sample,
2-3 per fold) produces an AUROC number, but it is not one to trust.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, load_config, read_csv# noqa: E402

# Below this many positives in the full sample, per-class AUROC is too noisy
# to report as a result. Flagged rather than hidden.
MIN_POSITIVES = 150


def load_fold_preds(out_dir: Path, model: str, suffix: str = ""
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    preds = []
    for path in sorted(out_dir.glob(f"{model}_fold*_val{suffix}.npz")):
        d = np.load(path)
        preds.append((d["y_true"], d["y_prob"]))
    return preds


def per_class_table(preds, model: str) -> pd.DataFrame:
    """AUROC per class per fold -> mean/std."""
    rows = []
    for fold, (y_true, y_prob) in enumerate(preds):
        for i, cls in enumerate(CLASSES):
            col = y_true[:, i]
            if 0 < col.sum() < len(col):
                rows.append({"model": model, "fold": fold, "class": cls,
                             "auroc": roc_auc_score(col, y_prob[:, i]),
                             "n_pos": int(col.sum())})
    return pd.DataFrame(rows)


def youden_thresholds(preds) -> pd.DataFrame:
    """Per-class operating point maximising (TPR - FPR), pooled over folds.

    A single 0.5 cut is wrong here: with pos_weight training and 2% prevalence
    classes, the useful threshold is nowhere near 0.5 and differs per class.
    """
    y_true = np.concatenate([t for t, _ in preds])
    y_prob = np.concatenate([p for _, p in preds])
    rows = []
    for i, cls in enumerate(CLASSES):
        col = y_true[:, i]
        if not (0 < col.sum() < len(col)):
            rows.append({"class": cls, "threshold": 0.5, "tpr": np.nan, "fpr": np.nan})
            continue
        fpr, tpr, thr = roc_curve(col, y_prob[:, i])
        j = int(np.argmax(tpr - fpr))
        rows.append({"class": cls, "threshold": float(thr[j]),
                     "tpr": float(tpr[j]), "fpr": float(fpr[j])})
    return pd.DataFrame(rows)


def plot_roc(preds, model: str, path: Path) -> None:
    y_true = np.concatenate([t for t, _ in preds])
    y_prob = np.concatenate([p for _, p in preds])
    fig, ax = plt.subplots(figsize=(7.5, 7))
    cmap = plt.get_cmap("tab20")
    for i, cls in enumerate(CLASSES):
        col = y_true[:, i]
        if not (0 < col.sum() < len(col)):
            continue
        fpr, tpr, _ = roc_curve(col, y_prob[:, i])
        auc = roc_auc_score(col, y_prob[:, i])
        ax.plot(fpr, tpr, lw=1.6, color=cmap(i % 20),
                label=f"{cls} ({auc:.3f}, n={int(col.sum())})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC by finding — {model}\n(pooled across CV folds)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_comparison(per_class: pd.DataFrame, counts: pd.Series, path: Path) -> None:
    piv = per_class.groupby(["class", "model"])["auroc"].mean().unstack()
    piv = piv.reindex(counts.sort_values(ascending=False).index)
    ax = piv.plot(kind="barh", figsize=(9, 8), width=0.8)
    ax.axvline(0.5, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC (mean over folds)")
    ax.set_ylabel("")
    labels = [f"{c}  (n={counts[c]})" + ("  ⚠" if counts[c] < MIN_POSITIVES else "")
              for c in piv.index]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Per-class AUROC by model\n⚠ = under 150 positives, underpowered")
    ax.legend(title="", fontsize=9)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=130)
    plt.close(ax.figure)


def paired_comparison(per_class: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Fold-paired comparison of each model against the baseline.

    Between-fold spread here (~0.035) dwarfs the between-model differences, so
    comparing means alone is underpowered. Every model saw identical folds, so
    pairing on fold cancels fold difficulty and tests the thing we care about.
    """
    from scipy import stats

    macro = per_class.groupby(["model", "fold"])["auroc"].mean().unstack()
    if baseline not in macro.index:
        return pd.DataFrame()

    rows = []
    for model in macro.index:
        if model == baseline:
            continue
        diff = (macro.loc[model] - macro.loc[baseline]).dropna()
        n = len(diff)
        t, p = stats.ttest_rel(macro.loc[model].dropna(), macro.loc[baseline].dropna())
        sem = diff.std(ddof=1) / np.sqrt(n)
        crit = stats.t.ppf(0.975, n - 1)
        rows.append({
            "model": model, "vs": baseline, "n_folds": n,
            "mean_diff": diff.mean(), "sd_diff": diff.std(ddof=1),
            "ci95_low": diff.mean() - crit * sem,
            "ci95_high": diff.mean() + crit * sem,
            "t": t, "p_value": p,
            "significant_at_0.05": bool(p < 0.05),
        })
    return pd.DataFrame(rows).round(5)


BASELINE = "densenet121_imagenet"


def discover_models(pred_dir: Path, suffix: str, configured) -> list[str]:
    """Every model with fold predictions on disk, not just the ones in config.

    Later experiments (RAD-DINO probe, full-data import) register themselves by
    writing <name>_fold<k>_val<suffix>.npz; listing the directory is what makes
    the comparison pick them up automatically. Baseline goes first so the paired
    table reads naturally.
    """
    import re
    found = set(configured)
    for path in pred_dir.glob(f"*_fold*_val{suffix}.npz"):
        m = re.match(r"(.+)_fold\d+_val", path.name)
        if m:
            found.add(m.group(1))
    found = [m for m in found if any(pred_dir.glob(f"{m}_fold*_val{suffix}.npz"))]
    return sorted(found, key=lambda m: (m != BASELINE, m))


def main() -> None:
    import argparse

    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="_tta to evaluate TTA predictions")
    args = ap.parse_args()
    out_dir = Path(cfg.dirs.metrics)
    pred_dir = Path(cfg.dirs.predictions)
    plot_dir = Path(cfg.dirs.plots)
    dist_dir = Path(cfg.dirs.distribution)
    counts = read_csv(dist_dir / "class_counts.csv", index_col=0)["positives"]

    all_pc, summary = [], []
    models = discover_models(pred_dir, args.suffix, cfg.models)
    print(f"models with predictions{args.suffix}: {models}")
    for model in models:
        preds = load_fold_preds(pred_dir, model, args.suffix)
        if not preds:
            print(f"skip {model}: no predictions found")
            continue
        print(f"{model}: {len(preds)} folds")
        pc = per_class_table(preds, model)
        all_pc.append(pc)

        well_powered = [c for c in CLASSES if counts[c] >= MIN_POSITIVES]
        by_fold = pc.groupby("fold")["auroc"].mean()
        by_fold_wp = pc[pc["class"].isin(well_powered)].groupby("fold")["auroc"].mean()
        summary.append({
            "model": model,
            "macro_auroc_all14": by_fold.mean(), "std_all14": by_fold.std(),
            "macro_auroc_wellpowered": by_fold_wp.mean(), "std_wellpowered": by_fold_wp.std(),
            "n_folds": len(preds),
        })
        plot_roc(preds, model, plot_dir / f"roc_{model}.png")
        youden_thresholds(preds).to_csv(out_dir / f"thresholds_{model}.csv", index=False)

    if not all_pc:
        print("nothing to evaluate -- run src.train first")
        return

    per_class = pd.concat(all_pc)
    agg = (per_class.groupby(["model", "class"])["auroc"]
           .agg(["mean", "std"]).round(4).reset_index())
    agg["n_pos_total"] = agg["class"].map(counts)
    agg["underpowered"] = agg["n_pos_total"] < MIN_POSITIVES
    agg.to_csv(out_dir / "per_class_auroc.csv", index=False)

    summary_df = pd.DataFrame(summary).round(4)
    summary_df.to_csv(out_dir / "model_comparison.csv", index=False)
    plot_comparison(per_class, counts, plot_dir / "auroc_by_class.png")

    paired = paired_comparison(per_class, baseline=BASELINE)
    if not paired.empty:
        paired.to_csv(out_dir / "paired_comparison.csv", index=False)

    print("\n=== model comparison ===")
    print(summary_df.to_string(index=False))
    if not paired.empty:
        print("\n=== paired vs densenet121_imagenet (same folds) ===")
        print(paired.to_string(index=False))
    print(f"\nwrote tables and figures to {out_dir}")


if __name__ == "__main__":
    main()
