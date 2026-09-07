"""Inspect predictions on a sample of images against ground truth.

    python scripts/test_predictions.py                 # 24 random held-out images
    python scripts/test_predictions.py --n 40 --seed 7
    python scripts/test_predictions.py --per-class     # one example per finding

Uses the pooled out-of-fold TTA predictions, so every image shown was scored by
the fold model that never trained on it. (The 5-model ensemble in
src.infer.predict_ensemble is for genuinely new images -- scoring it on this
dataset would be contaminated, since each image trained four of the five folds.)

Writes a contact sheet to outputs/predictions/ and a per-image CSV, and prints
a readable table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, N_CLASSES, load_config, read_csv# noqa: E402
from src.infer import prediction_grid  # noqa: E402


def load_oof(cfg, model: str, suffix: str = "_tta"):
    """Pooled out-of-fold predictions, aligned row-for-row with the fold rows."""
    folds = read_csv(Path(cfg.dirs.root) / "folds.csv")
    pred_dir = Path(cfg.dirs.predictions)
    dfs, trues, probs = [], [], []
    for fold in range(cfg.split.n_folds):
        path = pred_dir / f"{model}_fold{fold}_val{suffix}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        va = folds[folds.fold == fold]
        if len(va) != len(d["y_true"]):
            raise RuntimeError(f"fold {fold}: {len(va)} rows vs "
                               f"{len(d['y_true'])} predictions")
        dfs.append(va)
        trues.append(d["y_true"])
        probs.append(d["y_prob"])
    if not dfs:
        raise FileNotFoundError(f"no {suffix} predictions for {model}")
    return (pd.concat(dfs).reset_index(drop=True),
            np.concatenate(trues), np.concatenate(probs))


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.models[0])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class", action="store_true",
                    help="one positive example per finding instead of random")
    ap.add_argument("--suffix", default="_tta")
    args = ap.parse_args()

    df, y_true, y_prob = load_oof(cfg, args.model, args.suffix)
    metrics_dir = Path(cfg.dirs.metrics)
    thr_path = metrics_dir / f"thresholds_f1_{args.model}.csv"
    thr = (read_csv(thr_path).set_index("class").loc[CLASSES, "threshold"].to_numpy()
           if thr_path.exists() else np.full(N_CLASSES, 0.5))

    rng = np.random.default_rng(args.seed)
    if args.per_class:
        idx, tag = [], "per_class"
        for j, cls in enumerate(CLASSES):
            pos = np.flatnonzero(y_true[:, j] == 1)
            if len(pos):
                idx.append(int(rng.choice(pos)))
        idx = np.array(idx)
    else:
        idx = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
        tag = f"random{len(idx)}_seed{args.seed}"

    rows = []
    for i in idx:
        true_lbl = [CLASSES[j] for j in range(N_CLASSES) if y_true[i, j] == 1]
        pred_lbl = [CLASSES[j] for j in range(N_CLASSES) if y_prob[i, j] >= thr[j]]
        top = np.argsort(-y_prob[i])[:3]
        tp = set(true_lbl) & set(pred_lbl)
        rows.append({
            "image": df.iloc[i]["image"],
            "view": df.iloc[i]["view"],
            "true": "|".join(true_lbl) or "No Finding",
            "predicted": "|".join(pred_lbl) or "No Finding",
            "top3": ", ".join(f"{CLASSES[j]} {y_prob[i, j]:.2f}" for j in top),
            "n_true": len(true_lbl),
            "n_pred": len(pred_lbl),
            "hits": len(tp),
            "outcome": ("exact" if set(true_lbl) == set(pred_lbl)
                        else "partial" if tp
                        else "miss"),
        })
    out = pd.DataFrame(rows)

    pred_dir = Path(cfg.dirs.predictions)
    csv_path = pred_dir / f"test_samples_{tag}.csv"
    out.to_csv(csv_path, index=False)
    prediction_grid(df, y_true, y_prob, Path(cfg.data.images_small), idx,
                    f"Sampled held-out predictions ({tag}) — {args.model}",
                    pred_dir / f"test_samples_{tag}.png", thr)

    with pd.option_context("display.width", 200, "display.max_colwidth", 46):
        print(out[["image", "view", "true", "predicted", "outcome"]].to_string(index=False))

    print(f"\n--- outcome counts (n={len(out)}) ---")
    print(out["outcome"].value_counts().to_string())
    normals = out[out["true"] == "No Finding"]
    findings = out[out["true"] != "No Finding"]
    print(f"\nimages with a true finding: {len(findings)}   "
          f"of these, >=1 correct: {(findings['hits'] > 0).sum()}")
    if len(normals):
        print(f"truly normal images: {len(normals)}   "
              f"called normal: {(normals['n_pred'] == 0).sum()}")
    print(f"\nwrote {csv_path.name} and the contact sheet to {pred_dir}")


if __name__ == "__main__":
    main()
