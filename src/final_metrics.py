"""Final metrics for the deployed classifier (the Kaggle-trained full-data
DenseNet-121) on NIH's official test split.

    python -m src.final_metrics

Threshold-free metrics (AUROC, AUPRC) use all 25,596 test films. Threshold
metrics (accuracy, precision, recall, F1) need a per-finding cut-off, and
fitting it on the films you then score inflates the result; so the test split
is halved at random, thresholds are fitted (F1-optimal) on half A and every
threshold metric is reported on half B (12,798 films).

Writes:
    outputs/metrics/final_model_per_class.csv
    outputs/metrics/final_model_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, N_CLASSES, load_config  # noqa: E402

FULL = "densenet121_full224"


def f1_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    thr = np.full(N_CLASSES, 0.5)
    for i in range(N_CLASSES):
        col = y[:, i]
        if 0 < col.sum() < len(col):
            pr, rc, t = precision_recall_curve(col, p[:, i])
            f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
            if len(t):
                thr[i] = float(t[int(np.argmax(f1[:-1]))])
    return thr


def main() -> None:
    cfg = load_config()
    met = Path(cfg.dirs.metrics)
    d = np.load(Path(cfg.dirs.predictions) / f"{FULL}_test.npz")
    y, p = d["y_true"].astype(int), d["y_prob"].astype(np.float32)
    n = len(y)

    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    fit, rep = idx[: n // 2], idx[n // 2:]
    thr = f1_thresholds(y[fit], p[fit])
    pred = (p[rep] >= thr).astype(int)
    yr, pr_ = y[rep], p[rep]

    rows = []
    for i, c in enumerate(CLASSES):
        prec, recall, f1, sup = precision_recall_fscore_support(
            yr[:, i], pred[:, i], average="binary", zero_division=0)
        rows.append({
            "finding": c, "n_pos_test": int(y[:, i].sum()), "prevalence": float(y[:, i].mean()),
            "auroc": float(roc_auc_score(y[:, i], p[:, i])),
            "auprc": float(average_precision_score(y[:, i], p[:, i])),
            "threshold": float(thr[i]),
            "accuracy": float(accuracy_score(yr[:, i], pred[:, i])),
            "precision": float(prec), "recall": float(recall), "f1": float(f1),
            "n_pos_reported_half": int(yr[:, i].sum()),
        })
    df = pd.DataFrame(rows)
    df.round(4).to_csv(met / "final_model_per_class.csv", index=False)

    macro = {k: float(df[k].mean()) for k in ("auroc", "auprc", "accuracy", "precision", "recall", "f1")}
    mp, mr, mf, _ = precision_recall_fscore_support(yr, pred, average="micro", zero_division=0)
    # subset accuracy = every one of the 14 labels right on a film; label-wise accuracy = macro above
    exact = float((pred == yr).all(axis=1).mean())
    any_true = yr.sum(1) > 0
    detect = float(((pred == 1) & (yr == 1)).any(axis=1)[any_true].mean())
    clean = float((pred.sum(1) == 0)[~any_true].mean())

    summary = {
        "model": FULL, "checkpoint": f"checkpoints/{FULL}.pt",
        "trained_on": "77,988 NIH ChestX-ray14 films (official train_val minus a 10 % patient-grouped "
                      "validation split), 224 px, 8 epochs, one Kaggle T4 GPU-hour",
        "evaluated_on": "NIH official test split, 25,596 films (25,596 for AUROC/AUPRC; thresholds fitted "
                        "on a random half, threshold metrics reported on the other 12,798)",
        "n_test": int(n), "n_threshold_fit": int(len(fit)), "n_threshold_report": int(len(rep)),
        "macro": macro,
        "micro": {"precision": float(mp), "recall": float(mr), "f1": float(mf)},
        "film_level": {"exact_label_set_match": exact,
                       "abnormal_films_with_at_least_one_correct_finding": detect,
                       "normal_films_with_no_finding_reported": clean,
                       "share_abnormal_in_report_half": float(any_true.mean())},
        "notes": [
            "Accuracy per finding is dominated by the negatives (prevalence 0.2-18 %); it is reported "
            "because it was asked for, but AUROC / AUPRC / F1 are the informative numbers.",
            "Probabilities are uncalibrated (class-weighted training); thresholds are per finding.",
        ],
    }
    (met / "final_model_summary.json").write_text(json.dumps(summary, indent=2))

    with pd.option_context("display.width", 200):
        print(df[["finding", "n_pos_test", "auroc", "auprc", "threshold", "accuracy", "precision",
                  "recall", "f1"]].round(3).to_string(index=False))
    print("\nmacro:", {k: round(v, 3) for k, v in macro.items()})
    print("micro P/R/F1:", round(mp, 3), round(mr, 3), round(mf, 3))
    print("film-level:", {k: round(v, 3) for k, v in summary["film_level"].items()})


if __name__ == "__main__":
    main()
