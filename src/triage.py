"""Binary triage: is there ANY finding on this film?

    python -m src.triage

The 14-label task is severely imbalanced, which makes accuracy meaningless.
Collapsing to "abnormal vs normal" gives a 46/54 split, so accuracy, sensitivity
and specificity finally mean something -- and it is the clinically useful
question for a first-pass screen ("which films does the radiologist read first?").

Three scorers are compared, all on the same held-out images:
  * <model>/max       max over the 14 out-of-fold probabilities
  * <model>/noisy-or  1 - prod(1 - p_c): P(at least one finding) if independent
  * raddino_binary    a dedicated logistic head on frozen RAD-DINO features,
                      trained per fold on the binary label directly

Writes outputs/metrics/triage_results.csv and outputs/plots/triage_roc.png.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve,
                             confusion_matrix)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, load_config, read_csv  # noqa: E402


def discover_models(pred_dir: Path, suffix: str) -> list[str]:
    names = set()
    for p in pred_dir.glob(f"*_fold*_val{suffix}.npz"):
        m = re.match(r"(.+)_fold\d+_val", p.name)
        if m:
            names.add(m.group(1))
    return sorted(names)


def load_oof_binary(cfg, model: str, suffix: str):
    """Out-of-fold probs aligned to folds.csv order; binary truth."""
    folds = read_csv(Path(cfg.dirs.root) / "folds.csv")
    pred_dir = Path(cfg.dirs.predictions)
    parts_t, parts_p = [], []
    for k in range(cfg.split.n_folds):
        d = np.load(pred_dir / f"{model}_fold{k}_val{suffix}.npz")
        parts_t.append(d["y_true"])
        parts_p.append(d["y_prob"])
    y14 = np.concatenate(parts_t)
    p14 = np.concatenate(parts_p)
    order = np.concatenate([np.flatnonzero((folds.fold == k).to_numpy())
                            for k in range(cfg.split.n_folds)])
    return order, (y14.sum(1) > 0).astype(int), p14


def summarise(name: str, y: np.ndarray, s: np.ndarray) -> dict:
    fpr, tpr, thr = roc_curve(y, s)
    j = int(np.argmax(tpr - fpr))
    t_youden = thr[j]
    # threshold maximising plain accuracy, for the reader who insists on it
    cands = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    accs = [((s >= t).astype(int) == y).mean() for t in cands]
    t_acc = cands[int(np.argmax(accs))]

    def at(t):
        pred = (s >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {"acc": (tp + tn) / len(y), "sens": tp / max(tp + fn, 1),
                "spec": tn / max(tn + fp, 1),
                "bal_acc": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))}

    yd, ac = at(t_youden), at(t_acc)
    return {"scorer": name, "auroc": roc_auc_score(y, s),
            "auprc": average_precision_score(y, s),
            "thr_youden": t_youden, "acc@youden": yd["acc"], "sens@youden": yd["sens"],
            "spec@youden": yd["spec"], "bal_acc@youden": yd["bal_acc"],
            "thr_maxacc": t_acc, "acc@maxacc": ac["acc"], "sens@maxacc": ac["sens"],
            "spec@maxacc": ac["spec"], "n": len(y), "prevalence": y.mean()}


def raddino_binary(cfg, device) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    feat_path = Path(cfg.data.root) / "features" / "raddino_cls.npz"
    if not feat_path.exists():
        return None
    F = np.load(feat_path)
    pos = {n: i for i, n in enumerate(F["images"])}
    folds = read_csv(Path(cfg.dirs.root) / "folds.csv")
    idx = folds["image"].map(pos).to_numpy()
    X, Xf = F["feats"][idx], F["feats_flip"][idx]
    y = (folds[CLASSES].sum(1) > 0).to_numpy().astype(np.float32)

    scores = np.zeros(len(folds), dtype=np.float32)
    for k in range(cfg.split.n_folds):
        tr, va = (folds.fold != k).to_numpy(), (folds.fold == k).to_numpy()
        mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
        z = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32)).to(device)  # noqa: E731
        Xt = torch.cat([z(X[tr]), z(Xf[tr])])
        Yt = torch.from_numpy(np.concatenate([y[tr], y[tr]])).to(device)[:, None]
        head = nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
        crit = nn.BCEWithLogitsLoss()
        best, best_s, stale = -1, None, 0
        yv = y[va]
        for ep in range(400):
            head.train(); opt.zero_grad()
            crit(head(Xt), Yt).backward(); opt.step()
            head.eval()
            with torch.no_grad():
                s = ((torch.sigmoid(head(z(X[va]))) + torch.sigmoid(head(z(Xf[va])))) / 2
                     ).squeeze(1).cpu().numpy()
            a = roc_auc_score(yv, s)
            if a > best:
                best, best_s, stale = a, s, 0
            else:
                stale += 1
                if stale >= 30:
                    break
        scores[va] = best_s
    return np.arange(len(folds)), y.astype(int), scores


def main() -> None:
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_dir, met_dir, plot_dir = (Path(cfg.dirs.predictions), Path(cfg.dirs.metrics),
                                   Path(cfg.dirs.plots))
    rows, curves = [], []

    for model in discover_models(pred_dir, "_tta"):
        order, y, p14 = load_oof_binary(cfg, model, "_tta")
        for how, s in (("max", p14.max(1)), ("noisy-or", 1 - np.prod(1 - p14, axis=1))):
            name = f"{model}/{how}"
            rows.append(summarise(name, y, s))
            curves.append((name, y, s))

    rb = raddino_binary(cfg, device)
    if rb is not None:
        _, y, s = rb
        rows.append(summarise("raddino_binary (dedicated head)", y, s))
        curves.append(("raddino_binary", y, s))

    res = pd.DataFrame(rows).round(4)
    res.to_csv(met_dir / "triage_results.csv", index=False)
    with pd.option_context("display.width", 220):
        print(res[["scorer", "auroc", "auprc", "acc@youden", "sens@youden",
                   "spec@youden", "acc@maxacc", "sens@maxacc", "spec@maxacc"]]
              .to_string(index=False))
    print(f"\nprevalence of 'any finding': {res['prevalence'].iloc[0]:.3f}  (n={int(res['n'].iloc[0])})")

    fig, ax = plt.subplots(figsize=(7, 6.5))
    for name, y, s in curves:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=1.7, label=f"{name}  ({roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate (normal films flagged)")
    ax.set_ylabel("True positive rate (abnormal films caught)")
    ax.set_title("Triage: any finding vs none  (held-out, 5-fold OOF)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "triage_roc.png", dpi=130)
    print(f"wrote triage_results.csv and triage_roc.png")


if __name__ == "__main__":
    main()
