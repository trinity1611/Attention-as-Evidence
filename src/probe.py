"""Linear probe on frozen RAD-DINO features, same 5 folds as the CNN.

    python -m src.probe

One linear layer (768 -> 14) trained with the same weighted BCE as the CNN,
full-batch on GPU, early-stopped on held-out macro AUROC. Training uses both
the original and flipped embeddings (doubling the data); held-out predictions
are written twice, plain and flip-averaged (TTA), so src.evaluate / confusion /
plots compare it to the CNN with identical machinery via --suffix _tta.

The model is registered under the name `raddino_linear`, so its files never
collide with the DenseNet run. Writes:
    checkpoints/raddino_linear_fold<k>.pt          (scaler + weights, tiny)
    outputs/predictions/raddino_linear_fold<k>_val{,_tta}.npz
    outputs/metrics/raddino_linear_fold<k>_history.csv
    outputs/metrics/probe_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, N_CLASSES, load_config, read_csv  # noqa: E402
from src.dataset import pos_weight  # noqa: E402
from src.train import macro_auroc  # noqa: E402

MODEL_NAME = "raddino_linear"
EPOCHS, PATIENCE, LR, WD = 400, 30, 1e-3, 1e-3


def fit_fold(Xtr, Xtr_flip, Ytr, Xva, Xva_flip, Yva, pw, device):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    z = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32)).to(device)  # noqa: E731

    X = torch.cat([z(Xtr), z(Xtr_flip)])
    Y = torch.from_numpy(np.concatenate([Ytr, Ytr])).to(device)
    Xv, Xv_flip = z(Xva), z(Xva_flip)

    head = nn.Linear(X.shape[1], N_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))

    best, best_state, stale, hist = -1.0, None, 0, []
    for ep in range(1, EPOCHS + 1):
        head.train()
        opt.zero_grad()
        loss = crit(head(X), Y * 0.95 + 0.025)  # same label smoothing as the CNN
        loss.backward()
        opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            p_plain = torch.sigmoid(head(Xv))
            p_tta = (p_plain + torch.sigmoid(head(Xv_flip))) / 2
            vloss = crit(head(Xv), Yva_t := torch.from_numpy(Yva).to(device)).item()
        auroc, _ = macro_auroc(Yva, p_tta.cpu().numpy())
        hist.append({"stage": 2, "epoch": ep, "train_loss": loss.item(),
                     "val_loss": vloss, "val_auroc": auroc})
        if auroc > best:
            best, stale = auroc, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            best_preds = (p_plain.cpu().numpy(), p_tta.cpu().numpy())
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    head.load_state_dict(best_state)
    return head, mu, sd, best, best_preds, pd.DataFrame(hist)


def main() -> None:
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_path = Path(cfg.data.root) / "features" / "raddino_cls.npz"
    if not feat_path.exists():
        sys.exit(f"missing {feat_path} -- run `python -m src.features` first")
    F = np.load(feat_path)
    names, feats, feats_flip = list(F["images"]), F["feats"], F["feats_flip"]
    pos = {n: i for i, n in enumerate(names)}

    folds = read_csv(Path(cfg.dirs.root) / "folds.csv")
    idx = folds["image"].map(pos).to_numpy()
    X, Xf = feats[idx], feats_flip[idx]
    Y = folds[CLASSES].to_numpy(dtype=np.float32)

    pred_dir, met_dir = Path(cfg.dirs.predictions), Path(cfg.dirs.metrics)
    ckpt_dir = Path(cfg.paths.checkpoints)
    rows = []
    for k in range(cfg.split.n_folds):
        tr, va = (folds.fold != k).to_numpy(), (folds.fold == k).to_numpy()
        pw = pos_weight(Y[tr].sum(0), int(tr.sum()), cfg.train.pos_weight_cap)
        head, mu, sd, best, (p_plain, p_tta), hist = fit_fold(
            X[tr], Xf[tr], Y[tr], X[va], Xf[va], Y[va], pw, device)
        a_plain, _ = macro_auroc(Y[va], p_plain)
        print(f"fold {k}: plain {a_plain:.4f}  tta {best:.4f}  "
              f"({len(hist)} epochs)")
        rows.append({"model": MODEL_NAME, "fold": k, "auroc_plain": a_plain,
                     "auroc_tta": best, "epochs": len(hist)})
        np.savez(pred_dir / f"{MODEL_NAME}_fold{k}_val.npz", y_true=Y[va], y_prob=p_plain)
        np.savez(pred_dir / f"{MODEL_NAME}_fold{k}_val_tta.npz", y_true=Y[va], y_prob=p_tta)
        hist.to_csv(met_dir / f"{MODEL_NAME}_fold{k}_history.csv", index=False)
        torch.save({"model": MODEL_NAME, "fold": k, "backbone": "microsoft/rad-dino",
                    "state_dict": head.state_dict(), "mu": mu, "sd": sd,
                    "classes": CLASSES, "val_auroc": best},
                   ckpt_dir / f"{MODEL_NAME}_fold{k}.pt")

    res = pd.DataFrame(rows)
    res.to_csv(met_dir / "probe_results.csv", index=False)
    print(f"\n{MODEL_NAME}: macro AUROC {res.auroc_tta.mean():.4f} "
          f"± {res.auroc_tta.std():.4f} (TTA), {res.auroc_plain.mean():.4f} plain")


if __name__ == "__main__":
    main()
