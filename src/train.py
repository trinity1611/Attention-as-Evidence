"""Two-stage fine-tuning with patient-grouped cross-validation.

    python -m src.train                          # all models, all folds
    python -m src.train --models densenet121_chexpert --folds 0

Stage 1 trains only the classifier head on a frozen backbone; stage 2 unfreezes
everything with a much smaller backbone LR. With ~4,500 training images this
matters -- unfreezing from the start lets the random head push large gradients
into the pretrained features and wash them out.

Per fold we write:
    checkpoints/<model>_fold<k>.pt
    outputs/<model>_fold<k>_val.csv      per-image probabilities
    outputs/<model>_fold<k>_history.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, N_CLASSES, load_config  # noqa: E402
from src.dataset import make_loaders, pos_weight  # noqa: E402
from src.model import build_model  # noqa: E402


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    """Mean AUROC over classes that have both labels present in this split.

    A class with zero positives in a validation fold has undefined AUROC; we
    skip it rather than let sklearn raise or silently score it 0.5.
    """
    per_class = {}
    for i, cls in enumerate(CLASSES):
        col = y_true[:, i]
        if 0 < col.sum() < len(col):
            per_class[cls] = float(roc_auc_score(col, y_prob[:, i]))
    return float(np.mean(list(per_class.values()))) if per_class else float("nan"), per_class


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    probs, trues = [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
        if not train:
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            trues.append(y.cpu().numpy())

    if train:
        return total / n, None, None
    return total / n, np.concatenate(trues), np.concatenate(probs)


def train_fold(cfg, model_name: str, fold: int, device: torch.device,
               overwrite: bool = False) -> dict | None:
    tc = cfg.train
    ckpt_guard = Path(cfg.paths.checkpoints) / f"{model_name}_fold{fold}.pt"
    if ckpt_guard.exists() and not overwrite:
        # Protect finished runs: the 320px DenseNet is the baseline every later
        # experiment is compared against, so a casual re-run must not clobber it.
        print(f"  SKIP: {ckpt_guard.name} exists (pass --overwrite to retrain)")
        return None
    model, spec = build_model(model_name, N_CLASSES, tc.dropout)
    model.to(device)

    train_dl, val_dl, pos_counts = make_loaders(
        Path(cfg.dirs.root) / "folds.csv", Path(cfg.data.images_small),
        fold, tc.image_px, spec, tc.batch_size, tc.num_workers,
    )
    n_train = len(train_dl.dataset)
    pw = pos_weight(pos_counts, n_train, tc.pos_weight_cap).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Label smoothing for multi-label BCE: pull targets off the hard 0/1 rails.
    eps = tc.label_smoothing
    raw_criterion = criterion

    def smoothed(logits, y):
        return raw_criterion(logits, y * (1 - eps) + 0.5 * eps)

    scaler = torch.amp.GradScaler("cuda") if (tc.amp and device.type == "cuda") else None
    ckpt_dir, out_dir = Path(cfg.paths.checkpoints), Path(cfg.dirs.metrics)
    pred_dir = Path(cfg.dirs.predictions)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{model_name}_fold{fold}.pt"

    best = {"auroc": -1.0, "epoch": -1, "per_class": {}}
    history, stale = [], 0
    t0 = time.time()

    for stage in (1, 2):
        if stage == 1:
            model.freeze_backbone(True)
            optimizer = torch.optim.AdamW(
                model.classifier.parameters(), lr=tc.stage1_lr,
                weight_decay=tc.weight_decay)
            n_epochs = tc.stage1_epochs
        else:
            model.freeze_backbone(False)
            optimizer = torch.optim.AdamW(
                model.param_groups(tc.stage2_lr_backbone, tc.stage2_lr_head),
                weight_decay=tc.weight_decay)
            n_epochs = tc.stage2_epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        for ep in range(1, n_epochs + 1):
            tr_loss, _, _ = run_epoch(model, train_dl, smoothed, device, optimizer, scaler)
            va_loss, y_true, y_prob = run_epoch(model, val_dl, smoothed, device)
            sched.step()
            auroc, per_class = macro_auroc(y_true, y_prob)
            history.append({"stage": stage, "epoch": ep, "train_loss": tr_loss,
                            "val_loss": va_loss, "val_auroc": auroc})
            flag = ""
            if auroc > best["auroc"]:
                best = {"auroc": auroc, "epoch": len(history), "per_class": per_class}
                torch.save({"model": model_name, "fold": fold,
                            "state_dict": model.state_dict(),
                            "spec": {"in_channels": spec.in_channels, "norm": spec.norm},
                            "classes": CLASSES, "val_auroc": auroc}, ckpt_path)
                np.savez(pred_dir / f"{model_name}_fold{fold}_val.npz",
                         y_true=y_true, y_prob=y_prob)
                stale, flag = 0, "  *"
            else:
                stale += 1
            print(f"  s{stage} ep{ep:02d}  train {tr_loss:.4f}  val {va_loss:.4f} "
                  f" AUROC {auroc:.4f}{flag}")

            # Only early-stop in stage 2; stage 1 is a short fixed warmup.
            if stage == 2 and stale >= tc.early_stop_patience:
                print(f"  early stop (no gain in {stale} epochs)")
                break

    pd.DataFrame(history).to_csv(out_dir / f"{model_name}_fold{fold}_history.csv",
                                 index=False)
    mins = (time.time() - t0) / 60
    print(f"  best AUROC {best['auroc']:.4f} @ epoch {best['epoch']}  ({mins:.1f} min)")
    return {"model": model_name, "fold": fold, "auroc": best["auroc"],
            "minutes": round(mins, 1), **best["per_class"]}


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=cfg.models)
    ap.add_argument("--folds", nargs="+", type=int,
                    default=list(range(cfg.split.n_folds)))
    ap.add_argument("--overwrite", action="store_true",
                    help="retrain even if the fold checkpoint already exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    torch.manual_seed(cfg.split.seed)

    rows = []
    for name in args.models:
        for fold in args.folds:
            print(f"\n=== {name}  fold {fold} ===")
            res = train_fold(cfg, name, fold, device, overwrite=args.overwrite)
            if res is None:
                continue
            rows.append(res)
            pd.DataFrame(rows).to_csv(Path(cfg.dirs.metrics) / "cv_results.csv",
                                      index=False)

    if not rows:
        print("\nnothing trained (all checkpoints present; use --overwrite)")
        return
    df = pd.DataFrame(rows)
    print("\n=== summary (macro AUROC) ===")
    print(df.groupby("model")["auroc"].agg(["mean", "std", "count"]).to_string())
    (Path(cfg.dirs.metrics) / "cv_summary.json").write_text(
        json.dumps(df.groupby("model")["auroc"].agg(["mean", "std"]).to_dict(), indent=2))


if __name__ == "__main__":
    main()
