"""Register a Kaggle full-dataset run and compare it fairly with local models.

    python scripts/import_kaggle_run.py --prepare-only          # just write the comparison set
    python scripts/import_kaggle_run.py --dir ~/Downloads/kaggle_out

The full-data model and the sample-trained models were trained on different
data and evaluated on different splits, so their headline AUROCs are not
comparable. This script scores every model on the SAME images: the 1,293 films
that are in our 5,606-image sample AND in NIH's official test_list.txt.

  * sample models: out-of-fold predictions (never trained on the image)
  * full model:    test-set predictions (never trained on the image)

Copies the Kaggle artefacts into the project layout under the model name
`densenet121_full<img>` and writes:
    outputs/metrics/comparison_set.csv               the 1,293 image ids
    outputs/metrics/comparison_shared_test.csv       per-class AUROC per model
    outputs/metrics/comparison_shared_test_macro.csv macro + bootstrap CI vs baseline
    outputs/plots/comparison_shared_test.png
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, load_config, read_csv  # noqa: E402
from scripts.test_predictions import load_oof  # noqa: E402

BASELINE = "densenet121_imagenet"
MIN_POS = 10          # per-class AUROC needs at least this many positives
BOOT = 2000


def comparison_set(cfg) -> pd.DataFrame:
    folds = read_csv(Path(cfg.dirs.root) / "folds.csv")
    test_list = Path(cfg.data.root) / "meta" / "test_list.txt"
    if not test_list.exists():
        sys.exit(f"missing {test_list}; download test_list.txt from the full Kaggle dataset")
    test_names = set(test_list.read_text().split())
    shared = folds[folds.image.isin(test_names)].reset_index(drop=True)
    out = Path(cfg.dirs.metrics) / "comparison_set.csv"
    shared[["image", "patient", "view", "fold"]].to_csv(out, index=False)
    print(f"comparison set: {len(shared)} images, {shared.patient.nunique()} patients, "
          f"{(shared[CLASSES].sum(axis=1) > 0).mean():.1%} abnormal -> {out.name}")
    return shared


def local_model_names(pred_dir: Path, suffix: str) -> list[str]:
    names = set()
    for p in pred_dir.glob(f"*_fold*_val{suffix}.npz"):
        m = re.match(r"(.+)_fold\d+_val", p.name)
        if m:
            names.add(m.group(1))
    return sorted(names, key=lambda m: (m != BASELINE, m))


def import_run(cfg, run_dir: Path) -> str:
    ckpts = list(run_dir.glob("densenet121_full*.pt"))
    if not ckpts:
        sys.exit(f"no densenet121_full*.pt in {run_dir}")
    ckpt = ckpts[0]
    name = ckpt.stem                                   # densenet121_full224
    shutil.copy2(ckpt, Path(cfg.paths.checkpoints) / ckpt.name)
    for fn, dst in (("full_test_preds.npz", Path(cfg.dirs.predictions) / f"{name}_test.npz"),
                    ("history.csv", Path(cfg.dirs.metrics) / f"{name}_history.csv"),
                    ("summary.json", Path(cfg.dirs.metrics) / f"{name}_summary.json")):
        src = run_dir / fn
        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  ! {fn} not found in {run_dir}")
    summ = run_dir / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        print(f"imported {name}: trained on {s.get('n_train'):,} images, "
              f"official-test macro AUROC {s.get('test_macro_auroc_tta'):.4f}")
    return name


def scores_on_shared(cfg, shared: pd.DataFrame, suffix: str):
    """{model: (y_true, y_prob)} restricted to the shared image set, same row order."""
    pred_dir = Path(cfg.dirs.predictions)
    out = {}
    for m in local_model_names(pred_dir, suffix):
        df, yt, yp = load_oof(cfg, m, suffix)
        pos = {n: i for i, n in enumerate(df.image)}
        idx = shared.image.map(pos).to_numpy()
        out[m] = (yt[idx], yp[idx])
    for p in pred_dir.glob("densenet121_full*_test.npz"):
        d = np.load(p)
        pos = {n: i for i, n in enumerate(d["images"])}
        idx = shared.image.map(pos)
        if idx.isna().any():
            missing = int(idx.isna().sum())
            print(f"  ! {p.name}: {missing} shared images missing from full test preds")
        idx = idx.dropna().astype(int).to_numpy()
        out[p.stem.replace("_test", "")] = (d["y_true"][idx], d["y_prob"][idx])
    return out


def macro_over(y, p, classes_idx):
    return float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in classes_idx]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, help="folder with the downloaded Kaggle outputs")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--suffix", default="_tta")
    args = ap.parse_args()

    cfg = load_config()
    shared = comparison_set(cfg)
    if args.prepare_only:
        return
    if args.dir:
        import_run(cfg, args.dir)

    scores = scores_on_shared(cfg, shared, args.suffix)
    if not scores:
        sys.exit("no models to compare")
    y_ref = next(iter(scores.values()))[0]
    usable = [i for i in range(len(CLASSES)) if MIN_POS <= y_ref[:, i].sum() < len(y_ref)]
    skipped = [CLASSES[i] for i in range(len(CLASSES)) if i not in usable]
    print(f"classes scored: {len(usable)}  skipped (<{MIN_POS} positives): {skipped}")

    rows = []
    for m, (y, p) in scores.items():
        for i in usable:
            rows.append({"model": m, "class": CLASSES[i], "n_pos": int(y[:, i].sum()),
                         "auroc": roc_auc_score(y[:, i], p[:, i])})
    per_class = pd.DataFrame(rows)
    piv = per_class.pivot(index="class", columns="model", values="auroc")
    piv["n_pos"] = per_class.groupby("class")["n_pos"].first()
    piv = piv.sort_values("n_pos", ascending=False)
    piv.round(4).to_csv(Path(cfg.dirs.metrics) / "comparison_shared_test.csv")

    # macro AUROC + paired bootstrap CI for the difference against the baseline
    rng = np.random.default_rng(0)
    n = len(y_ref)
    macro_rows = []
    base = scores.get(BASELINE)
    for m, (y, p) in scores.items():
        macro = macro_over(y, p, usable)
        row = {"model": m, "n_images": n, "macro_auroc": macro}
        if base is not None and m != BASELINE:
            diffs = []
            for _ in range(BOOT):
                b = rng.integers(0, n, n)
                ok = [i for i in usable if 0 < y[b][:, i].sum() < n]
                if not ok:
                    continue
                diffs.append(macro_over(y[b], p[b], ok) - macro_over(base[0][b], base[1][b], ok))
            diffs = np.array(diffs)
            row.update({"diff_vs_baseline": macro - macro_over(base[0], base[1], usable),
                        "ci95_low": np.percentile(diffs, 2.5),
                        "ci95_high": np.percentile(diffs, 97.5),
                        "p_diff_le_0": float((diffs <= 0).mean())})
        macro_rows.append(row)
    macro_df = pd.DataFrame(macro_rows).round(4)
    macro_df.to_csv(Path(cfg.dirs.metrics) / "comparison_shared_test_macro.csv", index=False)

    with pd.option_context("display.width", 200):
        print("\n=== per-class AUROC on the shared test images ===")
        print(piv.round(3).to_string())
        print("\n=== macro AUROC, paired bootstrap vs baseline ===")
        print(macro_df.to_string(index=False))

    ax = piv.drop(columns="n_pos").plot(kind="barh", figsize=(9, 7), width=0.8)
    ax.axvline(0.5, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlim(0.4, 1.0)
    ax.set_yticklabels([f"{c}  (n={int(piv.loc[c, 'n_pos'])})" for c in piv.index], fontsize=9)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("")
    ax.set_title(f"Same {n} held-out images, every model\n"
                 "(sample models: out-of-fold; full-data model: official test split)")
    ax.legend(fontsize=8)
    ax.figure.tight_layout()
    ax.figure.savefig(Path(cfg.dirs.plots) / "comparison_shared_test.png", dpi=130)
    print(f"\nwrote comparison tables and plot")


if __name__ == "__main__":
    main()
