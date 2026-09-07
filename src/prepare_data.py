"""One-time preprocessing: resize images and build patient-grouped CV folds.

Run once before training:
    python -m src.prepare_data

Two jobs:
  1. Downscale the 1024px PNGs to 256px JPEG. Decoding full-size PNGs every
     epoch is CPU-bound and starves the GPU; 256px keeps room to random-crop
     to 224 and to try 320 later.
  2. Assign each image a CV fold, grouped by Patient ID and stratified on the
     rarest finding the image carries.

Writes:
    <images_small>/*.jpg
    outputs/folds.csv        image, patient, fold, one column per class
    outputs/class_counts.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, load_config, read_csv# noqa: E402


def resize_images(src_dir: Path, dst_dir: Path, px: int, quality: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    srcs = sorted(src_dir.glob("*.png"))
    done = 0
    for i, src in enumerate(srcs, start=1):
        dst = dst_dir / (src.stem + ".jpg")
        if dst.exists():
            continue
        img = Image.open(src).convert("L").resize((px, px), Image.BILINEAR)
        img.save(dst, quality=quality)
        done += 1
        if i % 500 == 0:
            print(f"  {i}/{len(srcs)}")
    print(f"resized {done} new / {len(srcs)} total -> {dst_dir}")
    return len(srcs)


def build_label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Explode 'Finding Labels' into 14 binary columns."""
    labels = df["Finding Labels"].str.split("|")
    for cls in CLASSES:
        df[cls] = labels.apply(lambda ls, c=cls: int(c in ls))
    return df


def rarest_label(row: pd.Series, order: list[str]) -> str:
    """Stratum = the rarest finding on this image, or 'NoFinding'.

    Stratifying on the rarest label is what keeps Hernia (13 positives) from
    landing entirely in one fold.
    """
    for cls in order:
        if row[cls] == 1:
            return cls
    return "NoFinding"


def plot_distribution(df: pd.DataFrame, order: list[str], dist_dir: Path,
                      group_col: str) -> None:
    """Label counts, fold balance and co-occurrence -- the dataset EDA figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = df[CLASSES].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(counts.index, counts.values, color="#3b6ea5")
    for b, v in zip(bars, counts.values):
        ax.text(v + 8, b.get_y() + b.get_height() / 2, f"{int(v):,}",
                va="center", fontsize=9)
    n_nf = int((df[CLASSES].sum(axis=1) == 0).sum())
    ax.set_title(f"Positives per finding (n={len(df):,} images, "
                 f"{n_nf:,} with no finding)")
    ax.set_xlabel("positive images")
    ax.set_xlim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(dist_dir / "class_distribution.png", dpi=130)
    plt.close(fig)

    # Findings per image: shows how much of the set is genuinely multi-label.
    per_img = df[CLASSES].sum(axis=1).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(per_img.index.astype(str), per_img.values, color="#3b6ea5")
    for x, v in zip(per_img.index.astype(str), per_img.values):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("findings on one image")
    ax.set_ylabel("images")
    ax.set_title("Label cardinality")
    fig.tight_layout()
    fig.savefig(dist_dir / "labels_per_image.png", dpi=130)
    plt.close(fig)

    # Co-occurrence: P(column | row). Explains why some classes are hard to
    # separate -- Consolidation rarely appears without Infiltration.
    m = np.zeros((len(CLASSES), len(CLASSES)))
    for i, a in enumerate(CLASSES):
        sub = df[df[a] == 1]
        if len(sub):
            m[i] = sub[CLASSES].mean().to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and m[i, j] >= 0.15:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(column | row)")
    ax.set_title("Finding co-occurrence")
    fig.tight_layout()
    fig.savefig(dist_dir / "cooccurrence.png", dpi=130)
    plt.close(fig)

    # Per-fold positive counts -- evidence the stratification worked.
    fold_counts = df.groupby("fold")[order].sum()
    fold_counts.to_csv(dist_dir / "positives_per_fold.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    fold_counts.T.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("positive images")
    ax.set_title("Positives per fold (patient-grouped, stratified)")
    ax.legend(title="fold", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(dist_dir / "positives_per_fold.png", dpi=130)
    plt.close(fig)

    # Patient-level view: how many images each patient contributes.
    per_pat = df.groupby(group_col).size().value_counts().sort_index()
    per_pat.to_csv(dist_dir / "images_per_patient.csv",
                   header=["patients"], index_label="images")


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg.dirs.root)
    dist_dir = Path(cfg.dirs.distribution)

    print("== resizing ==")
    n_imgs = resize_images(
        Path(cfg.data.images_full), Path(cfg.data.images_small),
        cfg.data.resize_px, cfg.data.jpeg_quality,
    )

    print("\n== labels ==")
    df = read_csv(cfg.data.labels_csv)
    df = build_label_matrix(df)
    counts = {c: int(df[c].sum()) for c in CLASSES}
    order = sorted(CLASSES, key=lambda c: counts[c])  # rarest first
    print("  positives:", {c: counts[c] for c in order})

    if len(df) != n_imgs:
        print(f"  ! csv rows ({len(df)}) != images ({n_imgs})")

    print("\n== folds ==")
    df["stratum"] = df.apply(rarest_label, axis=1, order=order)
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(
        n_splits=cfg.split.n_folds, shuffle=True, random_state=cfg.split.seed
    )
    for fold, (_, val_idx) in enumerate(
        sgkf.split(df, df["stratum"], groups=df[cfg.split.group_col])
    ):
        df.loc[df.index[val_idx], "fold"] = fold

    # A patient appearing in two folds means the grouping silently failed.
    leaked = df.groupby(cfg.split.group_col)["fold"].nunique()
    n_leaked = int((leaked > 1).sum())
    print(f"  patients spanning >1 fold: {n_leaked}  (must be 0)")
    assert n_leaked == 0, "patient leakage across folds"

    print(df.groupby("fold").size().to_string())
    print("\n  positives per fold:")
    print(df.groupby("fold")[order].sum().to_string())

    cols = ["Image Index", cfg.split.group_col, "View Position", "fold"] + CLASSES
    folds = df[cols].rename(
        columns={"Image Index": "image", cfg.split.group_col: "patient",
                 "View Position": "view"}
    )
    folds.to_csv(out_dir / "folds.csv", index=False)
    pd.Series(counts).sort_values(ascending=False).to_csv(
        dist_dir / "class_counts.csv", header=["positives"]
    )

    print("\n== distribution figures ==")
    plot_distribution(df, order, dist_dir, cfg.split.group_col)
    print(f"  -> {dist_dir}")
    print(f"\nwrote {out_dir / 'folds.csv'}")


if __name__ == "__main__":
    main()
