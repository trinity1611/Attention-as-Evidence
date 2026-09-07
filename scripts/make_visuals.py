"""Save sample chest X-rays for each of the 14 findings (plus 'No Finding').

For every class we pick N exemplars, preferring images whose only label is that
class -- a single-label Cardiomegaly film shows the finding far more clearly
than one that also has Effusion and Atelectasis on top of it. If a class does
not have enough single-label cases (Fibrosis and Hernia usually do not), we
fall back to multi-label images and note the co-findings in the caption.

Outputs
    outputs/sample_images/<Class>/<n>_<filename>.png   individual exemplars
    outputs/sample_images/_overview.png                one grid, 3 per class
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, read_csv  # noqa: E402

NO_FINDING = "No Finding"

DATA_ROOT = Path(r"C:\ml-data\nih-cxr")
LABELS_CSV = DATA_ROOT / "sample_labels.csv"
IMAGE_DIR = DATA_ROOT / "sample" / "images"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "sample_images"

# Long side of the saved exemplars. The source PNGs are 1024x1024; half that is
# plenty for eyeballing and keeps the folder small enough to sit in OneDrive.
EXEMPLAR_PX = 512


def load_labels(csv_path: Path) -> pd.DataFrame:
    df = read_csv(csv_path)
    df["labels"] = df["Finding Labels"].str.split("|")
    return df


def pick(df: pd.DataFrame, cls: str, n: int, seed: int) -> pd.DataFrame:
    """Up to n rows for `cls`, single-label ones first, deterministic order."""
    has_cls = df[df["labels"].apply(lambda ls: cls in ls)]
    solo = has_cls[has_cls["labels"].apply(len) == 1]
    multi = has_cls[has_cls["labels"].apply(len) > 1]

    chosen = solo.sample(min(n, len(solo)), random_state=seed)
    if len(chosen) < n and len(multi):
        extra = multi.sample(min(n - len(chosen), len(multi)), random_state=seed)
        chosen = pd.concat([chosen, extra])
    return chosen


def caption(row: pd.Series, cls: str) -> str:
    others = [l for l in row["labels"] if l != cls]
    tail = f"\n+ {', '.join(others)}" if others else ""
    return f"{row['Patient Age']} {row['Patient Gender']} {row['View Position']}{tail}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_labels(LABELS_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_rows: list[tuple[str, list[tuple[Path, str]]]] = []

    for cls in CLASSES + [NO_FINDING]:
        rows = pick(df, cls, args.per_class, args.seed)
        cls_dir = OUT_DIR / cls.replace(" ", "_")
        cls_dir.mkdir(exist_ok=True)

        picked: list[tuple[Path, str]] = []
        for i, (_, row) in enumerate(rows.iterrows(), start=1):
            src = IMAGE_DIR / row["Image Index"]
            if not src.exists():
                print(f"  ! missing {src.name}")
                continue
            img = Image.open(src).convert("L")
            img.thumbnail((EXEMPLAR_PX, EXEMPLAR_PX))
            dst = cls_dir / f"{i}_{row['Image Index']}"
            img.save(dst)
            picked.append((dst, caption(row, cls)))

        n_solo = sum(1 for _, c in picked if "+" not in c)
        print(f"{cls:<20} {len(picked)} saved ({n_solo} single-label)")
        grid_rows.append((cls, picked))

    # One overview figure: rows = classes, cols = exemplars.
    ncols = args.per_class
    nrows = len(grid_rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.4 * nrows))
    axes = axes.reshape(nrows, ncols)
    for r, (cls, picked) in enumerate(grid_rows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picked):
                path, cap = picked[c]
                ax.imshow(Image.open(path), cmap="gray")
                ax.set_title(cap, fontsize=7, color="#444")
            if c == 0:
                ax.text(-0.08, 0.5, cls.replace("_", " "), transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0, 1, 0.985))
    # suptitle after tight_layout, else it lands on top of the first row's captions
    fig.suptitle("NIH ChestX-ray14 (5% sample) — exemplars per finding",
                 fontsize=14, y=0.997)
    overview = OUT_DIR / "_overview.png"
    fig.savefig(overview, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\noverview -> {overview}")


if __name__ == "__main__":
    main()
