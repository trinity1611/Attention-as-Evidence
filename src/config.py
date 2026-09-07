"""Config loading and the canonical label list.

Everything imports CLASSES from here so the 14 columns keep a fixed order
across preprocessing, training, evaluation and the UI.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Alphabetical, matching the NIH label strings exactly. "No Finding" is not a
# class -- it is the all-zero vector.
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
N_CLASSES = len(CLASSES)


def read_csv(path, **kwargs) -> pd.DataFrame:
    """pd.read_csv with whitespace-tolerant headers and string values.

    Some CSVs in outputs/ have been column-aligned by an editor/CSV formatter,
    which pads both headers (' train_loss   ') and values ('AP  '). Numeric
    columns survive that, but string lookups and column names do not. Every
    read in this project goes through here so a cosmetic reformat cannot break
    a pipeline run.
    """
    df = pd.read_csv(path, skipinitialspace=True, **kwargs)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        # Do not test `dtype == object`: pandas 3 gives string columns a
        # dedicated "str" dtype, so that check silently skips them.
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass  # mixed-type object column; leave it alone
    return df


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


# outputs/ is split by artefact kind so the folder stays navigable as runs
# accumulate. Every script writes through cfg.dirs.* rather than building paths.
OUTPUT_SUBDIRS = {
    "sample_images": "sample_images",   # per-class exemplar X-rays
    "distribution": "distribution",     # label counts, co-occurrence, fold balance
    "predictions": "predictions",       # raw prediction dumps + actual-vs-predicted grids
    "plots": "plots",                   # confusion matrices, ROC, loss curves
    "metrics": "metrics",               # CSV/JSON result tables
}


def load_config(path: Path | str = CONFIG_PATH) -> SimpleNamespace:
    """Load config.yaml as a dot-accessible namespace, creating output dirs."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _ns(raw)
    # Resolve project-relative output dirs to absolute paths.
    cfg.paths.checkpoints = str(PROJECT_ROOT / cfg.paths.checkpoints)
    cfg.paths.outputs = str(PROJECT_ROOT / cfg.paths.outputs)

    out = Path(cfg.paths.outputs)
    cfg.dirs = _ns({k: str(out / v) for k, v in OUTPUT_SUBDIRS.items()})
    cfg.dirs.root = str(out)
    for d in OUTPUT_SUBDIRS.values():
        (out / d).mkdir(parents=True, exist_ok=True)
    return cfg
