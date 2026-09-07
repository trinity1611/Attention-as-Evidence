"""Check that a fresh clone is ready to run the app and (optionally) the pipeline.

    python scripts/check_setup.py

Prints one line per check: OK / WARN / FAIL. Everything marked WARN is optional
(the Streamlit app runs without the dataset and without Ollama); anything
marked FAIL must be fixed before `streamlit run app/Home.py` will work.
"""

from __future__ import annotations

import importlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results: list[tuple[str, str, str]] = []


def rec(level: str, name: str, detail: str = "") -> None:
    results.append((level, name, detail))
    print(f"[{level:4}] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------- python ---
v = sys.version_info
rec("OK" if (3, 10) <= v[:2] <= (3, 12) else "WARN", f"Python {v.major}.{v.minor}.{v.micro}",
    "" if (3, 10) <= v[:2] <= (3, 12) else "developed on 3.11; other versions are untested")

# -------------------------------------------------------------- packages ---
missing = []
for mod, pip_name in [("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"),
                      ("pandas", "pandas"), ("yaml", "PyYAML"), ("PIL", "pillow"), ("sklearn", "scikit-learn"),
                      ("matplotlib", "matplotlib"), ("pytorch_grad_cam", "grad-cam"), ("streamlit", "streamlit"),
                      ("plotly", "plotly"), ("reportlab", "reportlab"), ("requests", "requests")]:
    try:
        importlib.import_module(mod)
    except Exception:  # noqa: BLE001
        missing.append(pip_name)
if missing:
    rec("FAIL", "python packages", "missing: " + ", ".join(missing) + "  -> pip install -r requirements.txt")
else:
    import torch
    rec("OK", f"torch {torch.__version__}", "CUDA available: " + ("yes, " + torch.cuda.get_device_name(0)
                                                                   if torch.cuda.is_available()
                                                                   else "no (CPU inference, slower but fine)"))
for mod, pip_name, use in [("torchxrayvision", "torchxrayvision", "224 px CheXpert study"),
                           ("transformers", "transformers", "RAD-DINO probe"),
                           ("docx", "python-docx", "report builder")]:
    try:
        importlib.import_module(mod)
    except Exception:  # noqa: BLE001
        rec("WARN", pip_name, f"not installed; only needed for the {use}")

# ---------------------------------------------------------------- config ---
try:
    from src.config import CONFIG_PATH, load_config, meta_path
    cfg = load_config()
    rec("OK", f"config {CONFIG_PATH.name}", str(CONFIG_PATH))
except Exception as e:  # noqa: BLE001
    rec("FAIL", "config.yaml", repr(e))
    cfg = None

# ------------------------------------------------------------ checkpoint ---
ck = ROOT / "checkpoints" / "densenet121_full224.pt"
if ck.exists() and ck.stat().st_size > 10_000_000:
    rec("OK", "deployed classifier", f"{ck.relative_to(ROOT)} ({ck.stat().st_size / 1e6:.1f} MB)")
else:
    rec("FAIL", "deployed classifier", f"{ck} missing or truncated -- re-clone, or `git checkout -- checkpoints/`")
folds = sorted((ROOT / "checkpoints").glob("densenet121_imagenet_fold*.pt"))
rec("OK" if len(folds) == 5 else "WARN", f"sample-model folds: {len(folds)}/5",
    "" if len(folds) == 5 else "only needed for the comparative analysis, not for the app")

# --------------------------------------------------------------- outputs ---
need = ["outputs/folds.csv", "outputs/metrics/final_model_summary.json",
        "outputs/metrics/thresholds_f1_densenet121_full224_study.csv",
        "outputs/metrics/gradcam_summary_block3+4_layercam_none.csv",
        "outputs/metrics/gradcam_localization_block3+4_layercam_none.csv",
        "outputs/metrics/report_study_summary.csv", "outputs/metrics/gradcam_sweep.csv",
        "outputs/metrics/image_hashes.json", "outputs/hero/hero_assets.json"]
absent = [n for n in need if not (ROOT / n).exists()]
rec("FAIL" if absent else "OK", "result tables the app reads", ", ".join(absent) if absent else f"{len(need)} files present")

# ------------------------------------------------------------------ data ---
if cfg is not None:
    imgs = Path(cfg.data.images_full)
    n = len(list(imgs.glob("*.png"))) if imgs.exists() else 0
    if n >= 5606:
        rec("OK", "NIH sample images", f"{n} films at {imgs}")
    elif n:
        rec("WARN", "NIH sample images", f"only {n} films at {imgs} (expected 5,606)")
    else:
        rec("WARN", "NIH sample images", f"not found at {imgs}; app works without them (upload films), "
                                         "retraining needs them -- README -> Data")
    bb = meta_path(cfg, "BBox_List_2017.csv")
    rec("OK" if bb else "FAIL", "radiologist box list", str(bb) if bb else "missing from data root and repo data/")

# ---------------------------------------------------------------- ollama ---
try:
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
        tags = json.loads(r.read().decode())
    names = [m["name"] for m in tags.get("models", [])]
    if any(n.startswith("medgemma") for n in names):
        rec("OK", "Ollama + medgemma", ", ".join(n for n in names if n.startswith("medgemma")))
    else:
        rec("WARN", "Ollama running, medgemma not pulled", "run: ollama pull medgemma")
except Exception:  # noqa: BLE001
    rec("WARN", "Ollama not reachable on :11434", "optional; needed only for draft reports. "
                                                   "Install from ollama.com, then `ollama pull medgemma`")

# ---------------------------------------------------------------- verdict ---
fails = [r for r in results if r[0] == "FAIL"]
print()
if fails:
    print(f"{len(fails)} problem(s) to fix before the app will run.")
    sys.exit(1)
print("Ready: streamlit run app/Home.py")
