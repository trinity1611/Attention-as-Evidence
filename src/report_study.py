"""The grounding study: does telling MedGemma where the classifier looked reduce
the number of findings it asserts without support?

    python -m src.report_study --run --n 200        # generate (resumable, hours)
    python -m src.report_study --score              # metrics + figures from what exists

Films come from the 1,293-image shared test set, so the full-data model's
probabilities on them are honest. For each film we build:
  * probabilities   from the full-data model's TTA predictions on the official test split
  * flagged set     probabilities above per-finding F1-optimal thresholds, fitted on
                    the 24,000+ official-test films that are NOT in the study
  * zones           LayerCAM (adopted config) -> anatomical phrase, per flagged finding
and ask MedGemma to write under conditions A (image), B (+probabilities),
C (+zones). Every reply is appended to a JSONL file as it arrives, so the run
can be killed and resumed.

Scoring (per report, from the parsed JSON 'present' set):
  vs classifier flagged set : unsupported rate  = present but not flagged
                              (NOT called hallucination: it may be a correct reading the
                              classifier missed, a misreading, or prompt-driven assertion)
                              omission rate      = flagged but not present
  vs ground-truth labels    : precision / recall of the 'present' set
  localisation agreement    : condition C only -- does the model's stated
                              location share side and level with the zone given?

Outputs:
  outputs/reports/study_inputs.json           the films and their evidence
  outputs/reports/reports.jsonl               one line per (film, condition)
  outputs/reports/examples/<image>_<cond>.txt readable drafts for a handful of films
  outputs/metrics/report_study_per_report.csv
  outputs/metrics/report_study_summary.csv
  outputs/plots/report_study.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, N_CLASSES, load_config, read_csv  # noqa: E402

FULL = "densenet121_full224"
SIDES = ("right", "left", "bilateral", "central")
LEVELS = ("upper", "mid", "lower", "apex", "base", "basal", "costophrenic")


# --------------------------------------------------------------- inputs ---
def f1_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    from sklearn.metrics import precision_recall_curve
    thr = np.full(N_CLASSES, 0.5)
    for i in range(N_CLASSES):
        col = y_true[:, i]
        if not (0 < col.sum() < len(col)):
            continue
        p, r, t = precision_recall_curve(col, y_prob[:, i])
        f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
        if len(t):
            thr[i] = float(t[int(np.argmax(f1[:-1]))])
    return thr


def prepare_study_inputs(cfg, n_films: int, seed: int = 0, force: bool = False) -> list[dict]:
    """Select films, compute flagged sets and CAM zones. Cached to study_inputs.json."""
    out_dir = Path(cfg.dirs.root) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "study_inputs.json"
    if cache.exists() and not force:
        rows = json.loads(cache.read_text())
        if len(rows) >= n_films:
            for r in rows:
                r["probs"] = np.array(r["probs"], dtype=np.float32)
                r["image_path"] = Path(r["image_path"])
            return rows[:n_films]

    import torch
    from src.gradcam import CamEngine
    from src.medgemma import cam_to_zone, zone_phrase

    shared = read_csv(Path(cfg.dirs.metrics) / "comparison_set.csv")
    folds = read_csv(Path(cfg.dirs.root) / "folds.csv").set_index("image")
    full = np.load(Path(cfg.dirs.predictions) / f"{FULL}_test.npz")
    pos = {n: i for i, n in enumerate(full["images"])}

    # Balanced-ish selection: half abnormal, half normal, by ground truth.
    rng = np.random.default_rng(seed)
    shared = shared.copy()
    shared["abnormal"] = shared.image.map(lambda n: folds.loc[n, CLASSES].sum() > 0)
    n_ab = n_films // 2
    pick = pd.concat([
        shared[shared.abnormal].sample(min(n_ab, int(shared.abnormal.sum())), random_state=seed),
        shared[~shared.abnormal].sample(min(n_films - n_ab, int((~shared.abnormal).sum())),
                                        random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Thresholds fitted on the official test films NOT in the study.
    study_idx = np.array([pos[n] for n in pick.image])
    mask = np.ones(len(full["images"]), bool)
    mask[study_idx] = False
    thr = f1_thresholds(full["y_true"][mask], full["y_prob"][mask])
    pd.DataFrame({"class": CLASSES, "threshold": thr}).to_csv(
        Path(cfg.dirs.metrics) / f"thresholds_f1_{FULL}_study.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eng = CamEngine(cfg, device)
    img_dir = Path(cfg.data.images_full)
    from PIL import Image
    rows = []
    for k, r in pick.iterrows():
        probs = full["y_prob"][pos[r.image]].astype(np.float32)
        flagged = [CLASSES[i] for i in range(N_CLASSES) if probs[i] >= thr[i]]
        zones = {}
        if flagged:
            img = Image.open(img_dir / r.image).convert("L")
            _, cams = eng.run(FULL, None, img, [CLASSES.index(f) for f in flagged])
            for f, cam in zip(flagged, cams):
                zones[f] = zone_phrase(cam_to_zone(cam))
        truth = [c for c in CLASSES if folds.loc[r.image, c] == 1]
        rows.append({"image": r.image, "image_path": str(img_dir / r.image),
                     "view": str(folds.loc[r.image, "view"]), "truth": truth,
                     "probs": probs.round(4).tolist(), "flagged": flagged, "zones": zones})
        if (k + 1) % 25 == 0:
            print(f"  inputs {k + 1}/{len(pick)}")
    eng.release()
    cache.write_text(json.dumps(rows, indent=1))
    for r in rows:
        r["probs"] = np.array(r["probs"], dtype=np.float32)
        r["image_path"] = Path(r["image_path"])
    print(f"study inputs: {len(rows)} films "
          f"({sum(bool(r['truth']) for r in rows)} abnormal), "
          f"{sum(bool(r['flagged']) for r in rows)} with >=1 flagged finding")
    return rows


# ------------------------------------------------------------------ run ---
def run_study(cfg, n_films: int, conditions: tuple[str, ...], seed: int) -> None:
    from src.medgemma import generate_report
    rows = prepare_study_inputs(cfg, n_films, seed)
    out_dir = Path(cfg.dirs.root) / "reports"
    jsonl = out_dir / "reports.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["image"], d["condition"]))
    todo = [(r, c) for r in rows for c in conditions if (r["image"], c) not in done]
    print(f"{len(done)} reports already done; {len(todo)} to generate")
    t0 = time.time()
    with jsonl.open("a", encoding="utf-8") as fh:
        for k, (r, cond) in enumerate(todo, 1):
            try:
                out = generate_report(r["image_path"], cond, view=r["view"], probs=r["probs"],
                                      flagged=r["flagged"], zones=r["zones"])
            except Exception as e:  # noqa: BLE001 -- keep the run alive, record the failure
                out = {"condition": cond, "error": str(e), "findings": {}, "parse_ok": False,
                       "seconds": 0.0, "raw": "", "prose": "", "impression": "", "locations": {}}
            rec = {"image": r["image"], "condition": cond, "truth": r["truth"],
                   "flagged": r["flagged"], "zones": r["zones"], **out}
            rec.pop("prompt", None)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 5 == 0 or k == len(todo):
                el = (time.time() - t0) / 60
                print(f"  {k}/{len(todo)}  {el:.1f} min  (~{el / k * (len(todo) - k):.0f} min left)",
                      flush=True)


# ---------------------------------------------------------------- score ---
def _loc_agrees(stated: str, given: str) -> bool | None:
    if not stated or not given:
        return None
    s, g = stated.lower(), given.lower()
    side_g = next((w for w in SIDES if w in g), None)
    lvl_g = next((w for w in LEVELS if w in g), None)
    side_ok = side_g is None or side_g in s or (side_g == "bilateral" and "both" in s)
    lvl_ok = lvl_g is None or lvl_g in s or (lvl_g == "lower" and ("base" in s or "costophrenic" in s))
    return bool(side_ok and lvl_ok)


def score_study(cfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jsonl = Path(cfg.dirs.root) / "reports" / "reports.jsonl"
    recs = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for d in recs:
        present = {c for c, s in d.get("findings", {}).items() if s == "present"}
        flagged, truth = set(d["flagged"]), set(d["truth"])
        unsupported = present - flagged
        omit = flagged - present
        tp_truth = present & truth
        loc = [_loc_agrees(d.get("locations", {}).get(f, ""), d["zones"].get(f, ""))
               for f in present & set(d["zones"])] if d["condition"] == "C" else []
        loc = [x for x in loc if x is not None]
        rows.append({
            "image": d["image"], "condition": d["condition"], "parse_ok": d.get("parse_ok", False),
            "error": bool(d.get("error")), "seconds": d.get("seconds", 0),
            "n_present": len(present), "n_flagged": len(flagged), "n_truth": len(truth),
            "n_unsupported_vs_clf": len(unsupported), "n_omit_vs_clf": len(omit),
            # of the findings the classifier did not flag but MedGemma asserted:
            # how many were actually in the labels (a genuine catch) vs not
            "n_unsupported_but_true": len(unsupported & truth),
            "n_unsupported_and_false": len(unsupported - truth),
            "unsupported_rate_vs_clf": len(unsupported) / max(len(present), 1),
            "omit_rate_vs_clf": len(omit) / max(len(flagged), 1) if flagged else np.nan,
            "n_false_vs_truth": len(present - truth),
            "precision_vs_truth": len(tp_truth) / max(len(present), 1) if present else np.nan,
            "recall_vs_truth": len(tp_truth) / max(len(truth), 1) if truth else np.nan,
            "normal_film_kept_clean": (len(present) == 0) if not truth else np.nan,
            "loc_agree": np.mean(loc) if loc else np.nan, "n_loc_checked": len(loc),
        })
    df = pd.DataFrame(rows)
    met, plots = Path(cfg.dirs.metrics), Path(cfg.dirs.plots)
    df.round(4).to_csv(met / "report_study_per_report.csv", index=False)

    g = df.groupby("condition")
    summ = pd.DataFrame({
        "n_reports": g.size(),
        "parse_ok_rate": g.parse_ok.mean(),
        "mean_present_per_report": g.n_present.mean(),
        "unsupported_by_classifier_per_report": g.n_unsupported_vs_clf.mean(),
        "unsupported_rate_vs_classifier": g.unsupported_rate_vs_clf.mean(),
        "omission_rate_vs_classifier": g.omit_rate_vs_clf.mean(),
        "unsupported_but_true_per_report": g.n_unsupported_but_true.mean(),
        "unsupported_and_false_per_report": g.n_unsupported_and_false.mean(),
        "share_of_unsupported_that_were_true": (g.n_unsupported_but_true.sum()
                                                / g.n_unsupported_vs_clf.sum().clip(lower=1)),
        "false_findings_vs_truth_per_report": g.n_false_vs_truth.mean(),
        "precision_vs_truth": g.precision_vs_truth.mean(),
        "recall_vs_truth": g.recall_vs_truth.mean(),
        "normal_films_kept_clean": g.normal_film_kept_clean.mean(),
        "localisation_agreement": g.loc_agree.mean(),
        "mean_seconds": g.seconds.mean(),
    }).round(3)
    summ.to_csv(met / "report_study_summary.csv")
    with pd.option_context("display.width", 220):
        print(summ.T.to_string())

    # paired comparison across conditions on the same films
    piv = df.pivot_table(index="image", columns="condition", values="n_unsupported_vs_clf")
    if {"B", "C"} <= set(piv.columns):
        d_bc = (piv["C"] - piv["B"]).dropna()
        rng = np.random.default_rng(0)
        boots = [rng.choice(d_bc.to_numpy(), len(d_bc)).mean() for _ in range(2000)]
        print(f"\nfindings unsupported by the classifier, per report, C - B: {d_bc.mean():+.3f} "
              f"(95% CI [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}], "
              f"n={len(d_bc)} films)")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    labels = {"A": "A: image only", "B": "B: + probabilities", "C": "C: + Grad-CAM zone"}
    conds = [c for c in ("A", "B", "C") if c in summ.index]
    x = [labels[c] for c in conds]
    cols = ["#b3261e", "#d1873c", "#3b6ea5"][:len(conds)]
    axes[0].bar(x, summ.loc[conds, "unsupported_by_classifier_per_report"], color=cols)
    axes[0].set_title("Findings asserted that the classifier\ndid not flag (per report)")
    axes[1].bar(x, summ.loc[conds, "omission_rate_vs_classifier"], color=cols)
    axes[1].set_title("Share of flagged findings\nthe report omitted")
    axes[2].bar(x, summ.loc[conds, "false_findings_vs_truth_per_report"], color=cols)
    axes[2].set_title("Findings asserted that are not\nin the ground truth (per report)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"MedGemma grounding study - {int(summ.n_reports.max())} films per condition",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(plots / "report_study.png", dpi=130)
    print(f"\nwrote report_study_summary.csv, report_study_per_report.csv, report_study.png")

    # readable examples
    ex_dir = Path(cfg.dirs.root) / "reports" / "examples"
    ex_dir.mkdir(exist_ok=True)
    by_img: dict[str, dict] = {}
    for d in recs:
        by_img.setdefault(d["image"], {})[d["condition"]] = d
    for image, conds_d in list(by_img.items())[:8]:
        for c, d in conds_d.items():
            (ex_dir / f"{Path(image).stem}_{c}.txt").write_text(
                f"IMAGE {image}   CONDITION {c}\nTRUTH: {', '.join(d['truth']) or 'No finding'}\n"
                f"FLAGGED: {', '.join(d['flagged']) or 'none'}\nZONES: {d['zones']}\n\n"
                f"--- JSON findings ---\n{json.dumps(d.get('findings', {}), indent=1)}\n\n"
                f"--- report ---\n{d.get('prose', '')}\n", encoding="utf-8")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--conditions", default="ABC")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.run:
        run_study(cfg, args.n, tuple(args.conditions), args.seed)
    if args.score:
        score_study(cfg)


if __name__ == "__main__":
    main()
