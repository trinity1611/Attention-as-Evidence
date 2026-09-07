"""Inference service behind the app: one call turns an uploaded film into
probabilities, heatmaps, localisation text, evidence metrics and (optionally) a
MedGemma report with its scores.

Everything heavy is loaded once and cached on the module (the Streamlit page
wraps `get_service()` in st.cache_resource). The classifier is the full-data
DenseNet-121; the explanation is the sweep-adopted LayerCAM configuration.

Honesty rules baked in here rather than left to the UI:
  * hit / miss / IoU are computed ONLY when the film is one of the NIH films
    with a radiologist box (recognised by content hash). For any other film the
    service returns the per-finding reliability measured in our evaluation
    instead of inventing a metric.
  * ground-truth-based report scores are returned only for NIH films whose
    labels we have; otherwise only classifier-relative scores are given.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import CLASSES, N_CLASSES, load_config, meta_path, read_csv
from src.gradcam import BBOX_TO_CLASS, CamEngine, DEFAULT_CAM, cam_tag, localisation_metrics
from src.medgemma import cam_to_zone, generate_report, zone_phrase

FULL = "densenet121_full224"
VERSION = {"classifier": FULL, "cam": cam_tag(DEFAULT_CAM), "writer": "medgemma:latest (Ollama, Q4_K_M)"}


@dataclass
class Analysis:
    image_name: str
    view: str | None
    probs: np.ndarray
    thresholds: np.ndarray
    flagged: list[str]
    top3: list[str]
    cams: dict[str, np.ndarray]              # finding -> 1024x1024 map in 0..1
    zones: dict[str, str]                    # finding -> anatomical phrase
    cam_area: dict[str, float]               # finding -> fraction of image >= 50 % peak
    known_nih: bool = False
    truth: list[str] | None = None           # labels, if a known NIH film
    boxes: list[dict] = field(default_factory=list)   # radiologist boxes + metrics, if any
    reliability: dict[str, dict] = field(default_factory=dict)  # per-finding eval stats
    seconds: float = 0.0

    def summary_rows(self) -> list[dict]:
        rows = []
        for i, c in enumerate(CLASSES):
            rows.append({"finding": c, "probability": float(self.probs[i]),
                         "threshold": float(self.thresholds[i]),
                         "flagged": c in self.flagged, "zone": self.zones.get(c, "")})
        return sorted(rows, key=lambda r: -r["probability"])


class Service:
    def __init__(self):
        self.cfg = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = CamEngine(self.cfg, self.device)
        self.model, _, self.tf, self.px = self.engine.get(FULL, None)

        met = Path(self.cfg.dirs.metrics)
        thr_path = met / f"thresholds_f1_{FULL}_study.csv"
        self.thresholds = (read_csv(thr_path).set_index("class").loc[CLASSES, "threshold"].to_numpy()
                           if thr_path.exists() else np.full(N_CLASSES, 0.5))

        # per-finding localisation reliability from our bbox evaluation
        self.reliability = {}
        summ = met / f"gradcam_summary_{cam_tag(DEFAULT_CAM)}.csv"
        if summ.exists():
            s = read_csv(summ)
            s = s[s.model == FULL]
            for _, r in s.iterrows():
                self.reliability[r.finding] = {
                    "n_boxes": int(r.n), "hit_rate": float(r.pointing_hit_rate),
                    "hits": int(round(r.pointing_hit_rate * r.n)), "mean_iou": float(r.mean_iou)}

        # known NIH films: content hash -> name, labels, boxes
        self.folds = read_csv(Path(self.cfg.dirs.root) / "folds.csv").set_index("image")
        bb_path = meta_path(self.cfg, "BBox_List_2017.csv")   # data root, else the repo copy
        if bb_path is not None:
            bb = pd.read_csv(bb_path).iloc[:, :6]
            bb.columns = ["image", "finding", "x", "y", "w", "h"]
            bb["finding"] = bb["finding"].map(lambda f: BBOX_TO_CLASS.get(f, f))
            self.bboxes = bb[bb.image.isin(self.folds.index)]
        else:
            self.bboxes = pd.DataFrame(columns=["image", "finding", "x", "y", "w", "h"])
        self._hash_index: dict[str, str] | None = None

    # ---------------------------------------------------------- lookup ---
    def _build_hash_index(self) -> None:
        idx_file = Path(self.cfg.dirs.metrics) / "image_hashes.json"
        if idx_file.exists():
            self._hash_index = json.loads(idx_file.read_text())
            return
        img_dir = Path(self.cfg.data.images_full)
        index = {}
        if img_dir.exists():
            for name in self.folds.index:
                p = img_dir / name
                if p.exists():
                    index[hashlib.md5(p.read_bytes()).hexdigest()] = name
            idx_file.write_text(json.dumps(index))
        self._hash_index = index

    def identify(self, file_bytes: bytes) -> str | None:
        if self._hash_index is None:
            self._build_hash_index()
        return self._hash_index.get(hashlib.md5(file_bytes).hexdigest())

    # -------------------------------------------------------- analysis ---
    def analyse(self, image: Image.Image, file_bytes: bytes | None = None,
                view: str | None = None, name: str = "upload",
                extra_findings: list[str] | None = None) -> Analysis:
        t0 = time.time()
        img_gray = image.convert("L")
        if img_gray.size != (1024, 1024):
            img_gray = img_gray.resize((1024, 1024), Image.BILINEAR)

        known = self.identify(file_bytes) if file_bytes else None
        truth = None
        if known:
            name = known
            truth = [c for c in CLASSES if self.folds.loc[known, c] == 1]
            view = view or str(self.folds.loc[known, "view"])

        x = self.tf(img_gray).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p1 = torch.sigmoid(self.model(x))
            p2 = torch.sigmoid(self.model(torch.flip(x, dims=[3])))
            probs = ((p1 + p2) / 2).squeeze(0).cpu().numpy()

        flagged = [CLASSES[i] for i in range(N_CLASSES) if probs[i] >= self.thresholds[i]]
        top3 = [CLASSES[i] for i in np.argsort(-probs)[:3]]
        wanted = list(dict.fromkeys(flagged + top3 + (extra_findings or [])))
        _, cams_arr = self.engine.run(FULL, None, img_gray, [CLASSES.index(f) for f in wanted])
        cams = {f: cams_arr[k] for k, f in enumerate(wanted)}
        zones = {f: zone_phrase(cam_to_zone(cams[f])) for f in wanted}
        cam_area = {f: float((cams[f] >= 0.5 * cams[f].max()).mean()) if cams[f].max() > 0 else 0.0
                    for f in wanted}

        boxes = []
        if known:
            for _, r in self.bboxes[self.bboxes.image == known].iterrows():
                if r.finding not in cams:
                    _, extra = self.engine.run(FULL, None, img_gray, [CLASSES.index(r.finding)])
                    cams[r.finding] = extra[0]
                    zones[r.finding] = zone_phrase(cam_to_zone(extra[0]))
                m = localisation_metrics(cams[r.finding], (r.x, r.y, r.w, r.h))
                boxes.append({"finding": r.finding, "box": [float(r.x), float(r.y), float(r.w), float(r.h)],
                              "hit": bool(m["pointing_hit"]), "iou": float(m["iou"]),
                              "box_coverage": float(m["box_coverage"]),
                              "cam_mass_in_box": float(m["cam_mass_in_box"])})

        return Analysis(image_name=name, view=view, probs=probs, thresholds=self.thresholds,
                        flagged=flagged, top3=top3, cams=cams, zones=zones, cam_area=cam_area,
                        known_nih=bool(known), truth=truth, boxes=boxes,
                        reliability={f: self.reliability.get(f) for f in wanted if f in self.reliability},
                        seconds=time.time() - t0)

    # ---------------------------------------------------------- report ---
    def report(self, image_path: Path, a: Analysis, condition: str = "C") -> dict:
        zones = {f: a.zones[f] for f in a.flagged if f in a.zones} if condition == "C" else None
        out = generate_report(image_path, condition, view=a.view, probs=a.probs,
                              flagged=a.flagged, zones=zones)
        out["scores"] = score_report(out, a)
        out["version"] = VERSION
        return out


def score_report(rep: dict, a: Analysis) -> dict:
    present = {c for c, s in rep.get("findings", {}).items() if s == "present"}
    flagged = set(a.flagged)
    scores = {
        "findings_asserted": len(present),
        "asserted_and_flagged": len(present & flagged),
        "asserted_not_flagged": sorted(present - flagged),
        "flagged_not_asserted": sorted(flagged - present),
        "agreement_with_classifier_jaccard": (len(present & flagged) / len(present | flagged)
                                              if (present | flagged) else 1.0),
        "parse_ok": bool(rep.get("parse_ok")),
        "generation_seconds": rep.get("seconds"),
    }
    if rep.get("locations") and a.zones:
        from src.report_study import _loc_agrees
        checks = [_loc_agrees(rep["locations"].get(f, ""), a.zones.get(f, ""))
                  for f in present & set(a.zones)]
        checks = [c for c in checks if c is not None]
        scores["localisation_agreement"] = float(np.mean(checks)) if checks else None
    if a.truth is not None:
        truth = set(a.truth)
        scores["ground_truth_available"] = True
        scores["true_findings"] = sorted(truth)
        scores["precision_vs_truth"] = len(present & truth) / len(present) if present else None
        scores["recall_vs_truth"] = len(present & truth) / len(truth) if truth else None
        scores["false_vs_truth"] = sorted(present - truth)
    else:
        scores["ground_truth_available"] = False
    return scores


_SERVICE: Service | None = None


def get_service() -> Service:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = Service()
    return _SERVICE
