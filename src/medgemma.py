"""MedGemma report generation through Ollama: prompt building, CAM-to-anatomy
text, the HTTP call, and parsing of the structured reply.

    python -m src.medgemma --smoke 3        # 3 films x 3 conditions, printed

MedGemma is a WRITER here, not a second classifier. It receives the film plus
(depending on the experimental condition) the classifier's 14 probabilities and
the anatomical zone the LayerCAM heatmap points to, and returns a structured
JSON block followed by a radiology-style draft. Nothing is fine-tuned.

Conditions for the grounding study:
    A  image only
    B  image + classifier probabilities
    C  image + probabilities + Grad-CAM zone per flagged finding

Runs against the local Ollama server (default http://localhost:11434), model
`medgemma:latest` (4B-it, Q4_K_M), verified multimodal.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "medgemma:latest"
CONDITIONS = ("A", "B", "C")
STATUSES = ("present", "absent", "uncertain")

DISCLAIMER = ("This is an AI-generated draft produced for research and educational "
              "purposes only. It has not been reviewed by a radiologist and must not "
              "be used for clinical decision-making.")

SYSTEM_PROMPT = """You are drafting a chest radiograph report for review by a radiologist. You are a careful writer, not the final decision-maker.

Rules:
1. Report only what is supported by the image and by any classifier evidence you are given. Do not invent findings. If evidence is weak or conflicting, say "uncertain" rather than guessing.
2. Use hedged radiological language ("suggests", "consistent with", "cannot be excluded").
3. Classifier probabilities, when provided, are uncalibrated model scores, not clinical probabilities. A high score is a prompt to look, not a diagnosis.
4. Never state patient identity, age, sex or history unless it is given to you.
5. Use radiographic convention: the patient's right is on the left of the image.

Output format, exactly in this order:
First, a single JSON object on its own lines with three keys:
  "findings": an object with ALL of these 14 keys, each valued "present", "absent" or "uncertain":
  Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax
  "locations": an object mapping each "present" finding to a short anatomical location string
  "impression": one sentence
Then a blank line, then the prose report with the headings FINDINGS, IMPRESSION, RECOMMENDATION."""


# ---------------------------------------------------------- CAM -> anatomy ---
def cam_to_zone(cam: np.ndarray, thr: float = 0.5) -> dict:
    """Reduce a 0..1 heatmap to a radiographic zone phrase.

    Thresholds at `thr` of the peak, distributes the region's mass over a 3x3
    grid (upper/mid/lower x patient-right/central/patient-left) and names the
    cells holding at least 20 % of it. Patient right = image left.
    """
    H, W = cam.shape
    mask = cam >= thr * cam.max() if cam.max() > 0 else np.zeros_like(cam, bool)
    if not mask.any():
        return {"zone": "indeterminate", "hint": "", "rows": [], "cols": []}
    ys, xs = np.nonzero(mask)
    weights = cam[ys, xs]
    row_idx = np.minimum((ys / H * 3).astype(int), 2)
    col_idx = np.digitize(xs / W, [0.38, 0.62])          # 0 = image-left = patient RIGHT
    grid = np.zeros((3, 3))
    for r, c, w in zip(row_idx, col_idx, weights):
        grid[r, c] += w
    grid /= grid.sum()
    rows = [i for i in range(3) if grid[i].sum() >= 0.20]
    cols = [j for j in range(3) if grid[:, j].sum() >= 0.20]
    level = {0: "upper", 1: "mid", 2: "lower"}
    side = {0: "right", 1: "central", 2: "left"}
    rows = rows or [int(np.argmax(grid.sum(1)))]
    cols = cols or [int(np.argmax(grid.sum(0)))]
    level_txt = (f"{level[rows[0]]}-to-{level[rows[-1]]}" if len(rows) > 1 else level[rows[0]])
    if cols == [0, 1, 2]:
        side_txt = "bilateral"
    elif len(cols) == 2 and 1 in cols:
        side_txt = side[[c for c in cols if c != 1][0]] + " and central"
    elif len(cols) == 2:
        side_txt = "bilateral"
    else:
        side_txt = side[cols[0]]
    zone = f"{side_txt} {level_txt} zone"
    hint = ""
    if 1 in cols and 2 in rows and len(cols) == 1:
        hint = "cardiac silhouette / lower mediastinum"
    elif 1 in cols and len(cols) == 1:
        hint = "mediastinum / hila"
    elif 2 in rows and 1 not in cols:
        hint = "costophrenic region"
    elif rows == [0]:
        hint = "apex"
    return {"zone": zone, "hint": hint, "rows": rows, "cols": cols}


def zone_phrase(z: dict) -> str:
    return f"{z['zone']} ({z['hint']})" if z.get("hint") else z["zone"]


# --------------------------------------------------------------- prompting ---
def build_user_prompt(condition: str, view: str | None, probs: np.ndarray | None,
                      flagged: list[str] | None, zones: dict[str, str] | None) -> str:
    lines = ["Chest radiograph attached." + (f" View: {view}." if view else "")]
    if condition in ("B", "C"):
        assert probs is not None and flagged is not None
        lines.append("")
        lines.append("Classifier assessment (uncalibrated scores 0-1; * = above that "
                     "finding's decision threshold):")
        order = np.argsort(-probs)
        for i in order:
            mark = " *" if CLASSES[i] in flagged else ""
            lines.append(f"  {CLASSES[i].replace('_', ' ')}: {probs[i]:.2f}{mark}")
        if not flagged:
            lines.append("  (no finding reached its threshold)")
    if condition == "C":
        lines.append("")
        if zones:
            lines.append("Where the classifier's attention was concentrated for each flagged finding:")
            for f, z in zones.items():
                lines.append(f"  {f.replace('_', ' ')}: {z}")
        else:
            lines.append("No flagged finding, so no attention map is provided.")
    lines.append("")
    lines.append("Write the JSON block, then the report.")
    return "\n".join(lines)


# ------------------------------------------------------------------ client ---
def _post(payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call_medgemma(image_path: Path, user_prompt: str, *, temperature: float = 0.0,
                  timeout: int = 600, retries: int = 2, json_only: bool = False) -> dict:
    img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {
        "model": MODEL, "stream": False,
        "options": {"temperature": temperature, "num_predict": 900, "seed": 0},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_prompt, "images": [img_b64]}],
    }
    if json_only:
        payload["format"] = "json"
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            out = _post(payload, timeout)
            return {"text": out["message"]["content"], "seconds": time.time() - t0,
                    "eval_count": out.get("eval_count"), "attempt": attempt}
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Ollama call failed after {retries + 1} attempts: {last_err}")


# ----------------------------------------------------------------- parsing ---
def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _norm_key(k: str) -> str | None:
    k2 = re.sub(r"[^a-z]", "", k.lower())
    for c in CLASSES:
        if re.sub(r"[^a-z]", "", c.lower()) == k2:
            return c
    return None


def parse_reply(text: str) -> dict:
    """Return {'findings': {cls: status}, 'locations': {...}, 'impression': str,
    'prose': str, 'parse_ok': bool}. Missing findings become 'uncertain'."""
    blob = _first_json_object(text)
    findings = {c: "uncertain" for c in CLASSES}
    locations, impression, ok = {}, "", False
    if blob:
        try:
            data = json.loads(blob)
            ok = True
        except json.JSONDecodeError:
            try:  # common small-model slips: trailing commas
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
                ok = True
            except json.JSONDecodeError:
                data = {}
        for k, v in (data.get("findings") or {}).items():
            c = _norm_key(k)
            if c and isinstance(v, str):
                v = v.strip().lower()
                findings[c] = v if v in STATUSES else "uncertain"
        for k, v in (data.get("locations") or {}).items():
            c = _norm_key(k)
            if c and isinstance(v, str):
                locations[c] = v.strip()
        impression = str(data.get("impression", "")).strip()
    prose = text[text.find(blob) + len(blob):] if blob else text
    prose = re.sub(r"^\s*```[a-zA-Z]*\s*", "", prose).strip()   # closing/opening fences
    if not impression:
        # The model tends to put the impression in the prose rather than the JSON.
        m = re.search(r"IMPRESSION\s*:?\s*\n(.+?)(?:\n\s*\n|\nRECOMMENDATION|\Z)", prose,
                      flags=re.S | re.I)
        if m:
            impression = " ".join(m.group(1).split())
    return {"findings": findings, "locations": locations, "impression": impression,
            "prose": prose, "parse_ok": ok}


# ------------------------------------------------------------- one report ---
def generate_report(image_path: Path, condition: str, *, view: str | None,
                    probs: np.ndarray | None, flagged: list[str] | None,
                    zones: dict[str, str] | None, temperature: float = 0.0) -> dict:
    prompt = build_user_prompt(condition, view, probs, flagged, zones)
    raw = call_medgemma(image_path, prompt, temperature=temperature)
    parsed = parse_reply(raw["text"])
    if not parsed["parse_ok"]:
        # Fallback: ask for the JSON block alone with Ollama's JSON mode, so the
        # structured part is still scorable; the prose is kept from the first pass.
        raw2 = call_medgemma(image_path, prompt + "\nReturn ONLY the JSON object.",
                             temperature=temperature, json_only=True)
        parsed2 = parse_reply(raw2["text"])
        if parsed2["parse_ok"]:
            parsed2["prose"] = parsed["prose"]
            parsed2["fallback"] = True
            parsed = parsed2
            raw["seconds"] += raw2["seconds"]
    return {"condition": condition, "prompt": prompt, "raw": raw["text"],
            "seconds": round(raw["seconds"], 1), **parsed,
            "report": parsed["prose"] + "\n\n" + DISCLAIMER}


# ------------------------------------------------------------------- smoke ---
def _smoke(n: int) -> None:
    from src.report_study import prepare_study_inputs  # local import: heavy deps
    from src.config import load_config
    cfg = load_config()
    rows = prepare_study_inputs(cfg, n_films=n, seed=0)
    for r in rows:
        print("=" * 100)
        print(f"{r['image']}  view {r['view']}  truth: {', '.join(r['truth']) or 'No finding'}"
              f"  flagged: {', '.join(r['flagged']) or 'none'}")
        for cond in CONDITIONS:
            out = generate_report(r["image_path"], cond, view=r["view"], probs=r["probs"],
                                  flagged=r["flagged"], zones=r["zones"])
            present = [c for c, s in out["findings"].items() if s == "present"]
            print(f"--- condition {cond}  ({out['seconds']}s, parse_ok={out['parse_ok']}"
                  f"{', fallback' if out.get('fallback') else ''})")
            print(f"    present: {present}")
            print(f"    impression: {out['impression'][:160]}")
        print()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="run N films x 3 conditions and print")
    args = ap.parse_args()
    if args.smoke:
        _smoke(args.smoke)
