"""Pre-render the landing-page hero: real films, real heatmaps, real report lines.

    python scripts/make_hero_assets.py

For each demo film: the sepia-tinted base image, one alpha+contour layer per
flagged finding in that finding's colour, the probabilities, the labels, and
the MedGemma impression from the grounding study if that film was in it.
Everything is base64-embedded into outputs/hero/hero_assets.json so the hero
loads with no extra requests.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, load_config  # noqa: E402
from src.palette import colour, pretty  # noqa: E402
from src.service import get_service  # noqa: E402
from src.viz import alpha_layer, contour_layer, tint_film  # noqa: E402

DEMO = ["00023325_019.png",   # Cardiomegaly + Effusion, heatmap HIT
        "00023097_001.png",   # Atelectasis + Effusion, clean success
        "00014358_016.png",   # Pneumothorax, single finding
        "00001936_000.png"]   # normal film, nothing flagged
SIZE = 720


def b64(img: Image.Image, fmt="PNG", **kw) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    cfg = load_config()
    svc = get_service()
    img_dir = Path(cfg.data.images_full)
    out_dir = Path(cfg.dirs.root) / "hero"
    out_dir.mkdir(exist_ok=True)

    impressions = {}
    jsonl = Path(cfg.dirs.root) / "reports" / "reports.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d["condition"] == "C" and d.get("impression"):
                impressions[d["image"]] = d["impression"]

    films = []
    for name in DEMO:
        p = img_dir / name
        if not p.exists():
            print("skip missing", name)
            continue
        img = Image.open(p).convert("L")
        a = svc.analyse(img, file_bytes=p.read_bytes())
        small = img.resize((SIZE, SIZE), Image.LANCZOS)
        base = tint_film(small, "sepia")
        layers = []
        for f in a.flagged:
            cam = Image.fromarray((a.cams[f] * 255).astype("uint8")).resize((SIZE, SIZE), Image.BILINEAR)
            import numpy as np
            cam_np = np.asarray(cam, dtype="float32") / 255.0
            layer = Image.alpha_composite(alpha_layer(cam_np, colour(f), alpha_max=0.8),
                                          contour_layer(cam_np, colour(f), width=3))
            py, px = np.unravel_index(int(np.argmax(cam_np)), cam_np.shape)
            layers.append({"finding": f, "label": pretty(f), "colour": colour(f),
                           "prob": round(float(a.probs[CLASSES.index(f)]), 2),
                           "peak": [int(px), int(py)], "zone": a.zones.get(f, ""),
                           "png": b64(layer)})
        films.append({
            "name": name, "view": a.view, "truth": [pretty(t) for t in (a.truth or [])],
            "base_jpg": b64(base, "JPEG", quality=82),
            "layers": layers,
            "impression": impressions.get(name, ""),
            "top": [{"label": pretty(CLASSES[i]), "prob": round(float(a.probs[i]), 2),
                     "colour": colour(CLASSES[i])} for i in a.probs.argsort()[::-1][:4]],
        })
        print(f"{name}: {len(layers)} layers, impression {'yes' if impressions.get(name) else 'no'}")

    out = out_dir / "hero_assets.json"
    out.write_text(json.dumps({"size": SIZE, "films": films}))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
