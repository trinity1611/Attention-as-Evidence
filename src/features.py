"""Extract frozen RAD-DINO embeddings for every image, once.

    python -m src.features

RAD-DINO (microsoft/rad-dino) is a DINOv2 ViT-B/14 pretrained self-supervised
on ~880k chest X-rays. We use it as a fixed feature extractor: one 768-d CLS
embedding per image, plus the embedding of the horizontally flipped image so
the downstream probe can do feature-level TTA. Extracting once and caching
turns every later experiment into seconds of work.

CAVEAT, carried into the results: RAD-DINO's pretraining corpus included all of
NIH ChestX-ray14 -- images only, never labels. It has therefore *seen* the
pixels of our held-out images. That is a weaker form of leakage than label
leakage, but it must be disclosed alongside any number the probe produces.

Writes: <data.root>/features/raddino_cls.npz  {images, feats, feats_flip}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, read_csv  # noqa: E402

MODEL_ID = "microsoft/rad-dino"
BATCH = 16


@torch.no_grad()
def main() -> None:
    from transformers import AutoImageProcessor, AutoModel

    cfg = load_config()
    out_dir = Path(cfg.data.root) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raddino_cls.npz"

    folds = read_csv(Path(cfg.dirs.root) / "folds.csv")
    names = folds["image"].tolist()
    img_dir = Path(cfg.data.images_full)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    print(f"{MODEL_ID} loaded on {device}; {len(names)} images")

    feats = np.zeros((len(names), 768), dtype=np.float32)
    feats_flip = np.zeros_like(feats)
    t0 = time.time()
    for s in range(0, len(names), BATCH):
        batch_names = names[s:s + BATCH]
        ims = [Image.open(img_dir / n).convert("RGB") for n in batch_names]
        ims_flip = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in ims]
        for arr, imgs in ((feats, ims), (feats_flip, ims_flip)):
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(**inputs)
            arr[s:s + len(imgs)] = out.pooler_output.float().cpu().numpy()
        if (s // BATCH) % 25 == 0:
            done = s + len(batch_names)
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(names)}  {rate:.1f} img/s")

    np.savez(out_path, images=np.array(names), feats=feats, feats_flip=feats_flip)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
