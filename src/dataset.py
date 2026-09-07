"""Dataset and transforms for the 256px preprocessed X-rays.

Augmentation is deliberately conservative. Horizontal flip is fine (situs
inversus aside, left/right is not what the findings hinge on here), but
vertical flips and strong colour jitter produce anatomically impossible films
and hurt more than they help.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.config import CLASSES, read_csv
from src.model import PreprocSpec

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class XRVNormalize:
    """Scale [0, 1] tensors to the [-1024, 1024] range torchxrayvision expects.

    Must be a module-level class, not a lambda inside T.Lambda: Windows
    DataLoader workers spawn rather than fork, so the whole transform pipeline
    gets pickled, and a local lambda is unpicklable.
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t * 2048.0 - 1024.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def build_transforms(px: int, spec: PreprocSpec, train: bool) -> T.Compose:
    if train:
        geom = [
            T.RandomResizedCrop(px, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    else:
        geom = [T.Resize(px), T.CenterCrop(px)]

    if spec.norm == "imagenet":
        tail = [T.Grayscale(num_output_channels=3), T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    elif spec.norm == "xrv":
        # torchxrayvision backbones expect single-channel data scaled to
        # [-1024, 1024] rather than ImageNet statistics.
        tail = [T.Grayscale(num_output_channels=1), T.ToTensor(), XRVNormalize()]
    else:
        raise ValueError(f"unknown norm '{spec.norm}'")

    return T.Compose(geom + tail)


class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.targets = self.df[CLASSES].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # folds.csv stores the original .png name; on disk we have .jpg.
        name = Path(self.df.at[idx, "image"]).with_suffix(".jpg").name
        img = Image.open(self.image_dir / name).convert("L")
        return self.transform(img), torch.from_numpy(self.targets[idx])


def make_loaders(folds_csv: Path, image_dir: Path, fold: int, px: int,
                 spec: PreprocSpec, batch_size: int, num_workers: int
                 ) -> tuple[DataLoader, DataLoader, np.ndarray]:
    """Train/val loaders for one CV fold, plus per-class positive counts."""
    df = read_csv(folds_csv)
    tr, va = df[df.fold != fold], df[df.fold == fold]

    train_ds = CXRDataset(tr, image_dir, build_transforms(px, spec, train=True))
    val_ds = CXRDataset(va, image_dir, build_transforms(px, spec, train=False))

    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True, **common)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    return train_dl, val_dl, tr[CLASSES].sum().to_numpy(dtype=np.float32)


def pos_weight(pos_counts: np.ndarray, n_total: int, cap: float) -> torch.Tensor:
    """neg/pos per class, clamped.

    Hernia has ~10 positives in a training split, giving a raw weight near 450.
    Left uncapped it dominates the gradient and destabilises everything else.
    """
    neg = n_total - pos_counts
    w = np.divide(neg, np.maximum(pos_counts, 1.0))
    return torch.from_numpy(np.clip(w, 1.0, cap).astype(np.float32))
