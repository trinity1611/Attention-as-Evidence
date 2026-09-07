"""Model zoo for the 14-label chest X-ray classifier.

Three variants, chosen so the comparison isolates one factor at a time:

  densenet121_imagenet   generic pretraining          -- baseline
  densenet121_chexpert   chest X-ray pretraining      -- same architecture, different init
  efficientnet_b0        different architecture       -- same init as baseline

The CheXpert weights come from torchxrayvision. We deliberately do NOT use its
"nih" or "all" checkpoints: those were trained on NIH data and have already
seen images that end up in our test folds, which would leak. CheXpert is a
different institution and different patients, so we get the domain transfer
without contaminating the evaluation.

Each builder also returns a PreprocSpec, because the CheXpert backbone is
single-channel and expects a [-1024, 1024] input range rather than ImageNet
normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as tvm


@dataclass(frozen=True)
class PreprocSpec:
    in_channels: int
    norm: str  # "imagenet" | "xrv"


class CXRNet(nn.Module):
    """Pretrained conv backbone -> global pool -> dropout -> 14 logits."""

    def __init__(self, features: nn.Module, feat_dim: int, n_classes: int,
                 dropout: float, gradcam_layer: nn.Module):
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, n_classes)
        # Kept as an attribute so Grad-CAM can hook it without knowing the
        # architecture. Not registered as a child -- it lives inside features.
        object.__setattr__(self, "_gradcam_layer", gradcam_layer)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self._gradcam_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.drop(x))

    def freeze_backbone(self, frozen: bool = True) -> None:
        for p in self.features.parameters():
            p.requires_grad = not frozen

    def param_groups(self, lr_backbone: float, lr_head: float) -> list[dict]:
        return [
            {"params": self.features.parameters(), "lr": lr_backbone},
            {"params": self.classifier.parameters(), "lr": lr_head},
        ]


def _densenet_imagenet(n_classes: int, dropout: float):
    net = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
    model = CXRNet(net.features, 1024, n_classes, dropout,
                   gradcam_layer=net.features.denseblock4)
    return model, PreprocSpec(in_channels=3, norm="imagenet")


def _densenet_chexpert(n_classes: int, dropout: float):
    import torchxrayvision as xrv  # imported lazily; downloads weights on first use

    net = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model = CXRNet(net.features, 1024, n_classes, dropout,
                   gradcam_layer=net.features.denseblock4)
    return model, PreprocSpec(in_channels=1, norm="xrv")


def _efficientnet_b0(n_classes: int, dropout: float):
    net = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model = CXRNet(net.features, 1280, n_classes, dropout,
                   gradcam_layer=net.features[-1])
    return model, PreprocSpec(in_channels=3, norm="imagenet")


BUILDERS = {
    "densenet121_imagenet": _densenet_imagenet,
    "densenet121_chexpert": _densenet_chexpert,
    "efficientnet_b0": _efficientnet_b0,
}


def build_model(name: str, n_classes: int, dropout: float) -> tuple[CXRNet, PreprocSpec]:
    if name not in BUILDERS:
        raise KeyError(f"unknown model '{name}'; choose from {list(BUILDERS)}")
    return BUILDERS[name](n_classes, dropout)
