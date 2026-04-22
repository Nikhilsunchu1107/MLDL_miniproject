from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class GenderEfficientNet(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 1),
        )

        if freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.features.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        return logits
