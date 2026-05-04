"""Multi-task EfficientNet-B0: health classification + month regression + width regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class ModelOutput:
    health_logits: torch.Tensor   # (B, 3)
    month_logits:  torch.Tensor   # (B, 12)
    width_cm:      torch.Tensor   # (B, 1)  — always positive (Softplus)


class MultiTaskNet(nn.Module):
    """
    EfficientNet-B0 backbone with three task heads:
      - health_head  → 3-class classification
      - month_head   → 12-class classification
      - width_head   → scalar regression (cm), constrained positive via Softplus
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_health: int = 3,
        num_months: int = 12,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone

        # ── Backbone ──────────────────────────────────────────────────────────
        if backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            base = models.efficientnet_b0(weights=weights)
            feat_dim = base.classifier[1].in_features   # 1280
            base.classifier = nn.Identity()
            self.backbone = base
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            base = models.resnet50(weights=weights)
            feat_dim = base.fc.in_features               # 2048
            base.fc = nn.Identity()
            self.backbone = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}")

        self.feat_dim = feat_dim

        # ── Shared neck (BN + Dropout for regularisation) ─────────────────────
        self.neck = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p=dropout),
        )

        # ── Task heads ────────────────────────────────────────────────────────
        self.health_head = nn.Linear(feat_dim, num_health)
        self.month_head  = nn.Linear(feat_dim, num_months)

        # Width head — two-layer MLP; Softplus keeps output strictly positive
        self.width_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, 1),
            nn.Softplus(),   # guarantees width_cm > 0
        )

        self._init_heads()

    # ── Initialisation ────────────────────────────────────────────────────────
    def _init_heads(self) -> None:
        for head in (self.health_head, self.month_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        for m in self.width_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> ModelOutput:
        feats = self.backbone(x)           # (B, feat_dim)
        feats = self.neck(feats)           # (B, feat_dim)
        return ModelOutput(
            health_logits=self.health_head(feats),         # (B, 3)
            month_logits=self.month_head(feats),           # (B, 12)
            width_cm=self.width_head(feats).squeeze(-1),   # (B,)
        )

    # ── Freeze / unfreeze backbone ────────────────────────────────────────────
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


# ── CAM target layer lookup ───────────────────────────────────────────────────

def get_cam_layer(model: MultiTaskNet) -> nn.Module:
    """Return the last convolutional block used for Grad-CAM."""
    if model.backbone_name == "efficientnet_b0":
        return model.backbone.features[-1]
    if model.backbone_name == "resnet50":
        return model.backbone.layer4[-1]
    raise ValueError(f"No CAM layer defined for backbone={model.backbone_name!r}")
