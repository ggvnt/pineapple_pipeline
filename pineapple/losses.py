"""Loss functions and evaluation metrics for the three-task model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)


# ── Multi-task loss ───────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    total = alpha * L_health  +  beta * L_month  +  gamma * L_width

    Width loss is masked: samples where gt_width == 0.0 are excluded.
    """

    def __init__(
        self,
        health_weights: torch.Tensor | None = None,
        month_weights:  torch.Tensor | None = None,
        alpha:  float = 0.4,
        beta:   float = 0.4,
        gamma:  float = 0.2,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        self.loss_health = nn.CrossEntropyLoss(weight=health_weights, label_smoothing=label_smoothing)
        self.loss_month  = nn.CrossEntropyLoss(weight=month_weights,  label_smoothing=label_smoothing)
        self.loss_width  = nn.HuberLoss(reduction="none", delta=5.0)   # robust to outlier widths

    def forward(
        self,
        health_logits: torch.Tensor,   # (B, 3)
        month_logits:  torch.Tensor,   # (B, 12)
        width_pred:    torch.Tensor,   # (B,)
        y_health: torch.Tensor,        # (B,)
        y_month:  torch.Tensor,        # (B,)
        y_width:  torch.Tensor,        # (B,) — 0.0 means unknown
    ) -> dict[str, torch.Tensor]:
        l_h = self.loss_health(health_logits, y_health)
        l_m = self.loss_month(month_logits,  y_month)

        # Masked width loss
        mask = y_width > 0.0
        if mask.any():
            l_w = self.loss_width(width_pred[mask], y_width[mask]).mean()
        else:
            l_w = torch.tensor(0.0, device=width_pred.device)

        total = self.alpha * l_h + self.beta * l_m + self.gamma * l_w
        return {"total": total, "health": l_h, "month": l_m, "width": l_w}


# ── Class weight computation ──────────────────────────────────────────────────

def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    return torch.tensor(inv / inv.mean(), dtype=torch.float32)


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class ClassMetrics:
    accuracy: float
    f1_macro: float
    confusion: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {"accuracy": self.accuracy, "f1_macro": self.f1_macro, "confusion": self.confusion}


@dataclass
class RegressionMetrics:
    mae_cm:  float
    rmse_cm: float
    r2:      float

    def to_dict(self) -> dict[str, Any]:
        return {"mae_cm": self.mae_cm, "rmse_cm": self.rmse_cm, "r2": self.r2}


@dataclass
class EpochMetrics:
    loss_total:  float
    loss_health: float
    loss_month:  float
    loss_width:  float
    health: ClassMetrics | None       = None
    month:  ClassMetrics | None       = None
    width:  RegressionMetrics | None  = None
    month_mae_months: float           = 0.0   # month prediction error in calendar months

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "loss_total":  self.loss_total,
            "loss_health": self.loss_health,
            "loss_month":  self.loss_month,
            "loss_width":  self.loss_width,
            "month_mae_months": self.month_mae_months,
        }
        if self.health is not None:
            d["health"] = self.health.to_dict()
        if self.month is not None:
            d["month"] = self.month.to_dict()
        if self.width is not None:
            d["width"] = self.width.to_dict()
        return d


def compute_class_metrics(y_true: list[int], y_pred: list[int]) -> ClassMetrics:
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm  = confusion_matrix(y_true, y_pred).astype(int).tolist()
    return ClassMetrics(accuracy=acc, f1_macro=f1, confusion=cm)


def compute_regression_metrics(y_true: list[float], y_pred: list[float]) -> RegressionMetrics:
    yt = np.array(y_true, dtype=np.float32)
    yp = np.array(y_pred, dtype=np.float32)
    mae  = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2   = float(r2_score(yt, yp)) if len(yt) > 1 else float("nan")
    return RegressionMetrics(mae_cm=mae, rmse_cm=rmse, r2=r2)
