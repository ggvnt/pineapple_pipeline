"""Utility helpers for the Pineapple Analysis pipeline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Device ────────────────────────────────────────────────────────────────────

def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ── File helpers ──────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ── Early stopping ────────────────────────────────────────────────────────────

@dataclass
class EarlyStopping:
    patience:  int
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best: float | None = None
        self.bad: int = 0

    def step(self, metric: float) -> bool:
        """Return True if training should stop."""
        if self.best is None or metric < self.best - self.min_delta:
            self.best = metric
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def save_confusion_heatmaps(
    health_cm:  list[list[int]],
    month_cm:   list[list[int]],
    health_labels: list[str],
    month_labels:  list[str],
    out_path:  str | Path,
) -> None:
    """Save side-by-side confusion matrix heatmaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _plot_cm(ax, cm_raw, labels, title):
        cm = np.array(cm_raw, dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = cm / np.maximum(row_sums, 1)

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        n = len(labels)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                txt = f"{int(val)}\n({cm_norm[i,j]:.0%})"
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    _plot_cm(ax1, health_cm, health_labels, "Health Classification")
    _plot_cm(ax2, month_cm,  month_labels,  "Growth Stage (Month)")
    fig.suptitle("Pineapple Analysis — Confusion Matrices (row-normalised)", fontsize=14)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"✓ Saved confusion matrices → {out_path}")


# ── Training-curve plot ───────────────────────────────────────────────────────

def save_training_curves(history: list[dict], out_path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs    = [h["epoch"]              for h in history]
    tr_losses = [h.get("train_loss", 0)  for h in history]
    vl_losses = [h.get("val_loss",   0)  for h in history]
    vl_h_acc  = [h.get("val_health_acc",  0) for h in history]
    vl_m_acc  = [h.get("val_month_acc",   0) for h in history]
    vl_w_mae  = [h.get("val_width_mae",   0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, tr_losses, label="Train loss")
    axes[0].plot(epochs, vl_losses, label="Val loss")
    axes[0].set_title("Total Loss"); axes[0].legend()

    axes[1].plot(epochs, vl_h_acc, label="Health acc")
    axes[1].plot(epochs, vl_m_acc, label="Month acc")
    axes[1].set_title("Validation Accuracy"); axes[1].legend()

    axes[2].plot(epochs, vl_w_mae, color="tomato", label="Width MAE (cm)")
    axes[2].axhline(2.0, ls="--", c="grey", alpha=0.7, label="Target <2 cm")
    axes[2].set_title("Width MAE (cm)"); axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved training curves → {out_path}")
