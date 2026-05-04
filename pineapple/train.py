"""Training loop for the three-task Pineapple Analysis model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    _NEW_AMP = False

from .constants import HEALTH_CLASSES, MONTH_CLASSES
from .data import make_loaders, scan_dataset, split_samples
from .losses import (
    MultiTaskLoss,
    compute_class_metrics,
    compute_class_weights,
    compute_regression_metrics,
    EpochMetrics,
)
from .model import MultiTaskNet
from .utils import (
    EarlyStopping,
    ensure_dir,
    resolve_device,
    save_confusion_heatmaps,
    save_json,
    save_training_curves,
    set_seed,
)


# ── Single evaluation epoch ───────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(
    model: MultiTaskNet,
    loader,
    device: torch.device,
    criterion: MultiTaskLoss,
) -> EpochMetrics:
    model.eval()
    total_loss = total_lh = total_lm = total_lw = 0.0
    n_samples = 0

    y_true_h: list[int]   = []
    y_pred_h: list[int]   = []
    y_true_m: list[int]   = []
    y_pred_m: list[int]   = []
    y_true_w: list[float] = []
    y_pred_w: list[float] = []

    for x, y_h, y_m, y_w, _paths in loader:
        x   = x.to(device,   non_blocking=True)
        y_h = y_h.to(device, non_blocking=True)
        y_m = y_m.to(device, non_blocking=True)
        y_w = y_w.to(device, non_blocking=True)

        out = model(x)
        losses = criterion(out.health_logits, out.month_logits, out.width_cm, y_h, y_m, y_w)

        bs = x.size(0)
        total_loss += losses["total"].item() * bs
        total_lh   += losses["health"].item() * bs
        total_lm   += losses["month"].item()  * bs
        total_lw   += losses["width"].item()  * bs
        n_samples  += bs

        y_true_h.extend(y_h.cpu().tolist())
        y_pred_h.extend(out.health_logits.argmax(1).cpu().tolist())
        y_true_m.extend(y_m.cpu().tolist())
        y_pred_m.extend(out.month_logits.argmax(1).cpu().tolist())

        mask = y_w > 0.0
        y_true_w.extend(y_w[mask].cpu().tolist())
        y_pred_w.extend(out.width_cm[mask].cpu().tolist())

    n = max(1, n_samples)
    health_met = compute_class_metrics(y_true_h, y_pred_h)
    month_met  = compute_class_metrics(y_true_m, y_pred_m)
    width_met  = compute_regression_metrics(y_true_w, y_pred_w) if y_true_w else None

    month_mae_months = float(np.mean([abs(a - b) for a, b in zip(y_true_m, y_pred_m)]))

    return EpochMetrics(
        loss_total  = total_loss / n,
        loss_health = total_lh   / n,
        loss_month  = total_lm   / n,
        loss_width  = total_lw   / n,
        health      = health_met,
        month       = month_met,
        width       = width_met,
        month_mae_months = month_mae_months,
    )


# ── Full training run ─────────────────────────────────────────────────────────

def train_main(args: argparse.Namespace) -> None:
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    use_amp = device.type == "cuda"

    out_dir  = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(args.checkpoint_dir)
    print(f"▶  Device: {device}  |  AMP: {use_amp}  |  Output: {out_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    samples = scan_dataset(
        args.data_root,
        width_csv=getattr(args, "width_csv", None),
    )
    train_s, val_s, test_s = split_samples(samples, seed=int(args.seed))
    print(f"   Samples: train={len(train_s)}  val={len(val_s)}  test={len(test_s)}")

    train_loader, val_loader, test_loader = make_loaders(
        train_s, val_s, test_s,
        image_size   = int(args.image_size),
        batch_size   = int(args.batch_size),
        num_workers  = int(args.num_workers),
        strong_aug   = bool(getattr(args, "strong_aug", True)),
        sampler_mode = str(getattr(args, "sampler", "joint")),
        pin_memory   = device.type == "cuda",
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    health_w = compute_class_weights([s.health for s in train_s], len(HEALTH_CLASSES)).to(device)
    month_w  = compute_class_weights([s.month  for s in train_s], len(MONTH_CLASSES)).to(device)

    criterion = MultiTaskLoss(
        health_weights  = health_w,
        month_weights   = month_w,
        alpha           = float(getattr(args, "alpha", 0.4)),
        beta            = float(getattr(args, "beta",  0.4)),
        gamma           = float(getattr(args, "gamma", 0.2)),
        label_smoothing = float(getattr(args, "label_smoothing", 0.0)),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MultiTaskNet(
        backbone   = str(args.backbone),
        dropout    = float(getattr(args, "dropout", 0.3)),
    ).to(device)

    # Optional: freeze backbone for the first N epochs (warm-up heads only)
    freeze_epochs = int(getattr(args, "freeze_epochs", 0))
    if freeze_epochs > 0:
        model.freeze_backbone()
        print(f"   Backbone frozen for first {freeze_epochs} epochs.")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    def _make_optimizer():
        return AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

    optimizer = _make_optimizer()
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))
    early     = EarlyStopping(patience=int(args.patience))

    if _NEW_AMP:
        scaler = GradScaler("cuda", enabled=use_amp)
    else:
        scaler = GradScaler(enabled=use_amp)

    best_val  = float("inf")
    best_path = ckpt_dir / "best_model.pt"
    last_path = ckpt_dir / "last_model.pt"
    history: list[dict[str, Any]] = []

    # ── Training epochs ───────────────────────────────────────────────────────
    for epoch in range(1, int(args.epochs) + 1):

        # Unfreeze backbone after warm-up
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = _make_optimizer()
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs) - freeze_epochs))
            print(f"   Epoch {epoch}: backbone unfrozen.")

        model.train()
        running = n_total = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}")

        for x, y_h, y_m, y_w, _paths in pbar:
            x   = x.to(device,   non_blocking=True)
            y_h = y_h.to(device, non_blocking=True)
            y_m = y_m.to(device, non_blocking=True)
            y_w = y_w.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp and _NEW_AMP:
                amp_ctx = autocast(device_type="cuda", enabled=True)
            elif use_amp:
                amp_ctx = autocast(enabled=True)
            else:
                amp_ctx = nullcontext()

            with amp_ctx:
                out = model(x)
                losses = criterion(out.health_logits, out.month_logits, out.width_cm,
                                   y_h, y_m, y_w)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            running  += losses["total"].item() * bs
            n_total  += bs
            pbar.set_postfix({
                "loss": f"{running/n_total:.4f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = eval_epoch(model, val_loader, device, criterion)
        val_dict    = val_metrics.to_dict()
        save_json(out_dir / f"val_epoch_{epoch:03d}.json", val_dict)

        entry = {
            "epoch":           epoch,
            "train_loss":      running / max(1, n_total),
            "val_loss":        val_metrics.loss_total,
            "val_health_acc":  val_metrics.health.accuracy if val_metrics.health else 0.0,
            "val_month_acc":   val_metrics.month.accuracy  if val_metrics.month  else 0.0,
            "val_width_mae":   val_metrics.width.mae_cm    if val_metrics.width  else 0.0,
            "val_month_mae_months": val_metrics.month_mae_months,
        }
        history.append(entry)

        print(
            f"   val_loss={val_metrics.loss_total:.4f} | "
            f"health_acc={entry['val_health_acc']:.3f} | "
            f"month_acc={entry['val_month_acc']:.3f} | "
            f"width_mae={entry['val_width_mae']:.2f} cm"
        )

        # Save last checkpoint
        ckpt = {
            "model":      model.state_dict(),
            "backbone":   str(args.backbone),
            "image_size": int(args.image_size),
            "epoch":      epoch,
            "val":        val_dict,
        }
        torch.save(ckpt, last_path)

        # Save best checkpoint
        if val_metrics.loss_total < best_val:
            best_val = val_metrics.loss_total
            torch.save(ckpt, best_path)
            print(f"   ✓ New best ({best_val:.4f}) → saved best_model.pt")

        if early.step(val_metrics.loss_total):
            print(f"   Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    # ── Save training history ─────────────────────────────────────────────────
    save_json(out_dir / "training_history.json", {"history": history})
    save_training_curves(history, out_dir / "training_curves.png")

    # ── Final test evaluation ─────────────────────────────────────────────────
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = eval_epoch(model, test_loader, device, criterion)
    test_dict    = test_metrics.to_dict()
    save_json(out_dir / "test_metrics.json", test_dict)

    print("\n── Test metrics ──────────────────────────────────────────────")
    print(f"   Health: acc={test_metrics.health.accuracy:.3f}  f1={test_metrics.health.f1_macro:.3f}")
    print(f"   Month:  acc={test_metrics.month.accuracy:.3f}  f1={test_metrics.month.f1_macro:.3f}  "
          f"MAE={test_metrics.month_mae_months:.2f} months")
    if test_metrics.width:
        print(f"   Width:  MAE={test_metrics.width.mae_cm:.2f} cm  "
              f"RMSE={test_metrics.width.rmse_cm:.2f} cm  R²={test_metrics.width.r2:.3f}")

    # Confusion matrices
    save_confusion_heatmaps(
        health_cm     = test_metrics.health.confusion,
        month_cm      = test_metrics.month.confusion,
        health_labels = HEALTH_CLASSES,
        month_labels  = MONTH_CLASSES,
        out_path      = out_dir / "confusion_matrices.png",
    )

    # Sample predictions CSV
    _save_predictions_csv(model, test_loader, device, out_dir / "sample_predictions.csv")
    print("\n✓ Training complete. All outputs saved to:", out_dir)


# ── Sample predictions CSV ────────────────────────────────────────────────────

@torch.no_grad()
def _save_predictions_csv(model, loader, device, path: Path) -> None:
    import csv
    import torch.nn.functional as F
    from .constants import IDX_TO_HEALTH, IDX_TO_MONTH, EXPECTED_WIDTH_CM, STUNTING_THRESHOLD_FRACTION

    model.eval()
    rows = []
    for x, y_h, y_m, y_w, paths in loader:
        x = x.to(device)
        out = model(x)
        h_probs = F.softmax(out.health_logits, 1)
        m_probs = F.softmax(out.month_logits, 1)

        h_idx = h_probs.argmax(1).cpu().tolist()
        m_idx = m_probs.argmax(1).cpu().tolist()
        h_conf = h_probs.max(1).values.cpu().tolist()
        m_conf = m_probs.max(1).values.cpu().tolist()
        w_pred = out.width_cm.cpu().tolist()
        y_h_l  = y_h.cpu().tolist()
        y_m_l  = y_m.cpu().tolist()
        y_w_l  = y_w.cpu().tolist()

        for i, p in enumerate(paths):
            mon_num = m_idx[i] + 1
            exp_w   = EXPECTED_WIDTH_CM[mon_num]
            stunted = w_pred[i] < STUNTING_THRESHOLD_FRACTION * exp_w
            rows.append({
                "path":         p,
                "true_health":  IDX_TO_HEALTH[y_h_l[i]],
                "pred_health":  IDX_TO_HEALTH[h_idx[i]],
                "health_conf":  f"{h_conf[i]:.4f}",
                "true_month":   IDX_TO_MONTH[y_m_l[i]],
                "pred_month":   IDX_TO_MONTH[m_idx[i]],
                "month_conf":   f"{m_conf[i]:.4f}",
                "true_width_cm":  f"{y_w_l[i]:.2f}",
                "pred_width_cm":  f"{w_pred[i]:.2f}",
                "expected_width_cm": f"{exp_w:.1f}",
                "is_stunted":   str(stunted),
            })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ Sample predictions → {path}")
