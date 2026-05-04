"""
Inference module for the Pineapple Analysis System.

Public API
----------
predict(image_path, model_path, **kwargs) -> dict
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .advice import generate_advice
from .constants import (
    EXPECTED_WIDTH_CM,
    IDX_TO_HEALTH,
    IDX_TO_MONTH,
    STUNTING_THRESHOLD_FRACTION,
)
from .data import build_transforms
from .gradcam import GradCAM
from .model import MultiTaskNet, get_cam_layer
from .ruler import calibrate_from_image
from .utils import ensure_dir, resolve_device, save_json


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_rgb_u8(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _preprocess(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    tfm = build_transforms(image_size=image_size, train=False)
    x   = tfm(Image.fromarray(rgb)).unsqueeze(0)
    return x


def _load_model(model_path: str | Path, device: torch.device) -> tuple[MultiTaskNet, dict]:
    ckpt    = torch.load(model_path, map_location=device)
    backbone   = ckpt.get("backbone",   "efficientnet_b0")
    image_size = ckpt.get("image_size", 224)
    model = MultiTaskNet(backbone=backbone).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, {"backbone": backbone, "image_size": image_size}


# ── Main public function ──────────────────────────────────────────────────────

def predict(
    image_path: str | Path,
    model_path: str | Path,
    *,
    device:              str = "auto",
    save_dir:            str | Path | None = None,
    fallback_px_per_cm:  float = 30.0,
) -> dict[str, Any]:
    """
    Run full inference on a single image.

    Returns
    -------
    {
        'health':             str,
        'health_confidence':  float,
        'month':              str,
        'month_confidence':   float,
        'width_cm':           float,
        'is_stunted':         bool,
        'advice':             dict,
        'gradcam_path':       str | None,
        'calibration':        dict,
    }
    """
    dev   = resolve_device(device)
    model, meta = _load_model(model_path, dev)
    image_size  = meta["image_size"]

    rgb  = _load_rgb_u8(image_path)
    x    = _preprocess(rgb, image_size).to(dev)

    # ── Forward pass (no grad for base predictions) ───────────────────────────
    with torch.inference_mode():
        out = model(x)
        h_probs = F.softmax(out.health_logits, dim=1)
        m_probs = F.softmax(out.month_logits,  dim=1)
        h_idx   = int(h_probs.argmax(1).item())
        m_idx   = int(m_probs.argmax(1).item())
        h_conf  = float(h_probs[0, h_idx].item())
        m_conf  = float(m_probs[0, m_idx].item())
        w_pred  = float(out.width_cm.item())

    # ── Ruler calibration ─────────────────────────────────────────────────────
    calibration = calibrate_from_image(rgb, fallback_px_per_cm=fallback_px_per_cm)
    # Width from model is already in cm (trained directly). If ruler detected with
    # high confidence, optionally cross-check; for now we report both.
    width_cm = w_pred

    # ── Stunting detection ────────────────────────────────────────────────────
    month_number = m_idx + 1
    expected_cm  = EXPECTED_WIDTH_CM[month_number]
    is_stunted   = width_cm < STUNTING_THRESHOLD_FRACTION * expected_cm

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gradcam_path: str | None = None
    if save_dir is not None:
        save_dir = ensure_dir(save_dir)
        # Need a second forward pass with gradients enabled
        x_grad = _preprocess(rgb, image_size).to(dev)
        target_layer = get_cam_layer(model)
        cam = GradCAM(model, target_layer)
        out_grad = model(x_grad)
        cam_result = cam(
            input_tensor    = x_grad,
            class_idx       = h_idx,
            original_rgb_u8 = rgb,
            score_tensor    = out_grad.health_logits,
        )
        cam.close()

        overlay_path = save_dir / "gradcam_overlay.jpg"
        cv2.imwrite(str(overlay_path), cam_result.overlay_bgr)
        gradcam_path = str(overlay_path)

    # ── Farmer advice ─────────────────────────────────────────────────────────
    advice = generate_advice(
        health_label       = IDX_TO_HEALTH[h_idx],
        health_confidence  = h_conf,
        month_number       = month_number,
        width_cm           = width_cm,
        is_stunted         = is_stunted,
    )

    result: dict[str, Any] = {
        "health":            IDX_TO_HEALTH[h_idx],
        "health_confidence": round(h_conf, 4),
        "month":             IDX_TO_MONTH[m_idx],
        "month_number":      month_number,
        "month_confidence":  round(m_conf, 4),
        "width_cm":          round(width_cm, 2),
        "expected_width_cm": expected_cm,
        "is_stunted":        is_stunted,
        "stunting_threshold_fraction": STUNTING_THRESHOLD_FRACTION,
        "advice":            advice.to_dict(),
        "gradcam_path":      gradcam_path,
        "calibration": {
            "method":       calibration.method,
            "px_per_cm":    round(calibration.px_per_cm, 2),
            "confidence":   round(calibration.confidence, 2),
        },
    }

    if save_dir is not None:
        save_json(Path(save_dir) / "prediction.json", result)

    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

def infer_main(args: argparse.Namespace) -> None:
    result = predict(
        image_path         = args.image,
        model_path         = args.checkpoint,
        device             = str(args.device),
        save_dir           = getattr(args, "save_dir", None),
        fallback_px_per_cm = float(getattr(args, "fallback_px_per_cm", 30.0)),
    )
    import json
    print(json.dumps(result, indent=2))
