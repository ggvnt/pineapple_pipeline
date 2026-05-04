"""Grad-CAM for the MultiTaskNet — supports health and month tasks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class CamResult:
    heatmap:     np.ndarray   # H×W float32 in [0, 1]
    overlay_bgr: np.ndarray   # H×W×3 uint8


class GradCAM:
    """
    Usage
    -----
    cam = GradCAM(model, target_layer)
    result = cam(input_tensor, class_idx, original_rgb_uint8, score_tensor)
    cam.close()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._fh = target_layer.register_forward_hook(self._save_acts)
        self._bh = target_layer.register_full_backward_hook(self._save_grads)

    def close(self) -> None:
        self._fh.remove()
        self._bh.remove()

    def _save_acts(self, _m, _i, out) -> None:
        self._acts = out

    def _save_grads(self, _m, _i, grad_out) -> None:
        self._grads = grad_out[0]

    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        cam = np.maximum(cam, 0)
        denom = cam.max()
        if denom < 1e-8:
            return np.zeros_like(cam, dtype=np.float32)
        return (cam / denom).astype(np.float32)

    def __call__(
        self,
        input_tensor:     torch.Tensor,    # 1×3×H×W
        class_idx:        int,
        original_rgb_u8:  np.ndarray,      # H×W×3 uint8
        score_tensor:     torch.Tensor,    # 1×C logits
    ) -> CamResult:
        self.model.zero_grad(set_to_none=True)
        score = score_tensor[:, class_idx].sum()
        score.backward(retain_graph=True)

        acts  = self._acts
        grads = self._grads
        if acts is None or grads is None:
            raise RuntimeError("GradCAM hooks did not fire — check target_layer.")

        weights = grads.mean(dim=(2, 3), keepdim=True)   # 1×C×1×1
        cam_map = (weights * acts).sum(dim=1).squeeze(0)  # H×W
        cam_np  = self._normalize(cam_map.detach().cpu().numpy())

        h, w = original_rgb_u8.shape[:2]
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        bgr_orig = cv2.cvtColor(original_rgb_u8, cv2.COLOR_RGB2BGR)
        overlay  = cv2.addWeighted(bgr_orig, 0.55, heatmap, 0.45, 0)

        return CamResult(heatmap=cam_resized, overlay_bgr=overlay)
