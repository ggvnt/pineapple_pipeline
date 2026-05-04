"""Export trained model to ONNX or TorchScript."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from .model import MultiTaskNet
from .utils import ensure_dir


class _ExportWrapper(nn.Module):
    """Unwrap ModelOutput dataclass into plain tuple for ONNX/TorchScript."""

    def __init__(self, model: MultiTaskNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out.health_logits, out.month_logits, out.width_cm


def export_main(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    backbone   = ckpt.get("backbone",   "efficientnet_b0")
    image_size = int(getattr(args, "image_size", ckpt.get("image_size", 224)))

    model = MultiTaskNet(backbone=backbone)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    wrapper = _ExportWrapper(model)
    dummy   = torch.randn(1, 3, image_size, image_size)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)

    if args.format == "torchscript":
        traced = torch.jit.trace(wrapper, dummy)
        try:
            traced = torch.jit.optimize_for_inference(traced)
        except Exception:
            pass
        traced.save(str(out_path))
        print(f"✓ TorchScript saved → {out_path}")

    elif args.format == "onnx":
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            input_names=["image"],
            output_names=["health_logits", "month_logits", "width_cm"],
            dynamic_axes={
                "image":         {0: "batch"},
                "health_logits": {0: "batch"},
                "month_logits":  {0: "batch"},
                "width_cm":      {0: "batch"},
            },
            opset_version=int(args.opset),
        )
        print(f"✓ ONNX saved → {out_path}")
    else:
        raise ValueError(f"Unknown export format: {args.format!r}")
