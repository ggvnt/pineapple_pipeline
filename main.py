"""Command-line entry point for the Pineapple Analysis pipeline.

Usage
-----
python main.py train  --data_root /data/pineapple --output_dir runs/exp01 ...
python main.py infer  --image photo.jpg --checkpoint runs/exp01/checkpoints/best_model.pt ...
python main.py export --checkpoint best_model.pt --format onnx --out model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _train_parser(sub):
    p = sub.add_parser("train", help="Train the multi-task model")
    p.add_argument("--data_root",      required=True,  help="Root of dataset (M1/…/M12 structure)")
    p.add_argument("--width_csv",      default=None,   help="CSV with columns: path, width_cm")
    p.add_argument("--output_dir",     default="runs/output",      help="JSON / plots / CSV output dir")
    p.add_argument("--checkpoint_dir", default="runs/checkpoints", help="Checkpoint (.pt) output dir")
    p.add_argument("--backbone",       default="efficientnet_b0",  choices=["efficientnet_b0", "resnet50"])
    p.add_argument("--image_size",     default=224,  type=int)
    p.add_argument("--batch_size",     default=32,   type=int)
    p.add_argument("--epochs",         default=80,   type=int)
    p.add_argument("--lr",             default=3e-4, type=float)
    p.add_argument("--weight_decay",   default=1e-4, type=float)
    p.add_argument("--dropout",        default=0.3,  type=float)
    p.add_argument("--alpha",          default=0.4,  type=float, help="Health loss weight")
    p.add_argument("--beta",           default=0.4,  type=float, help="Month loss weight")
    p.add_argument("--gamma",          default=0.2,  type=float, help="Width loss weight")
    p.add_argument("--label_smoothing",default=0.0,  type=float)
    p.add_argument("--patience",       default=10,   type=int)
    p.add_argument("--seed",           default=42,   type=int)
    p.add_argument("--num_workers",    default=4,    type=int)
    p.add_argument("--device",         default="auto")
    p.add_argument("--freeze_epochs",  default=5,    type=int, help="Freeze backbone for N epochs")
    p.add_argument("--strong_aug",     action="store_true", default=True)
    p.add_argument("--sampler",        default="joint", choices=["joint", "health", "none"])
    return p


def _infer_parser(sub):
    p = sub.add_parser("infer", help="Run inference on a single image")
    p.add_argument("--image",              required=True)
    p.add_argument("--checkpoint",         required=True)
    p.add_argument("--save_dir",           default=None, help="Dir to save GradCAM + JSON")
    p.add_argument("--device",             default="auto")
    p.add_argument("--fallback_px_per_cm", default=30.0, type=float)
    return p


def _export_parser(sub):
    p = sub.add_parser("export", help="Export model to ONNX or TorchScript")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out",        required=True, help="Output file path")
    p.add_argument("--format",     default="onnx", choices=["onnx", "torchscript"])
    p.add_argument("--opset",      default=17, type=int, help="ONNX opset version")
    p.add_argument("--image_size", default=224, type=int)
    return p


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pineapple",
        description="Pineapple Plant Analysis System — Multi-task ML pipeline",
    )
    sub = parser.add_subparsers(dest="command")
    _train_parser(sub)
    _infer_parser(sub)
    _export_parser(sub)

    args = parser.parse_args(argv)

    if args.command == "train":
        from pineapple.train import train_main
        train_main(args)

    elif args.command == "infer":
        from pineapple.infer import infer_main
        infer_main(args)

    elif args.command == "export":
        from pineapple.export import export_main
        export_main(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
