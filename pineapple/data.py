"""Dataset scanning, transforms, and PyTorch Dataset / DataLoader helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from .constants import (
    HEALTH_TO_IDX,
    MONTH_TO_IDX,
    EXPECTED_WIDTH_CM,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Sample dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sample:
    path: Path
    health: int   # index into HEALTH_CLASSES
    month:  int   # index into MONTH_CLASSES  (0-based, so month_number = month+1)
    width_cm: float  # ground-truth width; 0.0 if unknown (will be masked in loss)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


# ── Dataset scanning ──────────────────────────────────────────────────────────

def scan_dataset(root: str | Path, width_csv: str | Path | None = None) -> list[Sample]:
    """
    Walk root / M{1-12} / {health_class} / *.jpg structure.

    If width_csv is provided it must contain columns: path, width_cm.
    Missing paths get width_cm = 0.0 (loss is masked for those samples).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    # Optional CSV lookup
    csv_lookup: dict[str, float] = {}
    if width_csv is not None:
        import csv
        with open(width_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                p = str(Path(row["path"]).resolve())
                csv_lookup[p] = float(row["width_cm"])

    samples: list[Sample] = []
    for month_dir in sorted(root.iterdir()):
        if not month_dir.is_dir():
            continue
        if month_dir.name not in MONTH_TO_IDX:
            continue
        month_idx = MONTH_TO_IDX[month_dir.name]
        month_num = month_idx + 1

        for health_dir in sorted(month_dir.iterdir()):
            if not health_dir.is_dir():
                continue
            if health_dir.name not in HEALTH_TO_IDX:
                continue
            health_idx = HEALTH_TO_IDX[health_dir.name]

            for img_path in health_dir.rglob("*"):
                if not _is_image(img_path):
                    continue
                key = str(img_path.resolve())
                width = csv_lookup.get(key, EXPECTED_WIDTH_CM[month_num])
                samples.append(Sample(
                    path=img_path,
                    health=health_idx,
                    month=month_idx,
                    width_cm=width,
                ))

    if not samples:
        raise RuntimeError(
            f"No images found under {root!r}.\n"
            "Expected layout: M1/healthy/*.jpg, M1/nitrogen_deficiency/*.jpg …"
        )
    return samples


# ── Train / val / test split ──────────────────────────────────────────────────

def split_samples(
    samples: list[Sample],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Stratified split on (health, month) joint label."""
    indices = list(range(len(samples)))
    y = [s.health * 12 + s.month for s in samples]

    def _split(idx_list, labels, test_size):
        try:
            return train_test_split(idx_list, test_size=test_size,
                                    stratify=labels, random_state=seed, shuffle=True)
        except Exception:
            rng = random.Random(seed)
            rng.shuffle(idx_list)
            n = int(len(idx_list) * (1 - test_size))
            return idx_list[:n], idx_list[n:]

    train_idx, tmp_idx = _split(indices, y, 1 - train_ratio)
    tmp_y = [y[i] for i in tmp_idx]
    val_share = val_ratio / (1 - train_ratio)
    val_idx, test_idx = _split(tmp_idx, tmp_y, 1 - val_share)

    return ([samples[i] for i in train_idx],
            [samples[i] for i in val_idx],
            [samples[i] for i in test_idx])


# ── Transforms ────────────────────────────────────────────────────────────────

def build_transforms(image_size: int, train: bool, strong_aug: bool = False) -> transforms.Compose:
    if train:
        ops: list = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAutocontrast(p=0.2),
        ]
        if strong_aug:
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        if strong_aug:
            ops.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.20), value="random"))
        return transforms.Compose(ops)

    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class PineappleDataset(Dataset):
    def __init__(self, samples: Iterable[Sample], tfm: transforms.Compose) -> None:
        self.samples = list(samples)
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.tfm(img)
        return (
            x,
            torch.tensor(s.health,   dtype=torch.long),
            torch.tensor(s.month,    dtype=torch.long),
            torch.tensor(s.width_cm, dtype=torch.float32),
            str(s.path),
        )


# ── Weighted sampler ──────────────────────────────────────────────────────────

def make_sampler(samples: list[Sample], mode: str = "joint") -> WeightedRandomSampler:
    """
    mode: 'health' | 'joint'
      health — oversample minority health classes
      joint  — oversample minority (health, month) combinations
    """
    from collections import Counter

    if mode == "health":
        keys = [s.health for s in samples]
    elif mode == "joint":
        keys = [s.health * 12 + s.month for s in samples]
    else:
        raise ValueError(f"Unknown sampler mode: {mode!r}")

    counts = Counter(keys)
    weights = torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── DataLoader factory ────────────────────────────────────────────────────────

def make_loaders(
    train_s: list[Sample],
    val_s:   list[Sample],
    test_s:  list[Sample],
    *,
    image_size:  int,
    batch_size:  int,
    num_workers: int,
    strong_aug:  bool,
    sampler_mode: str,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tfm = build_transforms(image_size, train=True,  strong_aug=strong_aug)
    eval_tfm  = build_transforms(image_size, train=False)

    train_ds = PineappleDataset(train_s, train_tfm)
    val_ds   = PineappleDataset(val_s,  eval_tfm)
    test_ds  = PineappleDataset(test_s, eval_tfm)

    sampler = make_sampler(train_s, mode=sampler_mode) if sampler_mode != "none" else None

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(train_ds, shuffle=(sampler is None), sampler=sampler, **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)
    return train_loader, val_loader, test_loader
