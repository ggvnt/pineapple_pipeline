"""Central constants for the Pineapple Plant Analysis System."""

from __future__ import annotations

# ── Health ────────────────────────────────────────────────────────────────────
HEALTH_CLASSES = ["healthy", "nitrogen_deficiency", "water_stress"]
HEALTH_TO_IDX  = {name: i for i, name in enumerate(HEALTH_CLASSES)}
IDX_TO_HEALTH  = {i: name for name, i in HEALTH_TO_IDX.items()}

# ── Growth Months ─────────────────────────────────────────────────────────────
MONTH_CLASSES = [f"M{i}" for i in range(1, 13)]
MONTH_TO_IDX  = {name: i for i, name in enumerate(MONTH_CLASSES)}
IDX_TO_MONTH  = {i: name for name, i in MONTH_TO_IDX.items()}

# Expected plant width (cm) per month — domain knowledge from agronomists
EXPECTED_WIDTH_CM: dict[int, float] = {
    1:  8.0,
    2: 12.0,
    3: 17.0,
    4: 23.0,
    5: 29.0,
    6: 35.0,
    7: 41.0,
    8: 46.0,
    9: 51.0,
    10: 55.0,
    11: 59.0,
    12: 62.0,
}

# Width tolerance for stunting detection (fraction of expected width)
STUNTING_THRESHOLD_FRACTION = 0.80   # flag if actual < 80 % of expected

# ImageNet normalisation stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
