"""
Ruler calibration: detect a ruler in an image and compute px_per_cm.

Strategy:
  1. Convert to grayscale + Canny edges
  2. Detect horizontal lines with HoughLinesP
  3. Find parallel tick-mark pattern → estimate pixels-per-cm
  4. Fallback: user-supplied px_per_cm constant
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    px_per_cm: float
    method: str          # "hough" | "fallback"
    confidence: float    # 0-1


def _detect_ruler_hough(
    gray: np.ndarray,
    *,
    expected_tick_px_range: tuple[int, int] = (8, 60),
) -> Optional[float]:
    """
    Try to detect ruler graduation marks using edge detection + Hough lines.
    Returns px_per_cm estimate or None.
    """
    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # Detect horizontal-ish line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=20,
        maxLineGap=8,
    )
    if lines is None or len(lines) < 4:
        return None

    # Keep nearly-horizontal segments
    horizontal = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 10 or angle > 170:
            horizontal.append((x1, y1, x2, y2))

    if len(horizontal) < 4:
        return None

    # Cluster x-positions of short segments (tick marks)
    x_centers = [int((x1 + x2) / 2) for x1, y1, x2, y2 in horizontal
                  if abs(x2 - x1) < 80]  # short ticks
    if len(x_centers) < 3:
        return None

    x_centers = sorted(x_centers)
    gaps = [x_centers[i + 1] - x_centers[i] for i in range(len(x_centers) - 1)]
    gaps = [g for g in gaps if expected_tick_px_range[0] <= g <= expected_tick_px_range[1]]
    if not gaps:
        return None

    median_gap = float(np.median(gaps))
    return median_gap   # one gap = 1 cm


def calibrate_from_image(
    image_rgb: np.ndarray,
    fallback_px_per_cm: float = 30.0,
) -> CalibrationResult:
    """
    Attempt ruler auto-detection. Falls back to `fallback_px_per_cm`.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    px_per_cm = _detect_ruler_hough(gray)

    if px_per_cm is not None and 5.0 < px_per_cm < 200.0:
        return CalibrationResult(
            px_per_cm=px_per_cm,
            method="hough",
            confidence=0.75,
        )

    return CalibrationResult(
        px_per_cm=fallback_px_per_cm,
        method="fallback",
        confidence=0.0,
    )


def pixel_width_to_cm(pixel_width: float, px_per_cm: float) -> float:
    """Convert a measured pixel width to centimetres."""
    if px_per_cm <= 0:
        raise ValueError(f"px_per_cm must be positive, got {px_per_cm}")
    return pixel_width / px_per_cm


def measure_plant_pixel_width(mask_uint8: np.ndarray) -> float:
    """
    Given a binary plant mask (0/255), return the maximum horizontal span
    of the largest connected component (proxy for plant width in pixels).
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    bbox_w = float(stats[best, cv2.CC_STAT_WIDTH])
    return bbox_w
