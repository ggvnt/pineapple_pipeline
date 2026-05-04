"""Generate safe, actionable farmer advice from model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import EXPECTED_WIDTH_CM, STUNTING_THRESHOLD_FRACTION


@dataclass(frozen=True)
class Advice:
    title: str
    confidence_note: str
    what_to_check: list[str]
    recovery_steps: list[str]
    when_to_escalate: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title":            self.title,
            "confidence_note":  self.confidence_note,
            "what_to_check":    self.what_to_check,
            "recovery_steps":   self.recovery_steps,
            "when_to_escalate": self.when_to_escalate,
        }


# ── Confidence note ───────────────────────────────────────────────────────────

def _conf_note(confidence: float) -> str:
    if confidence >= 0.75:
        return "High-confidence prediction based on the image."
    if confidence >= 0.55:
        return "Medium confidence. Consider a follow-up field check."
    return (
        "Low confidence — image conditions may be suboptimal. "
        "Retake in good natural light and consult an agronomist if unsure."
    )


# ── Per-condition advice tables ───────────────────────────────────────────────

_COMMON_ESCALATE = [
    "If symptoms spread rapidly within 3–7 days.",
    "If the crown, stem base, or roots appear dark or mushy.",
    "If more than 30 % of plants in the block are affected.",
    "Consult your local agronomist — local soil and weather context matters.",
]

_ADVICE: dict[str, dict[str, list[str]]] = {
    "healthy": {
        "what_to_check": [
            "Keep up weekly scouting — early problems are cheapest to fix.",
            "Confirm irrigation uniformity across the block.",
            "Check that leaf colour is uniform (no patchy pale areas).",
        ],
        "recovery": [
            "Maintain current fertiliser and irrigation schedule.",
            "If any yellowing or curling appears, re-photograph and run prediction again.",
        ],
    },
    "nitrogen_deficiency": {
        "what_to_check": [
            "Look for uniform pale-yellow or lime-green colouring starting on older (lower) leaves.",
            "Compare multiple plants — N deficiency typically appears across a whole zone, not isolated plants.",
            "Check recent rainfall or irrigation — excessive leaching can cause sudden N loss.",
        ],
        "recovery": [
            "Apply a nitrogen source in small split doses (follow local extension label rates).",
            "Avoid large single applications before forecast rain to reduce leaching.",
            "Foliar urea spray (1–2 %) can give fast relief while soil application takes effect.",
            "Re-photograph after 10–14 days to confirm colour recovery.",
            "Maintain consistent irrigation — drought or waterlogging slows N uptake.",
        ],
    },
    "water_stress": {
        "what_to_check": [
            "Check for leaf curl, rolling, or a dull grey-green colour during peak heat hours.",
            "Inspect drip lines or sprinkler heads for blockages or uneven coverage.",
            "Check soil moisture 10–15 cm deep, not just the surface.",
        ],
        "recovery": [
            "Shift to deeper, less frequent irrigation rather than frequent shallow wetting.",
            "Water early morning or late afternoon to reduce evaporation losses.",
            "Apply mulch around plants (not touching the crown) to retain soil moisture.",
            "If soil stays saturated for extended periods, reduce irrigation frequency and improve drainage.",
        ],
    },
}


# ── Width / stunting advice ───────────────────────────────────────────────────

def _stunting_advice(
    month_number: int,
    width_cm: float,
    expected_cm: float,
    is_stunted: bool,
) -> list[str]:
    if not is_stunted:
        return []
    shortfall = expected_cm - width_cm
    return [
        f"⚠️  STUNTED GROWTH DETECTED — measured {width_cm:.1f} cm vs expected "
        f"≈{expected_cm:.1f} cm for month {month_number} "
        f"(shortfall {shortfall:.1f} cm, threshold {STUNTING_THRESHOLD_FRACTION*100:.0f} %).",
        "Check for root restriction (compacted soil, nematodes) or combined water + nutrient stress.",
        "Compare plant height against adjacent same-age plants to confirm the flag.",
        "Consider soil sampling for nematodes, pH, and nutrient profile if the block is broadly affected.",
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_advice(
    *,
    health_label: str,
    health_confidence: float,
    month_number: int,
    width_cm: float,
    is_stunted: bool,
) -> Advice:
    expected_cm = EXPECTED_WIDTH_CM.get(month_number, 0.0)
    template = _ADVICE.get(health_label, {
        "what_to_check": ["Examine the plant carefully and compare with healthy reference plants."],
        "recovery":      ["Retake the photo with better lighting and consult an agronomist."],
    })

    what_to_check = list(template["what_to_check"])
    recovery      = list(template["recovery"])

    # Prepend stunting notes
    stunting_notes = _stunting_advice(month_number, width_cm, expected_cm, is_stunted)
    recovery = stunting_notes + recovery

    what_to_check.append(f"Model estimated growth stage: Month {month_number}. "
                          f"Expected width ≈ {expected_cm:.0f} cm.")

    label_pretty = health_label.replace("_", " ").title()
    title = f"Diagnosis: {label_pretty} — Month {month_number}"

    return Advice(
        title=title,
        confidence_note=_conf_note(health_confidence),
        what_to_check=what_to_check,
        recovery_steps=recovery,
        when_to_escalate=list(_COMMON_ESCALATE),
    )
