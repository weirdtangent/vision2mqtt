# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

from typing import TYPE_CHECKING

from vision2mqtt.models.events import DetectedObject, VisionResult

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt

# Maximum edge-to-edge gap (normalized coords) to consider two bboxes "near"
_PROXIMITY_THRESHOLD = 0.15

# Raw labels used by each composite type
_COMPOSITE_COMPANIONS: dict[str, set[str]] = {
    "dog_walker": {"dog"},
    "cyclist": {"bicycle"},
    "package_carrier": {"backpack", "suitcase", "handbag"},
}


def _bbox_gap(a: list[float], b: list[float]) -> float:
    """Minimum edge-to-edge gap between two [x1, y1, x2, y2] bboxes.

    Returns 0 or negative if the boxes overlap on both axes.
    """
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return (dx**2 + dy**2) ** 0.5


def _has_nearby_pair(persons: list[DetectedObject], companions: list[DetectedObject], threshold: float) -> bool:
    """Return True if any person bbox is within threshold of any companion bbox."""
    for p in persons:
        for c in companions:
            if _bbox_gap(p.bbox, c.bbox) <= threshold:
                return True
    return False


class CompositesMixin:
    def compute_composites(self: Vision2Mqtt, result: VisionResult) -> dict[str, bool]:
        """Compute composite presence states from single-frame spatial analysis.

        Uses all_detections (pre-label-filter) so companion objects like dog/bicycle/backpack
        are available even if not in the user's configured labels list.
        """
        enabled = self.vision_config.get("composites") or []
        if not enabled:
            return {}

        threshold = _PROXIMITY_THRESHOLD

        # Pre-extract persons and companion objects from all_detections
        persons = [d for d in result.all_detections if d.raw_label == "person"]
        composites: dict[str, bool] = {}

        for name in enabled:
            if name == "group":
                composites[name] = len(persons) >= 2
            elif name in _COMPOSITE_COMPANIONS:
                raw_labels = _COMPOSITE_COMPANIONS[name]
                companions = [d for d in result.all_detections if d.raw_label in raw_labels]
                composites[name] = _has_nearby_pair(persons, companions, threshold)
            else:
                self.logger.warning(f"unknown composite type: {name}")

        return composites
