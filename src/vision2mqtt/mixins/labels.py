# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt

# COCO class names -> specific categories
COCO_LABEL_MAP: dict[str, str] = {
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "cat": "cat",
    "dog": "dog",
    "horse": "horse",
    "cow": "cow",
    "sheep": "sheep",
    "bear": "bear",
    "elephant": "elephant",
    "zebra": "zebra",
    "giraffe": "giraffe",
    "bird": "bird",
    "backpack": "package",
    "suitcase": "package",
    "handbag": "package",
}

# Specific label -> parent group (for catch-all presence sensors)
LABEL_PARENTS: dict[str, str] = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "cow": "animal",
    "sheep": "animal",
    "bear": "animal",
    "elephant": "animal",
    "zebra": "animal",
    "giraffe": "animal",
}

# MDI icons per label type
LABEL_ICONS: dict[str, str] = {
    "person": "mdi:account",
    "vehicle": "mdi:car-side",
    "car": "mdi:car",
    "truck": "mdi:truck",
    "bus": "mdi:bus",
    "motorcycle": "mdi:motorbike",
    "bicycle": "mdi:bicycle",
    "animal": "mdi:paw",
    "cat": "mdi:cat",
    "dog": "mdi:dog-side",
    "bird": "mdi:bird",
    "package": "mdi:package-variant",
    "dog_walker": "mdi:dog-service",
    "group": "mdi:account-group",
    "cyclist": "mdi:bike",
    "student": "mdi:school",
    "package_carrier": "mdi:package-variant-closed",
}


class LabelsMixin:
    def get_simplified_label(self: Vision2Mqtt, raw_label: str) -> str | None:
        mapped = COCO_LABEL_MAP.get(raw_label)
        if not mapped:
            return None
        # Return specific label if it's directly in config
        if mapped in self.vision_config["labels"]:
            return mapped
        # Fall back to parent group if configured
        parent = LABEL_PARENTS.get(mapped)
        if parent and parent in self.vision_config["labels"]:
            return parent
        return None
