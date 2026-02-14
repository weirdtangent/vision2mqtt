# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt

# COCO class names -> simplified categories
COCO_LABEL_MAP: dict[str, str] = {
    "person": "person",
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
    "bird": "bird",
}


class LabelsMixin:
    def get_simplified_label(self: Vision2Mqtt, raw_label: str) -> str | None:
        simplified = COCO_LABEL_MAP.get(raw_label)
        if simplified and simplified in self.vision_config["labels"]:
            return simplified
        return None
