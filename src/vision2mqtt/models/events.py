# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from dataclasses import dataclass, field


@dataclass
class MotionEvent:
    camera_id: str
    camera_name: str
    event_id: str
    image_b64: str
    timestamp: str
    source: str


@dataclass
class DetectedObject:
    label: str
    raw_label: str
    confidence: float
    bbox: list[float] = field(default_factory=list)  # [x1, y1, x2, y2] normalized 0-1


@dataclass
class VisionResult:
    objects: list[DetectedObject] = field(default_factory=list)
    processing_time_ms: float = 0.0
