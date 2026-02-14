# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from vision2mqtt.models.events import DetectedObject, MotionEvent, VisionResult


class TestMotionEvent:
    def test_create(self, sample_motion_event):
        assert sample_motion_event.camera_id == "2BEFD0C907BB6BF2"
        assert sample_motion_event.camera_name == "Front Yard"
        assert sample_motion_event.event_id == "20260214-153045"
        assert sample_motion_event.source == "recording_snapshot"


class TestDetectedObject:
    def test_create(self):
        obj = DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.12, 0.34, 0.45, 0.89])
        assert obj.label == "person"
        assert obj.confidence == 0.87
        assert len(obj.bbox) == 4

    def test_default_bbox(self):
        obj = DetectedObject(label="vehicle", raw_label="car", confidence=0.72)
        assert obj.bbox == []


class TestVisionResult:
    def test_empty_result(self):
        result = VisionResult()
        assert result.objects == []
        assert result.processing_time_ms == 0.0

    def test_with_objects(self):
        objs = [
            DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.1, 0.2, 0.3, 0.4]),
            DetectedObject(label="vehicle", raw_label="car", confidence=0.72, bbox=[0.5, 0.1, 0.9, 0.5]),
        ]
        result = VisionResult(objects=objs, processing_time_ms=8.2)
        assert len(result.objects) == 2
        assert result.processing_time_ms == 8.2
