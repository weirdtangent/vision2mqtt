# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import base64
import io
import pytest
from unittest.mock import MagicMock, patch

from vision2mqtt.mixins.detector import DetectorMixin
from vision2mqtt.mixins.labels import LabelsMixin
from vision2mqtt.models.events import DetectedObject, MotionEvent


class FakeDetector(DetectorMixin, LabelsMixin):
    def __init__(self, vision_config):
        self.vision_config = vision_config
        self.logger = MagicMock()
        self._detector_model = None


def _make_tiny_jpeg_b64():
    """Create a minimal valid JPEG image (1x1 red pixel) as base64."""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestDetectorFiltering:
    @pytest.mark.asyncio
    async def test_filters_by_confidence(self, sample_vision_config):
        sample_vision_config["min_confidence"] = 0.5
        detector = FakeDetector(sample_vision_config)

        raw_objects = [
            DetectedObject(label="person", raw_label="person", confidence=0.8, bbox=[0.1, 0.2, 0.3, 0.4]),
            DetectedObject(label="person", raw_label="person", confidence=0.3, bbox=[0.5, 0.6, 0.7, 0.8]),
        ]

        with patch.object(detector, "_detect_ultralytics", return_value=raw_objects):
            event = MotionEvent("cam1", "Test", "ev1", _make_tiny_jpeg_b64(), "2026-01-01T00:00:00", "test")
            result = await detector.detect_objects(event)
            assert len(result.objects) == 1
            assert result.objects[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_filters_by_label(self, sample_vision_config):
        sample_vision_config["labels"] = ["person"]  # only person
        detector = FakeDetector(sample_vision_config)

        raw_objects = [
            DetectedObject(label="person", raw_label="person", confidence=0.8, bbox=[0.1, 0.2, 0.3, 0.4]),
            DetectedObject(label="chair", raw_label="chair", confidence=0.9, bbox=[0.5, 0.6, 0.7, 0.8]),
        ]

        with patch.object(detector, "_detect_ultralytics", return_value=raw_objects):
            event = MotionEvent("cam1", "Test", "ev1", _make_tiny_jpeg_b64(), "2026-01-01T00:00:00", "test")
            result = await detector.detect_objects(event)
            assert len(result.objects) == 1
            assert result.objects[0].label == "person"

    @pytest.mark.asyncio
    async def test_maps_car_to_vehicle(self, sample_vision_config):
        detector = FakeDetector(sample_vision_config)

        raw_objects = [
            DetectedObject(label="car", raw_label="car", confidence=0.75, bbox=[0.1, 0.2, 0.3, 0.4]),
        ]

        with patch.object(detector, "_detect_ultralytics", return_value=raw_objects):
            event = MotionEvent("cam1", "Test", "ev1", _make_tiny_jpeg_b64(), "2026-01-01T00:00:00", "test")
            result = await detector.detect_objects(event)
            assert len(result.objects) == 1
            assert result.objects[0].label == "vehicle"
            assert result.objects[0].raw_label == "car"

    @pytest.mark.asyncio
    async def test_processing_time_recorded(self, sample_vision_config):
        detector = FakeDetector(sample_vision_config)

        with patch.object(detector, "_detect_ultralytics", return_value=[]):
            event = MotionEvent("cam1", "Test", "ev1", _make_tiny_jpeg_b64(), "2026-01-01T00:00:00", "test")
            result = await detector.detect_objects(event)
            assert result.processing_time_ms >= 0
