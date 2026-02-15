# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import base64
import io

import numpy as np
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
        self._axcl_input_name = "images"
        self._axcl_input_size = 640
        self._axcl_nchw = False


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


class TestResolveInputLayout:
    def test_nhwc_shape(self):
        size, nchw = DetectorMixin._resolve_input_layout([1, 640, 640, 3])
        assert size == 640
        assert nchw is False

    def test_nchw_shape(self):
        size, nchw = DetectorMixin._resolve_input_layout([1, 3, 640, 640])
        assert size == 640
        assert nchw is True

    def test_single_channel_nhwc(self):
        size, nchw = DetectorMixin._resolve_input_layout([1, 320, 320, 1])
        assert size == 320
        assert nchw is False

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            DetectorMixin._resolve_input_layout([1, 640, 480, 3])

    def test_ambiguous_shape_raises(self):
        with pytest.raises(ValueError, match="Unable to determine"):
            DetectorMixin._resolve_input_layout([1, 640, 640, 640])

    def test_non_4d_raises(self):
        with pytest.raises(ValueError, match="4D"):
            DetectorMixin._resolve_input_layout([640, 640, 3])


class TestAxclBackend:
    @pytest.mark.asyncio
    async def test_detect_axcl_returns_objects(self, sample_vision_config):
        sample_vision_config["backend"] = "axcl"
        detector = FakeDetector(sample_vision_config)

        # mock session with one detection: person at bbox, confidence 0.9
        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[[320, 200, 500, 450, 0.9, 0]]], dtype=np.float32)]  # [batch, num_det, 6]
        detector._detector_model = mock_session

        event = MotionEvent("cam1", "Test", "ev1", _make_tiny_jpeg_b64(), "2026-01-01T00:00:00", "test")
        result = await detector.detect_objects(event)

        assert len(result.objects) == 1
        assert result.objects[0].label == "person"
        assert result.objects[0].confidence == 0.9

        # verify session.run was called with correct input name
        call_args = mock_session.run.call_args
        assert call_args[0][0] is None
        assert "images" in call_args[0][1]
        input_array = call_args[0][1]["images"]
        assert input_array.shape == (1, 640, 640, 3)

    @pytest.mark.asyncio
    async def test_detect_axcl_filters_low_confidence(self, sample_vision_config):
        sample_vision_config["backend"] = "axcl"
        sample_vision_config["min_confidence"] = 0.5
        detector = FakeDetector(sample_vision_config)

        mock_session = MagicMock()
        mock_session.run.return_value = [
            np.array(
                [
                    [[320, 200, 500, 450, 0.9, 0]],  # person, high conf
                    [[100, 100, 200, 200, 0.3, 2]],  # car, low conf
                ],
                dtype=np.float32,
            ).reshape(1, 2, 6)
        ]
        detector._detector_model = mock_session

        event = MotionEvent("cam1", "Test", "ev1", _make_tiny_jpeg_b64(), "2026-01-01T00:00:00", "test")
        result = await detector.detect_objects(event)

        assert len(result.objects) == 1
        assert result.objects[0].label == "person"

    @pytest.mark.asyncio
    async def test_detect_axcl_nchw_transpose(self, sample_vision_config):
        sample_vision_config["backend"] = "axcl"
        detector = FakeDetector(sample_vision_config)
        detector._axcl_nchw = True

        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[[320, 200, 500, 450, 0.9, 0]]], dtype=np.float32)]
        detector._detector_model = mock_session

        event = MotionEvent("cam1", "Test", "ev1", _make_tiny_jpeg_b64(), "2026-01-01T00:00:00", "test")
        await detector.detect_objects(event)

        # verify input was transposed to NCHW (1, 3, H, W)
        input_array = mock_session.run.call_args[0][1]["images"]
        assert input_array.shape == (1, 3, 640, 640)

    @pytest.mark.asyncio
    async def test_detect_axcl_empty_output(self, sample_vision_config):
        sample_vision_config["backend"] = "axcl"
        detector = FakeDetector(sample_vision_config)

        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([], dtype=np.float32).reshape(1, 0, 6)]
        detector._detector_model = mock_session

        event = MotionEvent("cam1", "Test", "ev1", _make_tiny_jpeg_b64(), "2026-01-01T00:00:00", "test")
        result = await detector.detect_objects(event)

        assert len(result.objects) == 0
