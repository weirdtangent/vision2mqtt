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


class TestPostprocessYoloRaw:
    def test_single_detection(self):
        """A single high-confidence detection in the center of a small grid."""
        # 2x2 grid, 144 channels (64 DFL + 80 classes), stride = 640/2 = 320
        head = np.zeros((1, 2, 2, 144), dtype=np.float32)
        # place a detection at grid cell (0, 0): set class 0 (person) logit high
        head[0, 0, 0, 64] = 10.0  # sigmoid(10) ≈ 1.0 for class 0
        # DFL: set bin 8 high for all 4 coords (center of range)
        for coord in range(4):
            head[0, 0, 0, coord * 16 + 8] = 10.0

        result = DetectorMixin._postprocess_yolo_raw([head], input_size=640, num_classes=80)
        assert result.shape[1] == 6
        assert len(result) >= 1
        # best detection should be class 0 (person) with high confidence
        assert int(result[0, 5]) == 0
        assert result[0, 4] > 0.9

    def test_empty_output(self):
        """All-zero feature maps should produce no confident detections."""
        head = np.zeros((1, 2, 2, 144), dtype=np.float32)
        result = DetectorMixin._postprocess_yolo_raw([head], input_size=640, num_classes=80)
        # class logits are all 0 -> sigmoid = 0.5, which is low; expect no confident detections
        # or at least all below typical threshold
        if len(result) > 0:
            assert result[:, 4].max() <= 0.5

    def test_nms_suppresses_overlapping(self):
        """Two overlapping detections of the same class should be suppressed to one."""
        # 4x4 grid, stride = 160
        head = np.zeros((1, 4, 4, 144), dtype=np.float32)
        # set all class logits very negative so only our targets have high confidence
        head[..., 64:] = -20.0
        # two adjacent cells with same high-confidence class 0
        for row in [1, 2]:
            head[0, row, 1, 64] = 10.0  # class 0 logit
            for coord in range(4):
                head[0, row, 1, coord * 16 + 4] = 10.0  # similar bbox

        result = DetectorMixin._postprocess_yolo_raw([head], input_size=640, num_classes=80, nms_iou=0.45)
        high_conf = result[result[:, 4] > 0.9]
        assert len(high_conf) == 1  # NMS keeps only one high-confidence detection

    def test_multi_head(self):
        """Multiple output heads are concatenated before NMS."""
        head1 = np.zeros((1, 4, 4, 144), dtype=np.float32)
        head2 = np.zeros((1, 2, 2, 144), dtype=np.float32)
        # one detection in each head, different locations
        head1[0, 0, 0, 64] = 10.0
        for c in range(4):
            head1[0, 0, 0, c * 16 + 2] = 10.0
        head2[0, 1, 1, 66] = 10.0  # class 2 (car)
        for c in range(4):
            head2[0, 1, 1, c * 16 + 12] = 10.0

        result = DetectorMixin._postprocess_yolo_raw([head1, head2], input_size=640, num_classes=80)
        classes = set(int(r[5]) for r in result if r[4] > 0.9)
        assert 0 in classes  # person from head1
        assert 2 in classes  # car from head2

    def test_chw_layout(self):
        """CHW layout heads should be handled correctly."""
        head_hwc = np.zeros((1, 4, 4, 144), dtype=np.float32)
        head_hwc[0, 0, 0, 64] = 10.0
        for c in range(4):
            head_hwc[0, 0, 0, c * 16 + 8] = 10.0

        # transpose to CHW
        head_chw = np.transpose(head_hwc, (0, 3, 1, 2))
        assert head_chw.shape == (1, 144, 4, 4)

        result = DetectorMixin._postprocess_yolo_raw([head_chw], input_size=640, num_classes=80)
        assert len(result) >= 1
        assert int(result[0, 5]) == 0

    def test_pre_nms_conf_filters_low_scores(self):
        """Pre-NMS confidence filter removes low-scoring detections before NMS."""
        head = np.zeros((1, 4, 4, 144), dtype=np.float32)
        # all class logits at 0 -> sigmoid = 0.5, just above default 0.25 threshold
        # with high threshold, all should be filtered
        result = DetectorMixin._postprocess_yolo_raw([head], input_size=640, num_classes=80, pre_nms_conf=0.9)
        assert len(result) == 0

    def test_all_heads_skipped_raises(self):
        """Raises ValueError when all output heads have wrong channel count."""
        bad_head = np.zeros((1, 4, 4, 100), dtype=np.float32)  # 100 != 144
        with pytest.raises(ValueError, match="channel mismatch"):
            DetectorMixin._postprocess_yolo_raw([bad_head], input_size=640, num_classes=80)


class TestAxclBackend:
    @pytest.mark.asyncio
    async def test_detect_axcl_returns_objects(self, sample_vision_config):
        sample_vision_config["backend"] = "axcl"
        detector = FakeDetector(sample_vision_config)

        mock_session = MagicMock()
        mock_session.run.return_value = [np.zeros((1, 2, 2, 144), dtype=np.float32)]
        detector._detector_model = mock_session

        # mock postprocess to return a known detection
        post_result = np.array([[320, 200, 500, 450, 0.9, 0]], dtype=np.float32)
        with patch.object(DetectorMixin, "_postprocess_yolo_raw", return_value=post_result):
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
        mock_session.run.return_value = [np.zeros((1, 2, 2, 144), dtype=np.float32)]
        detector._detector_model = mock_session

        post_result = np.array(
            [
                [320, 200, 500, 450, 0.9, 0],  # person, high conf
                [100, 100, 200, 200, 0.3, 2],  # car, low conf
            ],
            dtype=np.float32,
        )
        with patch.object(DetectorMixin, "_postprocess_yolo_raw", return_value=post_result):
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
        mock_session.run.return_value = [np.zeros((1, 2, 2, 144), dtype=np.float32)]
        detector._detector_model = mock_session

        with patch.object(DetectorMixin, "_postprocess_yolo_raw", return_value=np.empty((0, 6), dtype=np.float32)):
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
        mock_session.run.return_value = [np.zeros((1, 2, 2, 144), dtype=np.float32)]
        detector._detector_model = mock_session

        with patch.object(DetectorMixin, "_postprocess_yolo_raw", return_value=np.empty((0, 6), dtype=np.float32)):
            event = MotionEvent("cam1", "Test", "ev1", _make_tiny_jpeg_b64(), "2026-01-01T00:00:00", "test")
            result = await detector.detect_objects(event)

        assert len(result.objects) == 0


class TestPostprocessYolo26:
    """Tests for YOLO26 direct regression (84-channel) postprocessing."""

    def test_single_detection_direct_regression(self):
        """84-channel head (4 direct bbox + 80 classes) should decode without DFL."""
        # 2x2 grid, 84 channels (4 bbox + 80 classes), stride = 640/2 = 320
        head = np.zeros((1, 2, 2, 84), dtype=np.float32)
        # place a detection at grid cell (0, 0): set class 0 (person) logit high
        head[0, 0, 0, 4] = 10.0  # sigmoid(10) ≈ 1.0 for class 0
        # direct bbox: [left, top, right, bottom] distances from cell center
        head[0, 0, 0, 0] = 0.5  # left
        head[0, 0, 0, 1] = 0.5  # top
        head[0, 0, 0, 2] = 0.5  # right
        head[0, 0, 0, 3] = 0.5  # bottom

        result = DetectorMixin._postprocess_yolo_raw([head], input_size=640, num_classes=80)
        assert result.shape[1] == 6
        assert len(result) >= 1
        assert int(result[0, 5]) == 0  # class 0 (person)
        assert result[0, 4] > 0.9  # high confidence

    def test_empty_output_direct(self):
        """All-zero 84-channel feature maps should produce no confident detections."""
        head = np.zeros((1, 2, 2, 84), dtype=np.float32)
        result = DetectorMixin._postprocess_yolo_raw([head], input_size=640, num_classes=80)
        if len(result) > 0:
            assert result[:, 4].max() <= 0.5

    def test_chw_layout_direct(self):
        """CHW layout with 84 channels should be handled correctly."""
        head_hwc = np.zeros((1, 4, 4, 84), dtype=np.float32)
        head_hwc[0, 0, 0, 4] = 10.0  # class 0 logit
        head_hwc[0, 0, 0, 0:4] = 0.5  # bbox

        head_chw = np.transpose(head_hwc, (0, 3, 1, 2))
        assert head_chw.shape == (1, 84, 4, 4)

        result = DetectorMixin._postprocess_yolo_raw([head_chw], input_size=640, num_classes=80)
        assert len(result) >= 1
        assert int(result[0, 5]) == 0

    def test_multi_head_direct(self):
        """Multiple 84-channel heads are concatenated before NMS."""
        head1 = np.zeros((1, 4, 4, 84), dtype=np.float32)
        head2 = np.zeros((1, 2, 2, 84), dtype=np.float32)
        head1[0, 0, 0, 4] = 10.0  # class 0 (person)
        head1[0, 0, 0, 0:4] = 0.5
        head2[0, 1, 1, 6] = 10.0  # class 2 (car)
        head2[0, 1, 1, 0:4] = 0.5

        result = DetectorMixin._postprocess_yolo_raw([head1, head2], input_size=640, num_classes=80)
        classes = set(int(r[5]) for r in result if r[4] > 0.9)
        assert 0 in classes  # person from head1
        assert 2 in classes  # car from head2

    def test_mixed_format_heads(self):
        """A mix of 144-channel (DFL) and 84-channel (direct) heads should both decode."""
        head_dfl = np.zeros((1, 4, 4, 144), dtype=np.float32)
        head_direct = np.zeros((1, 2, 2, 84), dtype=np.float32)
        # DFL head: class 0 (person), DFL bin 8
        head_dfl[0, 0, 0, 64] = 10.0
        for c in range(4):
            head_dfl[0, 0, 0, c * 16 + 8] = 10.0
        # Direct head: class 2 (car)
        head_direct[0, 1, 1, 6] = 10.0
        head_direct[0, 1, 1, 0:4] = 0.5

        result = DetectorMixin._postprocess_yolo_raw([head_dfl, head_direct], input_size=640, num_classes=80)
        classes = set(int(r[5]) for r in result if r[4] > 0.9)
        assert 0 in classes  # person from DFL head
        assert 2 in classes  # car from direct head

    def test_split_bbox_class_heads(self):
        """Separate (H,W,4) bbox and (H,W,80) class tensors are merged and decoded.

        This matches the YOLO26 axmodel output format: 6 tensors (3 scales x 2).
        """
        # 3 scales: 8x8, 4x4, 2x2 — each with separate bbox and class tensors
        bbox_8 = np.zeros((1, 8, 8, 4), dtype=np.float32)
        cls_8 = np.full((1, 8, 8, 80), -20.0, dtype=np.float32)
        bbox_4 = np.zeros((1, 4, 4, 4), dtype=np.float32)
        cls_4 = np.full((1, 4, 4, 80), -20.0, dtype=np.float32)
        bbox_2 = np.zeros((1, 2, 2, 4), dtype=np.float32)
        cls_2 = np.full((1, 2, 2, 80), -20.0, dtype=np.float32)

        # person detection at 8x8 scale, cell (1,1)
        bbox_8[0, 1, 1, :] = [0.5, 0.5, 0.5, 0.5]
        cls_8[0, 1, 1, 0] = 10.0  # class 0 (person)

        # car detection at 2x2 scale, cell (0,0)
        bbox_2[0, 0, 0, :] = [0.5, 0.5, 0.5, 0.5]
        cls_2[0, 0, 0, 2] = 10.0  # class 2 (car)

        outputs = [bbox_8, cls_8, bbox_4, cls_4, bbox_2, cls_2]
        result = DetectorMixin._postprocess_yolo_raw(outputs, input_size=640, num_classes=80)
        classes = set(int(r[5]) for r in result if r[4] > 0.9)
        assert 0 in classes  # person
        assert 2 in classes  # car

    def test_split_heads_empty(self):
        """Split bbox/class heads with no confident detections return empty."""
        bbox = np.zeros((1, 4, 4, 4), dtype=np.float32)
        cls = np.zeros((1, 4, 4, 80), dtype=np.float32)  # all logits=0 → sigmoid=0.5

        result = DetectorMixin._postprocess_yolo_raw([bbox, cls], input_size=640, num_classes=80)
        if len(result) > 0:
            assert result[:, 4].max() <= 0.5

    def test_all_84ch_heads_skipped_raises(self):
        """Raises ValueError when all output heads have wrong channel count."""
        bad_head = np.zeros((1, 4, 4, 100), dtype=np.float32)  # 100 != 84 or 144
        with pytest.raises(ValueError, match="channel mismatch"):
            DetectorMixin._postprocess_yolo_raw([bad_head], input_size=640, num_classes=80)
