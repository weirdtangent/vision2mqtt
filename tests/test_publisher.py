# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch

from vision2mqtt.mixins.publish import PublishMixin
from vision2mqtt.models.events import DetectedObject, MotionEvent, VisionResult


class FakePublisher(PublishMixin):
    def __init__(self, vision_config):
        self.vision_config = vision_config
        self.service = "vision2mqtt"
        self.logger = MagicMock()
        self.mqtt_helper = MagicMock()
        self.mqtt_helper.safe_publish = MagicMock()


async def _fake_to_thread(fn, *args):
    """Replace asyncio.to_thread with synchronous call for testing."""
    return fn(*args)


class TestPublishVisionResult:
    @pytest.mark.asyncio
    async def test_publishes_objects_and_summary(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config)

        event = MotionEvent("cam1", "Front Yard", "ev1", "", "2026-02-14T15:30:45", "recording_snapshot")
        result = VisionResult(
            objects=[
                DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.12, 0.34, 0.45, 0.89]),
            ],
            processing_time_ms=8.2,
        )

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event, result)

        topics = [c.args[0] for c in pub.mqtt_helper.safe_publish.call_args_list]
        assert "vision2mqtt/cam1/ev1/objects" in topics
        assert "vision2mqtt/cam1/ev1/summary" in topics

    @pytest.mark.asyncio
    async def test_summary_contains_label_counts(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config)

        event = MotionEvent("cam1", "Front Yard", "ev1", "", "2026-02-14T15:30:45", "test")
        result = VisionResult(
            objects=[
                DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.1, 0.2, 0.3, 0.4]),
                DetectedObject(label="vehicle", raw_label="car", confidence=0.72, bbox=[0.5, 0.1, 0.9, 0.5]),
                DetectedObject(label="person", raw_label="person", confidence=0.65, bbox=[0.2, 0.3, 0.4, 0.5]),
            ],
            processing_time_ms=12.0,
        )

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event, result)

        for c in pub.mqtt_helper.safe_publish.call_args_list:
            if "summary" in c.args[0]:
                summary = json.loads(c.args[1])
                assert summary["labels"] == {"person": 2, "vehicle": 1}
                assert summary["object_count"] == 3
                assert summary["processing_time_ms"] == 12.0
                break
        else:
            pytest.fail("summary topic not published")

    @pytest.mark.asyncio
    async def test_presence_published_when_enabled(self, sample_vision_config):
        sample_vision_config["retain_presence"] = True
        pub = FakePublisher(sample_vision_config)

        event = MotionEvent("cam1", "Front Yard", "ev1", "", "2026-02-14T15:30:45", "test")
        result = VisionResult(
            objects=[
                DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.1, 0.2, 0.3, 0.4]),
            ],
            processing_time_ms=5.0,
        )

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event, result)

        topics = [c.args[0] for c in pub.mqtt_helper.safe_publish.call_args_list]
        assert "vision2mqtt/cam1/presence/person" in topics
        assert "vision2mqtt/cam1/presence/vehicle" in topics
        assert "vision2mqtt/cam1/presence/animal" in topics
        assert "vision2mqtt/cam1/presence/bird" in topics

        for c in pub.mqtt_helper.safe_publish.call_args_list:
            if c.args[0] == "vision2mqtt/cam1/presence/person":
                assert c.args[1] == "ON"
            elif "presence" in c.args[0]:
                assert c.args[1] == "OFF"
