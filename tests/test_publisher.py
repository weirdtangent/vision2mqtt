# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch

from vision2mqtt.mixins.composites import CompositesMixin
from vision2mqtt.mixins.publish import PublishMixin
from vision2mqtt.models.events import DetectedObject, MotionEvent, VisionResult


class FakePublisher(CompositesMixin, PublishMixin):
    def __init__(self, vision_config, ha_enabled=False):
        self.vision_config = vision_config
        self.service = "vision2mqtt"
        self.service_name = "vision2mqtt service"
        self.qos = 0
        self.ha_enabled = ha_enabled
        self.seen_cameras: set[str] = set()
        self._camera_discovery_lock = asyncio.Lock()
        self.config = {"version": "v0.1.0-test"}
        self.mqtt_config = {"discovery_prefix": "homeassistant"}
        self.logger = MagicMock()
        self.mqtt_helper = MagicMock()
        self.mqtt_helper.safe_publish = MagicMock()
        self.mqtt_helper.service_slug = "vision2mqtt"
        self.mqtt_helper.svc_unique_id = MagicMock(side_effect=lambda e: f"vision2mqtt_{e}")
        self.mqtt_helper.dev_unique_id = MagicMock(side_effect=lambda d, e: f"vision2mqtt_{d}_{e}")
        self.mqtt_helper.device_slug = MagicMock(side_effect=lambda d: f"vision2mqtt_{d}")
        self.mqtt_helper.stat_t = MagicMock(side_effect=lambda *args: "/".join(["vision2mqtt"] + [str(a) for a in args if a != "service"]))
        self.mqtt_helper.avty_t = MagicMock(return_value="vision2mqtt/availability")


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

    @pytest.mark.asyncio
    async def test_parent_presence_activated_by_specific_label(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo11n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "car", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": True,
            "debug_save_images": False,
            "composites": [],
        }
        pub = FakePublisher(config)

        event = MotionEvent("cam1", "Front Yard", "ev1", "", "2026-02-14T15:30:45", "test")
        result = VisionResult(
            objects=[
                DetectedObject(label="car", raw_label="car", confidence=0.85, bbox=[0.5, 0.1, 0.9, 0.5]),
            ],
            processing_time_ms=5.0,
        )

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event, result)

        presence_states = {}
        for c in pub.mqtt_helper.safe_publish.call_args_list:
            topic = c.args[0]
            if "/presence/" in topic:
                label = topic.rsplit("/", 1)[-1]
                presence_states[label] = c.args[1]

        assert presence_states["car"] == "ON"
        assert presence_states["vehicle"] == "ON"
        assert presence_states["person"] == "OFF"

    @pytest.mark.asyncio
    async def test_composite_presence_published(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo11n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": True,
            "debug_save_images": False,
            "composites": ["group"],
        }
        pub = FakePublisher(config)

        event = MotionEvent("cam1", "Front Yard", "ev1", "", "2026-02-14T15:30:45", "test")
        result = VisionResult(
            objects=[
                DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.1, 0.1, 0.3, 0.8]),
                DetectedObject(label="person", raw_label="person", confidence=0.75, bbox=[0.4, 0.1, 0.6, 0.8]),
            ],
            all_detections=[
                DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.1, 0.1, 0.3, 0.8]),
                DetectedObject(label="person", raw_label="person", confidence=0.75, bbox=[0.4, 0.1, 0.6, 0.8]),
            ],
            processing_time_ms=5.0,
        )

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event, result)

        presence_states = {}
        for c in pub.mqtt_helper.safe_publish.call_args_list:
            topic = c.args[0]
            if "/presence/" in topic:
                label = topic.rsplit("/", 1)[-1]
                presence_states[label] = c.args[1]

        assert presence_states["group"] == "ON"


class TestServiceDiscovery:
    @pytest.mark.asyncio
    async def test_publishes_when_ha_enabled(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_service_discovery()

        pub.mqtt_helper.safe_publish.assert_called_once()
        topic = pub.mqtt_helper.safe_publish.call_args.args[0]
        payload = json.loads(pub.mqtt_helper.safe_publish.call_args.args[1])
        assert topic == "homeassistant/device/vision2mqtt_service/config"
        assert payload["device"]["name"] == "vision2mqtt service"
        assert "cmps" in payload
        assert "server" in payload["cmps"]
        assert payload["cmps"]["server"]["p"] == "binary_sensor"
        assert payload["cmps"]["server"]["device_class"] == "connectivity"

    @pytest.mark.asyncio
    async def test_skips_when_ha_disabled(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=False)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_service_discovery()

        pub.mqtt_helper.safe_publish.assert_not_called()


class TestCameraDiscovery:
    @pytest.mark.asyncio
    async def test_triggered_on_first_detection(self, sample_vision_config):
        sample_vision_config["retain_presence"] = True
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        event = MotionEvent("cam1", "Front Yard", "ev1", "", "2026-02-14T15:30:45", "test")
        result = VisionResult(
            objects=[DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.1, 0.2, 0.3, 0.4])],
            processing_time_ms=5.0,
        )

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event, result)

        # camera discovery should have been published
        disc_calls = [c for c in pub.mqtt_helper.safe_publish.call_args_list if "homeassistant/device" in c.args[0] and "cam1" in c.args[0]]
        assert len(disc_calls) == 1
        payload = json.loads(disc_calls[0].args[1])
        assert payload["device"]["name"] == "Front Yard Vision"
        assert payload["device"]["via_device"] == "vision2mqtt"
        assert "cam1" in pub.seen_cameras

    @pytest.mark.asyncio
    async def test_not_repeated_on_second_detection(self, sample_vision_config):
        sample_vision_config["retain_presence"] = True
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        event = MotionEvent("cam1", "Front Yard", "ev1", "", "2026-02-14T15:30:45", "test")
        result = VisionResult(
            objects=[DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.1, 0.2, 0.3, 0.4])],
            processing_time_ms=5.0,
        )

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event, result)
            pub.mqtt_helper.safe_publish.reset_mock()

            event2 = MotionEvent("cam1", "Front Yard", "ev2", "", "2026-02-14T15:31:00", "test")
            await pub.publish_vision_result(event2, result)

        # no camera discovery on second call
        disc_calls = [c for c in pub.mqtt_helper.safe_publish.call_args_list if "homeassistant/device" in c.args[0]]
        assert len(disc_calls) == 0

    @pytest.mark.asyncio
    async def test_camera_discovery_has_label_sensors(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("cam1", "Front Yard")

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args.args[1])
        cmps = payload["cmps"]
        # binary sensor per label
        assert "presence_person" in cmps
        assert "presence_vehicle" in cmps
        assert "presence_animal" in cmps
        assert "presence_bird" in cmps
        # sensor entities
        assert "object_count" in cmps
        assert "last_detection" in cmps
        assert "processing_time" in cmps
        # verify types
        assert cmps["presence_person"]["p"] == "binary_sensor"
        assert cmps["object_count"]["p"] == "sensor"
        assert cmps["processing_time"]["unit_of_measurement"] == "ms"

    @pytest.mark.asyncio
    async def test_camera_discovery_uses_distinct_icons(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("cam1", "Front Yard")

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args.args[1])
        cmps = payload["cmps"]
        assert cmps["presence_person"]["icon"] == "mdi:account"
        assert cmps["presence_vehicle"]["icon"] == "mdi:car-side"
        assert cmps["presence_animal"]["icon"] == "mdi:paw"
        assert cmps["presence_bird"]["icon"] == "mdi:bird"

    @pytest.mark.asyncio
    async def test_camera_discovery_vision_suffix(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("cam1", "Front Yard")

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args.args[1])
        assert payload["device"]["name"] == "Front Yard Vision"

    @pytest.mark.asyncio
    async def test_camera_discovery_includes_composites(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo11n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": False,
            "debug_save_images": False,
            "composites": ["dog_walker", "group"],
        }
        pub = FakePublisher(config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("cam1", "Front Yard")

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args.args[1])
        cmps = payload["cmps"]
        assert "presence_dog_walker" in cmps
        assert cmps["presence_dog_walker"]["icon"] == "mdi:dog-service"
        assert "presence_group" in cmps
        assert cmps["presence_group"]["icon"] == "mdi:account-group"


class TestCameraState:
    @pytest.mark.asyncio
    async def test_publishes_sensor_values(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_state("cam1", 3, 12.345)

        topics = [c.args[0] for c in pub.mqtt_helper.safe_publish.call_args_list]
        assert "vision2mqtt/cam1/sensor/object_count" in topics
        assert "vision2mqtt/cam1/sensor/last_detection" in topics
        assert "vision2mqtt/cam1/sensor/processing_time" in topics

        for c in pub.mqtt_helper.safe_publish.call_args_list:
            if c.args[0] == "vision2mqtt/cam1/sensor/object_count":
                assert c.args[1] == "3"
            elif c.args[0] == "vision2mqtt/cam1/sensor/last_detection":
                assert "T" in c.args[1]  # ISO 8601 format
            elif c.args[0] == "vision2mqtt/cam1/sensor/processing_time":
                assert c.args[1] == "12.3"

    @pytest.mark.asyncio
    async def test_skips_when_ha_disabled(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=False)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_state("cam1", 3, 12.345)

        pub.mqtt_helper.safe_publish.assert_not_called()
