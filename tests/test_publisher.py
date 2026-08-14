# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import re
import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch

from vision2mqtt.mixins.composites import CompositesMixin
from vision2mqtt.mixins.helpers import HelpersMixin
from vision2mqtt.mixins.presence import PresenceTracker
from vision2mqtt.mixins.system_stats import SystemStatsMixin
from vision2mqtt.mixins.publish import PublishMixin
from vision2mqtt.models.events import CameraConfig, DetectedObject, MotionEvent, VisionResult


class FakePublisher(CompositesMixin, HelpersMixin, SystemStatsMixin, PublishMixin):
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
        self.mqtt_helper.obj_id = MagicMock(side_effect=lambda dev, e="": re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", f"{dev} {e}".lower())).strip("_"))
        self.mqtt_helper.svc_unique_id = MagicMock(side_effect=lambda e: f"vision2mqtt_{e}")
        self.mqtt_helper.dev_unique_id = MagicMock(side_effect=lambda d, e: f"vision2mqtt_{d}_{e}")
        self.mqtt_helper.device_slug = MagicMock(side_effect=lambda d: f"vision2mqtt_{d}")
        self.mqtt_helper.stat_t = MagicMock(side_effect=lambda *args: "/".join(["vision2mqtt"] + [str(a) for a in args if a != "service"]))
        self.mqtt_helper.avty_t = MagicMock(return_value="vision2mqtt/availability")
        self._presence_tracker = PresenceTracker()
        self._axcl_smi_path = None


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
            "model": "yolo26n.pt",
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
            "model": "yolo26n.pt",
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

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args_list[0].args[1])
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

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args_list[0].args[1])
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

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args_list[0].args[1])
        assert payload["device"]["name"] == "Front Yard Vision"

    @pytest.mark.asyncio
    async def test_camera_discovery_includes_composites(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo26n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": False,
            "debug_save_images": False,
            "composites": ["dog_walker", "group"],
            "cameras": {},
        }
        pub = FakePublisher(config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("cam1", "Front Yard")

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args_list[0].args[1])
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


class TestPresenceCooldown:
    @pytest.mark.asyncio
    async def test_presence_stays_on_within_cooldown(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo26n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": True,
            "debug_save_images": False,
            "composites": [],
            "presence_cooldown": 60,
            "frequency_window": 3600,
        }
        pub = FakePublisher(config)

        # First event: person detected
        event1 = MotionEvent("cam1", "Front Yard", "ev1", "", "2026-02-14T15:30:45", "test")
        result_with_person = VisionResult(
            objects=[DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.1, 0.2, 0.3, 0.4])],
            processing_time_ms=5.0,
        )

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event1, result_with_person)

        # Second event: no person (but cooldown should keep it ON)
        pub.mqtt_helper.safe_publish.reset_mock()
        event2 = MotionEvent("cam1", "Front Yard", "ev2", "", "2026-02-14T15:30:50", "test")
        result_empty = VisionResult(objects=[], processing_time_ms=3.0)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event2, result_empty)

        # Person presence should still be ON due to cooldown
        for c in pub.mqtt_helper.safe_publish.call_args_list:
            if c.args[0] == "vision2mqtt/cam1/presence/person":
                assert c.args[1] == "ON"
                break
        else:
            pytest.fail("person presence topic not published")

    @pytest.mark.asyncio
    async def test_zero_cooldown_gives_instant_off(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo26n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": True,
            "debug_save_images": False,
            "composites": [],
            "presence_cooldown": 0,
            "frequency_window": 3600,
        }
        pub = FakePublisher(config)

        # First event: person detected
        event1 = MotionEvent("cam1", "Front Yard", "ev1", "", "2026-02-14T15:30:45", "test")
        result_with_person = VisionResult(
            objects=[DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.1, 0.2, 0.3, 0.4])],
            processing_time_ms=5.0,
        )

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event1, result_with_person)

        # Second event: no person — should go OFF immediately
        pub.mqtt_helper.safe_publish.reset_mock()
        event2 = MotionEvent("cam1", "Front Yard", "ev2", "", "2026-02-14T15:30:50", "test")
        result_empty = VisionResult(objects=[], processing_time_ms=3.0)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event2, result_empty)

        for c in pub.mqtt_helper.safe_publish.call_args_list:
            if c.args[0] == "vision2mqtt/cam1/presence/person":
                assert c.args[1] == "OFF"
                break
        else:
            pytest.fail("person presence topic not published")


class TestFrequencySensor:
    @pytest.mark.asyncio
    async def test_frequency_published_with_presence(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo26n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": True,
            "debug_save_images": False,
            "composites": [],
            "presence_cooldown": 60,
            "frequency_window": 3600,
        }
        pub = FakePublisher(config)

        event = MotionEvent("cam1", "Front Yard", "ev1", "", "2026-02-14T15:30:45", "test")
        result = VisionResult(
            objects=[DetectedObject(label="person", raw_label="person", confidence=0.87, bbox=[0.1, 0.2, 0.3, 0.4])],
            processing_time_ms=5.0,
        )

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_vision_result(event, result)

        topics = [c.args[0] for c in pub.mqtt_helper.safe_publish.call_args_list]
        assert "vision2mqtt/cam1/sensor/frequency/person" in topics
        assert "vision2mqtt/cam1/sensor/frequency/vehicle" in topics
        assert "vision2mqtt/cam1/sensor/frequency/animal" in topics
        assert "vision2mqtt/cam1/sensor/frequency/bird" in topics

        # Person frequency should be > 0
        for c in pub.mqtt_helper.safe_publish.call_args_list:
            if c.args[0] == "vision2mqtt/cam1/sensor/frequency/person":
                assert float(c.args[1]) > 0
                break


class TestCameraDiscoveryFrequency:
    @pytest.mark.asyncio
    async def test_discovery_includes_frequency_sensors(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("cam1", "Front Yard")

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args_list[0].args[1])
        cmps = payload["cmps"]
        assert "frequency_person" in cmps
        assert "frequency_vehicle" in cmps
        assert "frequency_animal" in cmps
        assert "frequency_bird" in cmps
        assert cmps["frequency_person"]["p"] == "sensor"
        assert cmps["frequency_person"]["unit_of_measurement"] == "detections/h"
        assert cmps["frequency_person"]["state_class"] == "measurement"
        assert cmps["frequency_person"]["stat_t"] == "vision2mqtt/cam1/sensor/frequency/person"

    @pytest.mark.asyncio
    async def test_discovery_includes_composite_frequency(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo26n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": False,
            "debug_save_images": False,
            "composites": ["dog_walker"],
            "presence_cooldown": 60,
            "frequency_window": 3600,
            "cameras": {},
        }
        pub = FakePublisher(config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("cam1", "Front Yard")

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args_list[0].args[1])
        cmps = payload["cmps"]
        assert "frequency_dog_walker" in cmps
        assert cmps["frequency_dog_walker"]["unit_of_measurement"] == "detections/h"


class TestCheckPresenceCooldowns:
    @pytest.mark.asyncio
    async def test_expires_presence_and_publishes_off(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo26n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": True,
            "debug_save_images": False,
            "composites": [],
            "presence_cooldown": 60,
            "frequency_window": 3600,
        }
        pub = FakePublisher(config)
        tracker = pub._presence_tracker

        # Simulate a past detection + published ON state
        tracker.record_detection("cam1", "person", now=100.0)
        tracker.set_published_state("cam1", "person", True)

        # Fast-forward time so cooldown has expired
        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio, patch("time.monotonic", return_value=200.0):
            mock_asyncio.to_thread = _fake_to_thread
            await pub.check_presence_cooldowns()

        # Should have published OFF for person
        off_calls = [c for c in pub.mqtt_helper.safe_publish.call_args_list if c.args[0] == "vision2mqtt/cam1/presence/person"]
        assert len(off_calls) == 1
        assert off_calls[0].args[1] == "OFF"

    @pytest.mark.asyncio
    async def test_skips_when_cooldown_zero(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo26n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": True,
            "debug_save_images": False,
            "composites": [],
            "presence_cooldown": 0,
            "frequency_window": 3600,
        }
        pub = FakePublisher(config)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.check_presence_cooldowns()

        pub.mqtt_helper.safe_publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_retain_presence_disabled(self):
        config = {
            "backend": "ultralytics",
            "model": "yolo26n.pt",
            "subscribe_topics": ["+/vision/request"],
            "labels": ["person", "vehicle", "animal", "bird"],
            "min_confidence": 0.45,
            "concurrency": 1,
            "max_queue": 20,
            "retain_presence": False,
            "debug_save_images": False,
            "composites": [],
            "presence_cooldown": 60,
            "frequency_window": 3600,
        }
        pub = FakePublisher(config)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.check_presence_cooldowns()

        pub.mqtt_helper.safe_publish.assert_not_called()


class TestCameraModeDiscovery:
    @pytest.mark.asyncio
    async def test_discovery_includes_camera_mode_sensor(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("cam1", "Front Yard")

        payload = json.loads(pub.mqtt_helper.safe_publish.call_args_list[0].args[1])
        cmps = payload["cmps"]
        assert "camera_mode" in cmps
        assert cmps["camera_mode"]["p"] == "sensor"
        assert cmps["camera_mode"]["icon"] == "mdi:cog"
        assert cmps["camera_mode"]["entity_category"] == "diagnostic"
        assert cmps["camera_mode"]["stat_t"] == "vision2mqtt/cam1/sensor/camera_mode"

    @pytest.mark.asyncio
    async def test_mode_value_published_on_discovery(self, sample_vision_config):
        sample_vision_config["cameras"] = {
            "FEEDER": CameraConfig(mode="feeder"),
        }
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("FEEDER", "Bird Feeder")

        mode_calls = [c for c in pub.mqtt_helper.safe_publish.call_args_list if c.args[0] == "vision2mqtt/FEEDER/sensor/camera_mode"]
        assert len(mode_calls) == 1
        assert mode_calls[0].args[1] == "feeder"

    @pytest.mark.asyncio
    async def test_unconfigured_camera_publishes_default_mode(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("cam1", "Front Yard")

        mode_calls = [c for c in pub.mqtt_helper.safe_publish.call_args_list if c.args[0] == "vision2mqtt/cam1/sensor/camera_mode"]
        assert len(mode_calls) == 1
        assert mode_calls[0].args[1] == "default"

    @pytest.mark.asyncio
    async def test_camera_state_publishes_mode(self, sample_vision_config):
        sample_vision_config["cameras"] = {
            "FEEDER": CameraConfig(mode="feeder"),
        }
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_state("FEEDER", 2, 10.5)

        mode_calls = [c for c in pub.mqtt_helper.safe_publish.call_args_list if c.args[0] == "vision2mqtt/FEEDER/sensor/camera_mode"]
        assert len(mode_calls) == 1
        assert mode_calls[0].args[1] == "feeder"


class TestStableObjectIds:
    """entity_id must be pinned to the component key, never the display name.

    HA derives entity_id from the display name at first discovery and keeps it forever, keyed on
    unique_id. Confirmed on a live install that clearing discovery and waiting 25s still restores
    the same entity_id, so publishing obj_id at creation is the only point this can be fixed.
    """

    @pytest.mark.asyncio
    async def test_every_service_component_publishes_an_obj_id(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_service_discovery()

        cmps = json.loads(pub.mqtt_helper.safe_publish.call_args.args[1])["cmps"]
        missing = [k for k, c in cmps.items() if "obj_id" not in c]
        assert missing == [], f"components without obj_id: {missing}"

    @pytest.mark.asyncio
    async def test_every_camera_component_publishes_an_obj_id(self, sample_vision_config):
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("driveway", "Driveway Cam")

        cmps = json.loads(pub.mqtt_helper.safe_publish.call_args_list[0].args[1])["cmps"]
        missing = [k for k, c in cmps.items() if "obj_id" not in c]
        assert missing == [], f"components without obj_id: {missing}"

    @pytest.mark.asyncio
    async def test_camera_obj_ids_resolve_against_the_camera_not_the_service(self, sample_vision_config):
        """Camera components are built before their device block, so a naive backward scan
        attributes them to the service device instead."""
        pub = FakePublisher(sample_vision_config, ha_enabled=True)

        with patch("vision2mqtt.mixins.publish.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_camera_discovery("driveway", "Driveway Cam")

        cmps = json.loads(pub.mqtt_helper.safe_publish.call_args_list[0].args[1])["cmps"]
        for key, comp in cmps.items():
            assert "vision2mqtt_service" not in comp["obj_id"], f"{key} resolved against the service device"
            assert comp["obj_id"].startswith("driveway_cam"), f"{key} -> {comp['obj_id']}"
