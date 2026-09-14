# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from vision2mqtt.mixins.mqtt import MqttMixin
from vision2mqtt.models.events import MotionEvent


class FakeMqtt(MqttMixin):
    def __init__(self, vision_config, max_queue=20):
        self.vision_config = vision_config
        self.logger = MagicMock()
        self.queue = asyncio.Queue(maxsize=max_queue)
        self.mqtt_helper = MagicMock()
        self.mqtt_helper.service_slug = "vision2mqtt"
        self.handle_service_command = AsyncMock()


def _make_msg(topic, payload):
    """Create a fake MQTTMessage with the given topic and payload."""
    msg = MagicMock()
    msg.topic = topic
    if isinstance(payload, dict):
        msg.payload = json.dumps(payload).encode("utf-8")
    elif isinstance(payload, str):
        msg.payload = payload.encode("utf-8")
    else:
        msg.payload = payload
    return msg


VALID_PAYLOAD = {
    "camera_id": "ABC123",
    "camera_name": "Front Yard",
    "event_id": "20260214-153045",
    "image_b64": "dGVzdA==",
    "timestamp": "2026-02-14T15:30:45",
    "source": "recording_snapshot",
}


COMMAND_TOPIC = "vision2mqtt/service/+/command"


class TestMqttSubscriptionTopics:
    def test_returns_configured_topics_plus_the_command_topic(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        topics = mqtt.mqtt_subscription_topics()
        assert topics == ["+/vision/request", COMMAND_TOPIC]

    def test_returns_multiple_topics(self, sample_vision_config):
        sample_vision_config["subscribe_topics"] = ["amcrest2mqtt/vision/request", "blink2mqtt/vision/request"]
        mqtt = FakeMqtt(sample_vision_config)
        topics = mqtt.mqtt_subscription_topics()
        assert len(topics) == 3

    def test_command_topic_is_added_even_with_no_configured_topics(self, sample_vision_config):
        """The reset button must work regardless of subscribe_topics."""
        sample_vision_config["subscribe_topics"] = []
        mqtt = FakeMqtt(sample_vision_config)
        assert mqtt.mqtt_subscription_topics() == [COMMAND_TOPIC]


class TestServiceCommandRouting:
    @pytest.mark.asyncio
    async def test_service_command_is_routed_not_parsed_as_json(self, sample_vision_config):
        """A button press is a plain string, not vision-request JSON. Routing it before the
        json.loads() keeps every press from logging a decode warning."""
        mqtt = FakeMqtt(sample_vision_config)
        msg = _make_msg("vision2mqtt/service/reset_discovery/command", "PRESS")

        await mqtt.mqtt_on_message(None, None, msg)

        mqtt.handle_service_command.assert_awaited_once()
        assert mqtt.handle_service_command.await_args.args[0] == "reset_discovery"
        assert mqtt.queue.qsize() == 0
        mqtt.logger.warning.assert_not_called()

    @pytest.mark.asyncio
    async def test_vision_request_still_reaches_the_queue(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        await mqtt.mqtt_on_message(None, None, _make_msg("amcrest2mqtt/vision/request", VALID_PAYLOAD))
        assert mqtt.queue.qsize() == 1
        mqtt.handle_service_command.assert_not_awaited()


class TestMqttOnMessage:
    @pytest.mark.asyncio
    async def test_valid_payload_queued(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        msg = _make_msg("amcrest2mqtt/vision/request", VALID_PAYLOAD)

        await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 1
        event = mqtt.queue.get_nowait()
        assert isinstance(event, MotionEvent)
        assert event.camera_id == "ABC123"
        assert event.camera_name == "Front Yard"
        assert event.event_id == "20260214-153045"
        assert event.image_b64 == "dGVzdA=="
        assert event.timestamp == "2026-02-14T15:30:45"
        assert event.source == "recording_snapshot"

    @pytest.mark.asyncio
    async def test_invalid_json_rejected(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        msg = _make_msg("test/topic", "not json {{{")

        await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 0
        mqtt.logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_fields_rejected(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        incomplete = {"camera_id": "ABC123", "camera_name": "Front Yard"}
        msg = _make_msg("test/topic", incomplete)

        await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 0
        mqtt.logger.warning.assert_called_once()
        assert "missing required fields" in mqtt.logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_missing_single_field_rejected(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        # all fields except "source"
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "source"}
        msg = _make_msg("test/topic", payload)

        await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_empty_json_object_rejected(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        msg = _make_msg("test/topic", {})

        await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_binary_payload_rejected(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        msg = MagicMock()
        msg.topic = "test/topic"
        msg.payload = b"\x00\x01\x02\xff"

        await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_multiple_messages_queued(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)

        for i in range(3):
            payload = {**VALID_PAYLOAD, "event_id": f"event-{i}"}
            msg = _make_msg("test/topic", payload)
            await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 3


class TestMqttPayloadValidation:
    @pytest.mark.asyncio
    async def test_oversized_image_rejected(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        payload = {**VALID_PAYLOAD, "image_b64": "A" * 16_000_000}
        msg = _make_msg("test/topic", payload)

        await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 0
        mqtt.logger.warning.assert_called_once()
        assert "invalid or oversized" in mqtt.logger.warning.call_args[0][0]

    @pytest.mark.asyncio
    async def test_non_string_image_rejected(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        payload = {**VALID_PAYLOAD, "image_b64": 12345}
        msg = _make_msg("test/topic", payload)

        await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 0
        mqtt.logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_at_limit_accepted(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config)
        payload = {**VALID_PAYLOAD, "image_b64": "A" * 15_000_000}
        msg = _make_msg("test/topic", payload)

        await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 1


class TestMqttQueueFull:
    @pytest.mark.asyncio
    async def test_drops_oldest_when_full(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config, max_queue=2)

        # fill the queue
        for i in range(2):
            payload = {**VALID_PAYLOAD, "event_id": f"event-{i}", "camera_name": f"cam-{i}"}
            msg = _make_msg("test/topic", payload)
            await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 2

        # third message should drop the oldest
        payload = {**VALID_PAYLOAD, "event_id": "event-2", "camera_name": "cam-2"}
        msg = _make_msg("test/topic", payload)
        await mqtt.mqtt_on_message(None, None, msg)

        assert mqtt.queue.qsize() == 2
        mqtt.logger.warning.assert_called_once()
        assert "queue full" in mqtt.logger.warning.call_args[0][0]

        # verify oldest was dropped: first item should be event-1, not event-0
        first = mqtt.queue.get_nowait()
        assert first.event_id == "event-1"
        second = mqtt.queue.get_nowait()
        assert second.event_id == "event-2"

    @pytest.mark.asyncio
    async def test_queue_size_one(self, sample_vision_config):
        mqtt = FakeMqtt(sample_vision_config, max_queue=1)

        msg1 = _make_msg("test/topic", {**VALID_PAYLOAD, "event_id": "old"})
        await mqtt.mqtt_on_message(None, None, msg1)

        msg2 = _make_msg("test/topic", {**VALID_PAYLOAD, "event_id": "new"})
        await mqtt.mqtt_on_message(None, None, msg2)

        assert mqtt.queue.qsize() == 1
        event = mqtt.queue.get_nowait()
        assert event.event_id == "new"
