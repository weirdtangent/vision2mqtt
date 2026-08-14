# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
"""Tests for clearing/rebuilding HA discovery when the entity layout changes."""

import re
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from vision2mqtt.mixins.helpers import HelpersMixin
from vision2mqtt.mixins.mqtt import MqttMixin
from vision2mqtt.mixins.publish import PublishMixin


class FakeService(HelpersMixin, PublishMixin, MqttMixin):
    def __init__(self, cameras=None):
        self.logger = MagicMock()
        self.ha_enabled = True
        self.mqtt_config = {"discovery_prefix": "homeassistant"}
        self.mqtt_helper = MagicMock()
        self.mqtt_helper.service_slug = "vision2mqtt"
        self.mqtt_helper.obj_id = MagicMock(side_effect=lambda dev, e="": re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", f"{dev} {e}".lower())).strip("_"))
        self.seen_cameras = set(cameras or [])
        self._camera_discovery_lock = asyncio.Lock()
        self.publish_service_discovery = AsyncMock()
        self.publish_service_state = AsyncMock()


def _cleared_topics(svc):
    return [c.args[0] for c in svc.mqtt_helper.safe_publish.call_args_list if c.args[1] == ""]


class TestDiscoveryTopic:
    def test_service_and_camera_topics(self):
        svc = FakeService()

        assert svc.discovery_topic("service") == "homeassistant/device/vision2mqtt_service/config"
        assert svc.discovery_topic("driveway") == "homeassistant/device/vision2mqtt_driveway/config"

    def test_honours_a_custom_discovery_prefix(self):
        svc = FakeService()
        svc.mqtt_config = {"discovery_prefix": "ha"}

        assert svc.discovery_topic("service") == "ha/device/vision2mqtt_service/config"


class TestClearDiscovery:
    @pytest.mark.asyncio
    async def test_delegates_to_the_broker_sweep(self):
        """seen_cameras is empty at connect time, so the topic list must come from the broker."""
        svc = FakeService()
        svc.clear_retained_discovery = AsyncMock()

        await svc.clear_discovery()

        svc.clear_retained_discovery.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_clears_camera_topics_no_camera_has_reported_yet(self):
        svc = FakeService()  # no camera has sent an event, exactly as at mqtt_on_connect
        svc.collect_retained_discovery_topics = AsyncMock(
            return_value=[
                "homeassistant/device/vision2mqtt_driveway/config",
                "homeassistant/device/vision2mqtt_service/config",
            ]
        )

        await svc.clear_discovery()

        assert _cleared_topics(svc) == [
            "homeassistant/device/vision2mqtt_driveway/config",
            "homeassistant/device/vision2mqtt_service/config",
        ]

    @pytest.mark.asyncio
    async def test_clears_with_empty_payload_retained(self):
        """An empty payload removes the registry entry; None would publish the string "null"."""
        svc = FakeService()
        svc.collect_retained_discovery_topics = AsyncMock(return_value=["homeassistant/device/vision2mqtt_service/config"])

        await svc.clear_discovery()

        for c in svc.mqtt_helper.safe_publish.call_args_list:
            assert c.args[1] == ""
            assert c.kwargs == {"retain": True}

    @pytest.mark.asyncio
    async def test_forgets_seen_cameras_so_they_re_announce(self):
        svc = FakeService(cameras=["driveway", "porch"])
        svc.clear_retained_discovery = AsyncMock()

        await svc.clear_discovery()

        assert svc.seen_cameras == set()

    @pytest.mark.asyncio
    async def test_releases_the_camera_discovery_lock(self):
        """publish_vision_result takes the same lock; holding it would wedge every worker."""
        svc = FakeService(cameras=["driveway"])
        svc.clear_retained_discovery = AsyncMock()

        await svc.clear_discovery()

        assert not svc._camera_discovery_lock.locked()


class TestRediscoverAll:
    @pytest.mark.asyncio
    async def test_republishes_the_service_device(self):
        svc = FakeService()

        await svc.rediscover_all()

        svc.publish_service_discovery.assert_awaited_once()
        svc.publish_service_state.assert_awaited_once()


class TestSchemaVersion:
    def test_service_declares_a_schema_version(self):
        assert MqttMixin.DISCOVERY_SCHEMA_VERSION >= 1

    def test_version_topic_does_not_collide_with_the_vision_request_topics(self):
        svc = FakeService()
        svc.vision_config = {"subscribe_topics": ["+/vision/request"]}

        topic = svc.discovery_schema_version_topic()

        assert topic == "vision2mqtt/service/discovery_schema_version"
        assert topic not in svc.mqtt_subscription_topics()
