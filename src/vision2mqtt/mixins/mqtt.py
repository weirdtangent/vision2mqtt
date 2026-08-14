# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mqtt_helper import BaseMqttMixin
from paho.mqtt.client import Client, MQTTMessage

from vision2mqtt.models.events import MotionEvent

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt


class MqttMixin(BaseMqttMixin):
    # Bump ONLY when the entity layout changes: unique_ids, entity names, or the set of components
    # published. A bump clears retained discovery on next connect, which drops the entities from
    # HA's registry -- losing renames, areas, and hidden/disabled flags -- before recreating them.
    #
    # There is deliberately no reset_discovery command here: this service subscribes only to vision
    # request topics and has no command path, so the version gate is the whole mechanism.
    #
    # 1: baseline (2026-08) -- first version to carry a schema stamp
    DISCOVERY_SCHEMA_VERSION = 1

    def mqtt_subscription_topics(self: Vision2Mqtt) -> list[str]:
        return list(self.vision_config["subscribe_topics"])

    async def mqtt_on_message(self: Vision2Mqtt, client: Client, userdata: Any, msg: MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
            self.logger.warning(f"failed to decode MQTT payload on {msg.topic}")
            return

        # validate required fields
        required = ("camera_id", "camera_name", "event_id", "image_b64", "timestamp", "source")
        if not all(k in payload for k in required):
            self.logger.warning(f"vision request missing required fields on {msg.topic}: {list(payload.keys())}")
            return

        # reject non-string or oversized image payloads (~10MB decoded limit)
        max_image_b64_len = 15_000_000
        if not isinstance(payload["image_b64"], str) or len(payload["image_b64"]) > max_image_b64_len:
            self.logger.warning(f"invalid or oversized image_b64 on {msg.topic}, dropping")
            return

        event = MotionEvent(
            camera_id=payload["camera_id"],
            camera_name=payload["camera_name"],
            event_id=payload["event_id"],
            image_b64=payload["image_b64"],
            timestamp=payload["timestamp"],
            source=payload["source"],
        )

        # put on queue, drop oldest if full
        if self.queue.full():
            try:
                dropped = self.queue.get_nowait()
                self.logger.warning(f"queue full, dropped oldest event: {dropped.camera_name}/{dropped.event_id}")
            except Exception:
                pass

        await self.queue.put(event)
        self.logger.debug(f"queued vision request from '{event.camera_name}' ({event.source}) - queue size: {self.queue.qsize()}")
