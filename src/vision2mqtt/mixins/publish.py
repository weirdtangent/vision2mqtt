# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from vision2mqtt.mixins.labels import LABEL_ICONS, LABEL_PARENTS
from vision2mqtt.models.events import MotionEvent, VisionResult

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt


class PublishMixin:
    async def publish_service_discovery(self: Vision2Mqtt) -> None:
        if not self.ha_enabled:
            return

        device_id = "service"
        device = {
            "stat_t": self.mqtt_helper.stat_t(device_id, "service"),
            "avty_t": self.mqtt_helper.avty_t(device_id),
            "device": {
                "name": self.service_name,
                "identifiers": [self.mqtt_helper.service_slug],
                "manufacturer": "weirdTangent",
                "sw_version": self.config["version"],
            },
            "origin": {
                "name": self.service_name,
                "sw_version": self.config["version"],
                "support_url": "https://github.com/weirdTangent/vision2mqtt",
            },
            "qos": self.qos,
            "cmps": {
                "server": {
                    "p": "binary_sensor",
                    "name": self.service_name,
                    "uniq_id": self.mqtt_helper.svc_unique_id("server"),
                    "stat_t": self.mqtt_helper.stat_t(device_id, "service", "server"),
                    "payload_on": "online",
                    "payload_off": "offline",
                    "device_class": "connectivity",
                    "entity_category": "diagnostic",
                    "icon": "mdi:eye",
                },
                **self.build_system_stats_components(),
            },
        }

        discovery_prefix = self.mqtt_config.get("discovery_prefix", "homeassistant")
        topic = f"{discovery_prefix}/device/{self.mqtt_helper.service_slug}_{device_id}/config"
        await asyncio.to_thread(self.mqtt_helper.safe_publish, topic, json.dumps(device))
        self.logger.debug(f"published HA service discovery to {topic}")

    async def publish_service_availability(self: Vision2Mqtt, status: str = "online") -> None:
        if not self.ha_enabled:
            return
        await asyncio.to_thread(self.mqtt_helper.safe_publish, self.mqtt_helper.avty_t("service"), status)

    async def publish_service_state(self: Vision2Mqtt) -> None:
        if not self.ha_enabled:
            return
        await asyncio.to_thread(
            self.mqtt_helper.safe_publish,
            self.mqtt_helper.stat_t("service", "service", "server"),
            "online",
        )

    async def publish_camera_discovery(self: Vision2Mqtt, camera_id: str, camera_name: str) -> None:
        if not self.ha_enabled:
            return

        prefix = self.service
        labels = self.vision_config["labels"]
        composites = self.vision_config.get("composites") or []

        # binary sensors for each configured label
        cmps: dict[str, dict] = {}
        for label in labels:
            cmps[f"presence_{label}"] = {
                "p": "binary_sensor",
                "name": f"{label.title()} detected",
                "uniq_id": self.mqtt_helper.dev_unique_id(camera_id, f"presence_{label}"),
                "stat_t": f"{prefix}/{camera_id}/presence/{label}",
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "occupancy" if label == "person" else "motion",
                "icon": LABEL_ICONS.get(label, "mdi:motion-sensor"),
            }

        # binary sensors for each composite type
        for comp in composites:
            cmps[f"presence_{comp}"] = {
                "p": "binary_sensor",
                "name": f"{comp.replace('_', ' ').title()} detected",
                "uniq_id": self.mqtt_helper.dev_unique_id(camera_id, f"presence_{comp}"),
                "stat_t": f"{prefix}/{camera_id}/presence/{comp}",
                "payload_on": "ON",
                "payload_off": "OFF",
                "device_class": "motion",
                "icon": LABEL_ICONS.get(comp, "mdi:motion-sensor"),
            }

        # sensor: object count
        cmps["object_count"] = {
            "p": "sensor",
            "name": "Object count",
            "uniq_id": self.mqtt_helper.dev_unique_id(camera_id, "object_count"),
            "stat_t": f"{prefix}/{camera_id}/sensor/object_count",
            "icon": "mdi:counter",
        }

        # sensor: last detection timestamp
        cmps["last_detection"] = {
            "p": "sensor",
            "name": "Last detection",
            "uniq_id": self.mqtt_helper.dev_unique_id(camera_id, "last_detection"),
            "stat_t": f"{prefix}/{camera_id}/sensor/last_detection",
            "device_class": "timestamp",
            "icon": "mdi:clock-outline",
        }

        # sensor: processing time
        cmps["processing_time"] = {
            "p": "sensor",
            "name": "Processing time",
            "uniq_id": self.mqtt_helper.dev_unique_id(camera_id, "processing_time"),
            "stat_t": f"{prefix}/{camera_id}/sensor/processing_time",
            "unit_of_measurement": "ms",
            "icon": "mdi:timer-outline",
        }

        # camera: annotated snapshot with bounding boxes
        cmps["annotated_image"] = {
            "p": "camera",
            "name": "Annotated",
            "uniq_id": self.mqtt_helper.dev_unique_id(camera_id, "annotated_image"),
            "t": f"{prefix}/{camera_id}/image/annotated",
            "image_encoding": "b64",
        }

        device = {
            "stat_t": self.mqtt_helper.stat_t(camera_id, "state"),
            "avty_t": self.mqtt_helper.avty_t("service"),
            "device": {
                "name": f"{camera_name} Vision",
                "identifiers": [self.mqtt_helper.device_slug(camera_id)],
                "via_device": self.mqtt_helper.service_slug,
            },
            "origin": {
                "name": self.service_name,
                "sw_version": self.config["version"],
                "support_url": "https://github.com/weirdTangent/vision2mqtt",
            },
            "qos": self.qos,
            "cmps": cmps,
        }

        discovery_prefix = self.mqtt_config.get("discovery_prefix", "homeassistant")
        topic = f"{discovery_prefix}/device/{self.mqtt_helper.service_slug}_{camera_id}/config"
        await asyncio.to_thread(self.mqtt_helper.safe_publish, topic, json.dumps(device))
        self.logger.info(f"published HA camera discovery for '{camera_name} Vision' ({camera_id})")

    async def publish_camera_state(self: Vision2Mqtt, camera_id: str, object_count: int, processing_time_ms: float) -> None:
        if not self.ha_enabled:
            return

        prefix = self.service
        now_iso = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(self.mqtt_helper.safe_publish, f"{prefix}/{camera_id}/sensor/object_count", str(object_count))
        await asyncio.to_thread(self.mqtt_helper.safe_publish, f"{prefix}/{camera_id}/sensor/last_detection", now_iso)
        await asyncio.to_thread(self.mqtt_helper.safe_publish, f"{prefix}/{camera_id}/sensor/processing_time", str(round(processing_time_ms, 1)))

    async def publish_vision_result(self: Vision2Mqtt, event: MotionEvent, result: VisionResult) -> None:
        prefix = self.service

        # lazy camera discovery on first detection (lock guards concurrent workers)
        if self.ha_enabled and event.camera_id not in self.seen_cameras:
            async with self._camera_discovery_lock:
                if event.camera_id not in self.seen_cameras:
                    try:
                        await self.publish_camera_discovery(event.camera_id, event.camera_name)
                    except Exception:
                        self.logger.exception("Failed to publish camera discovery for '%s'", event.camera_id)
                    else:
                        self.seen_cameras.add(event.camera_id)

        # publish objects list
        objects_topic = f"{prefix}/{event.camera_id}/{event.event_id}/objects"
        objects_payload = [
            {
                "label": obj.label,
                "raw_label": obj.raw_label,
                "confidence": obj.confidence,
                "bbox": obj.bbox,
            }
            for obj in result.objects
        ]
        await asyncio.to_thread(self.mqtt_helper.safe_publish, objects_topic, json.dumps(objects_payload))

        # publish summary
        label_counts: dict[str, int] = {}
        for obj in result.objects:
            label_counts[obj.label] = label_counts.get(obj.label, 0) + 1

        summary_topic = f"{prefix}/{event.camera_id}/{event.event_id}/summary"
        summary_payload = {
            "camera_id": event.camera_id,
            "camera_name": event.camera_name,
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "labels": label_counts,
            "object_count": len(result.objects),
            "processing_time_ms": result.processing_time_ms,
            "source": event.source,
        }
        await asyncio.to_thread(self.mqtt_helper.safe_publish, summary_topic, json.dumps(summary_payload))

        # publish per-label presence (optional, retained)
        if self.vision_config.get("retain_presence"):
            # Collect active specific labels and their parent groups
            active_labels: set[str] = set()
            for obj in result.objects:
                active_labels.add(obj.label)
                parent = LABEL_PARENTS.get(obj.label)
                if parent:
                    active_labels.add(parent)

            for label in self.vision_config["labels"]:
                presence_topic = f"{prefix}/{event.camera_id}/presence/{label}"
                state = "ON" if label in active_labels else "OFF"
                await asyncio.to_thread(self.mqtt_helper.safe_publish, presence_topic, state)

            # publish composite presence
            composites = self.compute_composites(result)
            for comp_name, comp_state in composites.items():
                presence_topic = f"{prefix}/{event.camera_id}/presence/{comp_name}"
                state = "ON" if comp_state else "OFF"
                await asyncio.to_thread(self.mqtt_helper.safe_publish, presence_topic, state)

        # publish annotated snapshot image
        if result.annotated_image_b64:
            image_topic = f"{prefix}/{event.camera_id}/image/annotated"
            await asyncio.to_thread(self.mqtt_helper.safe_publish, image_topic, result.annotated_image_b64)

        # publish camera sensor state for HA
        await self.publish_camera_state(event.camera_id, len(result.objects), result.processing_time_ms)

        self.logger.info(f"published results for '{event.camera_name}' ({event.event_id}): " f"{len(result.objects)} objects, {result.processing_time_ms}ms")
