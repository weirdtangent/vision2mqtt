# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from vision2mqtt.models.events import MotionEvent, VisionResult

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt


class PublishMixin:
    async def publish_vision_result(self: Vision2Mqtt, event: MotionEvent, result: VisionResult) -> None:
        prefix = self.service

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
            active_labels = set(obj.label for obj in result.objects)
            for label in self.vision_config["labels"]:
                presence_topic = f"{prefix}/{event.camera_id}/presence/{label}"
                state = "ON" if label in active_labels else "OFF"
                await asyncio.to_thread(self.mqtt_helper.safe_publish, presence_topic, state)

        self.logger.info(f"published results for '{event.camera_name}' ({event.event_id}): " f"{len(result.objects)} objects, {result.processing_time_ms}ms")
