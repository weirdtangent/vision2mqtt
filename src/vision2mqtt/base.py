# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

import psutil
from json_logging import get_logger
from mqtt_helper import MqttHelper
from paho.mqtt.client import Client

from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt
from vision2mqtt.mixins.presence import PresenceTracker
from vision2mqtt.models.events import MotionEvent


class Base:
    def __init__(self: Vision2Mqtt, args: argparse.Namespace | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.loop = asyncio.get_running_loop()
        self.loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=16))

        self.args = args
        self.logger = get_logger(__name__)

        # now load self.config right away
        cfg_arg = getattr(args, "config", None)
        self.config = self.load_config(cfg_arg)

        if not self.config["mqtt"] or not self.config["vision"]:
            raise ValueError("config was not loaded")

        # down in trenches if we have to
        if self.config.get("debug"):
            self.logger.setLevel(logging.DEBUG)

        self.mqtt_config = self.config["mqtt"]
        self.vision_config = self.config["vision"]

        self.service = self.mqtt_config["prefix"]
        self.service_name = f"{self.service} service"
        self.qos = self.mqtt_config["qos"]

        self.mqtt_helper = MqttHelper(self.service, default_qos=self.qos, default_retain=True)

        self.running = False

        self.mqttc: Client
        self.client_id = self.mqtt_helper.client_id()

        self.ha_enabled: bool = self.config.get("home_assistant", True)
        # camera_id -> the display name we last published discovery with. Keyed on the id so a
        # rename is detected and re-announced; a bare set froze the name at first detection.
        self.seen_cameras: dict[str, str] = {}
        # Frames run through the detector TODAY. Rolls over at LOCAL midnight, matching
        # amcrest2mqtt's api_calls -- a monotonic lifetime total reads as a meaningless large
        # number, whereas "today" is directly useful at a glance. state_class stays
        # total_increasing: HA handles the daily reset and still derives correct long-term
        # statistics from it.
        self.images_annotated: int = 0
        self.images_annotated_date = datetime.now(UTC).astimezone()
        # Serialises increment+publish. With vision.concurrency > 1 the workers each await
        # inside the publish, so without this the executor can deliver "2" before "1" -- and
        # because the topic is RETAINED, the stored value would end up lower than one already
        # published, which breaks the total_increasing contract and corrupts HA statistics.
        self._images_annotated_lock = asyncio.Lock()
        self._camera_discovery_lock = asyncio.Lock()

        self._presence_tracker = PresenceTracker()

        max_queue = self.vision_config.get("max_queue", 20)
        self.queue: asyncio.Queue[MotionEvent] = asyncio.Queue(maxsize=max_queue)

    async def __aenter__(self: Self) -> Vision2Mqtt:
        super_enter = getattr(super(), "__enter__", None)
        if callable(super_enter):
            super_enter()

        # Restore today's image counter before anything can publish, so a restart mid-day does
        # not zero a number the user reads as "today".
        cast(Any, self).restore_state()

        # Prime cpu_percent so first telemetry reading isn't 0
        psutil.cpu_percent(interval=None)
        # Probe for axcl-smi before MQTT connect so discovery includes NPU sensors
        cast(Any, self)._axcl_smi_path = cast(Any, self)._probe_axcl_smi()
        if cast(Any, self)._axcl_smi_path:
            self.logger.info(f"axcl-smi found at {cast(Any, self)._axcl_smi_path} — NPU telemetry enabled")

        await cast(Any, self).mqttc_create()
        await cast(Any, self).init_detector()
        self.running = True

        return cast(Vision2Mqtt, self)

    def save_state(self: Vision2Mqtt) -> None:
        """Persist today's counter so a restart does not zero the day's total.

        Atomic write (mkstemp + fsync + os.replace) copied from amcrest2mqtt: a plain truncate
        -then-write can leave an empty .dat behind if the write fails partway.
        """
        data_file = Path(self.config["config_path"]) / "vision2mqtt.dat"
        state = {
            "images_annotated": self.images_annotated,
            "images_annotated_date": str(self.images_annotated_date),
        }
        tmp_path = None
        try:
            fd, tmp_name = tempfile.mkstemp(dir=str(data_file.parent), prefix=f".{data_file.name}.", suffix=".tmp")
            tmp_path = Path(tmp_name)
            try:
                file = os.fdopen(fd, "w", encoding="utf-8")
            except BaseException:
                with suppress(OSError):
                    os.close(fd)
                raise
            with file:
                json.dump(state, file, indent=4)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, data_file)
            self.logger.info(f"saved state to {data_file}: {self.images_annotated} images today")
        except OSError as err:
            self.logger.error(f"could not save state to {data_file}: {err!r}")
            if tmp_path is not None:
                with suppress(OSError):
                    os.unlink(tmp_path)

    def restore_state(self: Vision2Mqtt) -> None:
        """Restore today's counter. A stored count from a PREVIOUS day is discarded, not carried
        forward -- restarting after midnight must not resurrect yesterday's total."""
        data_file = Path(self.config["config_path"]) / "vision2mqtt.dat"
        if not data_file.exists():
            return
        try:
            state = json.loads(data_file.read_text(encoding="utf-8"))
            stored_date = datetime.fromisoformat(state["images_annotated_date"])
            if stored_date.tzinfo is None:
                stored_date = stored_date.astimezone()
            if stored_date.date() == datetime.now(UTC).astimezone().date():
                self.images_annotated = int(state["images_annotated"])
                self.images_annotated_date = stored_date
                self.logger.info(f"restored state from {data_file}: {self.images_annotated} images today")
            else:
                self.logger.info(f"discarded stale state from {data_file} ({stored_date.date()}) -- new day")
        except (ValueError, KeyError, TypeError, OSError) as err:
            self.logger.warning(f"could not restore state from {data_file}: {err} -- starting fresh")

    async def __aexit__(self: Self, exc_type: BaseException | None, exc_val: BaseException | None, exc_tb: TracebackType) -> None:
        super_exit = getattr(super(), "__exit__", None)
        if callable(super_exit):
            super_exit(exc_type, exc_val, exc_tb)

        self.running = False

        if cast(Any, self).mqttc is not None:
            if self.ha_enabled:
                try:
                    await cast(Any, self).publish_service_availability("offline")
                except Exception as err:
                    self.logger.debug(f"publish offline failed: {err!r}")

            try:
                cast(Any, self).mqttc.loop_stop()
            except Exception as err:
                self.logger.debug(f"mqtt loop_stop failed: {err!r}")

            if cast(Any, self).mqttc.is_connected():
                try:
                    cast(Any, self).mqttc.disconnect()
                    self.logger.info("disconnected from MQTT broker")
                except Exception as err:
                    self.logger.warning(f"error during MQTT disconnect: {err!r}")

        cast(Any, self).save_state()
        self.logger.info("exiting gracefully")
