# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import asyncio
import argparse
import concurrent.futures
import logging
import psutil
from json_logging import get_logger
from mqtt_helper import MqttHelper
from paho.mqtt.client import Client
from types import TracebackType

from typing import Any, Self, cast

from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt
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
        self.seen_cameras: set[str] = set()
        self._camera_discovery_lock = asyncio.Lock()

        max_queue = self.vision_config.get("max_queue", 20)
        self.queue: asyncio.Queue[MotionEvent] = asyncio.Queue(maxsize=max_queue)

    async def __aenter__(self: Self) -> Vision2Mqtt:
        super_enter = getattr(super(), "__enter__", None)
        if callable(super_enter):
            super_enter()

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

        self.logger.info("exiting gracefully")
