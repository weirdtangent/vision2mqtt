# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

import logging
import os
import pathlib
import signal
import threading
from types import FrameType
from typing import TYPE_CHECKING, Any, cast
import yaml

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt

READY_FILE = os.getenv("READY_FILE", "/tmp/vision2mqtt.ready")


class ConfigError(ValueError):
    """Raised when the configuration file is invalid."""

    pass


class HelpersMixin:
    def mark_ready(self: Vision2Mqtt) -> None:
        pathlib.Path(READY_FILE).touch()

    def heartbeat_ready(self: Vision2Mqtt) -> None:
        pathlib.Path(READY_FILE).touch()

    def read_file(self: Vision2Mqtt, file_name: str) -> str:
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_name}")

    def handle_signal(self: Vision2Mqtt, signum: int, _: FrameType | None) -> Any:
        sig_name = signal.Signals(signum).name
        self.logger.warning(f"{sig_name} received - stopping service loop")
        self.running = False

        def _force_exit() -> None:
            self.logger.warning("force-exiting process after signal")
            os._exit(0)

        threading.Timer(5.0, _force_exit).start()

    def load_config(self: Vision2Mqtt, config_arg: Any | None) -> dict[str, Any]:
        version = os.getenv("APP_VERSION", self.read_file("VERSION"))
        tier = os.getenv("APP_TIER", "prod")
        if tier == "dev":
            version += ":DEV"

        config_from = "env"
        config: dict[str, str | bool | int | dict] = {}

        # Determine config file path
        config_path = config_arg or "/config"
        config_path = os.path.expanduser(config_path)
        config_path = os.path.abspath(config_path)

        if os.path.isdir(config_path):
            config_file = os.path.join(config_path, "config.yaml")
        elif os.path.isfile(config_path):
            config_file = config_path
            config_path = os.path.dirname(config_file)
        else:
            if config_path.endswith(".yaml"):
                config_file = config_path
            else:
                config_file = os.path.join(config_path, "config.yaml")

        # Try to load from YAML
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f) or {}
                config_from = "file"
            except Exception as err:
                raise ConfigError(f"found {config_file} but failed to load: {err}")
        else:
            logging.warning(f"Config file not found at {config_file}, falling back to environment vars")

        # Merge with environment vars
        mqtt = cast(dict[str, Any], config.get("mqtt", {}))
        vision = cast(dict[str, Any], config.get("vision", {}))

        # fmt: off
        mqtt = {
            "host":             str(mqtt.get("host")             or os.getenv("MQTT_HOST", "localhost")),
            "port":         int(str(mqtt.get("port")             or os.getenv("MQTT_PORT", 1883))),
            "qos":          int(str(mqtt.get("qos")              or os.getenv("MQTT_QOS", 0))),
            "protocol_version": str(mqtt.get("protocol_version") or os.getenv("MQTT_PROTOCOL", "5")),
            "username":         str(mqtt.get("username")         or os.getenv("MQTT_USERNAME", "")),
            "password":         str(mqtt.get("password")         or os.getenv("MQTT_PASSWORD", "")),
            "tls_enabled":     bool(mqtt.get("tls_enabled")      or (os.getenv("MQTT_TLS_ENABLED", "false").lower() == "true")),
            "tls_ca_cert":      str(mqtt.get("tls_ca_cert")      or os.getenv("MQTT_TLS_CA_CERT", "")),
            "tls_cert":         str(mqtt.get("tls_cert")         or os.getenv("MQTT_TLS_CERT", "")),
            "tls_key":          str(mqtt.get("tls_key")          or os.getenv("MQTT_TLS_KEY", "")),
            "prefix":           str(mqtt.get("prefix")           or os.getenv("MQTT_PREFIX", "vision2mqtt")),
        }

        vision = {
            "backend":           str(vision.get("backend")           or os.getenv("VISION_BACKEND", "ultralytics")),
            "model":             str(vision.get("model")             or os.getenv("VISION_MODEL", "yolo11n.pt")),
            "subscribe_topics":  list(vision.get("subscribe_topics") or ["+/vision/request"]),
            "labels":            list(vision.get("labels")           or ["person", "vehicle", "animal", "bird"]),
            "min_confidence": float(str(vision.get("min_confidence") or os.getenv("VISION_MIN_CONFIDENCE", "0.45"))),
            "concurrency":   int(str(vision.get("concurrency")      or os.getenv("VISION_CONCURRENCY", "1"))),
            "max_queue":     int(str(vision.get("max_queue")        or os.getenv("VISION_MAX_QUEUE", "20"))),
            "retain_presence":  bool(vision.get("retain_presence",   os.getenv("VISION_RETAIN_PRESENCE", "").lower() == "true")),
            "debug_save_images": bool(vision.get("debug_save_images", os.getenv("VISION_DEBUG_SAVE", "").lower() == "true")),
        }

        config = {
            "mqtt":        mqtt,
            "vision":      vision,
            "debug":       bool(config.get("debug", os.getenv("DEBUG", "").lower() == "true")),
            "config_from": config_from,
            "config_path": config_path,
            "version":     version,
        }
        # fmt: on

        # Validate
        if vision["backend"] not in ("ultralytics", "axcl"):
            raise ConfigError(f"`vision.backend` must be 'ultralytics' or 'axcl', got '{vision['backend']}'")

        return config
