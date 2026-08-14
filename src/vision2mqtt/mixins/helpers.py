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
from mqtt_helper import ConfigError

from vision2mqtt.models.events import CameraConfig

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt

READY_FILE = os.getenv("READY_FILE", "/tmp/vision2mqtt.ready")

_CAMERA_MODE_PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "feeder": {
        "min_confidence": {"person": 0.90, "vehicle": 0.95, "bird": 0.20, "animal": 0.25},
        "default_label": "bird",
    },
    "baby": {
        "min_confidence": {"person": 0.25, "vehicle": 0.99, "bird": 0.99, "animal": 0.80},
        "default_label": None,
    },
    "security": {
        "min_confidence": {"person": 0.25, "vehicle": 0.25, "bird": 0.25, "animal": 0.25},
        "default_label": None,
    },
}


class HelpersMixin:
    async def clear_discovery(self: Vision2Mqtt) -> None:
        """Delete every retained discovery topic we own, service device included.

        The topic list comes from the broker rather than seen_cameras: this runs from
        mqtt_on_connect, when no camera has reported yet, so seen_cameras is empty and every
        per-camera topic would survive -- reducing a schema bump to an in-place update that cannot
        dislodge a squatted entity_id.

        Dropping seen_cameras is still what makes the republish work. Cameras are discovered lazily
        on first detection (see publish_vision_result) and their display names only arrive with the
        event, so there is nothing to replay here -- each re-announces itself on its next event.
        """
        await self.clear_retained_discovery()
        async with self._camera_discovery_lock:
            self.seen_cameras.clear()

    async def rediscover_all(self: Vision2Mqtt) -> None:
        await self.publish_service_discovery()
        await self.publish_service_state()

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

    def _read_version_file(self: Vision2Mqtt) -> str:
        try:
            with open("VERSION", "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "dev"

    def load_config(self: Vision2Mqtt, config_arg: Any | None) -> dict[str, Any]:
        version = os.getenv("APP_VERSION") or self._read_version_file()
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
        raw_cameras: dict[str, Any] = vision.get("cameras", {})

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
            "discovery_prefix": str(mqtt.get("discovery_prefix") or os.getenv("MQTT_DISCOVERY_PREFIX", "homeassistant")),
        }

        vision = {
            "backend":           str(vision.get("backend")           or os.getenv("VISION_BACKEND", "ultralytics")),
            "model":             str(vision.get("model")             or os.getenv("VISION_MODEL", "yolo26n.pt")),
            "subscribe_topics":  list(vision.get("subscribe_topics") or ["+/vision/request"]),
            "labels":            list(vision.get("labels")           or ["person", "vehicle", "animal", "bird", "package"]),
            "min_confidence": float(str(vision.get("min_confidence") or os.getenv("VISION_MIN_CONFIDENCE", "0.45"))),
            "concurrency":   int(str(vision.get("concurrency")      or os.getenv("VISION_CONCURRENCY", "1"))),
            "max_queue":     int(str(vision.get("max_queue")        or os.getenv("VISION_MAX_QUEUE", "20"))),
            "retain_presence":  bool(vision.get("retain_presence",   os.getenv("VISION_RETAIN_PRESENCE", "").lower() == "true")),
            "debug_save_images": bool(vision.get("debug_save_images", os.getenv("VISION_DEBUG_SAVE", "").lower() == "true")),
            "composites":        list(vision.get("composites")       or []),
            "presence_cooldown": int(str(vision.get("presence_cooldown") or os.getenv("VISION_PRESENCE_COOLDOWN", "60"))),
            "frequency_window":  int(str(vision.get("frequency_window")  or os.getenv("VISION_FREQUENCY_WINDOW", "3600"))),
        }

        # Expand per-camera config overrides
        cameras: dict[str, CameraConfig] = {}
        for cam_id, cam_raw in raw_cameras.items():
            if not isinstance(cam_raw, dict):
                raise ConfigError(f"vision.cameras.{cam_id}: expected a mapping, got {type(cam_raw).__name__}")
            mode = str(cam_raw.get("mode", "default"))
            if mode not in _CAMERA_MODE_PRESETS:
                raise ConfigError(f"vision.cameras.{cam_id}: unknown mode '{mode}' (valid: {', '.join(_CAMERA_MODE_PRESETS)})")

            # Start from preset defaults
            preset = _CAMERA_MODE_PRESETS[mode]
            merged_conf: dict[str, float] | float | None = None
            if "min_confidence" in preset:
                merged_conf = dict(preset["min_confidence"]) if isinstance(preset["min_confidence"], dict) else preset["min_confidence"]

            # Overlay explicit min_confidence from camera block
            raw_conf = cam_raw.get("min_confidence")
            if raw_conf is not None:
                if isinstance(raw_conf, dict):
                    if merged_conf is None or isinstance(merged_conf, (int, float)):
                        merged_conf = {}
                    for k, v in raw_conf.items():
                        merged_conf[k] = float(v)
                elif isinstance(raw_conf, (int, float)):
                    merged_conf = float(raw_conf)
                else:
                    raise ConfigError(f"vision.cameras.{cam_id}.min_confidence: expected dict or number, got {type(raw_conf).__name__}")

            default_label = cam_raw.get("default_label", preset.get("default_label"))
            cameras[cam_id] = CameraConfig(mode=mode, min_confidence=merged_conf, default_label=default_label)

        vision["cameras"] = cameras

        ha_raw = config.get("home_assistant")
        home_assistant = str(ha_raw if ha_raw is not None else os.getenv("HOME_ASSISTANT", "true")).lower() == "true"

        config = {
            "mqtt":           mqtt,
            "vision":         vision,
            "home_assistant": home_assistant,
            "debug":          bool(config.get("debug", os.getenv("DEBUG", "").lower() == "true")),
            "config_from":    config_from,
            "config_path":    config_path,
            "version":        version,
        }
        # fmt: on

        # HA discovery requires retained presence topics
        if home_assistant and not vision["retain_presence"]:
            vision["retain_presence"] = True

        # Validate
        if vision["backend"] not in ("ultralytics", "axcl"):
            raise ConfigError(f"`vision.backend` must be 'ultralytics' or 'axcl', got '{vision['backend']}'")

        return config

    def _get_camera_min_confidence(self: Vision2Mqtt, camera_id: str, label: str) -> float:
        cameras: dict[str, CameraConfig] = self.vision_config.get("cameras", {})
        cam_cfg = cameras.get(camera_id)
        if cam_cfg and cam_cfg.min_confidence is not None:
            if isinstance(cam_cfg.min_confidence, dict):
                if label in cam_cfg.min_confidence:
                    return cam_cfg.min_confidence[label]
                # per-label dict but label not listed — fall through to global
            else:
                return cam_cfg.min_confidence
        return float(self.vision_config["min_confidence"])

    def _get_camera_default_label(self: Vision2Mqtt, camera_id: str) -> str | None:
        cameras: dict[str, CameraConfig] = self.vision_config.get("cameras", {})
        cam_cfg = cameras.get(camera_id)
        if cam_cfg:
            return cam_cfg.default_label
        return None

    def _get_camera_mode(self: Vision2Mqtt, camera_id: str) -> str:
        cameras: dict[str, CameraConfig] = self.vision_config.get("cameras", {})
        cam_cfg = cameras.get(camera_id)
        if cam_cfg:
            return cam_cfg.mode
        return "default"
