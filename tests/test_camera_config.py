# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import pytest
from unittest.mock import MagicMock

from mqtt_helper import ConfigError

from vision2mqtt.mixins.helpers import HelpersMixin, _CAMERA_MODE_PRESETS
from vision2mqtt.mixins.labels import LabelsMixin
from vision2mqtt.models.events import CameraConfig


class FakeHelper(HelpersMixin, LabelsMixin):
    def __init__(self, vision_config):
        self.vision_config = vision_config
        self.logger = MagicMock()


class TestCameraModePresets:
    def test_feeder_preset_has_expected_thresholds(self):
        preset = _CAMERA_MODE_PRESETS["feeder"]
        assert preset["min_confidence"]["person"] == 0.90
        assert preset["min_confidence"]["bird"] == 0.20
        assert preset["default_label"] == "bird"

    def test_baby_preset_has_expected_thresholds(self):
        preset = _CAMERA_MODE_PRESETS["baby"]
        assert preset["min_confidence"]["person"] == 0.25
        assert preset["min_confidence"]["vehicle"] == 0.99
        assert preset["default_label"] is None

    def test_security_preset_has_low_thresholds(self):
        preset = _CAMERA_MODE_PRESETS["security"]
        for label in ("person", "vehicle", "bird", "animal"):
            assert preset["min_confidence"][label] == 0.25
        assert preset["default_label"] is None

    def test_default_preset_is_empty(self):
        preset = _CAMERA_MODE_PRESETS["default"]
        assert preset == {}


class TestCameraConfigLoading:
    def _load(self, yaml_vision_block, tmp_path):
        """Helper to run load_config with a YAML vision block."""
        import yaml

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"vision": yaml_vision_block}))

        helper = FakeHelper({})
        helper._read_version_file = lambda: "test"  # type: ignore[assignment]
        config = helper.load_config(str(config_file))
        return config["vision"]["cameras"]

    def test_feeder_mode_expands(self, tmp_path):
        cameras = self._load({"cameras": {"CAM1": {"mode": "feeder"}}}, tmp_path)
        assert cameras["CAM1"].mode == "feeder"
        assert cameras["CAM1"].min_confidence["person"] == 0.90
        assert cameras["CAM1"].min_confidence["bird"] == 0.20
        assert cameras["CAM1"].default_label == "bird"

    def test_baby_mode_expands(self, tmp_path):
        cameras = self._load({"cameras": {"CAM1": {"mode": "baby"}}}, tmp_path)
        assert cameras["CAM1"].mode == "baby"
        assert cameras["CAM1"].min_confidence["person"] == 0.25
        assert cameras["CAM1"].default_label is None

    def test_security_mode_expands(self, tmp_path):
        cameras = self._load({"cameras": {"CAM1": {"mode": "security"}}}, tmp_path)
        assert cameras["CAM1"].mode == "security"
        assert cameras["CAM1"].min_confidence["person"] == 0.25
        assert cameras["CAM1"].min_confidence["animal"] == 0.25

    def test_mode_with_overlay_merges(self, tmp_path):
        cameras = self._load({"cameras": {"CAM1": {"mode": "feeder", "min_confidence": {"bird": 0.15}}}}, tmp_path)
        assert cameras["CAM1"].min_confidence["bird"] == 0.15
        assert cameras["CAM1"].min_confidence["person"] == 0.90  # from preset

    def test_unknown_mode_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown mode 'bogus'"):
            self._load({"cameras": {"CAM1": {"mode": "bogus"}}}, tmp_path)

    def test_per_label_dict_loads(self, tmp_path):
        cameras = self._load({"cameras": {"CAM1": {"min_confidence": {"person": 0.80, "bird": 0.10}}}}, tmp_path)
        assert cameras["CAM1"].mode == "default"
        assert cameras["CAM1"].min_confidence == {"person": 0.80, "bird": 0.10}

    def test_uniform_float_override(self, tmp_path):
        cameras = self._load({"cameras": {"CAM1": {"min_confidence": 0.30}}}, tmp_path)
        assert cameras["CAM1"].min_confidence == 0.30

    def test_invalid_confidence_type_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="expected dict or number"):
            self._load({"cameras": {"CAM1": {"min_confidence": "high"}}}, tmp_path)

    def test_no_cameras_key_gives_empty(self, tmp_path):
        cameras = self._load({}, tmp_path)
        assert cameras == {}

    def test_default_label_from_camera_block(self, tmp_path):
        cameras = self._load({"cameras": {"CAM1": {"default_label": "animal"}}}, tmp_path)
        assert cameras["CAM1"].default_label == "animal"


class TestGetCameraMinConfidence:
    def test_per_label_override(self, sample_vision_config):
        sample_vision_config["cameras"] = {
            "CAM1": CameraConfig(min_confidence={"person": 0.90, "bird": 0.20}),
        }
        helper = FakeHelper(sample_vision_config)
        assert helper._get_camera_min_confidence("CAM1", "person") == 0.90
        assert helper._get_camera_min_confidence("CAM1", "bird") == 0.20

    def test_per_label_falls_through_to_global(self, sample_vision_config):
        sample_vision_config["cameras"] = {
            "CAM1": CameraConfig(min_confidence={"person": 0.90}),
        }
        helper = FakeHelper(sample_vision_config)
        # "vehicle" not in per-label dict — should fall through to global 0.45
        assert helper._get_camera_min_confidence("CAM1", "vehicle") == 0.45

    def test_uniform_float_override(self, sample_vision_config):
        sample_vision_config["cameras"] = {
            "CAM1": CameraConfig(min_confidence=0.30),
        }
        helper = FakeHelper(sample_vision_config)
        assert helper._get_camera_min_confidence("CAM1", "person") == 0.30
        assert helper._get_camera_min_confidence("CAM1", "bird") == 0.30

    def test_unconfigured_camera_uses_global(self, sample_vision_config):
        helper = FakeHelper(sample_vision_config)
        assert helper._get_camera_min_confidence("UNKNOWN_CAM", "person") == 0.45

    def test_no_min_confidence_on_camera_uses_global(self, sample_vision_config):
        sample_vision_config["cameras"] = {
            "CAM1": CameraConfig(mode="default"),
        }
        helper = FakeHelper(sample_vision_config)
        assert helper._get_camera_min_confidence("CAM1", "person") == 0.45


class TestGetCameraDefaultLabel:
    def test_returns_configured_value(self, sample_vision_config):
        sample_vision_config["cameras"] = {
            "FEEDER": CameraConfig(default_label="bird"),
        }
        helper = FakeHelper(sample_vision_config)
        assert helper._get_camera_default_label("FEEDER") == "bird"

    def test_returns_none_for_unconfigured(self, sample_vision_config):
        helper = FakeHelper(sample_vision_config)
        assert helper._get_camera_default_label("UNKNOWN") is None

    def test_returns_none_when_not_set(self, sample_vision_config):
        sample_vision_config["cameras"] = {
            "CAM1": CameraConfig(mode="baby"),
        }
        helper = FakeHelper(sample_vision_config)
        assert helper._get_camera_default_label("CAM1") is None


class TestGetCameraMode:
    def test_returns_feeder(self, sample_vision_config):
        sample_vision_config["cameras"] = {
            "FEEDER": CameraConfig(mode="feeder"),
        }
        helper = FakeHelper(sample_vision_config)
        assert helper._get_camera_mode("FEEDER") == "feeder"

    def test_returns_baby(self, sample_vision_config):
        sample_vision_config["cameras"] = {
            "NURSERY": CameraConfig(mode="baby"),
        }
        helper = FakeHelper(sample_vision_config)
        assert helper._get_camera_mode("NURSERY") == "baby"

    def test_returns_default_for_unconfigured(self, sample_vision_config):
        helper = FakeHelper(sample_vision_config)
        assert helper._get_camera_mode("UNKNOWN") == "default"
