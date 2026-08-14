# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import os
import pytest
from unittest.mock import MagicMock

from mqtt_helper import ConfigError
from vision2mqtt.mixins.helpers import HelpersMixin


class FakeHelpers(HelpersMixin):
    def __init__(self):
        self.logger = MagicMock()
        self.running = True


class TestLoadConfigFromFile:
    def test_loads_valid_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
mqtt:
  host: 10.10.10.1
  port: 1883
  username: mqtt
  password: secret
  prefix: vision2mqtt

vision:
  backend: ultralytics
  model: yolo26n.pt
  labels:
    - person
    - vehicle
  min_confidence: 0.5
  concurrency: 2
  max_queue: 10
""")
        # write a VERSION file in cwd for read_file
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert config["mqtt"]["host"] == "10.10.10.1"
        assert config["mqtt"]["port"] == 1883
        assert config["mqtt"]["username"] == "mqtt"
        assert config["mqtt"]["prefix"] == "vision2mqtt"
        assert config["vision"]["backend"] == "ultralytics"
        assert config["vision"]["model"] == "yolo26n.pt"
        assert config["vision"]["labels"] == ["person", "vehicle"]
        assert config["vision"]["min_confidence"] == 0.5
        assert config["vision"]["concurrency"] == 2
        assert config["vision"]["max_queue"] == 10
        assert config["config_from"] == "file"
        assert config["version"] == "v0.1.0"

    def test_loads_from_file_path_directly(self, tmp_path):
        config_file = tmp_path / "myconfig.yaml"
        config_file.write_text("""
mqtt:
  host: 192.168.1.1
vision:
  backend: ultralytics
""")
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.2.0")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(config_file))
        finally:
            os.chdir(old_cwd)

        assert config["mqtt"]["host"] == "192.168.1.1"
        assert config["config_from"] == "file"


class TestLoadConfigDefaults:
    def test_defaults_when_no_file(self, tmp_path):
        """When no config file exists, env vars and defaults are used."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        # check defaults
        assert config["mqtt"]["host"] == "localhost"
        assert config["mqtt"]["port"] == 1883
        assert config["mqtt"]["qos"] == 0
        assert config["mqtt"]["protocol_version"] == "5"
        assert config["mqtt"]["prefix"] == "vision2mqtt"
        assert config["vision"]["backend"] == "ultralytics"
        assert config["vision"]["model"] == "yolo26n.pt"
        assert config["vision"]["subscribe_topics"] == ["+/vision/request"]
        assert config["vision"]["labels"] == ["person", "vehicle", "animal", "bird", "package"]
        assert config["vision"]["min_confidence"] == 0.45
        assert config["vision"]["concurrency"] == 1
        assert config["vision"]["max_queue"] == 20
        assert config["vision"]["retain_presence"] is True  # forced by home_assistant=True default
        assert config["vision"]["debug_save_images"] is False
        assert config["home_assistant"] is True
        assert config["mqtt"]["discovery_prefix"] == "homeassistant"
        assert config["config_from"] == "env"

    def test_env_var_overrides(self, tmp_path, monkeypatch):
        """Environment variables should override defaults when no config file."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")

        monkeypatch.setenv("MQTT_HOST", "mqtt.example.com")
        monkeypatch.setenv("MQTT_PORT", "8883")
        monkeypatch.setenv("VISION_BACKEND", "ultralytics")
        monkeypatch.setenv("VISION_MIN_CONFIDENCE", "0.7")
        monkeypatch.setenv("VISION_CONCURRENCY", "4")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert config["mqtt"]["host"] == "mqtt.example.com"
        assert config["mqtt"]["port"] == 8883
        assert config["vision"]["min_confidence"] == 0.7
        assert config["vision"]["concurrency"] == 4


class TestLoadConfigValidation:
    def test_invalid_backend_raises(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
mqtt:
  host: localhost
vision:
  backend: tensorflow
""")
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with pytest.raises(ConfigError, match="must be 'ultralytics' or 'axcl'"):
                helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

    def test_axcl_backend_accepted(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
mqtt:
  host: localhost
vision:
  backend: axcl
  model: /models/yolo26s.axmodel
""")
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert config["vision"]["backend"] == "axcl"
        assert config["vision"]["model"] == "/models/yolo26s.axmodel"

    def test_malformed_yaml_raises(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(": : : not valid yaml [[[")
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with pytest.raises(ConfigError, match="failed to load"):
                helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)


class TestLoadConfigVersion:
    def test_dev_tier_appends_suffix(self, tmp_path, monkeypatch):
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")
        monkeypatch.setenv("APP_TIER", "dev")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert config["version"] == "v0.1.0:DEV"

    def test_app_version_env_overrides_file(self, tmp_path, monkeypatch):
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")
        monkeypatch.setenv("APP_VERSION", "v9.9.9")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert config["version"] == "v9.9.9"


class TestReadFile:
    def test_reads_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("  hello world  \n")

        helpers = FakeHelpers()
        assert helpers.read_file(str(f)) == "hello world"

    def test_missing_file_raises(self):
        helpers = FakeHelpers()
        with pytest.raises(FileNotFoundError):
            helpers.read_file("/nonexistent/file.txt")


class TestHomeAssistantConfig:
    def test_ha_defaults_to_true(self, tmp_path):
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert config["home_assistant"] is True

    def test_ha_forces_retain_presence(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
mqtt:
  host: localhost
vision:
  backend: ultralytics
  retain_presence: false
""")
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert config["home_assistant"] is True
        assert config["vision"]["retain_presence"] is True

    def test_ha_disabled_preserves_retain_presence(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
home_assistant: false
mqtt:
  host: localhost
vision:
  backend: ultralytics
  retain_presence: false
""")
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert config["home_assistant"] is False
        assert config["vision"]["retain_presence"] is False

    def test_discovery_prefix_configurable(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
mqtt:
  host: localhost
  discovery_prefix: custom_prefix
vision:
  backend: ultralytics
""")
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert config["mqtt"]["discovery_prefix"] == "custom_prefix"

    def test_ha_env_var_override(self, tmp_path, monkeypatch):
        version_file = tmp_path / "VERSION"
        version_file.write_text("v0.1.0")
        monkeypatch.setenv("HOME_ASSISTANT", "false")

        helpers = FakeHelpers()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = helpers.load_config(str(tmp_path))
        finally:
            os.chdir(old_cwd)

        assert config["home_assistant"] is False


class TestHandleSignal:
    def test_sets_running_false(self):
        helpers = FakeHelpers()
        assert helpers.running is True

        helpers.handle_signal(2, None)  # SIGINT = 2

        assert helpers.running is False
        helpers.logger.warning.assert_called_once()
