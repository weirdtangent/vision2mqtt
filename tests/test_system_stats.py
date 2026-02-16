# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import pytest
from unittest.mock import MagicMock, patch, mock_open

from vision2mqtt.mixins.system_stats import SystemStatsMixin, HOST_SENSORS, NPU_SENSORS


class FakeStatsPublisher(SystemStatsMixin):
    def __init__(self, axcl_smi_path=None):
        self.service = "vision2mqtt"
        self.service_name = "vision2mqtt service"
        self.qos = 0
        self.ha_enabled = True
        self.config = {"version": "v0.1.0-test"}
        self.mqtt_config = {"discovery_prefix": "homeassistant"}
        self.logger = MagicMock()
        self.mqtt_helper = MagicMock()
        self.mqtt_helper.safe_publish = MagicMock()
        self.mqtt_helper.service_slug = "vision2mqtt"
        self.mqtt_helper.svc_unique_id = MagicMock(side_effect=lambda e: f"vision2mqtt_{e}")
        self._axcl_smi_path = axcl_smi_path


async def _fake_to_thread(fn, *args):
    return fn(*args)


SAMPLE_AXCL_SMI_OUTPUT = """\
+-------------------------------+----------------------+----------------------------------------+
| NPU   Name       Firmware     | Bus-Id               | CMM-Memory-Usage                       |
|       TDP   Temp   Pwr / Cap  | NPU%   VPU%          | NPU-Memory-Usage                       |
|===============================+======================+========================================|
| 0     AX650N     V2.18.0                | 0001:81:00.0 | 181 MiB / 954 MiB                    |
|       --   52C   -- / --                | 3%    0%     | 22 MiB / 3072 MiB                    |
+-------------------------------+----------------------+----------------------------------------+
"""


class TestCollectSystemStats:
    @patch("vision2mqtt.mixins.system_stats.psutil")
    @patch("vision2mqtt.mixins.system_stats.os")
    @patch("vision2mqtt.mixins.system_stats.time")
    def test_returns_expected_host_keys(self, mock_time, mock_os, mock_psutil):
        mock_psutil.cpu_percent.return_value = 23.5
        mock_psutil.virtual_memory.return_value = MagicMock(percent=45.2)
        mock_psutil.disk_usage.return_value = MagicMock(percent=62.1)
        mock_psutil.boot_time.return_value = 1000.0
        mock_psutil.sensors_temperatures.return_value = {"cpu-thermal": [MagicMock(current=55.3)]}
        mock_os.getloadavg.return_value = (1.23, 0.98, 0.75)
        mock_time.time.return_value = 4600.0

        pub = FakeStatsPublisher()
        stats = pub.collect_system_stats()

        assert stats["cpu_usage"] == 23.5
        assert stats["cpu_temperature"] == 55.3
        assert stats["memory_usage"] == 45.2
        assert stats["disk_usage"] == 62.1
        assert stats["load_avg_1m"] == 1.23
        assert stats["load_avg_5m"] == 0.98
        assert stats["uptime"] == 3600.0 / 3600  # 1.0 hours

    @patch("vision2mqtt.mixins.system_stats.psutil")
    @patch("vision2mqtt.mixins.system_stats.os")
    @patch("vision2mqtt.mixins.system_stats.time")
    def test_missing_cpu_temp_excluded(self, mock_time, mock_os, mock_psutil):
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=30.0)
        mock_psutil.disk_usage.return_value = MagicMock(percent=50.0)
        mock_psutil.boot_time.return_value = 0.0
        mock_psutil.sensors_temperatures.side_effect = NotImplementedError
        mock_os.getloadavg.return_value = (0.5, 0.5, 0.5)
        mock_time.time.return_value = 3600.0

        pub = FakeStatsPublisher()
        # Also need to handle the sysfs fallback
        with patch("builtins.open", side_effect=OSError):
            stats = pub.collect_system_stats()

        assert "cpu_temperature" not in stats
        assert "cpu_usage" in stats


class TestReadCpuTemperature:
    @patch("vision2mqtt.mixins.system_stats.psutil")
    def test_psutil_thermal(self, mock_psutil):
        mock_psutil.sensors_temperatures.return_value = {"cpu-thermal": [MagicMock(current=62.5)]}
        result = SystemStatsMixin._read_cpu_temperature()
        assert result == 62.5

    @patch("vision2mqtt.mixins.system_stats.psutil")
    def test_sysfs_fallback(self, mock_psutil):
        mock_psutil.sensors_temperatures.side_effect = NotImplementedError
        with patch("builtins.open", mock_open(read_data="54000\n")):
            result = SystemStatsMixin._read_cpu_temperature()
        assert result == 54.0

    @patch("vision2mqtt.mixins.system_stats.psutil")
    def test_returns_none_when_unavailable(self, mock_psutil):
        mock_psutil.sensors_temperatures.side_effect = NotImplementedError
        with patch("builtins.open", side_effect=OSError):
            result = SystemStatsMixin._read_cpu_temperature()
        assert result is None


class TestCollectNpuStats:
    @patch("vision2mqtt.mixins.system_stats.subprocess.run")
    def test_parses_axcl_smi_output(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout=SAMPLE_AXCL_SMI_OUTPUT)
        pub = FakeStatsPublisher(axcl_smi_path="/usr/bin/axcl/axcl-smi")
        stats = pub._collect_npu_stats()

        assert stats["npu_temperature"] == 52.0
        assert stats["npu_utilization"] == 3.0
        assert stats["npu_memory_used"] == 22.0
        assert stats["npu_memory_total"] == 3072.0

    def test_returns_empty_when_not_found(self):
        pub = FakeStatsPublisher(axcl_smi_path=None)
        stats = pub._collect_npu_stats()
        assert stats == {}

    @patch("vision2mqtt.mixins.system_stats.subprocess.run")
    def test_returns_empty_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        pub = FakeStatsPublisher(axcl_smi_path="/usr/bin/axcl/axcl-smi")
        stats = pub._collect_npu_stats()
        assert stats == {}

    @patch("vision2mqtt.mixins.system_stats.subprocess.run")
    def test_returns_empty_on_exception(self, mock_run):
        mock_run.side_effect = OSError("not found")
        pub = FakeStatsPublisher(axcl_smi_path="/usr/bin/axcl/axcl-smi")
        stats = pub._collect_npu_stats()
        assert stats == {}


class TestBuildSystemStatsComponents:
    def test_host_sensors_only_without_npu(self):
        pub = FakeStatsPublisher(axcl_smi_path=None)
        cmps = pub.build_system_stats_components()

        assert len(cmps) == len(HOST_SENSORS)
        for desc in HOST_SENSORS:
            key = f"telemetry_{desc.key}"
            assert key in cmps
            assert cmps[key]["p"] == "sensor"
            assert cmps[key]["entity_category"] == "diagnostic"
            assert cmps[key]["stat_t"] == f"vision2mqtt/service/telemetry/{desc.key}"

    def test_includes_npu_when_available(self):
        pub = FakeStatsPublisher(axcl_smi_path="/usr/bin/axcl/axcl-smi")
        cmps = pub.build_system_stats_components()

        assert len(cmps) == len(HOST_SENSORS) + len(NPU_SENSORS)
        assert "telemetry_npu_temperature" in cmps
        assert "telemetry_npu_utilization" in cmps
        assert "telemetry_npu_memory_used" in cmps
        assert "telemetry_npu_memory_total" in cmps

    def test_sensor_attributes(self):
        pub = FakeStatsPublisher(axcl_smi_path=None)
        cmps = pub.build_system_stats_components()

        cpu_temp = cmps["telemetry_cpu_temperature"]
        assert cpu_temp["device_class"] == "temperature"
        assert cpu_temp["unit_of_measurement"] == "\u00b0C"
        assert cpu_temp["state_class"] == "measurement"
        assert cpu_temp["icon"] == "mdi:thermometer"

        uptime = cmps["telemetry_uptime"]
        assert uptime["device_class"] == "duration"
        assert uptime["state_class"] == "total_increasing"
        assert uptime["unit_of_measurement"] == "h"


class TestPublishSystemStats:
    @pytest.mark.asyncio
    async def test_publishes_to_correct_topics(self):
        pub = FakeStatsPublisher(axcl_smi_path=None)

        fake_metrics = {
            "cpu_usage": 23.5,
            "cpu_temperature": 55.3,
            "memory_usage": 45.2,
            "disk_usage": 62.1,
            "load_avg_1m": 1.23,
            "load_avg_5m": 0.98,
            "uptime": 24.56,
        }

        with patch("vision2mqtt.mixins.system_stats.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            with patch.object(pub, "collect_system_stats", return_value=fake_metrics):
                await pub.publish_system_stats()

        topics = [c.args[0] for c in pub.mqtt_helper.safe_publish.call_args_list]
        assert "vision2mqtt/service/telemetry/cpu_usage" in topics
        assert "vision2mqtt/service/telemetry/cpu_temperature" in topics
        assert "vision2mqtt/service/telemetry/memory_usage" in topics
        assert "vision2mqtt/service/telemetry/disk_usage" in topics
        assert "vision2mqtt/service/telemetry/load_avg_1m" in topics
        assert "vision2mqtt/service/telemetry/load_avg_5m" in topics
        assert "vision2mqtt/service/telemetry/uptime" in topics

    @pytest.mark.asyncio
    async def test_formats_values_with_precision(self):
        pub = FakeStatsPublisher(axcl_smi_path=None)

        fake_metrics = {
            "cpu_usage": 23.456,
            "load_avg_1m": 1.2345,
            "uptime": 24.5678,
        }

        with patch("vision2mqtt.mixins.system_stats.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            with patch.object(pub, "collect_system_stats", return_value=fake_metrics):
                await pub.publish_system_stats()

        payloads = {c.args[0]: c.args[1] for c in pub.mqtt_helper.safe_publish.call_args_list}
        assert payloads["vision2mqtt/service/telemetry/cpu_usage"] == "23.5"
        assert payloads["vision2mqtt/service/telemetry/load_avg_1m"] == "1.23"
        assert payloads["vision2mqtt/service/telemetry/uptime"] == "24.57"

    @pytest.mark.asyncio
    async def test_skips_when_ha_disabled(self):
        pub = FakeStatsPublisher(axcl_smi_path=None)
        pub.ha_enabled = False

        with patch("vision2mqtt.mixins.system_stats.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            await pub.publish_system_stats()

        pub.mqtt_helper.safe_publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_includes_npu_topics_when_available(self):
        pub = FakeStatsPublisher(axcl_smi_path="/usr/bin/axcl/axcl-smi")

        fake_metrics = {
            "cpu_usage": 10.0,
            "memory_usage": 30.0,
            "disk_usage": 50.0,
            "load_avg_1m": 0.5,
            "load_avg_5m": 0.5,
            "uptime": 1.0,
            "npu_temperature": 52.0,
            "npu_utilization": 3.0,
            "npu_memory_used": 22.0,
            "npu_memory_total": 3072.0,
        }

        with patch("vision2mqtt.mixins.system_stats.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _fake_to_thread
            with patch.object(pub, "collect_system_stats", return_value=fake_metrics):
                await pub.publish_system_stats()

        topics = [c.args[0] for c in pub.mqtt_helper.safe_publish.call_args_list]
        assert "vision2mqtt/service/telemetry/npu_temperature" in topics
        assert "vision2mqtt/service/telemetry/npu_utilization" in topics
        assert "vision2mqtt/service/telemetry/npu_memory_used" in topics
        assert "vision2mqtt/service/telemetry/npu_memory_total" in topics


class TestParseAxclSmiOutput:
    def test_parses_standard_output(self):
        stats = SystemStatsMixin._parse_axcl_smi_output(SAMPLE_AXCL_SMI_OUTPUT)
        assert stats["npu_temperature"] == 52.0
        assert stats["npu_utilization"] == 3.0
        assert stats["npu_memory_used"] == 22.0
        assert stats["npu_memory_total"] == 3072.0

    def test_handles_empty_output(self):
        stats = SystemStatsMixin._parse_axcl_smi_output("")
        assert stats == {}

    def test_handles_partial_output(self):
        partial = "| 0     AX650N     V2.18.0 | 0001:81:00.0 | 181 MiB / 954 MiB |"
        stats = SystemStatsMixin._parse_axcl_smi_output(partial)
        assert "npu_memory_used" in stats
        assert stats["npu_memory_used"] == 181.0
        assert stats["npu_memory_total"] == 954.0
