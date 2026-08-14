# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess  # nosec B404 - axcl-smi is a known system tool
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from vision2mqtt.interface import VisionServiceProtocol as Vision2Mqtt

AXCL_SMI_PATHS = ["/usr/bin/axcl/axcl-smi", "/usr/local/bin/axcl-smi"]


@dataclass(frozen=True)
class TelemetryDescriptor:
    key: str
    name: str
    unit: str | None
    device_class: str | None
    state_class: str
    icon: str
    precision: int


HOST_SENSORS: list[TelemetryDescriptor] = [
    TelemetryDescriptor(key="cpu_usage", name="CPU Usage", unit="%", device_class=None, state_class="measurement", icon="mdi:cpu-64-bit", precision=1),
    TelemetryDescriptor(
        key="cpu_temperature",
        name="CPU Temperature",
        unit="\u00b0C",
        device_class="temperature",
        state_class="measurement",
        icon="mdi:thermometer",
        precision=1,
    ),
    TelemetryDescriptor(key="memory_usage", name="Memory Usage", unit="%", device_class=None, state_class="measurement", icon="mdi:memory", precision=1),
    TelemetryDescriptor(key="disk_usage", name="Disk Usage", unit="%", device_class=None, state_class="measurement", icon="mdi:harddisk", precision=1),
    TelemetryDescriptor(key="load_avg_1m", name="Load Avg (1m)", unit=None, device_class=None, state_class="measurement", icon="mdi:chart-line", precision=2),
    TelemetryDescriptor(key="load_avg_5m", name="Load Avg (5m)", unit=None, device_class=None, state_class="measurement", icon="mdi:chart-line", precision=2),
    TelemetryDescriptor(key="uptime", name="Uptime", unit="h", device_class="duration", state_class="total_increasing", icon="mdi:timer", precision=2),
]

NPU_SENSORS: list[TelemetryDescriptor] = [
    TelemetryDescriptor(
        key="npu_temperature", name="NPU Temperature", unit="\u00b0C", device_class="temperature", state_class="measurement", icon="mdi:chip", precision=0
    ),
    TelemetryDescriptor(
        key="npu_utilization", name="NPU Utilization", unit="%", device_class=None, state_class="measurement", icon="mdi:expansion-card", precision=0
    ),
    TelemetryDescriptor(
        key="npu_memory_used", name="NPU Memory Used", unit="MiB", device_class="data_size", state_class="measurement", icon="mdi:memory", precision=0
    ),
    TelemetryDescriptor(
        key="npu_memory_total", name="NPU Memory Total", unit="MiB", device_class="data_size", state_class="measurement", icon="mdi:memory", precision=0
    ),
]


class SystemStatsMixin:
    _axcl_smi_path: str | None = None

    def _probe_axcl_smi(self: Vision2Mqtt) -> str | None:
        path = shutil.which("axcl-smi")
        if path:
            return path
        for candidate in AXCL_SMI_PATHS:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    @staticmethod
    def _read_cpu_temperature() -> float | None:
        try:
            temps = psutil.sensors_temperatures()
        except (NotImplementedError, AttributeError, OSError):
            temps = {}

        for key in ("cpu-thermal", "soc_thermal", "gpu", "coretemp", "arm"):
            entries = temps.get(key)
            if entries:
                current = entries[0].current
                if current is not None:
                    return float(current)

        candidate_paths = [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/devices/virtual/thermal/thermal_zone0/temp",
        ]
        for path in candidate_paths:
            try:
                with open(path, encoding="utf-8") as handle:
                    raw = handle.read().strip()
                    value = float(raw) / (1000 if len(raw) > 3 else 1)
                    return value
            except (OSError, ValueError):
                continue
        return None

    def _collect_npu_stats(self: Vision2Mqtt) -> dict[str, float]:
        if not self._axcl_smi_path:
            return {}
        try:
            result = subprocess.run(  # nosec B603 - path validated at startup
                [self._axcl_smi_path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return {}
            return self._parse_axcl_smi_output(result.stdout)
        except (subprocess.SubprocessError, OSError):
            return {}

    @staticmethod
    def _parse_axcl_smi_output(output: str) -> dict[str, float]:
        stats: dict[str, float] = {}

        # Temperature: look for pattern like "52C"
        temp_match = re.search(r"(\d+)C", output)
        if temp_match:
            stats["npu_temperature"] = float(temp_match.group(1))

        # NPU utilization: look for percentage like "3%" at start of the second data line
        util_match = re.search(r"\|\s+(\d+)%\s+\d+%", output)
        if util_match:
            stats["npu_utilization"] = float(util_match.group(1))

        # NPU memory: parse "22 MiB / 3072 MiB" from second data line
        # The dashboard has two memory columns; we want the second one (NPU memory)
        mem_matches = re.findall(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", output)
        if len(mem_matches) >= 2:
            stats["npu_memory_used"] = float(mem_matches[1][0])
            stats["npu_memory_total"] = float(mem_matches[1][1])
        elif len(mem_matches) == 1:
            stats["npu_memory_used"] = float(mem_matches[0][0])
            stats["npu_memory_total"] = float(mem_matches[0][1])

        return stats

    def collect_system_stats(self: Vision2Mqtt) -> dict[str, float]:
        metrics: dict[str, float] = {}

        metrics["cpu_usage"] = psutil.cpu_percent(interval=None)

        cpu_temp = self._read_cpu_temperature()
        if cpu_temp is not None:
            metrics["cpu_temperature"] = cpu_temp

        metrics["memory_usage"] = psutil.virtual_memory().percent
        metrics["disk_usage"] = psutil.disk_usage("/").percent

        load1, load5, _ = os.getloadavg()
        metrics["load_avg_1m"] = load1
        metrics["load_avg_5m"] = load5

        uptime_seconds = max(0, int(time.time() - psutil.boot_time()))
        metrics["uptime"] = uptime_seconds / 3600

        npu_stats = self._collect_npu_stats()
        metrics.update(npu_stats)

        return metrics

    def build_system_stats_components(self: Vision2Mqtt) -> dict[str, dict[str, Any]]:
        expire_after = 185  # 60s interval * 3 + 5s
        cmps: dict[str, dict[str, Any]] = {}

        sensors = list(HOST_SENSORS)
        if self._axcl_smi_path:
            sensors.extend(NPU_SENSORS)

        for desc in sensors:
            entry: dict[str, Any] = {
                "p": "sensor",
                "name": desc.name,
                "uniq_id": self.mqtt_helper.svc_unique_id(f"telemetry_{desc.key}"),
                "obj_id": self.mqtt_helper.obj_id(self.service_name, f"telemetry_{desc.key}"),
                "stat_t": f"{self.service}/service/telemetry/{desc.key}",
                "entity_category": "diagnostic",
                "icon": desc.icon,
                "expire_after": expire_after,
            }
            if desc.unit:
                entry["unit_of_measurement"] = desc.unit
            if desc.device_class:
                entry["device_class"] = desc.device_class
            entry["state_class"] = desc.state_class
            cmps[f"telemetry_{desc.key}"] = entry

        return cmps

    async def publish_system_stats(self: Vision2Mqtt) -> None:
        if not self.ha_enabled:
            return

        metrics = await asyncio.to_thread(self.collect_system_stats)
        prefix = self.service

        for desc in HOST_SENSORS + (NPU_SENSORS if self._axcl_smi_path else []):
            value = metrics.get(desc.key)
            if value is None:
                continue
            topic = f"{prefix}/service/telemetry/{desc.key}"
            fmt = f"{{:.{desc.precision}f}}"
            payload = fmt.format(value)
            await asyncio.to_thread(self.mqtt_helper.safe_publish, topic, payload)
