# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
import pytest

from vision2mqtt.models.events import CameraConfig


@pytest.fixture
def sample_vision_config():
    return {
        "backend": "ultralytics",
        "model": "yolo26n.pt",
        "subscribe_topics": ["+/vision/request"],
        "labels": ["person", "vehicle", "animal", "bird"],
        "min_confidence": 0.45,
        "concurrency": 1,
        "max_queue": 20,
        "retain_presence": False,
        "debug_save_images": False,
        "composites": [],
        "presence_cooldown": 60,
        "frequency_window": 3600,
        "cameras": {},
    }


@pytest.fixture
def feeder_vision_config(sample_vision_config):
    sample_vision_config["cameras"] = {
        "FEEDER_CAM": CameraConfig(
            mode="feeder",
            min_confidence={"person": 0.90, "vehicle": 0.95, "bird": 0.20, "animal": 0.25},
            default_label="bird",
        ),
    }
    return sample_vision_config


@pytest.fixture
def sample_motion_event():
    from vision2mqtt.models.events import MotionEvent

    return MotionEvent(
        camera_id="2BEFD0C907BB6BF2",
        camera_name="Front Yard",
        event_id="20260214-153045",
        image_b64="",
        timestamp="2026-02-14T15:30:45",
        source="recording_snapshot",
    )
