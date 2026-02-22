# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from vision2mqtt.mixins.presence import PresenceTracker


class TestRecordDetection:
    def test_records_last_seen(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        assert t._last_seen[("cam1", "person")] == 100.0

    def test_appends_to_detection_times(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        t.record_detection("cam1", "person", now=105.0)
        assert list(t._detection_times[("cam1", "person")]) == [100.0, 105.0]


class TestIsPresenceOn:
    def test_on_within_cooldown(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        assert t.is_presence_on("cam1", "person", cooldown=60, now=130.0) is True

    def test_off_after_cooldown(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        assert t.is_presence_on("cam1", "person", cooldown=60, now=160.0) is False

    def test_off_at_exact_boundary(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        assert t.is_presence_on("cam1", "person", cooldown=60, now=160.0) is False

    def test_never_seen(self):
        t = PresenceTracker()
        assert t.is_presence_on("cam1", "person", cooldown=60, now=100.0) is False

    def test_renewed_by_new_detection(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        t.record_detection("cam1", "person", now=150.0)
        # Should still be on at 200 (150 + 50 < 150 + 60)
        assert t.is_presence_on("cam1", "person", cooldown=60, now=200.0) is True
        # Should be off at 211 (150 + 61)
        assert t.is_presence_on("cam1", "person", cooldown=60, now=211.0) is False


class TestGetFrequency:
    def test_empty(self):
        t = PresenceTracker()
        assert t.get_frequency("cam1", "person", window=3600, now=100.0) == 0.0

    def test_single_detection(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        freq = t.get_frequency("cam1", "person", window=3600, now=100.0)
        assert freq == 1.0  # 1 detection in 3600s window = 1/h

    def test_multiple_detections(self):
        t = PresenceTracker()
        for i in range(10):
            t.record_detection("cam1", "person", now=100.0 + i)
        freq = t.get_frequency("cam1", "person", window=3600, now=110.0)
        assert freq == 10.0  # 10 detections in 3600s = 10/h

    def test_window_pruning(self):
        t = PresenceTracker()
        # Old detection outside window
        t.record_detection("cam1", "person", now=100.0)
        # Recent detection inside window
        t.record_detection("cam1", "person", now=4000.0)
        freq = t.get_frequency("cam1", "person", window=3600, now=4000.0)
        assert freq == 1.0  # only 1 detection in window


class TestGetExpiredPresenceKeys:
    def test_returns_expired_keys(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        t.set_published_state("cam1", "person", True)
        expired = t.get_expired_presence_keys(cooldown=60, now=161.0)
        assert ("cam1", "person") in expired

    def test_skips_non_expired(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        t.set_published_state("cam1", "person", True)
        expired = t.get_expired_presence_keys(cooldown=60, now=130.0)
        assert len(expired) == 0

    def test_skips_already_off(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        t.set_published_state("cam1", "person", False)
        expired = t.get_expired_presence_keys(cooldown=60, now=200.0)
        assert len(expired) == 0

    def test_skips_never_published(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        expired = t.get_expired_presence_keys(cooldown=60, now=200.0)
        assert len(expired) == 0


class TestPerCameraIsolation:
    def test_separate_cameras(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        t.record_detection("cam2", "person", now=200.0)
        # cam1 detected at 100, still within 60s cooldown at 150
        assert t.is_presence_on("cam1", "person", cooldown=60, now=150.0) is True
        # cam1 expired at 161 (100 + 60 = 160)
        assert t.is_presence_on("cam1", "person", cooldown=60, now=161.0) is False
        # cam2 detected at 200, still within 60s cooldown at 250
        assert t.is_presence_on("cam2", "person", cooldown=60, now=250.0) is True
        # cam2 expired at 261 (200 + 60 = 260)
        assert t.is_presence_on("cam2", "person", cooldown=60, now=261.0) is False


class TestTrackedKeys:
    def test_returns_all_keys(self):
        t = PresenceTracker()
        t.record_detection("cam1", "person", now=100.0)
        t.record_detection("cam1", "vehicle", now=100.0)
        t.record_detection("cam2", "person", now=100.0)
        assert t.tracked_keys() == {("cam1", "person"), ("cam1", "vehicle"), ("cam2", "person")}


class TestPublishedState:
    def test_set_and_get(self):
        t = PresenceTracker()
        t.set_published_state("cam1", "person", True)
        assert t.get_published_state("cam1", "person") is True
        t.set_published_state("cam1", "person", False)
        assert t.get_published_state("cam1", "person") is False

    def test_none_when_never_set(self):
        t = PresenceTracker()
        assert t.get_published_state("cam1", "person") is None
