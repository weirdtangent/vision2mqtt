# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from __future__ import annotations

import time
from collections import deque


class PresenceTracker:
    """Tracks presence cooldowns and detection frequency per (camera, label) key.

    Pure data class — not a mixin. Owns all presence/frequency state so that
    publish and heartbeat code can query it without duplicating logic.
    """

    def __init__(self) -> None:
        self._last_seen: dict[tuple[str, str], float] = {}
        self._detection_times: dict[tuple[str, str], deque[float]] = {}
        self._presence_state: dict[tuple[str, str], bool] = {}

    def record_detection(self, camera_id: str, label: str, *, now: float | None = None) -> None:
        """Record that *label* was detected on *camera_id* at monotonic time *now*."""
        now = now if now is not None else time.monotonic()
        key = (camera_id, label)
        self._last_seen[key] = now
        if key not in self._detection_times:
            self._detection_times[key] = deque()
        self._detection_times[key].append(now)

    def is_presence_on(self, camera_id: str, label: str, cooldown: int, *, now: float | None = None) -> bool:
        """Return True if *label* was seen within *cooldown* seconds on *camera_id*."""
        now = now if now is not None else time.monotonic()
        key = (camera_id, label)
        last = self._last_seen.get(key)
        if last is None:
            return False
        return (now - last) < cooldown

    def get_frequency(self, camera_id: str, label: str, window: int, *, now: float | None = None) -> float:
        """Return detections-per-hour for *label* on *camera_id* within the rolling *window* (seconds)."""
        now = now if now is not None else time.monotonic()
        key = (camera_id, label)
        dq = self._detection_times.get(key)
        if not dq:
            return 0.0
        # Prune entries outside the window
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if not dq:
            return 0.0
        # Scale count to per-hour rate
        return len(dq) * (3600.0 / window)

    def get_expired_presence_keys(self, cooldown: int, *, now: float | None = None) -> list[tuple[str, str]]:
        """Return (camera_id, label) keys whose presence has expired but was last published as ON."""
        now = now if now is not None else time.monotonic()
        expired: list[tuple[str, str]] = []
        for key, last in self._last_seen.items():
            if self._presence_state.get(key, False) and (now - last) >= cooldown:
                expired.append(key)
        return expired

    def set_published_state(self, camera_id: str, label: str, state: bool) -> None:
        """Record the last-published presence state to avoid redundant OFF publishes."""
        self._presence_state[(camera_id, label)] = state

    def get_published_state(self, camera_id: str, label: str) -> bool | None:
        """Return the last-published presence state, or None if never published."""
        return self._presence_state.get((camera_id, label))

    def tracked_keys(self) -> set[tuple[str, str]]:
        """Return all (camera_id, label) keys that have been seen at least once."""
        return set(self._last_seen.keys())
