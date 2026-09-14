# SPDX-License-Identifier: MIT
"""Persistence of today's image counter across a restart."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from vision2mqtt.base import Base


class FakeState(Base):
    def __init__(self, tmp_path):
        self.logger = MagicMock()
        self.config = {"config_path": str(tmp_path)}
        self.images_annotated = 0
        self.images_annotated_date = datetime.now(UTC).astimezone()


class TestStatePersistence:
    def test_round_trip_preserves_today(self, tmp_path):
        a = FakeState(tmp_path)
        a.images_annotated = 412
        a.save_state()

        b = FakeState(tmp_path)
        b.restore_state()
        assert b.images_annotated == 412, "a restart mid-day must not zero today's count"

    def test_stale_day_is_discarded(self, tmp_path):
        """Restarting after midnight must not resurrect yesterday's total."""
        a = FakeState(tmp_path)
        a.images_annotated = 412
        a.images_annotated_date = datetime.now(UTC).astimezone() - timedelta(days=1)
        a.save_state()

        b = FakeState(tmp_path)
        b.restore_state()
        assert b.images_annotated == 0

    def test_missing_file_is_not_an_error(self, tmp_path):
        b = FakeState(tmp_path)
        b.restore_state()
        assert b.images_annotated == 0

    def test_corrupt_file_starts_fresh_without_raising(self, tmp_path):
        (tmp_path / "vision2mqtt.dat").write_text("{not json", encoding="utf-8")
        b = FakeState(tmp_path)
        b.restore_state()
        assert b.images_annotated == 0
        b.logger.warning.assert_called_once()

    def test_write_is_atomic_leaving_no_temp_files(self, tmp_path):
        a = FakeState(tmp_path)
        a.images_annotated = 7
        a.save_state()
        assert json.loads((tmp_path / "vision2mqtt.dat").read_text())["images_annotated"] == 7
        assert not [p for p in Path(tmp_path).iterdir() if p.name.startswith(".")], "temp file left behind"
