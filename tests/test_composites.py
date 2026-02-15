# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from unittest.mock import MagicMock

from vision2mqtt.mixins.composites import CompositesMixin, _bbox_gap
from vision2mqtt.models.events import DetectedObject, VisionResult


class FakeComposites(CompositesMixin):
    def __init__(self, composites):
        self.vision_config = {"composites": composites}
        self.logger = MagicMock()


def _det(raw_label: str, bbox: list[float]) -> DetectedObject:
    return DetectedObject(label=raw_label, raw_label=raw_label, confidence=0.9, bbox=bbox)


class TestBboxGap:
    def test_overlapping_boxes_zero_gap(self):
        assert _bbox_gap([0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7]) == 0.0

    def test_adjacent_boxes_zero_gap(self):
        assert _bbox_gap([0.0, 0.0, 0.5, 0.5], [0.5, 0.0, 1.0, 0.5]) == 0.0

    def test_separated_horizontally(self):
        gap = _bbox_gap([0.0, 0.0, 0.3, 0.5], [0.5, 0.0, 0.8, 0.5])
        assert abs(gap - 0.2) < 0.001

    def test_separated_diagonally(self):
        gap = _bbox_gap([0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3])
        expected = (0.1**2 + 0.1**2) ** 0.5
        assert abs(gap - expected) < 0.001


class TestDogWalker:
    def test_person_near_dog(self):
        fc = FakeComposites(["dog_walker"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.1, 0.1, 0.3, 0.8]),
                _det("dog", [0.3, 0.5, 0.5, 0.8]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["dog_walker"] is True

    def test_person_far_from_dog(self):
        fc = FakeComposites(["dog_walker"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.0, 0.0, 0.1, 0.3]),
                _det("dog", [0.7, 0.7, 0.9, 0.9]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["dog_walker"] is False

    def test_no_dog(self):
        fc = FakeComposites(["dog_walker"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.1, 0.1, 0.3, 0.8]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["dog_walker"] is False


class TestGroup:
    def test_two_people(self):
        fc = FakeComposites(["group"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.1, 0.1, 0.3, 0.8]),
                _det("person", [0.4, 0.1, 0.6, 0.8]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["group"] is True

    def test_single_person(self):
        fc = FakeComposites(["group"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.1, 0.1, 0.3, 0.8]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["group"] is False

    def test_no_people(self):
        fc = FakeComposites(["group"])
        result = VisionResult(all_detections=[])
        composites = fc.compute_composites(result)
        assert composites["group"] is False


class TestCyclist:
    def test_person_near_bicycle(self):
        fc = FakeComposites(["cyclist"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.3, 0.1, 0.5, 0.7]),
                _det("bicycle", [0.3, 0.5, 0.6, 0.9]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["cyclist"] is True

    def test_person_far_from_bicycle(self):
        fc = FakeComposites(["cyclist"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.0, 0.0, 0.1, 0.3]),
                _det("bicycle", [0.7, 0.7, 0.9, 0.9]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["cyclist"] is False


class TestStudent:
    def test_person_near_backpack(self):
        fc = FakeComposites(["student"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.2, 0.1, 0.4, 0.8]),
                _det("backpack", [0.35, 0.2, 0.5, 0.5]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["student"] is True

    def test_person_alone(self):
        fc = FakeComposites(["student"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.2, 0.1, 0.4, 0.8]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["student"] is False


class TestPackageCarrier:
    def test_person_near_suitcase(self):
        fc = FakeComposites(["package_carrier"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.2, 0.1, 0.4, 0.8]),
                _det("suitcase", [0.4, 0.5, 0.6, 0.8]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["package_carrier"] is True

    def test_no_package(self):
        fc = FakeComposites(["package_carrier"])
        result = VisionResult(
            all_detections=[
                _det("person", [0.2, 0.1, 0.4, 0.8]),
            ]
        )
        composites = fc.compute_composites(result)
        assert composites["package_carrier"] is False


class TestNoComposites:
    def test_empty_composites_returns_empty(self):
        fc = FakeComposites([])
        result = VisionResult(all_detections=[_det("person", [0.1, 0.1, 0.3, 0.8])])
        assert fc.compute_composites(result) == {}

    def test_none_composites_returns_empty(self):
        fc = FakeComposites(None)
        result = VisionResult(all_detections=[_det("person", [0.1, 0.1, 0.3, 0.8])])
        assert fc.compute_composites(result) == {}
