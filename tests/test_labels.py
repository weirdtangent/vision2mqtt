# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from vision2mqtt.mixins.labels import COCO_LABEL_MAP, LabelsMixin


class FakeVision(LabelsMixin):
    def __init__(self, labels):
        self.vision_config = {"labels": labels}


class TestLabelMapping:
    def test_person_maps_to_person(self):
        v = FakeVision(["person"])
        assert v.get_simplified_label("person") == "person"

    def test_car_maps_to_vehicle(self):
        v = FakeVision(["vehicle"])
        assert v.get_simplified_label("car") == "vehicle"

    def test_truck_maps_to_vehicle(self):
        v = FakeVision(["vehicle"])
        assert v.get_simplified_label("truck") == "vehicle"

    def test_dog_maps_to_animal(self):
        v = FakeVision(["animal"])
        assert v.get_simplified_label("dog") == "animal"

    def test_bird_maps_to_bird(self):
        v = FakeVision(["bird"])
        assert v.get_simplified_label("bird") == "bird"

    def test_unknown_label_returns_none(self):
        v = FakeVision(["person", "vehicle"])
        assert v.get_simplified_label("chair") is None

    def test_label_not_in_config_returns_none(self):
        v = FakeVision(["person"])  # only person enabled
        assert v.get_simplified_label("car") is None  # car -> vehicle, but vehicle not in labels

    def test_all_vehicle_labels(self):
        v = FakeVision(["vehicle"])
        for label in ("car", "truck", "bus", "motorcycle", "bicycle"):
            assert v.get_simplified_label(label) == "vehicle"

    def test_all_animal_labels(self):
        v = FakeVision(["animal"])
        for label in ("cat", "dog", "horse", "cow", "sheep", "bear", "elephant", "zebra", "giraffe"):
            assert v.get_simplified_label(label) == "animal"

    def test_coco_map_has_expected_entries(self):
        assert len(COCO_LABEL_MAP) == 16
