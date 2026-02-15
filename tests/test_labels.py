# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jeff Culverhouse
from vision2mqtt.mixins.labels import COCO_LABEL_MAP, LABEL_ICONS, LABEL_PARENTS, LabelsMixin


class FakeVision(LabelsMixin):
    def __init__(self, labels):
        self.vision_config = {"labels": labels}


class TestLabelMapping:
    def test_person_maps_to_person(self):
        v = FakeVision(["person"])
        assert v.get_simplified_label("person") == "person"

    def test_car_maps_to_vehicle_via_parent(self):
        v = FakeVision(["vehicle"])
        assert v.get_simplified_label("car") == "vehicle"

    def test_car_maps_to_car_when_specific(self):
        v = FakeVision(["car", "vehicle"])
        assert v.get_simplified_label("car") == "car"

    def test_truck_maps_to_vehicle_via_parent(self):
        v = FakeVision(["vehicle"])
        assert v.get_simplified_label("truck") == "vehicle"

    def test_truck_maps_to_truck_when_specific(self):
        v = FakeVision(["truck"])
        assert v.get_simplified_label("truck") == "truck"

    def test_dog_maps_to_animal_via_parent(self):
        v = FakeVision(["animal"])
        assert v.get_simplified_label("dog") == "animal"

    def test_dog_maps_to_dog_when_specific(self):
        v = FakeVision(["dog", "animal"])
        assert v.get_simplified_label("dog") == "dog"

    def test_bird_maps_to_bird(self):
        v = FakeVision(["bird"])
        assert v.get_simplified_label("bird") == "bird"

    def test_unknown_label_returns_none(self):
        v = FakeVision(["person", "vehicle"])
        assert v.get_simplified_label("chair") is None

    def test_label_not_in_config_returns_none(self):
        v = FakeVision(["person"])
        assert v.get_simplified_label("car") is None

    def test_all_vehicle_labels_fallback(self):
        v = FakeVision(["vehicle"])
        for label in ("car", "truck", "bus", "motorcycle", "bicycle"):
            assert v.get_simplified_label(label) == "vehicle"

    def test_all_animal_labels_fallback(self):
        v = FakeVision(["animal"])
        for label in ("cat", "dog", "horse", "cow", "sheep", "bear", "elephant", "zebra", "giraffe"):
            assert v.get_simplified_label(label) == "animal"

    def test_backpack_maps_to_package(self):
        v = FakeVision(["package"])
        assert v.get_simplified_label("backpack") == "package"

    def test_suitcase_maps_to_package(self):
        v = FakeVision(["package"])
        assert v.get_simplified_label("suitcase") == "package"

    def test_handbag_maps_to_package(self):
        v = FakeVision(["package"])
        assert v.get_simplified_label("handbag") == "package"

    def test_coco_map_has_expected_entries(self):
        assert len(COCO_LABEL_MAP) == 19

    def test_label_parents_has_expected_entries(self):
        assert len(LABEL_PARENTS) == 14

    def test_label_icons_has_expected_entries(self):
        assert len(LABEL_ICONS) == 16
        for key in (
            "person",
            "vehicle",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "animal",
            "cat",
            "dog",
            "bird",
            "package",
            "dog_walker",
            "group",
            "cyclist",
            "package_carrier",
        ):
            assert key in LABEL_ICONS
