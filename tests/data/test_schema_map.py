import pytest

from agastya.data.schema_map import map_source_label
from agastya.schema.classes import name_to_id


def test_roboflow_with_helmet_maps_to_helmet():
    assert map_source_label("roboflow", "with_helmet") == name_to_id("helmet")


def test_roboflow_without_helmet_maps_to_no_helmet():
    assert map_source_label("roboflow", "without_helmet") == name_to_id("no-helmet")


def test_roboflow_number_plate_maps_to_license_plate():
    assert map_source_label("roboflow", "number_plate") == name_to_id("license-plate")


def test_roboflow_triple_riding_maps():
    assert map_source_label("roboflow", "Triple_riding") == name_to_id("triple-riding")


def test_unknown_source_label_returns_none():
    assert map_source_label("roboflow", "animal") is None


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        map_source_label("kitti", "car")
