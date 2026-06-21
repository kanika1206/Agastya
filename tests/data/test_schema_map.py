import pytest

from agastya.data.schema_map import map_source_label
from agastya.schema.classes import name_to_id


def test_triple_with_helmet_maps_to_helmet():
    assert map_source_label("triple", "with_helmet") == name_to_id("helmet")


def test_triple_without_helmet_maps_to_no_helmet():
    assert map_source_label("triple", "without_helmet") == name_to_id("no-helmet")


def test_triple_number_plate_maps_to_license_plate():
    assert map_source_label("triple", "number_plate") == name_to_id("license-plate")


def test_triple_triple_riding_maps():
    assert map_source_label("triple", "Triple_riding") == name_to_id("triple-riding")


def test_triple_motorcycle_maps():
    assert map_source_label("triple", "motorcycle") == name_to_id("motorcycle")


def test_safety_bike_maps_to_motorcycle():
    assert map_source_label("safety", "bike") == name_to_id("motorcycle")


def test_safety_number_plate_maps_to_license_plate():
    assert map_source_label("safety", "number-plate") == name_to_id("license-plate")


def test_safety_helmet_maps():
    assert map_source_label("safety", "helmet") == name_to_id("helmet")


def test_safety_no_helmet_maps():
    assert map_source_label("safety", "no-helmet") == name_to_id("no-helmet")


def test_unknown_source_label_returns_none():
    assert map_source_label("triple", "animal") is None


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        map_source_label("kitti", "car")
