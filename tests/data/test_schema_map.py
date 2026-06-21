import pytest

from agastya.data.schema_map import map_source_label
from agastya.schema.classes import name_to_id


def test_aicity_motorbike_maps_to_motorcycle():
    assert map_source_label("aicity", "motorbike") == name_to_id("motorcycle")


def test_aicity_dnohelmet_maps_to_no_helmet():
    assert map_source_label("aicity", "DNoHelmet") == name_to_id("no-helmet")


def test_aicity_dhelmet_maps_to_helmet():
    assert map_source_label("aicity", "DHelmet") == name_to_id("helmet")


def test_idd_autorickshaw_maps():
    assert map_source_label("idd", "autorickshaw") == name_to_id("auto-rickshaw")


def test_unknown_source_label_returns_none():
    assert map_source_label("idd", "animal") is None


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        map_source_label("kitti", "car")
