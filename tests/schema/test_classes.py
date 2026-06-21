import pytest

from agastya.schema.classes import CLASSES, name_to_id, id_to_name, validate_class


def test_four_classes_in_fixed_order():
    assert CLASSES == (
        "helmet",
        "no-helmet",
        "license-plate",
        "triple-riding",
    )


def test_name_id_roundtrip():
    for idx, name in enumerate(CLASSES):
        assert name_to_id(name) == idx
        assert id_to_name(idx) == name


def test_validate_rejects_unknown():
    with pytest.raises(ValueError):
        validate_class("scooter")
