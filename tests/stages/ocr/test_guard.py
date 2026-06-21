from agastya.stages.ocr.guard import guard_reading, is_valid_plate, normalize_plate


def test_normalize_uppercases_and_strips_separators():
    assert normalize_plate(" ka 01-ab 1234 ") == "KA01AB1234"


def test_valid_plate_standard_format():
    assert is_valid_plate("KA01AB1234") is True
    assert is_valid_plate("MH12DE7777") is True
    assert is_valid_plate("DL3CAB1234") is True


def test_invalid_plate_wrong_shape():
    assert is_valid_plate("ABCD") is False
    assert is_valid_plate("12AB34CD") is False
    assert is_valid_plate("KA01AB123") is False


def test_guard_accepts_valid_confident_reading():
    reading = guard_reading("KA01AB1234", confidence=0.9, min_confidence=0.5)
    assert reading.text == "KA01AB1234"
    assert reading.abstained is False


def test_guard_abstains_on_low_confidence():
    reading = guard_reading("KA01AB1234", confidence=0.3, min_confidence=0.5)
    assert reading.abstained is True


def test_guard_abstains_on_invalid_format():
    reading = guard_reading("NOTAPLATE", confidence=0.99, min_confidence=0.5)
    assert reading.abstained is True


def test_guard_normalizes_before_validating():
    reading = guard_reading("ka 01 ab 1234", confidence=0.9, min_confidence=0.5)
    assert reading.text == "KA01AB1234"
    assert reading.abstained is False
