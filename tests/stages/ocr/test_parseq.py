from agastya.stages.ocr.parseq import ParseqOcr


class _FakeParseq(ParseqOcr):
    def __init__(self, text: str, confidence: float, min_confidence: float = 0.5) -> None:
        super().__init__(min_confidence=min_confidence)
        self._text = text
        self._confidence = confidence

    def _recognize(self, pixels: bytes) -> tuple[str, float]:
        return self._text, self._confidence


def test_parseq_applies_guard_to_valid_reading():
    reading = _FakeParseq("KA01AB1234", 0.9).read(b"pixels")
    assert reading.text == "KA01AB1234"
    assert reading.abstained is False


def test_parseq_abstains_on_invalid_format():
    reading = _FakeParseq("garbage", 0.99).read(b"pixels")
    assert reading.abstained is True


def test_parseq_abstains_below_min_confidence():
    reading = _FakeParseq("KA01AB1234", 0.2).read(b"pixels")
    assert reading.abstained is True
