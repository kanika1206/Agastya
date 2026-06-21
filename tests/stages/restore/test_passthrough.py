from agastya.stages.restore.passthrough import PassthroughRestorer


def test_passthrough_returns_input_unchanged():
    restorer = PassthroughRestorer()
    data = b"\x89PNG-bytes-here"
    assert restorer.restore(data) == data


def test_passthrough_returns_same_object_identity():
    restorer = PassthroughRestorer()
    data = b"raw"
    assert restorer.restore(data) is data
