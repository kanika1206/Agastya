import numpy as np
import pytest

from agastya.stages.restore.errors import RestorerUnavailable
from agastya.stages.restore.nafnet import NafnetRestorer


def _png_bytes() -> bytes:
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_nafnet_without_weights_raises():
    restorer = NafnetRestorer(weights=None)
    with pytest.raises(RestorerUnavailable):
        restorer.restore(b"anything")


def test_nafnet_missing_weights_file_raises():
    restorer = NafnetRestorer(weights="/no/such/best.pt")
    with pytest.raises(RestorerUnavailable):
        restorer.restore(b"anything")


def test_nafnet_roundtrip_with_mocked_model(monkeypatch):
    pytest.importorskip("cv2")
    restorer = NafnetRestorer(weights="/fake/best.pt")
    monkeypatch.setattr(restorer, "_load_model", lambda: (lambda arr: arr))
    out = restorer.restore(_png_bytes())
    assert isinstance(out, bytes)
    assert len(out) > 0


def test_nafnet_torch_load_and_restore(tmp_path):
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from agastya.stages.restore._nafnet_arch import build_width32_gopro

    model = build_width32_gopro()
    weights = tmp_path / "w.pth"
    torch.save(model.state_dict(), str(weights))
    restorer = NafnetRestorer(weights=str(weights), device="cpu")
    out = restorer.restore(_png_bytes())
    arr = cv2.imdecode(np.frombuffer(out, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert arr.shape == (8, 8, 3)
