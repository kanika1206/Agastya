from __future__ import annotations

import os
from typing import Callable

import numpy as np

from agastya.stages.restore.errors import RestorerUnavailable


class NafnetRestorer:
    def __init__(self, weights: str | None, device: str = "cpu") -> None:
        self.weights = weights
        self.device = device
        self._model: Callable[[np.ndarray], np.ndarray] | None = None

    def _load_model(self) -> Callable[[np.ndarray], np.ndarray]:
        if not self.weights or not os.path.exists(self.weights):
            raise RestorerUnavailable(f"NAFNet weights not found: {self.weights}")
        try:
            import torch
        except ImportError as exc:
            raise RestorerUnavailable("torch is not installed") from exc

        import cv2

        from agastya.stages.restore._nafnet_arch import build_width32_gopro

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        ckpt = torch.load(self.weights, map_location="cpu")
        state = ckpt.get("params", ckpt) if isinstance(ckpt, dict) else ckpt
        model = build_width32_gopro()
        model.load_state_dict(state, strict=True)
        model.eval().to(device)

        def run(arr: np.ndarray) -> np.ndarray:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tensor)
            out = pred.clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
            return cv2.cvtColor((out * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        return run

    def restore(self, pixels: bytes) -> bytes:
        import cv2

        if self._model is None:
            self._model = self._load_model()
        arr = cv2.imdecode(np.frombuffer(pixels, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("failed to decode image bytes")
        out = self._model(arr)
        ok, buf = cv2.imencode(".png", out)
        if not ok:
            raise ValueError("failed to encode restored image")
        return buf.tobytes()
