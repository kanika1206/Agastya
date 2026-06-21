from __future__ import annotations

import os
from typing import Callable

import numpy as np

from agastya.stages.gate.errors import GateUnavailable


class ArniqaGate:
    def __init__(self, weights: str | None, device: str = "cpu") -> None:
        self.weights = weights
        self.device = device
        self._metric: Callable[[np.ndarray], float] | None = None

    def _load_metric(self) -> Callable[[np.ndarray], float]:
        if not self.weights or not os.path.exists(self.weights):
            raise GateUnavailable(f"ARNIQA weights not found: {self.weights}")
        try:
            import torch
        except ImportError as exc:
            raise GateUnavailable("torch is not installed") from exc
        try:
            import pyiqa
        except ImportError as exc:
            raise GateUnavailable("pyiqa is not installed") from exc

        import cv2

        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        metric = pyiqa.create_metric("arniqa", device=device)
        metric.eval()

        def run(arr: np.ndarray) -> float:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                value = float(metric(tensor).item())
            return min(1.0, max(0.0, value))

        return run

    def score_image(self, pixels: bytes) -> float:
        if self._metric is None:
            self._metric = self._load_metric()
        import cv2

        arr = cv2.imdecode(np.frombuffer(pixels, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise GateUnavailable("could not decode image for quality gating")
        return self._metric(arr)
