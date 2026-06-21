from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from agastya.stages.associate.errors import AssociatorUnavailable
from agastya.stages.associate.overlap import count_overlapping_masks
from agastya.stages.associate.rules import TRIPLE_RIDING_MIN
from agastya.types import BBox, Detection


class Sam2Associator:
    def __init__(
        self,
        model: str,
        weights: str | None = None,
        device: str = "cpu",
        min_overlap: float = 0.1,
    ) -> None:
        self.model = model
        self.weights = weights
        self.device = device
        self.min_overlap = min_overlap
        self._predictor: object | None = None

    def _load_predictor(self) -> object:
        try:
            from ultralytics import SAM
        except ImportError as exc:
            raise AssociatorUnavailable("ultralytics not installed") from exc
        try:
            return SAM(self.weights or self.model)
        except Exception as exc:
            raise AssociatorUnavailable(f"failed to load SAM weights: {exc}") from exc

    def _ensure_predictor(self) -> None:
        if self._predictor is None:
            self._predictor = self._load_predictor()

    def _segment(self, pixels: bytes, boxes: Sequence[BBox]) -> list[np.ndarray]:
        import cv2

        self._ensure_predictor()
        image = cv2.imdecode(np.frombuffer(pixels, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise AssociatorUnavailable("could not decode image bytes")
        bbox_list = [[box.x1, box.y1, box.x2, box.y2] for box in boxes]
        results = self._predictor(image, bboxes=bbox_list, device=self.device, verbose=False)
        masks = results[0].masks
        if masks is None:
            return []
        data = masks.data.cpu().numpy().astype(bool)
        return [data[i] for i in range(data.shape[0])]

    def is_triple_riding(
        self, motorcycle: BBox, persons: Sequence[Detection], pixels: bytes
    ) -> bool:
        if not persons:
            return False
        boxes = [motorcycle, *(person.box for person in persons)]
        masks = self._segment(pixels, boxes)
        if not masks:
            return False
        moto_mask, person_masks = masks[0], masks[1:]
        count = count_overlapping_masks(moto_mask, person_masks, self.min_overlap)
        return count >= TRIPLE_RIDING_MIN
