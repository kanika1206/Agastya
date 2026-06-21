from __future__ import annotations

from collections.abc import Sequence

from agastya.stages.associate.rules import is_triple_riding
from agastya.types import BBox, Detection


class BoxOverlapAssociator:
    def __init__(self, min_overlap: float) -> None:
        self.min_overlap = min_overlap

    def is_triple_riding(
        self, motorcycle: BBox, persons: Sequence[Detection], pixels: bytes
    ) -> bool:
        return is_triple_riding(motorcycle, persons, self.min_overlap)
