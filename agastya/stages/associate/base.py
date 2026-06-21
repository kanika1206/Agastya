from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agastya.types import BBox, Detection


class Associator(Protocol):
    def is_triple_riding(
        self, motorcycle: BBox, persons: Sequence[Detection], pixels: bytes
    ) -> bool: ...
