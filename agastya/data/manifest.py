from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

_SPLIT_RESOLUTION = 10_000


@dataclass(frozen=True)
class DatasetItem:
    source: str
    image_path: str
    split: str


def assign_split(image_id: str, val_fraction: float) -> str:
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be in [0, 1]")
    digest = hashlib.sha256(image_id.encode()).hexdigest()
    bucket = int(digest, 16) % _SPLIT_RESOLUTION
    return "val" if bucket < val_fraction * _SPLIT_RESOLUTION else "train"


def build_manifest(
    entries: Iterable[tuple[str, str]], val_fraction: float
) -> tuple[DatasetItem, ...]:
    items: list[DatasetItem] = []
    for source, image_path in entries:
        split = assign_split(image_path, val_fraction)
        items.append(DatasetItem(source=source, image_path=image_path, split=split))
    return tuple(items)
