from __future__ import annotations

import hashlib
from collections.abc import Sequence


def leaf_hash(data: bytes) -> str:
    return hashlib.sha256(b"\x00" + data).hexdigest()


def _pair_hash(left: str, right: str) -> str:
    return hashlib.sha256(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def merkle_root(leaves: Sequence[bytes]) -> str:
    if not leaves:
        raise ValueError("merkle_root requires at least one leaf")
    level = [leaf_hash(item) for item in leaves]
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(_pair_hash(left, right))
        level = nxt
    return level[0]


def verify_leaf(data: bytes, leaves: Sequence[bytes], root: str) -> bool:
    if leaf_hash(data) not in {leaf_hash(item) for item in leaves}:
        return False
    return merkle_root(leaves) == root
