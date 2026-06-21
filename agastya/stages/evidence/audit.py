from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

GENESIS_PREV_HASH = "0" * 64

_FIELD_SEP = "\x1f"


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    event: str
    payload_hash: str
    prev_hash: str
    entry_hash: str


def _payload_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _entry_hash(seq: int, event: str, payload_hash: str, prev_hash: str) -> str:
    material = _FIELD_SEP.join((str(seq), event, payload_hash, prev_hash))
    return hashlib.sha256(material.encode()).hexdigest()


def append(entries: Sequence[AuditEntry], event: str, payload: dict) -> tuple[AuditEntry, ...]:
    seq = len(entries)
    prev_hash = entries[-1].entry_hash if entries else GENESIS_PREV_HASH
    payload_hash = _payload_hash(payload)
    entry = AuditEntry(
        seq=seq,
        event=event,
        payload_hash=payload_hash,
        prev_hash=prev_hash,
        entry_hash=_entry_hash(seq, event, payload_hash, prev_hash),
    )
    return (*entries, entry)


def verify_chain(entries: Sequence[AuditEntry]) -> bool:
    prev_hash = GENESIS_PREV_HASH
    for index, entry in enumerate(entries):
        if entry.seq != index or entry.prev_hash != prev_hash:
            return False
        recomputed = _entry_hash(entry.seq, entry.event, entry.payload_hash, entry.prev_hash)
        if entry.entry_hash != recomputed:
            return False
        prev_hash = entry.entry_hash
    return True
