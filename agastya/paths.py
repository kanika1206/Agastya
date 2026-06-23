from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT_ENV = "AGASTYA_EVIDENCE_ROOT"


def evidence_root() -> Path:
    return Path(os.environ.get(EVIDENCE_ROOT_ENV, str(REPO_ROOT)))


def resolve_evidence_path(image_path: str | None) -> str | None:
    if not image_path:
        return None
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    return str(evidence_root() / path)


def to_relative(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_absolute():
        return image_path
    try:
        return str(path.relative_to(evidence_root()))
    except ValueError:
        return image_path
