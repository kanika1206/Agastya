from __future__ import annotations

from pathlib import Path

import yaml

from agastya.schema.classes import CLASSES


def write_data_yaml(out_path: Path, root: Path, train_dir: str, val_dir: str) -> None:
    payload = {
        "path": str(root),
        "train": train_dir,
        "val": val_dir,
        "nc": len(CLASSES),
        "names": list(CLASSES),
    }
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False))
