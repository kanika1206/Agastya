from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import yaml

from agastya.data.data_yaml import write_data_yaml
from agastya.data.manifest import build_manifest
from agastya.data.schema_map import map_source_label
from agastya.data.yolo_format import polygon_to_bbox_yolo

_SOURCES = ("triple", "safety", "tvd2", "overload")


def label_path_for_image(image_path: Path) -> Path:
    parts = [p if p != "images" else "labels" for p in image_path.parts]
    return Path(*parts).with_suffix(".txt")


def remap_label_text(text: str, source: str, src_names: Sequence[str]) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        values = [float(p) for p in parts[1:]]
        if len(values) == 4:
            cx, cy, w, h = values
        else:
            cx, cy, w, h = polygon_to_bbox_yolo(values)
        unified_id = map_source_label(source, src_names[class_id])
        if unified_id is None:
            continue
        out_lines.append(f"{unified_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(out_lines)


def load_source_names(raw_root: Path, source: str) -> list[str]:
    data_yaml = raw_root / source / "data.yaml"
    parsed = yaml.safe_load(data_yaml.read_text())
    return list(parsed["names"])


def discover_images(raw_root: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for source in _SOURCES:
        source_dir = raw_root / source
        if not source_dir.exists():
            continue
        for image_path in sorted(source_dir.rglob("*.jpg")):
            entries.append((source, str(image_path)))
    return entries


def assemble_dataset(raw_root: Path, out_root: Path, val_fraction: float) -> int:
    entries = discover_images(raw_root)
    manifest = build_manifest(entries, val_fraction=val_fraction)
    names_cache: dict[str, list[str]] = {}
    for item in manifest:
        src_names = names_cache.setdefault(
            item.source, load_source_names(raw_root, item.source)
        )
        image_src = Path(item.image_path)
        stem = f"{item.source}_{image_src.stem}"
        image_dst = out_root / "images" / item.split / f"{stem}.jpg"
        label_dst = out_root / "labels" / item.split / f"{stem}.txt"
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_src, image_dst)
        label_src = label_path_for_image(image_src)
        label_text = label_src.read_text() if label_src.exists() else ""
        label_dst.write_text(remap_label_text(label_text, item.source, src_names))
    out_root.mkdir(parents=True, exist_ok=True)
    write_data_yaml(
        out_root / "data.yaml",
        root=out_root,
        train_dir="images/train",
        val_dir="images/val",
    )
    return len(manifest)
