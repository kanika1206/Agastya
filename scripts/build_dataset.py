from __future__ import annotations

import argparse
from pathlib import Path

from agastya.data.data_yaml import write_data_yaml
from agastya.data.manifest import build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble unified AGASTYA YOLO dataset")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_images(raw_root: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for source in ("idd", "aicity", "anpr"):
        source_dir = raw_root / source
        if not source_dir.exists():
            continue
        for image_path in source_dir.rglob("*.jpg"):
            entries.append((source, str(image_path)))
    return entries


def main() -> None:
    args = parse_args()
    entries = discover_images(args.raw_root)
    manifest = build_manifest(entries, val_fraction=args.val_fraction)
    print(f"discovered {len(manifest)} images across sources")
    if args.dry_run:
        print("dry run: no files written")
        return
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_data_yaml(
        args.out_root / "data.yaml",
        root=args.out_root,
        train_dir="images/train",
        val_dir="images/val",
    )
    print(f"wrote {args.out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
