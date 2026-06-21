from __future__ import annotations

import argparse
from pathlib import Path

from agastya.data.assemble import assemble_dataset, discover_images
from agastya.data.manifest import build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble unified AGASTYA YOLO dataset")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        manifest = build_manifest(discover_images(args.raw_root), val_fraction=args.val_fraction)
        print(f"discovered {len(manifest)} images across sources")
        print("dry run: no files written")
        return
    count = assemble_dataset(args.raw_root, args.out_root, val_fraction=args.val_fraction)
    print(f"assembled {count} images -> {args.out_root}")
    print(f"wrote {args.out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
