from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AGASTYA YOLO26 baseline")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo26-p2.yaml")
    parser.add_argument("--weights", default="yolo26m.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm:
        print("dry run: re-run with --confirm to start training")
        print(f"  data={args.data} model={args.model} weights={args.weights}")
        print(f"  imgsz={args.imgsz} epochs={args.epochs} batch={args.batch} device={args.device}")
        return
    from ultralytics import YOLO

    model = YOLO(args.model).load(args.weights)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        amp=True,
    )


if __name__ == "__main__":
    main()
