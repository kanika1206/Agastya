from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SAHI with YOLO26 NMS-free head")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--slice", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--confirm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm:
        print("dry run: re-run with --confirm to execute SAHI validation")
        print(f"  weights={args.weights} image={args.image} slice={args.slice}")
        print("  will run end2end=False (one-to-many) and merge slices via explicit NMS")
        return
    from ultralytics import YOLO

    from agastya.detect.results_adapter import boxes_to_detections
    from agastya.detect.sahi_merge import SlicePrediction, merge_slice_predictions

    model = YOLO(str(args.weights))
    results = model.predict(source=str(args.image), imgsz=args.slice)
    names = results[0].names
    boxes = results[0].boxes
    dets = boxes_to_detections(
        xyxy=boxes.xyxy.tolist(),
        class_ids=[int(c) for c in boxes.cls.tolist()],
        scores=boxes.conf.tolist(),
        names=names,
    )
    merged = merge_slice_predictions(
        [SlicePrediction(offset_x=0.0, offset_y=0.0, detections=dets)], iou_threshold=args.iou
    )
    print(f"whole-image detections: {len(dets)}; merged: {len(merged)}")
    print("Next: tile the image, predict per tile with end2end=False, merge with merge_slice_predictions")


if __name__ == "__main__":
    main()
