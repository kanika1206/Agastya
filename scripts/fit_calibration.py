from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

from agastya.eval.prf import label_predictions
from agastya.eval.yolo_data import (
    label_path_for,
    load_data_yaml,
    load_truths,
    predictions_for,
)
from agastya.verify.calibration import fit_calibrator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit production Calibrator (temperature + conformal qhat) from val predictions"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models/calibration.json"))
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_dir, names = load_data_yaml(args.data)
    images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    rng = random.Random(args.seed)
    if args.samples < len(images):
        images = rng.sample(images, args.samples)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    confidences: list[float] = []
    labels: list[int] = []
    for idx, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        truths = load_truths(label_path_for(image_path), names)
        result = model.predict(image, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        preds = predictions_for(result, names)
        for det, correct in label_predictions(preds, truths, args.iou):
            confidences.append(det.score)
            labels.append(int(correct))
        if idx % 50 == 0:
            print(f"  processed {idx}/{len(images)}  predictions={len(confidences)}", flush=True)

    if not confidences:
        raise SystemExit("no predictions collected; check weights/data/conf threshold")

    calibrator = fit_calibrator(confidences, labels, args.alpha)
    payload = {
        "temperature": calibrator.temperature,
        "qhat": calibrator.qhat,
        "alpha": args.alpha,
        "n_predictions": len(confidences),
        "n_correct": sum(labels),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        f"\npredictions: {len(confidences)}  correct: {sum(labels)}  alpha: {args.alpha}\n"
        f"temperature: {calibrator.temperature:.4f}  qhat: {calibrator.qhat:.4f}\n"
        f"wrote {args.out}"
    )


if __name__ == "__main__":
    main()
