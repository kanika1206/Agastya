from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from agastya.eval.degrade import motion_blur
from agastya.eval.scoring import accumulate, f1_of, new_counts, overall_f1
from agastya.eval.yolo_data import (
    label_path_for,
    load_data_yaml,
    load_truths,
    predictions_for,
)
from agastya.stages.restore.nafnet import NafnetRestorer

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep motion-blur angles; report degraded vs NAFNet F1 robustness"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--nafnet-weights", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kernel", type=int, default=15)
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 30, 60, 90, 120, 150])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def sample_images(val_dir: Path, samples: int, seed: int) -> list[Path]:
    images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no val images under {val_dir}")
    rng = random.Random(seed)
    if samples < len(images):
        images = rng.sample(images, samples)
    return sorted(images)


def predict_dets(model, image: np.ndarray, names: list[str], imgsz: int, conf: float):
    result = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
    return predictions_for(result, names)


def main() -> None:
    args = parse_args()
    val_dir, names = load_data_yaml(args.data)
    images = sample_images(val_dir, args.samples, args.seed)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    restorer = NafnetRestorer(str(args.nafnet_weights), device=args.device)

    degraded = {a: new_counts(names) for a in args.angles}
    restored = {a: new_counts(names) for a in args.angles}

    for idx, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        if original is None:
            continue
        truths = load_truths(label_path_for(image_path), names)
        for angle in args.angles:
            blurred = motion_blur(original, args.kernel, angle)
            encoded = cv2.imencode(".png", blurred)[1].tobytes()
            deblurred = cv2.imdecode(
                np.frombuffer(restorer.restore(encoded), np.uint8), cv2.IMREAD_COLOR
            )
            accumulate(degraded[angle], predict_dets(model, blurred, names, args.imgsz, args.conf), truths, names, args.iou)
            accumulate(restored[angle], predict_dets(model, deblurred, names, args.imgsz, args.conf), truths, names, args.iou)
        if idx % 25 == 0:
            print(f"  processed {idx}/{len(images)}", flush=True)

    print(f"\nimages: {len(images)}  kernel={args.kernel}  samples={args.samples}")
    print(f"{'angle':>6} {'degF1':>8} {'nafF1':>8} {'lift':>8}   {'degPers':>8} {'nafPers':>8}")
    for angle in args.angles:
        d_all = overall_f1(degraded[angle], names)
        n_all = overall_f1(restored[angle], names)
        d_p = f1_of(degraded[angle], "person") if "person" in names else float("nan")
        n_p = f1_of(restored[angle], "person") if "person" in names else float("nan")
        print(f"{angle:>6.0f} {d_all:>8.3f} {n_all:>8.3f} {n_all - d_all:>+8.3f}   {d_p:>8.3f} {n_p:>8.3f}")


if __name__ == "__main__":
    main()
