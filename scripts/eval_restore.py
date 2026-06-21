from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from agastya.eval.degrade import motion_blur
from agastya.eval.prf import precision_recall_f1
from agastya.eval.scoring import accumulate, f1_of, new_counts
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
        description="Compare detection P/R/F1 on clean vs degraded vs NAFNet-restored frames"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--nafnet-weights", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kernel", type=int, default=15)
    parser.add_argument("--angle", type=float, default=30.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--person-only", action="store_true")
    return parser.parse_args()


def sample_images(
    val_dir: Path, names: list[str], samples: int, seed: int, person_only: bool
) -> list[Path]:
    images = sorted(
        p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if person_only and "person" in names:
        images = [p for p in images if any(t.label == "person" for t in load_truths(label_path_for(p), names))]
    if not images:
        raise SystemExit(f"no val images under {val_dir}")
    rng = random.Random(seed)
    if samples < len(images):
        images = rng.sample(images, samples)
    return sorted(images)


def predict_dets(model, image: np.ndarray, names: list[str], imgsz: int, conf: float):
    result = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
    return predictions_for(result, names)


def print_table(title: str, counts: dict[str, list[int]], names: list[str]) -> None:
    total = [0, 0, 0]
    print(f"\n== {title} ==")
    print(f"{'class':<14} {'P':>7} {'R':>7} {'F1':>7}   {'TP':>6} {'FP':>6} {'FN':>6}")
    for name in names:
        tp, fp, fn = counts[name]
        total[0] += tp
        total[1] += fp
        total[2] += fn
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        print(f"{name:<14} {p:>7.3f} {r:>7.3f} {f1:>7.3f}   {tp:>6} {fp:>6} {fn:>6}")
    p, r, f1 = precision_recall_f1(*total)
    print(
        f"{'overall':<14} {p:>7.3f} {r:>7.3f} {f1:>7.3f}   "
        f"{total[0]:>6} {total[1]:>6} {total[2]:>6}"
    )


def main() -> None:
    args = parse_args()
    val_dir, names = load_data_yaml(args.data)
    images = sample_images(val_dir, names, args.samples, args.seed, args.person_only)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    restorer = NafnetRestorer(str(args.nafnet_weights), device=args.device)

    clean = new_counts(names)
    degraded = new_counts(names)
    restored = new_counts(names)

    for idx, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        if original is None:
            continue
        truths = load_truths(label_path_for(image_path), names)
        blurred = motion_blur(original, args.kernel, args.angle)
        encoded = cv2.imencode(".png", blurred)[1].tobytes()
        deblurred = cv2.imdecode(
            np.frombuffer(restorer.restore(encoded), np.uint8), cv2.IMREAD_COLOR
        )

        accumulate(clean, predict_dets(model, original, names, args.imgsz, args.conf), truths, names, args.iou)
        accumulate(degraded, predict_dets(model, blurred, names, args.imgsz, args.conf), truths, names, args.iou)
        accumulate(restored, predict_dets(model, deblurred, names, args.imgsz, args.conf), truths, names, args.iou)
        if idx % 25 == 0:
            print(f"  processed {idx}/{len(images)}", flush=True)

    print(f"\nimages: {len(images)}  motion-blur kernel={args.kernel} angle={args.angle}")
    print_table("clean (control)", clean, names)
    print_table("degraded (passthrough)", degraded, names)
    print_table("degraded -> NAFNet", restored, names)

    print("\n== F1 delta (NAFNet - degraded) ==")
    for name in names:
        lift = f1_of(restored, name) - f1_of(degraded, name)
        recovered = f1_of(restored, name) - f1_of(clean, name)
        print(f"{name:<14} deblur_lift={lift:+.3f}  vs_clean={recovered:+.3f}")


if __name__ == "__main__":
    main()
