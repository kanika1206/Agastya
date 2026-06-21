from __future__ import annotations

import argparse
import random
import time
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
from agastya.stages.gate.arniqa import ArniqaGate
from agastya.stages.gate.router import score_to_decision
from agastya.stages.restore.nafnet import NafnetRestorer

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality-gate ablation: gated vs always-on vs never restoration"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--nafnet-weights", type=Path, required=True)
    parser.add_argument("--arniqa-weights", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kernel", type=int, default=15)
    parser.add_argument("--angle", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.36)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def predict_dets(model, image: np.ndarray, names: list[str], imgsz: int, conf: float):
    result = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
    return predictions_for(result, names)


def main() -> None:
    args = parse_args()
    val_dir, names = load_data_yaml(args.data)
    images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    rng = random.Random(args.seed)
    if args.samples < len(images):
        images = rng.sample(images, args.samples)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    restorer = NafnetRestorer(str(args.nafnet_weights), device=args.device)
    gate = ArniqaGate(str(args.arniqa_weights), device=args.device)

    counts = {p: new_counts(names) for p in ("never", "always", "gated")}
    restore_calls = {"never": 0, "always": 0, "gated": 0}
    restore_seconds = {"never": 0.0, "always": 0.0, "gated": 0.0}
    n_degraded = 0

    def restore_bytes(policy: str, encoded: bytes) -> np.ndarray:
        restore_calls[policy] += 1
        start = time.perf_counter()
        out = restorer.restore(encoded)
        restore_seconds[policy] += time.perf_counter() - start
        return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)

    for idx, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        if original is None:
            continue
        truths = load_truths(label_path_for(image_path), names)
        is_degraded = rng.random() < 0.5
        frame = motion_blur(original, args.kernel, args.angle) if is_degraded else original
        n_degraded += int(is_degraded)
        encoded = cv2.imencode(".png", frame)[1].tobytes()

        accumulate(counts["never"], predict_dets(model, frame, names, args.imgsz, args.conf), truths, names, args.iou)

        always_img = restore_bytes("always", encoded)
        accumulate(counts["always"], predict_dets(model, always_img, names, args.imgsz, args.conf), truths, names, args.iou)

        score = gate.score_image(encoded)
        if score_to_decision(score, args.threshold).degraded:
            gated_img = restore_bytes("gated", encoded)
        else:
            gated_img = frame
        accumulate(counts["gated"], predict_dets(model, gated_img, names, args.imgsz, args.conf), truths, names, args.iou)
        if idx % 25 == 0:
            print(f"  processed {idx}/{len(images)}", flush=True)

    n = len([p for p in images])
    print(f"\nimages: {n}  degraded: {n_degraded}  clean: {n - n_degraded}  threshold={args.threshold}")
    print(f"{'policy':<8} {'overallF1':>10} {'personF1':>9} {'restores':>9} {'restore_s':>10}")
    for policy in ("never", "always", "gated"):
        print(
            f"{policy:<8} {overall_f1(counts[policy], names):>10.3f} "
            f"{f1_of(counts[policy], 'person'):>9.3f} "
            f"{restore_calls[policy]:>9} {restore_seconds[policy]:>10.2f}"
        )


if __name__ == "__main__":
    main()
