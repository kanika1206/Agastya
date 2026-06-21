from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import cv2

from agastya.eval.degrade import motion_blur
from agastya.eval.yolo_data import load_data_yaml
from agastya.stages.gate.arniqa import ArniqaGate

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score clean vs motion-blurred val frames with ARNIQA; suggest gate threshold"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--arniqa-weights", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kernel", type=int, default=15)
    parser.add_argument("--angle", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def summary(label: str, scores: list[float]) -> None:
    print(
        f"{label:<8} n={len(scores)} mean={statistics.mean(scores):.4f} "
        f"min={min(scores):.4f} max={max(scores):.4f} median={statistics.median(scores):.4f}"
    )


def accuracy_at(threshold: float, clean: list[float], blur: list[float]) -> float:
    correct = sum(1 for s in clean if s >= threshold) + sum(1 for s in blur if s < threshold)
    return correct / (len(clean) + len(blur))


def main() -> None:
    args = parse_args()
    val_dir, _ = load_data_yaml(args.data)
    images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    rng = random.Random(args.seed)
    if args.samples < len(images):
        images = rng.sample(images, args.samples)

    gate = ArniqaGate(str(args.arniqa_weights), device=args.device)
    clean_scores: list[float] = []
    blur_scores: list[float] = []
    for image_path in images:
        original = cv2.imread(str(image_path))
        if original is None:
            continue
        blurred = motion_blur(original, args.kernel, args.angle)
        clean_scores.append(gate.score_image(cv2.imencode(".png", original)[1].tobytes()))
        blur_scores.append(gate.score_image(cv2.imencode(".png", blurred)[1].tobytes()))

    summary("clean", clean_scores)
    summary("blur", blur_scores)
    midpoint = (statistics.mean(clean_scores) + statistics.mean(blur_scores)) / 2
    candidates = sorted({round(s, 3) for s in clean_scores + blur_scores})
    best_t, best_acc = midpoint, accuracy_at(midpoint, clean_scores, blur_scores)
    for t in candidates:
        acc = accuracy_at(t, clean_scores, blur_scores)
        if acc > best_acc:
            best_t, best_acc = t, acc
    print(f"\nmidpoint_threshold={midpoint:.4f} acc={accuracy_at(midpoint, clean_scores, blur_scores):.3f}")
    print(f"best_threshold={best_t:.4f} acc={best_acc:.3f}")
    print(f"acc_at_0.5={accuracy_at(0.5, clean_scores, blur_scores):.3f}")


if __name__ == "__main__":
    main()
