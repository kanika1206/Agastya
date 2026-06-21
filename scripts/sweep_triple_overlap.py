from __future__ import annotations

import argparse
from pathlib import Path

from agastya.config import PipelineConfig
from agastya.eval.e2e import ViolationCounts, gt_violations
from agastya.eval.yolo_data import label_path_for, load_data_yaml, load_truths
from agastya.stages.associate.rules import is_triple_riding
from agastya.stages.detect.factory import build_detector
from agastya.types import BBox, Detection

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triple-riding overlap-threshold sweep")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--gt-overlap", type=float, default=0.1)
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=0.6)
    parser.add_argument("--step", type=float, default=0.05)
    return parser.parse_args()


def collect(args: argparse.Namespace) -> list[tuple[list[BBox], list[Detection], bool]]:
    config = PipelineConfig(
        detect_backend="yolo",
        detector_weights=str(args.weights),
        detect_conf=args.conf,
        restore_device=args.device,
    )
    detector = build_detector(config)
    val_dir, names = load_data_yaml(args.data)
    images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    images = images[: args.samples]

    samples: list[tuple[list[BBox], list[Detection], bool]] = []
    for image_path in images:
        detections = detector.detect(image_path.read_bytes())
        motos = [d.box for d in detections if d.label == "motorcycle"]
        persons = [d for d in detections if d.label == "person"]
        truths = load_truths(label_path_for(image_path), names)
        gt_triple = "triple-riding" in gt_violations(truths, args.gt_overlap)
        samples.append((motos, persons, gt_triple))
    return samples


def pred_triple(motos: list[BBox], persons: list[Detection], overlap: float) -> bool:
    return any(is_triple_riding(moto, persons, overlap) for moto in motos)


def thresholds(low: float, high: float, step: float) -> list[float]:
    values: list[float] = []
    t = low
    while t <= high + 1e-9:
        values.append(round(t, 2))
        t += step
    return values


def main() -> int:
    args = parse_args()
    samples = collect(args)
    n_pos = sum(1 for _, _, gt in samples if gt)
    print(f"images: {len(samples)}  triple-riding GT positives: {n_pos}")
    print(f"GT overlap held fixed at {args.gt_overlap:.2f}\n")
    print(f"{'overlap':>7} {'P':>7} {'R':>7} {'F1':>7}   {'TP':>4} {'FP':>4} {'FN':>4}")

    best = None
    for overlap in thresholds(args.low, args.high, args.step):
        counts = ViolationCounts()
        for motos, persons, gt_present in samples:
            counts.update(pred_triple(motos, persons, overlap), gt_present)
        precision, recall, f1 = counts.prf()
        print(
            f"{overlap:>7.2f} {precision:>7.3f} {recall:>7.3f} {f1:>7.3f}   "
            f"{counts.tp:>4} {counts.fp:>4} {counts.fn:>4}"
        )
        if best is None or f1 > best[1]:
            best = (overlap, f1, precision, recall)

    if best is not None:
        print(
            f"\nbest F1 @ overlap {best[0]:.2f}: F1 {best[1]:.3f} "
            f"(P {best[2]:.3f} R {best[3]:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
