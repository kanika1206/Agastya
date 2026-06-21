from __future__ import annotations

import argparse
import time
from pathlib import Path

from agastya.config import PipelineConfig
from agastya.eval.e2e import VIOLATION_TYPES, ViolationCounts, accumulate, gt_violations
from agastya.eval.yolo_data import label_path_for, load_data_yaml, load_truths
from agastya.stages.associate.factory import build_associator
from agastya.stages.detect.factory import build_detector
from agastya.stages.gate.factory import build_gate
from agastya.stages.gate.router import score_to_decision
from agastya.stages.ocr.factory import build_ocr
from agastya.stages.restore.factory import build_restorer
from agastya.verify.calibration import Calibrator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGASTYA end-to-end baseline")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--gate-backend", type=str, default="never")
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--restore-backend", type=str, default="passthrough")
    parser.add_argument("--ocr-backend", type=str, default="none")
    parser.add_argument("--associate-backend", type=str, default="box")
    parser.add_argument("--triple-overlap", type=float, default=0.1)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--arniqa-weights", type=Path, default=None)
    parser.add_argument("--nafnet-weights", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        gate_threshold=args.gate_threshold,
        gate_backend=args.gate_backend,
        arniqa_weights=str(args.arniqa_weights) if args.arniqa_weights else None,
        restore_backend=args.restore_backend,
        nafnet_weights=str(args.nafnet_weights) if args.nafnet_weights else None,
        restore_device=args.device,
        detect_backend="yolo",
        detector_weights=str(args.weights),
        detect_conf=args.conf,
        ocr_backend=args.ocr_backend,
        associate_backend=args.associate_backend,
        triple_riding_overlap=args.triple_overlap,
    )


class Timers:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def add(self, stage: str, seconds: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + seconds
        self.calls[stage] = self.calls.get(stage, 0) + 1

    def mean_ms(self, stage: str) -> float:
        calls = self.calls.get(stage, 0)
        return 1000.0 * self.totals[stage] / calls if calls else 0.0


def main() -> int:
    args = parse_args()
    config = build_config(args)
    val_dir, names = load_data_yaml(args.data)
    images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no val images under {val_dir}")
    images = images[: args.samples]

    gate = build_gate(config)
    restorer = build_restorer(config)
    detector = build_detector(config)
    associator = build_associator(config)
    ocr = build_ocr(config)
    calibrator = Calibrator.from_json(str(args.calibration)) if args.calibration else None

    timers = Timers()
    stage_counts = {
        "degraded": 0,
        "restore_invoked": 0,
        "detect_hit": 0,
        "triple_flagged": 0,
        "ocr_abstained": 0,
        "needs_review": 0,
    }
    per_type = {name: ViolationCounts() for name in VIOLATION_TYPES}
    calibrated_confidences: list[float] = []

    for image_path in images:
        pixels = image_path.read_bytes()

        start = time.perf_counter()
        score = gate.score_image(pixels)
        decision = score_to_decision(score, config.gate_threshold)
        timers.add("gate", time.perf_counter() - start)
        if decision.degraded:
            stage_counts["degraded"] += 1
            start = time.perf_counter()
            pixels = restorer.restore(pixels)
            timers.add("restore", time.perf_counter() - start)
            stage_counts["restore_invoked"] += 1

        start = time.perf_counter()
        detections = detector.detect(pixels)
        timers.add("detect", time.perf_counter() - start)
        if detections:
            stage_counts["detect_hit"] += 1

        pred_set: set[str] = set()
        if any(
            d.label == "no-helmet" and d.score >= config.no_helmet_min_conf
            for d in detections
        ):
            pred_set.add("no-helmet")

        persons = [d for d in detections if d.label == "person"]
        start = time.perf_counter()
        for moto in detections:
            if moto.label == "motorcycle" and associator.is_triple_riding(
                moto.box, persons, pixels
            ):
                pred_set.add("triple-riding")
                break
        timers.add("associate", time.perf_counter() - start)
        if "triple-riding" in pred_set:
            stage_counts["triple_flagged"] += 1

        start = time.perf_counter()
        reading = ocr.read(pixels)
        timers.add("ocr", time.perf_counter() - start)
        if reading.abstained:
            stage_counts["ocr_abstained"] += 1

        if calibrator is not None:
            for violation_type in pred_set:
                result = calibrator.evaluate(0.9, violation_type)
                calibrated_confidences.append(result.confidence)
                if result.needs_review:
                    stage_counts["needs_review"] += 1

        truths = load_truths(label_path_for(image_path), names)
        accumulate(per_type, pred_set, gt_violations(truths, config.triple_riding_overlap))

    _report(len(images), per_type, timers, stage_counts, calibrated_confidences)
    return 0


def _report(
    n_images: int,
    per_type: dict[str, ViolationCounts],
    timers: Timers,
    stage_counts: dict[str, int],
    calibrated_confidences: list[float],
) -> None:
    print(f"images: {n_images}")
    print("\nviolation metrics (image-level presence):")
    print(f"{'type':<14} {'P':>7} {'R':>7} {'F1':>7}   {'TP':>5} {'FP':>5} {'FN':>5}")
    for name, counts in per_type.items():
        precision, recall, f1 = counts.prf()
        print(
            f"{name:<14} {precision:>7.3f} {recall:>7.3f} {f1:>7.3f}   "
            f"{counts.tp:>5} {counts.fp:>5} {counts.fn:>5}"
        )
    print("\nper-stage mean latency (ms):")
    for stage in ("gate", "restore", "detect", "associate", "ocr"):
        if timers.calls.get(stage):
            print(f"  {stage:<10} {timers.mean_ms(stage):>8.2f}   (calls={timers.calls[stage]})")
    print("\nstage counts:")
    for name, value in stage_counts.items():
        print(f"  {name:<16} {value}")
    if calibrated_confidences:
        mean_conf = sum(calibrated_confidences) / len(calibrated_confidences)
        print(f"\ncalibration: mean calibrated confidence {mean_conf:.3f} "
              f"over {len(calibrated_confidences)} violations")


if __name__ == "__main__":
    raise SystemExit(main())
