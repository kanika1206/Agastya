from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import torch

from agastya.eval.ablation import (
    ARMS,
    CellResult,
    build_arm_config,
    degrade_bytes,
    rank_arms,
    render_matrix,
    restore_rate,
)
from agastya.eval.e2e import ViolationCounts, accumulate, gt_violations
from agastya.eval.yolo_data import label_path_for, load_data_yaml, load_truths
from agastya.stages.associate.factory import build_associator
from agastya.stages.detect.factory import build_detector
from agastya.stages.gate.factory import build_gate
from agastya.stages.gate.router import score_to_decision
from agastya.stages.ocr.factory import build_ocr
from agastya.stages.restore.factory import build_restorer
from agastya.verify.calibration import Calibrator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CONDITIONS = ("clean", "degraded")


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGASTYA Phase 4 ablation matrix")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--restore-device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--arms", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--nafnet-weights", type=Path, default=Path("models/NAFNet-GoPro-width32.pth")
    )
    parser.add_argument(
        "--calibration", type=Path, default=Path("models/calibration.json")
    )
    parser.add_argument(
        "--arniqa-weights",
        type=Path,
        default=Path("~/.cache/torch/hub/checkpoints/ARNIQA.pth").expanduser(),
    )
    parser.add_argument(
        "--runbook", type=Path, default=Path("docs/runbooks/phase4-ablation.md")
    )
    return parser.parse_args()


def run_cell(arm, condition, images, names, args) -> CellResult:
    config = build_arm_config(
        arm,
        str(args.weights),
        str(args.nafnet_weights),
        args.device,
        args.conf,
        arniqa_weights=str(args.arniqa_weights),
    )
    restore_config = config.model_copy(update={"restore_device": args.restore_device})
    gate = build_gate(restore_config)
    restorer = build_restorer(restore_config)
    detector = build_detector(config)
    associator = build_associator(config)
    ocr = build_ocr(config)
    calibrator = (
        Calibrator.from_json(str(args.calibration)) if arm.use_calibration else None
    )

    per_type = {name: ViolationCounts() for name in ("no-helmet", "triple-riding")}
    restore_invoked = 0
    detect_seconds = 0.0

    n_total = len(images)
    cell_start = time.perf_counter()
    _log(f"CELL START {arm.name}/{condition} n={n_total} rss={_rss_gb():.2f}GB")

    for index, image_path in enumerate(images, start=1):
        pixels = image_path.read_bytes()
        if condition == "degraded":
            pixels = degrade_bytes(pixels)

        score = gate.score_image(pixels)
        if score_to_decision(score, config.gate_threshold).degraded:
            torch.set_num_threads(args.cpu_threads)
            pixels = restorer.restore(pixels)
            restore_invoked += 1

        start = time.perf_counter()
        detections = detector.detect(pixels)
        detect_seconds += time.perf_counter() - start

        pred_set: set[str] = set()
        if any(
            d.label == "no-helmet" and d.score >= config.no_helmet_min_conf
            for d in detections
        ):
            pred_set.add("no-helmet")
        persons = [d for d in detections if d.label == "person"]
        for moto in detections:
            if moto.label == "motorcycle" and associator.is_triple_riding(
                moto.box, persons, pixels
            ):
                pred_set.add("triple-riding")
                break

        ocr.read(pixels)
        if calibrator is not None:
            for violation_type in pred_set:
                calibrator.evaluate(0.9, violation_type)

        truths = load_truths(label_path_for(image_path), names)
        accumulate(per_type, pred_set, gt_violations(truths, config.triple_riding_overlap))

        if index % args.log_every == 0 or index == n_total:
            elapsed = time.perf_counter() - cell_start
            _log(
                f"  {arm.name}/{condition} {index}/{n_total} "
                f"restores={restore_invoked} elapsed={elapsed:.1f}s "
                f"({elapsed / index:.2f}s/img) rss={_rss_gb():.2f}GB"
            )

    cell_seconds = time.perf_counter() - cell_start
    _log(
        f"CELL DONE {arm.name}/{condition} {cell_seconds:.1f}s "
        f"restores={restore_invoked} rss={_rss_gb():.2f}GB"
    )

    n = len(images)
    _, _, nohelmet_f1 = per_type["no-helmet"].prf()
    _, _, triple_f1 = per_type["triple-riding"].prf()
    return CellResult(
        arm_name=arm.name,
        condition=condition,
        nohelmet_f1=nohelmet_f1,
        triple_f1=triple_f1,
        mean_latency_ms=1000.0 * detect_seconds / n if n else 0.0,
        restore_invoked_count=restore_invoked,
        restore_invoked_rate=restore_rate(restore_invoked, n),
    )


def _print_ranking(results: list[CellResult], condition: str) -> None:
    print(f"\nranking ({condition}):")
    for r in rank_arms(results, condition):
        print(
            f"  {r.arm_name:<16} dTripleF1 {r.d_triple_f1:+.3f}  "
            f"dNoHelmetF1 {r.d_nohelmet_f1:+.3f}  dLat {r.d_latency_ms:+.2f}"
        )


def main() -> int:
    args = parse_args()
    val_dir, names = load_data_yaml(args.data)
    all_images = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    subset = all_images[: args.samples]
    if not subset:
        raise SystemExit(f"no val images under {val_dir}")

    selected = ARMS
    if args.arms:
        wanted = {name.strip() for name in args.arms.split(",")} | {"control"}
        selected = tuple(arm for arm in ARMS if arm.name in wanted)

    results: list[CellResult] = []
    for arm in selected:
        for condition in CONDITIONS:
            results.append(run_cell(arm, condition, subset, names, args))

    matrix = render_matrix(results)
    print(f"images per cell: {len(subset)}\n")
    print(matrix)
    _print_ranking(results, "clean")
    _print_ranking(results, "degraded")

    args.runbook.write_text(
        f"# AGASTYA Phase 4 — Ablation Matrix\n\n"
        f"Add-one-in fractional ablation, {len(subset)} val images per cell.\n\n"
        f"{matrix}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
