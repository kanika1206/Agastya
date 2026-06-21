from __future__ import annotations

import argparse
import os
from pathlib import Path

from agastya.config import PipelineConfig
from agastya.ingest.runner import IngestRunner
from agastya.pipeline import Pipeline
from agastya.stages.associate.factory import build_associator
from agastya.stages.detect.factory import build_detector
from agastya.stages.gate.factory import build_gate
from agastya.stages.ocr.factory import build_ocr
from agastya.stages.restore.factory import build_restorer
from agastya.verify.calibration import Calibrator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SIGNING_KEY_ENV = "AGASTYA_SIGNING_KEY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGASTYA ingest: image(s) -> pipeline -> store")
    parser.add_argument("source", type=Path, help="image file or folder of images")
    parser.add_argument("--db", type=str, default="agastya_violations.db")
    parser.add_argument("--evidence-dir", type=str, default="evidence")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--detect-backend", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--gate-backend", type=str, default="never")
    parser.add_argument("--restore-backend", type=str, default="passthrough")
    parser.add_argument("--ocr-backend", type=str, default="none")
    parser.add_argument("--associate-backend", type=str, default="box")
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--signing-key", type=str, default=None)
    return parser.parse_args()


def resolve_signing_key(explicit: str | None) -> bytes:
    value = explicit or os.environ.get(SIGNING_KEY_ENV)
    if not value:
        raise SystemExit(
            f"signing key required: pass --signing-key or set {SIGNING_KEY_ENV}"
        )
    return value.encode()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    detect_backend = args.detect_backend or ("yolo" if args.weights else "stub")
    return PipelineConfig(
        gate_backend=args.gate_backend,
        restore_backend=args.restore_backend,
        restore_device=args.device,
        detect_backend=detect_backend,
        detector_weights=str(args.weights) if args.weights else None,
        detect_conf=args.conf,
        ocr_backend=args.ocr_backend,
        associate_backend=args.associate_backend,
    )


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    args = parse_args()
    signing_key = resolve_signing_key(args.signing_key)
    config = build_config(args)
    images = collect_images(args.source)
    if not images:
        raise SystemExit(f"no images found at {args.source}")

    from agastya.store.sqlite_store import ViolationStore

    pipeline = Pipeline(
        config=config,
        gate=build_gate(config),
        restorer=build_restorer(config),
        detector=build_detector(config),
        ocr=build_ocr(config),
        calibrator=Calibrator.from_json(str(args.calibration)) if args.calibration else None,
        associator=build_associator(config),
    )
    store = ViolationStore(args.db)
    runner = IngestRunner(
        pipeline,
        store,
        signing_key=signing_key,
        model_versions={"detector": config.detector},
        evidence_dir=args.evidence_dir,
    )

    total_violations = 0
    for image_path in images:
        result = runner.ingest_image(image_path.name, image_path.read_bytes())
        total_violations += result.count
        print(f"{image_path.name}: {result.count} violation(s) -> ids {list(result.violation_ids)}")

    print(f"\ningested {len(images)} image(s), {total_violations} violation row(s) into {args.db}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
