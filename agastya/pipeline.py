from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from agastya.config import PipelineConfig
from agastya.stages.associate.base import Associator
from agastya.stages.associate.factory import build_associator
from agastya.stages.evidence.manifest import build_manifest
from agastya.stages.evidence.merkle import merkle_root
from agastya.stages.gate.router import score_to_decision
from agastya.stages.violations.rules import detect_scene_violations
from agastya.stages.violations.scene import SceneContext
from agastya.types import Detection, PlateReading, ViolationRecord
from agastya.verify.calibration import Calibrator


class GateStage(Protocol):
    def score_image(self, pixels: bytes) -> float:
        ...


class RestoreStage(Protocol):
    def restore(self, pixels: bytes) -> bytes:
        ...


class DetectStage(Protocol):
    def detect(self, pixels: bytes) -> list[Detection]:
        ...


class OCRStage(Protocol):
    def read(self, pixels: bytes) -> PlateReading:
        ...


@dataclass(frozen=True)
class PipelineInput:
    image_id: str
    pixels: bytes


@dataclass(frozen=True)
class PipelineResult:
    image_id: str
    records: tuple[ViolationRecord, ...]
    merkle_root: str
    manifests: tuple[dict, ...] = field(default_factory=tuple)


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig,
        gate: GateStage,
        restorer: RestoreStage,
        detector: DetectStage,
        ocr: OCRStage,
        calibrator: Calibrator | None = None,
        associator: Associator | None = None,
        scene: SceneContext | None = None,
    ) -> None:
        self.config = config
        self.gate = gate
        self.restorer = restorer
        self.detector = detector
        self.ocr = ocr
        self.calibrator = calibrator
        self.associator = associator or build_associator(config)
        self.scene = scene

    def run(self, item: PipelineInput) -> PipelineResult:
        pixels = item.pixels
        decision = score_to_decision(self.gate.score_image(pixels), self.config.gate_threshold)
        if decision.degraded:
            pixels = self.restorer.restore(pixels)
        detections = self.detector.detect(pixels)
        plate = self.ocr.read(pixels)
        records = self._build_records(detections, plate, pixels)
        manifests = tuple(
            build_manifest(record, model_versions={"detector": self.config.detector})
            for record in records
        )
        leaves = [json.dumps(m, sort_keys=True).encode() for m in manifests] or [item.image_id.encode()]
        return PipelineResult(
            image_id=item.image_id,
            records=records,
            merkle_root=merkle_root(leaves),
            manifests=manifests,
        )

    def _build_records(
        self, detections: list[Detection], plate: PlateReading, pixels: bytes
    ) -> tuple[ViolationRecord, ...]:
        records: list[ViolationRecord] = []
        for det in detections:
            if det.label == "no-helmet" and det.score >= self.config.no_helmet_min_conf:
                records.append(self._make_record("no-helmet", det.score, plate, (det,)))
        persons = [d for d in detections if d.label == "person"]
        for moto in detections:
            if moto.label != "motorcycle":
                continue
            if self.associator.is_triple_riding(moto.box, persons, pixels):
                records.append(
                    self._make_record("triple-riding", moto.score, plate, (moto, *persons))
                )
        if self.scene is not None:
            for candidate in detect_scene_violations(
                detections, self.scene, seatbelt_min_conf=self.config.no_helmet_min_conf
            ):
                records.append(
                    self._make_record(
                        candidate.violation_type, candidate.score, plate, candidate.detections
                    )
                )
        return tuple(records)

    def _make_record(
        self,
        violation_type: str,
        raw_confidence: float,
        plate: PlateReading,
        detections: tuple[Detection, ...],
    ) -> ViolationRecord:
        if self.calibrator is None:
            return ViolationRecord(
                violation_type=violation_type,
                confidence=raw_confidence,
                plate=plate,
                detections=detections,
            )
        calibrated = self.calibrator.evaluate(raw_confidence, violation_type)
        return ViolationRecord(
            violation_type=violation_type,
            confidence=calibrated.confidence,
            plate=plate,
            detections=detections,
            metadata={
                "raw_confidence": str(raw_confidence),
                "conformal_set": ",".join(sorted(calibrated.prediction_set)),
                "human_review": str(calibrated.needs_review).lower(),
            },
        )
