from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from agastya.config import PipelineConfig
from agastya.stages.evidence.manifest import build_manifest
from agastya.stages.evidence.merkle import merkle_root
from agastya.stages.gate.router import score_to_decision
from agastya.types import Detection, PlateReading, ViolationRecord


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
    ) -> None:
        self.config = config
        self.gate = gate
        self.restorer = restorer
        self.detector = detector
        self.ocr = ocr

    def run(self, item: PipelineInput) -> PipelineResult:
        pixels = item.pixels
        decision = score_to_decision(self.gate.score_image(pixels), self.config.gate_threshold)
        if decision.degraded:
            pixels = self.restorer.restore(pixels)
        detections = self.detector.detect(pixels)
        plate = self.ocr.read(pixels)
        records = self._build_records(detections, plate)
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
        self, detections: list[Detection], plate: PlateReading
    ) -> tuple[ViolationRecord, ...]:
        records: list[ViolationRecord] = []
        for det in detections:
            if det.label == "no-helmet":
                records.append(
                    ViolationRecord(
                        violation_type="no-helmet",
                        confidence=det.score,
                        plate=plate,
                        detections=(det,),
                    )
                )
        return tuple(records)
