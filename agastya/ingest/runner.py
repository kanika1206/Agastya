from __future__ import annotations

import os
from dataclasses import dataclass

from agastya.ingest.annotate import annotate_violation
from agastya.pipeline import Pipeline, PipelineInput
from agastya.stages.evidence.record import build_evidence_bundle
from agastya.store.sqlite_store import ViolationStore, compute_dedup_key
from agastya.types import ViolationRecord


@dataclass(frozen=True)
class IngestResult:
    image_id: str
    violation_ids: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.violation_ids)


class IngestRunner:
    def __init__(
        self,
        pipeline: Pipeline,
        store: ViolationStore,
        *,
        signing_key: bytes,
        model_versions: dict[str, str],
        evidence_dir: str | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._store = store
        self._signing_key = signing_key
        self._model_versions = dict(model_versions)
        self._evidence_dir = evidence_dir
        if evidence_dir is not None:
            os.makedirs(evidence_dir, exist_ok=True)

    def ingest_image(self, image_id: str, pixels: bytes) -> IngestResult:
        result = self._pipeline.run(PipelineInput(image_id=image_id, pixels=pixels))
        ids: list[int] = []
        for record in result.records:
            bundle = build_evidence_bundle(
                record, pixels, self._model_versions, self._signing_key
            )
            dedup_key = compute_dedup_key(
                bundle["content_hash"], record.violation_type, _boxes(record)
            )
            violation_id = self._store.save(bundle, dedup_key=dedup_key)
            self._write_evidence(violation_id, pixels, record)
            ids.append(violation_id)
        return IngestResult(image_id=image_id, violation_ids=tuple(ids))

    def _write_evidence(self, violation_id: int, pixels: bytes, record: ViolationRecord) -> None:
        if self._evidence_dir is None:
            return
        annotated = annotate_violation(pixels, record)
        path = os.path.join(self._evidence_dir, f"{violation_id}.jpg")
        with open(path, "wb") as handle:
            handle.write(annotated)
        self._store.set_image_path(violation_id, path)


def _boxes(record: ViolationRecord) -> list[tuple[float, float, float, float]]:
    return [(d.box.x1, d.box.y1, d.box.x2, d.box.y2) for d in record.detections]
