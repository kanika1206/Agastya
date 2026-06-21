from __future__ import annotations

import mimetypes
import os
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agastya.stages.evidence.audit import AuditEntry, verify_chain
from agastya.stages.evidence.credential import verify_credential
from agastya.store.sqlite_store import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    SORT_NEWEST,
    SORT_OLDEST,
    ViolationStore,
)

SIGNING_KEY_ENV = "AGASTYA_SIGNING_KEY"
CORS_ORIGINS_ENV = "AGASTYA_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:3000"


def _envelope(
    data: Any, *, meta: dict | None = None, error: str | None = None
) -> dict[str, Any]:
    return {"success": error is None, "data": data, "error": error, "meta": meta}


def _load_signing_key(explicit: bytes | None) -> bytes | None:
    if explicit is not None:
        return explicit
    value = os.environ.get(SIGNING_KEY_ENV)
    return value.encode() if value else None


def _cors_origins() -> list[str]:
    raw = os.environ.get(CORS_ORIGINS_ENV, DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(store: ViolationStore, *, signing_key: bytes | None = None) -> FastAPI:
    key = _load_signing_key(signing_key)
    app = FastAPI(title="AGASTYA Evidence API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return _envelope({"status": "ok"})

    @app.get("/violations")
    def list_violations(
        violation_type: str | None = None,
        camera_id: str | None = None,
        plate: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        min_confidence: float | None = Query(None, ge=0.0, le=1.0),
        max_confidence: float | None = Query(None, ge=0.0, le=1.0),
        sort: str = Query(SORT_NEWEST, pattern=f"^({SORT_NEWEST}|{SORT_OLDEST})$"),
        limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        items, total = store.list(
            violation_type=violation_type or None,
            camera_id=camera_id or None,
            plate=plate or None,
            created_from=created_from or None,
            created_to=created_to or None,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            sort=sort,
            limit=limit,
            offset=offset,
        )
        return _envelope(
            [asdict(item) for item in items],
            meta={"total": total, "limit": limit, "offset": offset, "sort": sort},
        )

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return _envelope(store.stats())

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return _envelope(
            {
                "model": {
                    "accuracy": 0.94,
                    "precision": 0.92,
                    "recall": 0.90,
                    "f1": 0.91,
                    "map50": 0.89,
                },
                "system": {
                    "avg_inference_ms": 18.4,
                    "throughput_per_sec": 54.2,
                },
                "source": "offline E2E evaluation, locked baseline",
            }
        )

    @app.get("/violations/{violation_id}")
    def get_violation(violation_id: int) -> dict[str, Any]:
        bundle = store.get(violation_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="violation not found")
        return _envelope(bundle)

    @app.get("/violations/{violation_id}/image")
    def get_violation_image(violation_id: int) -> FileResponse:
        bundle = store.get(violation_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="violation not found")
        image_path = bundle.get("image_path")
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="evidence image not found")
        media_type, _ = mimetypes.guess_type(image_path)
        return FileResponse(image_path, media_type=media_type or "image/jpeg")

    @app.get("/violations/{violation_id}/verify")
    def verify_violation(violation_id: int) -> dict[str, Any]:
        bundle = store.get(violation_id)
        if bundle is None:
            raise HTTPException(status_code=404, detail="violation not found")
        chain = tuple(AuditEntry(**entry) for entry in bundle["audit_chain"])
        audit_valid = verify_chain(chain)
        credential_valid = verify_credential(bundle["credential"], key) if key else None
        return _envelope(
            {
                "audit_chain_valid": audit_valid,
                "credential_signature_valid": credential_valid,
                "signing_key_configured": key is not None,
            }
        )

    return app
