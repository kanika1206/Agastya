from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from agastya.store.schema import ADD_COLUMNS, CREATE_INDEXES, CREATE_VIOLATIONS

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 500
SORT_OLDEST = "oldest"
SORT_NEWEST = "newest"


@dataclass(frozen=True)
class StoredViolation:
    id: int
    content_hash: str
    violation_type: str
    confidence: float
    plate: str | None
    camera_id: str | None
    captured_at: str | None
    evidence_root: str
    created_at: str
    image_path: str | None


def compute_dedup_key(
    content_hash: str,
    violation_type: str,
    boxes: Sequence[tuple[float, float, float, float]],
) -> str:
    ordered = sorted(tuple(round(coord, 3) for coord in box) for box in boxes)
    payload = json.dumps([content_hash, violation_type, ordered], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit_to_json(audit_chain: object) -> str:
    entries = [entry if isinstance(entry, dict) else asdict(entry) for entry in audit_chain]
    return json.dumps(entries, sort_keys=True)


def _row_to_stored(row: sqlite3.Row) -> StoredViolation:
    return StoredViolation(
        id=row["id"],
        content_hash=row["content_hash"],
        violation_type=row["violation_type"],
        confidence=row["confidence"],
        plate=row["plate"],
        camera_id=row["camera_id"],
        captured_at=row["captured_at"],
        evidence_root=row["evidence_root"],
        created_at=row["created_at"],
        image_path=row["image_path"],
    )


class ViolationStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(CREATE_VIOLATIONS)
        self._migrate()
        for statement in CREATE_INDEXES:
            self._conn.execute(statement)
        self._conn.commit()

    def _migrate(self) -> None:
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(violations)")}
        for column, statement in ADD_COLUMNS:
            if column not in existing:
                self._conn.execute(statement)

    def close(self) -> None:
        self._conn.close()

    def save(
        self,
        bundle: dict,
        *,
        created_at: str | None = None,
        dedup_key: str | None = None,
        image_path: str | None = None,
    ) -> int:
        manifest = bundle["credential"]["manifest"]
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO violations (content_hash, violation_type, confidence, "
            "plate, camera_id, captured_at, evidence_root, credential, audit_chain, "
            "created_at, image_path, dedup_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                bundle["content_hash"],
                manifest["violation_type"],
                manifest["confidence"],
                manifest.get("plate"),
                manifest.get("camera_id"),
                manifest.get("timestamp"),
                bundle["evidence_root"],
                json.dumps(bundle["credential"], sort_keys=True),
                _audit_to_json(bundle["audit_chain"]),
                created_at or _utc_now(),
                image_path,
                dedup_key,
            ),
        )
        self._conn.commit()
        if cursor.rowcount == 0 and dedup_key is not None:
            existing = self._conn.execute(
                "SELECT id FROM violations WHERE dedup_key = ?", (dedup_key,)
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
        return int(cursor.lastrowid)

    def set_image_path(self, violation_id: int, image_path: str) -> None:
        self._conn.execute(
            "UPDATE violations SET image_path = ? WHERE id = ?", (image_path, violation_id)
        )
        self._conn.commit()

    def list(
        self,
        *,
        violation_type: str | None = None,
        camera_id: str | None = None,
        plate: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        sort: str = SORT_NEWEST,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[list[StoredViolation], int]:
        limit = max(1, min(limit, MAX_PAGE_LIMIT))
        offset = max(0, offset)
        clauses: list[str] = []
        params: list[object] = []
        if violation_type is not None:
            clauses.append("violation_type = ?")
            params.append(violation_type)
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if plate is not None:
            clauses.append("plate LIKE ?")
            params.append(f"%{plate}%")
        if created_from is not None:
            clauses.append("created_at >= ?")
            params.append(created_from)
        if created_to is not None:
            clauses.append("created_at <= ?")
            params.append(created_to)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        if max_confidence is not None:
            clauses.append("confidence <= ?")
            params.append(max_confidence)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order = "ASC" if sort == SORT_OLDEST else "DESC"
        total = self._conn.execute(
            f"SELECT COUNT(*) AS c FROM violations{where}", params
        ).fetchone()["c"]
        rows = self._conn.execute(
            f"SELECT * FROM violations{where} ORDER BY id {order} LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [_row_to_stored(row) for row in rows], int(total)

    def get(self, violation_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM violations WHERE id = ?", (violation_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "content_hash": row["content_hash"],
            "evidence_root": row["evidence_root"],
            "credential": json.loads(row["credential"]),
            "audit_chain": json.loads(row["audit_chain"]),
            "created_at": row["created_at"],
            "image_path": row["image_path"],
        }

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) AS c FROM violations").fetchone()["c"]
        by_type = {
            row["violation_type"]: row["n"]
            for row in self._conn.execute(
                "SELECT violation_type, COUNT(*) AS n FROM violations GROUP BY violation_type"
            )
        }
        by_day = {
            row["day"]: row["n"]
            for row in self._conn.execute(
                "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS n "
                "FROM violations GROUP BY day ORDER BY day"
            )
        }
        mean_confidence = self._conn.execute(
            "SELECT AVG(confidence) AS a FROM violations"
        ).fetchone()["a"]
        return {
            "total": int(total),
            "by_type": by_type,
            "by_day": by_day,
            "mean_confidence": mean_confidence,
        }
