from __future__ import annotations

CREATE_VIOLATIONS = """
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    plate TEXT,
    camera_id TEXT,
    captured_at TEXT,
    evidence_root TEXT NOT NULL,
    credential TEXT NOT NULL,
    audit_chain TEXT NOT NULL,
    created_at TEXT NOT NULL,
    image_path TEXT,
    dedup_key TEXT
)
"""

ADD_COLUMNS = (
    ("image_path", "ALTER TABLE violations ADD COLUMN image_path TEXT"),
    ("dedup_key", "ALTER TABLE violations ADD COLUMN dedup_key TEXT"),
)

CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_violations_type ON violations(violation_type)",
    "CREATE INDEX IF NOT EXISTS idx_violations_camera ON violations(camera_id)",
    "CREATE INDEX IF NOT EXISTS idx_violations_created ON violations(created_at)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_violations_dedup ON violations(dedup_key)",
)
