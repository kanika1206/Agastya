"""Convert absolute evidence image_path values to repo-relative for portability.

Idempotent: already-relative paths and paths outside the evidence root are left
unchanged. Run once against any DB seeded before paths were made portable.

Usage:
    python scripts/migrate_evidence_paths.py [--db PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import sqlite3

from agastya.paths import to_relative


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="agastya_violations.db")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, image_path FROM violations WHERE image_path IS NOT NULL"
    ).fetchall()

    changed = 0
    for row in rows:
        relative = to_relative(row["image_path"])
        if relative != row["image_path"]:
            changed += 1
            if not args.dry_run:
                conn.execute(
                    "UPDATE violations SET image_path = ? WHERE id = ?",
                    (relative, row["id"]),
                )
    if not args.dry_run:
        conn.commit()
    conn.close()

    mode = "dry-run" if args.dry_run else "applied"
    print(f"db={args.db} scanned={len(rows)} changed={changed} mode={mode}")


if __name__ == "__main__":
    main()
