"""Seed a realistic demo database with signed violation bundles + evidence images.

Usage:
    AGASTYA_SIGNING_KEY=dev-key python scripts/seed_demo.py [--db PATH] [--n N]
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta, timezone

from agastya.stages.evidence.record import build_evidence_bundle
from agastya.store.sqlite_store import ViolationStore, compute_dedup_key
from agastya.types import PlateReading, ViolationRecord

VTYPES = [
    ("no-helmet", 2),
    ("triple-riding", 3),
    ("no-seatbelt", 2),
    ("wrong-side-driving", 3),
    ("stop-line-violation", 1),
    ("red-light-violation", 3),
    ("illegal-parking", 1),
]
CAMERAS = [
    "cam_silkboard_01",
    "cam_orr_whitefield",
    "cam_kr_puram",
    "cam_hebbal_flyover",
    "cam_marathahalli",
    "cam_mg_road_02",
]
MODELS = {"detector": "agastya-det-v4.2.1", "ocr": "plate-ocr-v2.7.0", "calibrator": "conformal-cal-v1.3.0"}
TIER_COLOR = {3: (216, 38, 28), 2: (230, 167, 0), 1: (31, 157, 77)}
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def _plate(rng: random.Random) -> str:
    d = lambda: str(rng.randint(0, 9))
    return "KA" + d() + d() + rng.choice(LETTERS) + rng.choice(LETTERS) + d() + d() + d() + d()


def _evidence_png(rng: random.Random, color: tuple[int, int, int], idx: int) -> bytes:
    """Tiny self-contained PNG (no external libs) tinted by severity tier."""
    import struct
    import zlib

    w = h = 96
    r, g, b = color
    raw = bytearray()
    jx, jy = idx * 7 % 17, idx * 13 % 19  # per-record jitter -> unique bytes -> unique content_hash
    for y in range(h):
        raw.append(0)  # filter byte per scanline
        for x in range(w):
            # gradient + a darker "evidence box" rectangle
            base = 40 + (x + y + idx) % 60
            in_box = (24 + jx) <= x <= (72 + jx) and (30 + jy) <= y <= (78 + jy)
            if in_box and (x in (24 + jx, 72 + jx) or y in (30 + jy, 78 + jy)):
                raw += bytes((r, g, b))  # box border in tier color
            elif in_box:
                raw += bytes((base // 2, base // 2, base // 2))
            else:
                raw += bytes((base, base + 8, base + 16))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 9)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="agastya_violations.db")
    ap.add_argument("--n", type=int, default=140)
    ap.add_argument("--images", default="web/assets/evidence")
    args = ap.parse_args()

    key = os.environ.get("AGASTYA_SIGNING_KEY", "dev-key").encode()
    rng = random.Random(20260620)
    os.makedirs(args.images, exist_ok=True)

    store = ViolationStore(args.db)
    now = datetime.now(timezone.utc)
    inserted = 0

    for i in range(args.n):
        vtype, tier = rng.choices(
            VTYPES, weights=[28, 22, 14, 10, 9, 9, 8], k=1
        )[0]
        cam = rng.choice(CAMERAS)
        cal_conf = round(0.62 + rng.random() * 0.37, 2)
        abstained = cal_conf < 0.72
        has_plate = rng.random() > 0.14
        captured = now - timedelta(
            days=rng.randint(0, 13), hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
        )
        ts = captured.isoformat()

        plate = None
        if has_plate:
            plate = PlateReading(text=_plate(rng), confidence=round(0.7 + rng.random() * 0.29, 2), abstained=abstained)

        record = ViolationRecord(
            violation_type=vtype,
            confidence=cal_conf,
            plate=plate,
            detections=(),
            metadata={"camera_id": cam, "timestamp": ts},
        )
        img = _evidence_png(rng, TIER_COLOR[tier], i)
        bundle = build_evidence_bundle(record, img, MODELS, key)

        dedup = compute_dedup_key(
            bundle["content_hash"], vtype, [(0.3 + i * 1e-4, 0.4, 0.26, 0.34)]
        )
        vid = store.save(bundle, created_at=ts, dedup_key=dedup)

        img_path = os.path.abspath(os.path.join(args.images, f"{vid}.png"))
        with open(img_path, "wb") as fh:
            fh.write(img)
        store.set_image_path(vid, img_path)
        inserted += 1

    s = store.stats()
    store.close()
    print(f"seeded db={args.db} records_attempted={args.n} total_in_db={s['total']}")
    print(f"by_type={s['by_type']}")
    print(f"mean_confidence={s['mean_confidence']:.3f}")
    print(f"images -> {os.path.abspath(args.images)}")


if __name__ == "__main__":
    main()
