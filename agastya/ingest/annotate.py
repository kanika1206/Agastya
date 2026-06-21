from __future__ import annotations

import cv2
import numpy as np

from agastya.types import ViolationRecord

_COLORS = {
    "no-helmet": (0, 0, 255),
    "triple-riding": (0, 165, 255),
    "seatbelt": (255, 0, 0),
    "illegal-parking": (255, 0, 255),
    "stop-line": (0, 255, 255),
    "red-light": (0, 0, 139),
    "wrong-side": (128, 0, 255),
}
_DEFAULT_COLOR = (0, 255, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate_violation(pixels: bytes, record: ViolationRecord) -> bytes:
    frame = cv2.imdecode(np.frombuffer(pixels, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode image bytes")
    color = _COLORS.get(record.violation_type, _DEFAULT_COLOR)
    for detection in record.detections:
        box = detection.box
        p1 = (int(box.x1), int(box.y1))
        p2 = (int(box.x2), int(box.y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        cv2.putText(frame, detection.label, (p1[0], max(p1[1] - 5, 10)), _FONT, 0.5, color, 1)
    caption = f"{record.violation_type} {record.confidence:.2f}"
    cv2.putText(frame, caption, (10, 25), _FONT, 0.7, color, 2)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("could not encode annotated image")
    return encoded.tobytes()
