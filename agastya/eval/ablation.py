from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from agastya.config import PipelineConfig
from agastya.eval.degrade import motion_blur


@dataclass(frozen=True)
class ArmSpec:
    name: str
    gate_backend: str
    gate_threshold: float
    restore_backend: str
    associate_backend: str
    use_calibration: bool


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("control", "never", 0.5, "passthrough", "box", False),
    ArmSpec("restore_blanket", "always", 0.5, "nafnet", "box", False),
    ArmSpec("gate_routed", "arniqa", 0.36, "nafnet", "box", False),
    ArmSpec("sam2", "never", 0.5, "passthrough", "sam2", False),
    ArmSpec("calibration", "never", 0.5, "passthrough", "box", True),
)


def build_arm_config(
    arm: ArmSpec,
    weights: str,
    nafnet_weights: str | None,
    device: str,
    conf: float = 0.25,
    arniqa_weights: str | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        gate_backend=arm.gate_backend,
        gate_threshold=arm.gate_threshold,
        arniqa_weights=arniqa_weights,
        restore_backend=arm.restore_backend,
        nafnet_weights=nafnet_weights,
        restore_device=device,
        detect_backend="yolo",
        detector_weights=weights,
        detect_conf=conf,
        associate_backend=arm.associate_backend,
    )


@dataclass(frozen=True)
class CellResult:
    arm_name: str
    condition: str
    nohelmet_f1: float
    triple_f1: float
    mean_latency_ms: float
    restore_invoked_count: int
    restore_invoked_rate: float


@dataclass(frozen=True)
class RankedArm:
    arm_name: str
    condition: str
    d_triple_f1: float
    d_nohelmet_f1: float
    d_latency_ms: float


def restore_rate(count: int, n_images: int) -> float:
    if n_images <= 0:
        return 0.0
    return count / n_images


def rank_arms(results: list[CellResult], condition: str) -> list[RankedArm]:
    cells = [r for r in results if r.condition == condition]
    control = next(r for r in cells if r.arm_name == "control")
    ranked = [
        RankedArm(
            r.arm_name,
            condition,
            r.triple_f1 - control.triple_f1,
            r.nohelmet_f1 - control.nohelmet_f1,
            r.mean_latency_ms - control.mean_latency_ms,
        )
        for r in cells
        if r.arm_name != "control"
    ]
    ranked.sort(key=lambda r: (-r.d_triple_f1, -r.d_nohelmet_f1, r.d_latency_ms))
    return ranked


def degrade_bytes(pixels: bytes, kernel_size: int = 15, angle_deg: float = 30.0) -> bytes:
    arr = cv2.imdecode(np.frombuffer(pixels, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("could not decode image bytes")
    blurred = motion_blur(arr, kernel_size, angle_deg)
    ok, buffer = cv2.imencode(".jpg", blurred)
    if not ok:
        raise ValueError("could not encode degraded image")
    return buffer.tobytes()


def _control_for(results: list[CellResult], condition: str) -> CellResult:
    return next(r for r in results if r.condition == condition and r.arm_name == "control")


def render_matrix(results: list[CellResult]) -> str:
    header = (
        "| arm | condition | no-helmet F1 | ΔF1 | triple F1 | ΔF1 | "
        "lat ms | Δlat | restore n | restore rate |"
    )
    sep = "|" + "---|" * 10
    rows = [header, sep]
    for r in results:
        control = _control_for(results, r.condition)
        d_nohelmet = r.nohelmet_f1 - control.nohelmet_f1
        d_triple = r.triple_f1 - control.triple_f1
        d_lat = r.mean_latency_ms - control.mean_latency_ms
        rows.append(
            f"| {r.arm_name} | {r.condition} | {r.nohelmet_f1:.3f} | {d_nohelmet:+.3f} | "
            f"{r.triple_f1:.3f} | {d_triple:+.3f} | {r.mean_latency_ms:.2f} | {d_lat:+.2f} | "
            f"{r.restore_invoked_count} | {r.restore_invoked_rate:.3f} |"
        )
    return "\n".join(rows)
