from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

LOW_VRAM_GB = 10.0


class VRAMProfile(str, Enum):
    FULL = "full"
    LOW = "low"


def select_profile(available_gb: float) -> VRAMProfile:
    return VRAMProfile.LOW if available_gb < LOW_VRAM_GB else VRAMProfile.FULL


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile: VRAMProfile = VRAMProfile.FULL
    gate_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    triple_riding_overlap: float = Field(default=0.1, ge=0.0, le=1.0)
    no_helmet_min_conf: float = Field(default=0.25, ge=0.0, le=1.0)
    detector: str = ""
    segmenter: str = ""
    use_diffusion_sr: bool = False
    restore_backend: str = "passthrough"
    nafnet_weights: str | None = None
    restore_device: str = "cpu"
    gate_backend: str = "never"
    arniqa_weights: str | None = None
    ocr_backend: str = "none"
    parseq_weights: str | None = None
    ocr_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    associate_backend: str = "box"
    sam2_model: str = "sam2_b.pt"
    sam2_weights: str | None = None
    detect_backend: str = "stub"
    detector_weights: str | None = None
    detect_conf: float = Field(default=0.25, ge=0.0, le=1.0)
    detect_imgsz: int = Field(default=640, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _apply_profile_defaults(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw_profile = data.get("profile", VRAMProfile.FULL)
        profile = raw_profile if isinstance(raw_profile, VRAMProfile) else VRAMProfile(raw_profile)
        if not data.get("detector"):
            data["detector"] = "yolo26-s" if profile is VRAMProfile.LOW else "yolo26-m"
        if not data.get("segmenter"):
            data["segmenter"] = "mobilesam" if profile is VRAMProfile.LOW else "sam2"
        return data
