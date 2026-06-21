import pytest
from pydantic import ValidationError

from agastya.config import PipelineConfig, VRAMProfile, select_profile


def test_high_vram_selects_full_profile():
    assert select_profile(16.0) == VRAMProfile.FULL


def test_low_vram_selects_fallback_profile():
    assert select_profile(8.0) == VRAMProfile.LOW


def test_full_profile_uses_yolo26_m():
    cfg = PipelineConfig(profile=VRAMProfile.FULL)
    assert cfg.detector == "yolo26-m"
    assert cfg.use_diffusion_sr is False


def test_low_profile_swaps_to_light_models():
    cfg = PipelineConfig(profile=VRAMProfile.LOW)
    assert cfg.detector == "yolo26-s"
    assert cfg.segmenter == "mobilesam"


def test_gate_threshold_must_be_unit_interval():
    with pytest.raises(ValueError):
        PipelineConfig(profile=VRAMProfile.FULL, gate_threshold=2.0)


def test_config_is_frozen():
    cfg = PipelineConfig(profile=VRAMProfile.FULL)
    with pytest.raises(ValidationError):
        cfg.detector = "tampered"


def test_restore_defaults_are_passthrough_cpu():
    cfg = PipelineConfig()
    assert cfg.restore_backend == "passthrough"
    assert cfg.nafnet_weights is None
    assert cfg.restore_device == "cpu"


def test_restore_fields_are_settable():
    cfg = PipelineConfig(restore_backend="nafnet", nafnet_weights="/w/best.pt", restore_device="cuda")
    assert cfg.restore_backend == "nafnet"
    assert cfg.nafnet_weights == "/w/best.pt"
    assert cfg.restore_device == "cuda"
