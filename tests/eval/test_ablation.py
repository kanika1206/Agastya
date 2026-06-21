import cv2
import numpy as np

from agastya.eval.ablation import (
    ARMS,
    CellResult,
    RankedArm,
    build_arm_config,
    degrade_bytes,
    rank_arms,
    render_matrix,
    restore_rate,
)


def test_arms_has_five_with_control_first():
    assert len(ARMS) == 5
    assert ARMS[0].name == "control"


def test_control_arm_config_matches_locked_defaults():
    arm = next(a for a in ARMS if a.name == "control")
    config = build_arm_config(arm, "best.pt", None, "cuda")
    assert config.gate_backend == "never"
    assert config.restore_backend == "passthrough"
    assert config.associate_backend == "box"
    assert config.detect_conf == 0.25
    assert config.no_helmet_min_conf == 0.25
    assert config.triple_riding_overlap == 0.10


def test_gate_routed_arm_enables_arniqa_and_nafnet():
    arm = next(a for a in ARMS if a.name == "gate_routed")
    config = build_arm_config(arm, "best.pt", "models/NAFNet-GoPro-width32.pth", "cuda")
    assert config.gate_backend == "arniqa"
    assert config.gate_threshold == 0.36
    assert config.restore_backend == "nafnet"
    assert config.nafnet_weights == "models/NAFNet-GoPro-width32.pth"


def test_restore_blanket_arm_uses_always_gate():
    arm = next(a for a in ARMS if a.name == "restore_blanket")
    assert arm.gate_backend == "always"
    assert arm.restore_backend == "nafnet"


def test_sam2_arm_selects_sam2_associator():
    arm = next(a for a in ARMS if a.name == "sam2")
    config = build_arm_config(arm, "best.pt", None, "cuda")
    assert config.associate_backend == "sam2"


def test_calibration_arm_flag_set():
    arm = next(a for a in ARMS if a.name == "calibration")
    assert arm.use_calibration is True


def test_restore_rate_basic():
    assert restore_rate(125, 500) == 0.25


def test_restore_rate_zero_images_guard():
    assert restore_rate(0, 0) == 0.0


def _cell(name, triple, nohelmet, latency):
    return CellResult(name, "clean", nohelmet, triple, latency, 0, 0.0)


def test_rank_excludes_control_and_orders_by_triple_then_nohelmet_then_latency():
    results = [
        _cell("control", 0.893, 0.967, 12.0),
        _cell("sam2", 0.910, 0.967, 30.0),
        _cell("calibration", 0.893, 0.967, 12.5),
        _cell("gate_routed", 0.893, 0.975, 40.0),
    ]
    ranked = rank_arms(results, "clean")
    assert [r.arm_name for r in ranked] == ["sam2", "gate_routed", "calibration"]
    assert ranked[0].d_triple_f1 == 0.910 - 0.893
    assert ranked[1].d_nohelmet_f1 == 0.975 - 0.967
    assert ranked[2].d_latency_ms == 12.5 - 12.0


def test_rank_only_uses_cells_of_requested_condition():
    results = [
        CellResult("control", "clean", 0.967, 0.893, 12.0, 0, 0.0),
        CellResult("control", "degraded", 0.800, 0.700, 13.0, 0, 0.0),
        CellResult("sam2", "degraded", 0.700, 0.750, 31.0, 0, 0.0),
    ]
    ranked = rank_arms(results, "degraded")
    assert [r.arm_name for r in ranked] == ["sam2"]
    assert ranked[0].d_triple_f1 == 0.750 - 0.700


def _encode(arr):
    return cv2.imencode(".jpg", arr)[1].tobytes()


def test_degrade_bytes_returns_decodable_same_shape():
    arr = (np.random.rand(64, 96, 3) * 255).astype("uint8")
    out = degrade_bytes(_encode(arr))
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (64, 96, 3)


def test_degrade_bytes_changes_pixels():
    arr = (np.random.rand(64, 96, 3) * 255).astype("uint8")
    clean = cv2.imdecode(np.frombuffer(_encode(arr), np.uint8), cv2.IMREAD_COLOR)
    blurred = cv2.imdecode(
        np.frombuffer(degrade_bytes(_encode(arr)), np.uint8), cv2.IMREAD_COLOR
    )
    assert not np.array_equal(clean, blurred)


def test_render_matrix_has_header_and_delta_columns():
    results = [
        CellResult("control", "clean", 0.967, 0.893, 12.0, 0, 0.0),
        CellResult("sam2", "clean", 0.967, 0.910, 30.0, 0, 0.0),
    ]
    table = render_matrix(results)
    assert "arm" in table and "condition" in table
    assert "no-helmet F1" in table
    assert "triple F1" in table
    assert "ΔF1" in table
    assert "Δlat" in table
    assert "restore rate" in table


def test_render_matrix_control_deltas_zero_and_sam2_triple_delta():
    results = [
        CellResult("control", "clean", 0.967, 0.893, 12.0, 0, 0.0),
        CellResult("sam2", "clean", 0.967, 0.910, 30.0, 0, 0.0),
    ]
    table = render_matrix(results)
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    control_row = next(ln for ln in lines if "control" in ln)
    sam2_row = next(ln for ln in lines if "sam2" in ln)
    assert "+0.000" in control_row
    assert "+0.017" in sam2_row


def test_rankedarm_fields():
    r = RankedArm("sam2", "clean", 0.017, 0.0, 18.0)
    assert r.arm_name == "sam2"
    assert r.condition == "clean"
