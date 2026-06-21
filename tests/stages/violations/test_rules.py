from agastya.stages.violations.rules import (
    detect_illegal_parking,
    detect_red_light,
    detect_scene_violations,
    detect_seatbelt,
    detect_stop_line,
    detect_wrong_side,
)
from agastya.stages.violations.scene import SceneContext, StopLine
from agastya.types import BBox, Detection


def _vehicle(label="motorcycle", score=0.9, box=BBox(4.0, 4.0, 6.0, 6.0)):
    return Detection(label=label, score=score, box=box)


def test_seatbelt_emits_on_class_label():
    dets = [Detection(label="no-seatbelt", score=0.8, box=BBox(1.0, 1.0, 2.0, 2.0))]
    out = detect_seatbelt(dets, 0.5)
    assert [c.violation_type for c in out] == ["seatbelt"]


def test_seatbelt_abstains_without_class():
    assert detect_seatbelt([_vehicle()], 0.5) == []


def test_illegal_parking_inside_zone():
    scene = SceneContext(no_parking_zones=(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),))
    out = detect_illegal_parking([_vehicle()], scene)
    assert [c.violation_type for c in out] == ["illegal-parking"]


def test_illegal_parking_abstains_without_zone():
    assert detect_illegal_parking([_vehicle()], SceneContext()) == []


def test_stop_line_crossed():
    scene = SceneContext(stop_line=StopLine(a=(0.0, 5.0), b=(10.0, 5.0), violation_sign=1.0))
    crossing = _vehicle(box=BBox(4.0, 6.0, 6.0, 9.0))
    out = detect_stop_line([crossing], scene)
    assert [c.violation_type for c in out] == ["stop-line"]


def test_stop_line_not_crossed():
    scene = SceneContext(stop_line=StopLine(a=(0.0, 5.0), b=(10.0, 5.0), violation_sign=1.0))
    behind = _vehicle(box=BBox(4.0, 1.0, 6.0, 3.0))
    assert detect_stop_line([behind], scene) == []


def test_red_light_requires_red_signal():
    line = StopLine(a=(0.0, 5.0), b=(10.0, 5.0), violation_sign=1.0)
    crossing = _vehicle(box=BBox(4.0, 6.0, 6.0, 9.0))
    assert detect_red_light([crossing], SceneContext(stop_line=line, signal_state="green")) == []
    out = detect_red_light([crossing], SceneContext(stop_line=line, signal_state="red"))
    assert [c.violation_type for c in out] == ["red-light"]


def test_wrong_side_with_heading():
    scene = SceneContext(allowed_direction=(1.0, 0.0), headings={0: (-1.0, 0.0)})
    out = detect_wrong_side([_vehicle()], scene)
    assert [c.violation_type for c in out] == ["wrong-side"]


def test_wrong_side_abstains_without_heading():
    scene = SceneContext(allowed_direction=(1.0, 0.0))
    assert detect_wrong_side([_vehicle()], scene) == []


def test_scene_violations_empty_context_is_silent_except_seatbelt():
    out = detect_scene_violations([_vehicle()], SceneContext(), seatbelt_min_conf=0.5)
    assert out == []
