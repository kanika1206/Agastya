from __future__ import annotations

from pathlib import Path

from agastya.eval.yolo_data import label_path_for, load_truths, norm_box


def test_norm_box_builds_corner_box():
    box = norm_box(0.5, 0.5, 0.2, 0.4)
    assert box is not None
    assert box.x1 == 0.4
    assert box.y1 == 0.3
    assert box.x2 == 0.6
    assert box.y2 == 0.7


def test_norm_box_rejects_degenerate():
    assert norm_box(0.5, 0.5, 0.0, 0.4) is None


def test_label_path_for_swaps_images_dir_and_suffix():
    img = Path("dataset/images/val/frame_001.jpg")
    assert label_path_for(img) == Path("dataset/labels/val/frame_001.txt")


def test_load_truths_missing_file_returns_empty():
    assert load_truths(Path("/nonexistent/label.txt"), ["person"]) == []


def test_load_truths_parses_lines(tmp_path):
    label = tmp_path / "lbl.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n")
    truths = load_truths(label, ["person", "car"])
    assert [t.label for t in truths] == ["person", "car"]
    assert all(t.score == 1.0 for t in truths)
