import pytest

from agastya.data.assemble import (
    assemble_dataset,
    discover_images,
    label_path_for_image,
    load_source_names,
    remap_label_text,
)


def _make_triple_source(raw_root):
    src = raw_root / "triple"
    (src / "train" / "images").mkdir(parents=True)
    (src / "train" / "labels").mkdir(parents=True)
    (src / "data.yaml").write_text(
        "nc: 5\nnames: ['Triple_riding', 'motorcycle', 'number_plate', "
        "'with_helmet', 'without_helmet']\n"
    )
    (src / "train" / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (src / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n")
    return src


def test_discover_images_finds_jpgs_per_source(tmp_path):
    _make_triple_source(tmp_path)
    entries = discover_images(tmp_path)
    assert entries == [("triple", str(tmp_path / "triple" / "train" / "images" / "a.jpg"))]


def test_assemble_dataset_copies_images_and_remaps_labels(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_triple_source(raw)
    out = tmp_path / "out"

    count = assemble_dataset(raw, out, val_fraction=0.0)

    assert count == 1
    image_out = out / "images" / "train" / "triple_a.jpg"
    label_out = out / "labels" / "train" / "triple_a.txt"
    assert image_out.read_bytes() == b"\xff\xd8\xff"
    lines = label_out.read_text().splitlines()
    assert lines[0].startswith("3 ")  # Triple_riding -> triple-riding
    assert lines[1].startswith("4 ")  # motorcycle -> motorcycle
    assert (out / "data.yaml").exists()


def test_label_path_for_image_swaps_images_dir_and_suffix(tmp_path):
    image = tmp_path / "train" / "images" / "frame_001.jpg"
    label = label_path_for_image(image)
    assert label == tmp_path / "train" / "labels" / "frame_001.txt"


def test_remap_label_text_remaps_triple_class_ids():
    src_names = ["Triple_riding", "motorcycle", "number_plate", "with_helmet", "without_helmet"]
    text = "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n3 0.4 0.4 0.3 0.3\n"
    remapped = remap_label_text(text, source="triple", src_names=src_names)
    lines = remapped.splitlines()
    assert lines[0].startswith("3 ")  # Triple_riding -> triple-riding (id 3)
    assert lines[1].startswith("4 ")  # motorcycle -> motorcycle (id 4)
    assert lines[2].startswith("0 ")  # with_helmet -> helmet (id 0)


def test_remap_label_text_drops_unmapped_classes():
    src_names = ["motorcycle", "background"]
    text = "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n"
    remapped = remap_label_text(text, source="triple", src_names=src_names)
    lines = remapped.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("4 ")  # motorcycle kept, background dropped


def test_remap_label_text_converts_polygon_to_bbox():
    src_names = ["motorcycle"]
    text = "0 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7\n"
    remapped = remap_label_text(text, source="triple", src_names=src_names)
    parts = remapped.split()
    assert parts[0] == "4"  # motorcycle -> id 4
    assert parts[1] == "0.400000"  # cx from polygon extremes
    assert parts[3] == "0.400000"  # width


def test_remap_label_text_handles_empty_label():
    remapped = remap_label_text("", source="triple", src_names=["motorcycle"])
    assert remapped == ""


def test_load_source_names_reads_data_yaml(tmp_path):
    source_dir = tmp_path / "safety"
    source_dir.mkdir()
    (source_dir / "data.yaml").write_text(
        "nc: 4\nnames: ['bike', 'helmet', 'no-helmet', 'number-plate']\n"
    )
    names = load_source_names(tmp_path, "safety")
    assert names == ["bike", "helmet", "no-helmet", "number-plate"]


def test_load_source_names_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_source_names(tmp_path, "nonexistent")
