from agastya.data.data_yaml import write_data_yaml
from agastya.schema.classes import CLASSES


def test_write_data_yaml_contains_classes_and_paths(tmp_path):
    out = tmp_path / "data.yaml"
    write_data_yaml(out, root=tmp_path, train_dir="images/train", val_dir="images/val")
    text = out.read_text()
    assert "nc: 4" in text
    for name in CLASSES:
        assert name in text
    assert "images/train" in text
    assert "images/val" in text


def test_write_data_yaml_is_parseable(tmp_path):
    import yaml

    out = tmp_path / "data.yaml"
    write_data_yaml(out, root=tmp_path, train_dir="images/train", val_dir="images/val")
    parsed = yaml.safe_load(out.read_text())
    assert parsed["nc"] == len(CLASSES)
    assert parsed["names"] == list(CLASSES)
