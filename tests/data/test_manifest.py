from agastya.data.manifest import DatasetItem, assign_split, build_manifest


def test_assign_split_is_deterministic():
    first = assign_split("img-0001", val_fraction=0.2)
    second = assign_split("img-0001", val_fraction=0.2)
    assert first == second
    assert first in {"train", "val"}


def test_assign_split_respects_fraction_roughly():
    ids = [f"img-{i:05d}" for i in range(2000)]
    val = sum(1 for i in ids if assign_split(i, val_fraction=0.2) == "val")
    assert 300 <= val <= 500


def test_build_manifest_tags_source_and_split():
    items = build_manifest(
        [("idd", "/data/idd/a.jpg"), ("aicity", "/data/aicity/b.jpg")],
        val_fraction=0.2,
    )
    assert all(isinstance(item, DatasetItem) for item in items)
    assert {item.source for item in items} == {"idd", "aicity"}
    assert all(item.split in {"train", "val"} for item in items)
