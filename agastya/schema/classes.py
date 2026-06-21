CLASSES: tuple[str, ...] = (
    "helmet",
    "no-helmet",
    "license-plate",
    "motorcycle",
    "person",
)

_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def validate_class(name: str) -> str:
    if name not in _NAME_TO_ID:
        raise ValueError(f"unknown class: {name}")
    return name


def name_to_id(name: str) -> int:
    return _NAME_TO_ID[validate_class(name)]


def id_to_name(class_id: int) -> str:
    if class_id < 0 or class_id >= len(CLASSES):
        raise ValueError(f"class id out of range: {class_id}")
    return CLASSES[class_id]
