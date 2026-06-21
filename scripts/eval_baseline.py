from __future__ import annotations

import argparse
from pathlib import Path

from agastya.eval.prf import match_detections, precision_recall_f1
from agastya.eval.yolo_data import (
    label_path_for,
    load_data_yaml,
    load_truths,
    predictions_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-class P/R/F1 for AGASTYA detector")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_dir, names = load_data_yaml(args.data)
    images = sorted(
        p for p in val_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"no val images under {val_dir}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    counts = {name: [0, 0, 0] for name in names}

    for image_path in images:
        result = model.predict(
            str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False
        )[0]
        preds = predictions_for(result, names)
        truths = load_truths(label_path_for(image_path), names)
        for name in names:
            p_c = [d for d in preds if d.label == name]
            t_c = [d for d in truths if d.label == name]
            if not p_c and not t_c:
                continue
            tp, fp, fn = match_detections(p_c, t_c, args.iou)
            counts[name][0] += tp
            counts[name][1] += fp
            counts[name][2] += fn

    total = [0, 0, 0]
    print(f"{'class':<14} {'P':>7} {'R':>7} {'F1':>7}   {'TP':>6} {'FP':>6} {'FN':>6}")
    for name in names:
        tp, fp, fn = counts[name]
        total[0] += tp
        total[1] += fp
        total[2] += fn
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        print(f"{name:<14} {p:>7.3f} {r:>7.3f} {f1:>7.3f}   {tp:>6} {fp:>6} {fn:>6}")
    p, r, f1 = precision_recall_f1(*total)
    print(
        f"{'overall':<14} {p:>7.3f} {r:>7.3f} {f1:>7.3f}   "
        f"{total[0]:>6} {total[1]:>6} {total[2]:>6}"
    )


if __name__ == "__main__":
    main()
