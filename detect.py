#!/usr/bin/env python3
"""
detect.py - Run the Zoysia Seed Head Detector on a single image or a directory of images.

Usage:
    python detect.py --model path/to/weights.pt --source path/to/image.jpg
    python detect.py --model path/to/weights.pt --source path/to/image_dir/ --output results/ --save-csv --save-yolo-labels

Outputs (written to --output):
    annotated/<image>_detected.jpg   Annotated copies showing predicted bounding boxes (default on)
    summary.csv                      One row per image with total seed head count (--save-csv)
    labels/<image>.txt               Per-image bounding boxes in YOLO format (--save-yolo-labels)
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Error: ultralytics package not found. Install with: pip install ultralytics")

try:
    import cv2
except ImportError:
    sys.exit("Error: opencv-python not found. Install with: pip install opencv-python")


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run seed head detection on a single image or a directory of images."
    )
    parser.add_argument(
        "--model", "--weights", dest="model", required=True,
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to a single image file or a directory containing images "
             "(subdirectories are searched recursively)",
    )
    parser.add_argument(
        "--output", default="results",
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.10,
        help="Confidence threshold for detections (default: 0.10)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for non-max suppression (default: 0.45)",
    )
    parser.add_argument(
        "--max_det", type=int, default=2000,
        help="Maximum number of detections per image (default: 2000)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to run inference on, e.g. 'cpu', '0', '0,1'. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--no-save-images", action="store_true",
        help="Skip saving annotated images",
    )
    parser.add_argument(
        "--save-csv", action="store_true",
        help="Write summary.csv with the seed head count per image",
    )
    parser.add_argument(
        "--save-yolo-labels", action="store_true",
        help="Write a YOLO-format .txt label file per image (labels/<image>.txt), "
             "one line per detection: 'class x_center y_center width height' "
             "(normalized 0-1)",
    )
    return parser.parse_args()


def gather_images(source: str) -> list[Path]:
    source_path = Path(source)

    if source_path.is_file():
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            sys.exit(f"Error: '{source_path}' does not look like a supported image file.")
        return [source_path]

    if source_path.is_dir():
        images = sorted(
            p for p in source_path.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            sys.exit(f"Error: no images found under '{source_path}'.")
        return images

    sys.exit(f"Error: '{source}' is not a valid file or directory.")


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Error: model weights not found at '{model_path}'.")

    images = gather_images(args.source)
    print(f"Found {len(images)} image(s) to process.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = output_dir / "annotated"
    if not args.no_save_images:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = output_dir / "labels"
    if args.save_yolo_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from '{model_path}'...")
    model = YOLO(str(model_path))

    summary_rows = []

    for img_path in images:
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        num_detections = len(result.boxes)

        print(f"  {img_path.name}: {num_detections} seed head(s) detected")

        summary_rows.append({"image": str(img_path), "seed_head_count": num_detections})

        if not args.no_save_images:
            annotated = result.plot()
            out_path = annotated_dir / f"{img_path.stem}_detected.jpg"
            cv2.imwrite(str(out_path), annotated)

        if args.save_yolo_labels:
            img_h, img_w = result.orig_shape  # (height, width)
            label_path = labels_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    f.write(
                        f"{cls_id} {x_center:.6f} {y_center:.6f} "
                        f"{width:.6f} {height:.6f}\n"
                    )

    if args.save_csv:
        summary_path = output_dir / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "seed_head_count"])
            writer.writeheader()
            writer.writerows(summary_rows)

    total = sum(row["seed_head_count"] for row in summary_rows)
    print(f"\nDone. {len(images)} image(s) processed, {total} total seed head(s) detected.")
    if not args.no_save_images:
        print(f"  Annotated images: {annotated_dir}/")
    if args.save_csv:
        print(f"  Summary CSV:      {output_dir / 'summary.csv'}")
    if args.save_yolo_labels:
        print(f"  YOLO labels:      {labels_dir}/")


if __name__ == "__main__":
    main()