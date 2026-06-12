#!/usr/bin/env python3
"""
detect.py - Run the Zoysia Seed Head Detector on a single image or a directory of images.

Usage:
    python detect.py --model path/to/weights.pt --source path/to/image.jpg
    python detect.py --model path/to/weights.pt --source path/to/image_dir/ --output results/ \
        --save-csv --save-yolo-labels --save-detections-csv

Outputs (written to --output, auto-incremented if it already exists, e.g. results -> results2):
    annotated/<image>_detected.jpg   Annotated copies showing predicted bounding boxes (default on)
    summary.csv                      One row per image with total seed head count (--save-csv)
    labels/<image>.txt               Per-image detections in standard YOLO format:
                                      'class x_center y_center width height' (normalized 0-1).
                                      (--save-yolo-labels)
                                      Add --yolo-labels-with-conf to append confidence as a
                                      6th column. Note: standard YOLO training expects 5
                                      columns, so strip the confidence column if reusing
                                      these as training labels.
    detections.csv                   One row per detection across all images, written
                                      incrementally so memory use stays flat for large batches.
                                      Columns: image, x_center, y_center, width, height,
                                      confidence, class (--save-detections-csv)
    config.json                      The settings used for this run (default on; disable with
                                      --no-save-config)
"""

import argparse
import csv
import json
import sys
from datetime import datetime
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
        help="Directory to save results (default: results/). If it already exists, "
             "a number is appended (results2, results3, ...) so previous runs aren't overwritten.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.10,
        help="Confidence threshold for detections (default: 0.10)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.35,
        help="IoU threshold for non-max suppression (default: 0.35)",
    )
    parser.add_argument(
        "--max-det", type=int, default=3000,
        help="Maximum number of detections per image (default: 3000)",
    )
    parser.add_argument(
         "--line-width", type=int, default=2,
         help="Line width for annotated boxes (default: 2)",
    )
    parser.add_argument(
        "--show-conf", action="store_true",
        help="Show confidence scores on annotated images (default: False)",
    )
    parser.add_argument(
        "--end2end", action="store_true",
        help="Use end-to-end (NMS-free) inference mode, if supported by the model "
             "architecture (default: False)",
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
    parser.add_argument(
        "--yolo-labels-with-conf", action="store_true",
        help="Append confidence as a 6th column in the YOLO label files "
             "(requires --save-yolo-labels). Default is standard 5-column format.",
    )
    parser.add_argument(
        "--save-detections-csv", action="store_true",
        help="Write detections.csv with one row per detection across all images "
             "(written incrementally; suitable for large batches)",
    )
    parser.add_argument(
        "--no-save-config", dest="save_config", action="store_false",
        help="Don't write config.json with the settings used for this run "
             "(saved by default)",
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


def get_unique_output_dir(path: Path) -> Path:
    """Return `path` if it doesn't exist, otherwise append 2, 3, 4, ... until unique."""
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.parent / f"{path.name}{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    args = parse_args()

    if args.yolo_labels_with_conf and not args.save_yolo_labels:
        sys.exit("Error: --yolo-labels-with-conf requires --save-yolo-labels.")

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Error: model weights not found at '{model_path}'.")

    images = gather_images(args.source)
    print(f"Found {len(images)} image(s) to process.")

    output_dir = get_unique_output_dir(Path(args.output))
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Writing results to '{output_dir}/'")

    if args.save_config:
        config = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_images": len(images),
            "output_dir": str(output_dir),
            **{k: v for k, v in vars(args).items() if k != "save_config"},
        }
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    annotated_dir = output_dir / "annotated"
    if not args.no_save_images:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = output_dir / "labels"
    if args.save_yolo_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)

    detections_csv_file = None
    detections_writer = None
    if args.save_detections_csv:
        detections_csv_path = output_dir / "detections.csv"
        detections_csv_file = open(detections_csv_path, "w", newline="")
        detections_writer = csv.DictWriter(
            detections_csv_file,
            fieldnames=["image", "x_center", "y_center", "width", "height", "confidence", "class"],
        )
        detections_writer.writeheader()

    print(f"Loading model from '{model_path}'...")
    model = YOLO(str(model_path))

    summary_rows = []

    try:
        for img_path in images:
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
                agnostic_nms=False,
                end2end=args.end2end,
            )
            result = results[0]
            num_detections = len(result.boxes)

            print(f"  {img_path.name}: {num_detections} seed head(s) detected")

            summary_rows.append({"image": str(img_path), "seed_head_count": num_detections})

            if not args.no_save_images:
                annotated = result.plot(
                    line_width=args.line_width,
                    labels=False,        # Don't draw class labels since we only have one class and it can overlap a lot
                    conf=args.show_conf,  # Confidence scores on the image; off by default to reduce clutter
                )
                out_path = annotated_dir / f"{img_path.stem}_detected.jpg"
                cv2.imwrite(str(out_path), annotated)

            if args.save_yolo_labels or args.save_detections_csv:
                img_h, img_w = result.orig_shape  # (height, width)

                yolo_lines = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    if args.save_yolo_labels:
                        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        if args.yolo_labels_with_conf:
                            line += f" {conf:.4f}"
                        yolo_lines.append(line)

                    if args.save_detections_csv:
                        detections_writer.writerow({
                            "image": str(img_path),
                            "x_center": round(x_center, 6),
                            "y_center": round(y_center, 6),
                            "width": round(width, 6),
                            "height": round(height, 6),
                            "confidence": round(conf, 4),
                            "class": cls_id,
                        })

                if args.save_yolo_labels:
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    with open(label_path, "w") as f:
                        f.write("\n".join(yolo_lines))
                        if yolo_lines:
                            f.write("\n")

                if args.save_detections_csv:
                    detections_csv_file.flush()
    finally:
        if detections_csv_file is not None:
            detections_csv_file.close()

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
    if args.save_detections_csv:
        print(f"  Detections CSV:   {output_dir / 'detections.csv'}")
    if args.save_config:
        print(f"  Config:           {output_dir / 'config.json'}")


if __name__ == "__main__":
    main()