# Running Detection

This page covers all command-line options for `detect.py`, what each output file contains, and a few example commands for common workflows.

## Basic usage

```bash
python detect.py --model path/to/weights.pt --source path/to/image_or_folder
```

`--source` can be a single image file or a directory. If it's a directory, all images in it (including subdirectories) are processed.

## Required arguments

| Argument | Description |
|---|---|
| `--model`, `--weights` | Path to trained model weights (`.pt` file) |
| `--source` | Path to a single image or a directory of images (subdirectories are searched recursively) |

## Inference settings

| Argument | Default | Description |
|---|---|---|
| `--conf` | `0.10` | Confidence threshold for detections. See [Model Performance Notes](model_performance.md) for why this is lower than the typical 0.5 default. |
| `--iou` | `0.35` | IoU threshold used for non-max suppression |
| `--max-det` | `3000` | Maximum number of detections per image (raise this for very dense images if detections are being capped) |
| `--end2end` | off | Use end-to-end (NMS-free) inference, if supported by the model architecture |
| `--device` | auto | Device to run on, e.g. `cpu`, `0`, `0,1`. Defaults to auto-detect (GPU if available) |

## Output options

By default, running `detect.py` creates a results folder containing annotated images and a `config.json` recording the settings used. Everything else is opt-in.

| Argument | Default | Description |
|---|---|---|
| `--output` | `results` | Output directory. If it already exists, a number is appended (`results2`, `results3`, ...) so previous runs are never overwritten. |
| `--no-save-images` | off | Skip saving annotated images |
| `--line-width` | `2` | Line width (in pixels) for the bounding boxes drawn on annotated images |
| `--show-conf` | off | Show confidence scores on annotated images |
| `--save-csv` | off | Write `summary.csv` — one row per image with its total seed head count |
| `--save-yolo-labels` | off | Write a YOLO-format `.txt` label file per image |
| `--yolo-labels-with-conf` | off | Append confidence as a 6th column in the YOLO label files (requires `--save-yolo-labels`) |
| `--save-detections-csv` | off | Write `detections.csv` — one row per detection across all images, with coordinates and confidence |
| `--no-save-config` | off | Don't write `config.json` (it's saved by default) |

## Output structure

```
results/
├── config.json            # Settings used for this run (default on)
├── annotated/              # Images with predicted boxes drawn (default on)
│   └── <image>_detected.jpg
├── summary.csv             # --save-csv: one row per image, with seed head count
├── detections.csv          # --save-detections-csv: one row per detection, all images
└── labels/                 # --save-yolo-labels: one .txt file per image
    └── <image>.txt
```

`summary.csv` columns: `image`, `seed_head_count`

`detections.csv` columns: `image`, `x_center`, `y_center`, `width`, `height`, `confidence`, `class` (all coordinates normalized 0–1, same convention as YOLO label files)

`labels/<image>.txt`: one line per detection, `class x_center y_center width height` (and `confidence` as a 6th column if `--yolo-labels-with-conf` is set). Note that standard YOLO training expects exactly 5 columns, so strip the confidence column before using these as training labels.

## Examples

**Quick check on a single image** (just view the annotated result):

```bash
python detect.py --model zoysia-seedhead-yolov8m-v1.pt --source image1.jpg
```

**Process a folder of plot images and get per-image counts:**

```bash
python detect.py --model zoysia-seedhead-yolov8m-v1.pt --source plot_images/ --save-csv
```

**Full output for downstream analysis** (counts, per-detection data, and YOLO labels for potential retraining):

```bash
python detect.py --model zoysia-seedhead-yolov8m-v1.pt --source plot_images/ \
    --save-csv --save-detections-csv --save-yolo-labels
```

**Large batch run, skip saving annotated images to save time/disk space:**

```bash
python detect.py --model zoysia-seedhead-yolov8m-v1.pt --source plot_images/ \
    --no-save-images --save-csv --save-detections-csv
```

**Generating pseudo-labels for a future training round** (includes confidence so a human can filter by it before using them as training labels):

```bash
python detect.py --model zoysia-seedhead-yolov8m-v1.pt --source new_images/ \
    --save-yolo-labels --yolo-labels-with-conf --no-save-images
```