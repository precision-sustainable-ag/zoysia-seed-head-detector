# Zoysia Seed Head Detector

A computer vision model for detecting seed heads in zoysia grass images, developed for image-based phenotyping.

![Example detection](docs/assets/example1.jpg)

## Overview

Manually counting seed heads across large field trials is slow and labor-intensive. This repository provides a trained YOLO-based detector for identifying zoysia seed heads in RGB field images. The model is intended to support seed-head localization, visual review, count estimation, and downstream phenotyping workflows.

For details on the training data, annotation process, training procedure, and validation results, see the [Model Card](MODEL_CARD.md).

## Repository Contents

| Resource                                                                                                                                | Description                                                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [Model Card](docs/model_card_zoysia-seedhead-yolov8m-v1.md)                                                                                                             | Dataset composition, annotation history, training procedure, performance summary, and version information |
| [Model Performance Notes](docs/model_performance.md)                                                                                    | Explanation of mAP, confidence thresholds, and recommended inference settings                             |
| [Running Detection](docs/usage.md)                                                                                                      | Inference commands and available script arguments                                                         |
| [Colab demo](https://colab.research.google.com/github/precision-sustainable-ag/zoysia-seed-head-detector/blob/develop/colab_demo.ipynb) | Run the detector without installing locally (model must be manually downloaded from NFS storage)                                                               |

## Model Details

| Item                         | Description                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| **Model name**               | `zoysia-seedhead-yolov8m-v1.pt`                                                      |
| **Architecture**             | YOLOv8m / YOLO detection model                                                       |
| **Input**                    | RGB images collected by the monocam, approximately 4032 × 3040 px                    |
| **Output**                   | Bounding boxes and confidence scores for detected seed heads                         |
| **Training/validation data** | 310 annotated images from semifield BenchBot and farmer-mode field image collections |
| **Training approach**        | Baseline YOLO training followed by fine-tuning from the best baseline weights        |
| **Primary use**              | Seed-head detection, visual review, count estimation, and density estimation         |

Held-out test set performance:

| Metric       | Value |
| ------------ | ----: |
| mAP@0.5      |  0.61 |
| mAP@0.5:0.95 |  0.40 |
| mAP@0.75     |  0.45 |
| Precision    |  0.75 |
| Recall       |  0.43 |

> **Note:** mAP scores may look low compared to typical object detection benchmarks. This task involves small, dense, clustered objects, so useful detections may occur at lower confidence thresholds than the typical `0.5` default. See [Model Performance Notes](docs/model_performance.md) for more context and recommended inference settings.

## Quick Start

### Try it without installing anything

[Open in Colab](https://colab.research.google.com/github/precision-sustainable-ag/zoysia-seed-head-detector/blob/develop/colab_demo.ipynb)

### Run locally

```bash
git clone https://github.com/precision-sustainable-ag/zoysia-seed-head-detector.git
cd zoysia-seed-head-detector
pip install -r requirements.txt
```

Run detection on a single image or folder of images:

```bash
python detect.py \
  --model path/to/weights.pt \
  --source path/to/your/image_or_folder \
  --output results/ \
  --conf 0.10 \
  --iou 0.35 \
  --max-det 3000
```

Recommended starting inference settings:

```yaml
conf: 0.10
iou: 0.35
max-det: 3000
```

These settings are intentionally different from typical object detection defaults because seed heads are small, dense, and often visually ambiguous. Adjust the confidence threshold depending on whether the goal is higher recall, cleaner visual outputs, or count estimation.

For the full list of arguments, including options to save per-image counts, YOLO-format labels, and detections CSV files for batch runs, see [Running Detection](docs/usage.md).

## Model Weights

The current model weights are available to NCSU collaborators with cluster access at:

```text
/rsstu/users/s/srmilla/NIFA_Zoysia/zoysia-seed-head-detector/model/zoysia-seedhead-yolov8m-v1.pt
```

## Limitations

* **Image quality matters most.** Blurry images significantly reduce detection accuracy. Improving image capture quality is the most direct way to improve results.
* **Dense clusters are difficult.** In areas where many seed heads are packed closely together, the model may underestimate counts, miss individual seed heads, or merge multiple seed heads into one detection.
* **Some detections are intentionally low confidence.** Many useful detections may occur below `0.5` confidence. Start with the recommended settings above and adjust based on the image set and desired precision/recall balance.
* **New image settings should be validated.** Images from different cameras, angles, lighting conditions, or growth stages may require additional validation or fine-tuning.

Example limitation case: [dense seed-head cluster example](docs/assets/example2.jpg). In this image, the detector identifies more than 2,000 seed heads, but still misses additional seed heads in dense clusters that are difficult to visually separate.

## License

[TODO: specify a license — e.g., MIT, Apache 2.0, or a research-data-specific license]
