# Zoysia Seed Head Detector

This repository provides documentation and model handoff materials for a trained object detection model that identifies zoysia seed heads in field imagery.

The model was developed for an image-based phenotyping workflow where seed heads need to be detected, counted, reviewed, and summarized from images collected under outdoor conditions.

## Project Summary

The goal of this project is to provide a reusable seed head detection model that can support phenotyping and image analysis workflows.

The model is intended to help users:

* detect visible seed heads in zoysia imagery
* generate prediction outputs for review
* support seed head counting and downstream phenotyping summaries
* evaluate model behavior across different image collection settings

## Intended Use

This model is intended for zoysia seed head detection in images similar to the training and validation data used during model development.

Recommended uses include:

* batch prediction on field images
* visual review of seed head detections
* seed head counting workflows
* comparison of model outputs across image sets
* integration into downstream phenotyping pipelines

## Not Intended For

This model should not be assumed to work reliably on unrelated crops, grass species, imaging systems, or field conditions without additional validation.

The model may require further testing before use on:

* new camera systems
* new field sites
* different growth stages
* different lighting conditions
* images with very different scale, resolution, or viewing angle
* species other than zoysia

## Model Files

Model weights and release notes will be made available through this repository or linked from an external storage location.

Current model version:

```text
Model version: TBD
Model file: TBD
Training date: TBD
Maintainer: TBD
```

## Example Predictions

Example model outputs will be added here to show typical behavior.

Suggested examples to include:

* a strong prediction example
* a difficult image with dense seed heads
* an image with missed detections
* an image with false positives
* a low-confidence prediction example

```text
docs/assets/example_prediction.jpg
```

## Performance Summary

A summary of model performance will be added after validation results are finalized.

| Metric               | Value |
| -------------------- | ----: |
| mAP50                |   TBD |
| mAP50-95             |   TBD |
| Precision            |   TBD |
| Recall               |   TBD |
| Validation images    |   TBD |
| Validation instances |   TBD |

## Recommended Inference Settings

Recommended inference settings will be updated as the model is finalized.

```text
image size: TBD
confidence threshold: TBD
IoU threshold: TBD
model weights: TBD
```

## Known Limitations

Known limitations include:

* seed heads may be missed when they are small, blurry, occluded, or densely overlapping
* predictions may have relatively low confidence even when visually correct
* performance may vary across field conditions, lighting, camera angle, and image resolution
* the model should be validated before being used for new image collection protocols

## Reporting Issues

Questions, bugs, or model improvement requests should be submitted through GitHub Issues.

When reporting an issue, please include:

* model version
* image or batch name
* inference settings used
* description of the problem
* example image or screenshot, if possible

## Status

This repository is intended as a model handoff and documentation space. Future updates may be added as new validation results, model versions, or usage examples become available.
