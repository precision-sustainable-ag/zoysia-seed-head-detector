# Model Performance Notes

## Interpreting the mAP scores

The mAP score for this model is lower than would typically be expected for a well-performing detector, but that's partly a limitation of the evaluation metric for this particular problem. Seed heads are small, dense, and sometimes ambiguous. The model often places boxes on the correct structures, but small differences in box placement or low confidence scores can reduce mAP even when the detection is biologically correct.

In practice, the detector appears most useful when evaluated visually and when using a lower confidence threshold — around 0.1–0.3 — rather than the conventional 0.5 default. For this project, mAP should be treated as one diagnostic metric among several, not the sole measure of usefulness. Standard mAP metrics are highly sensitive to exact bounding-box overlap and confidence thresholds, so biologically useful detections may still be penalized.

**Recommended starting inference settings:**

```yaml
conf: 0.10
iou: 0.25
max_det: 2000
```

The model is best viewed as a practical seed-head detection tool whose outputs require threshold calibration, rather than as a generic object detector expected to perform best at a 0.5 confidence threshold.

## Why mAP may be low even when the model is useful

### 1. Seed heads are small objects

For small objects, a few pixels of box shift can greatly reduce IoU. A bounding box can appear to identify the correct seed head, but if it's slightly offset, the mAP calculation may treat it as a poor match even though the detection is correct in practice.

### 2. The images are extremely dense

Some images contain hundreds, or even over a thousand, seed heads. In dense scenes, objects overlap, touch, or are partially hidden, making exact one-to-one box matching between predictions and labels difficult — even when the model has correctly identified the seed-head region.

### 3. Confidence scores are low, but still meaningful

Many correct predictions fall in the 0.1–0.3 confidence range, meaning the conventional 0.5 threshold is too strict for this detector. A confidence of 0.15 may still represent a valid seed-head detection — the operating threshold should be chosen based on validation results, not assumed from generic object detection defaults.

### 4. Some labels may be incomplete or ambiguous

If a real seed head is visible in an image but wasn't labeled, the model is penalized for detecting it. Dense biological datasets often have this issue, so some apparent "false positives" may actually be real seed heads missing from the annotation set.

### 5. The biological task differs from the evaluation metric

The practical goal is to estimate seed-head presence, density, or counts — mAP evaluates exact bounding-box agreement. The model can still be useful for phenotyping if its predicted counts or spatial density patterns match field reality, even if strict box-level mAP is lower.