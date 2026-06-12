# Model Card: Zoysia Seed Head Detector

## 1. Model Overview

This model detects zoysia seed heads in outdoor RGB field images for image-based phenotyping. The current handoff model is:

```text
zoysia-seedhead-yolov8m-v1.pt
```

The model was trained using a two-stage YOLO detection workflow:

1. **Baseline training:** `yolo26m_baseline_001`, initialized from `yolo26m.pt`.
2. **Fine-tuning:** `yolo26m_finetune_001`, initialized from the best baseline weights.

The model was trained and validated on **310 annotated images** collected across **three outdoor image types/settings**. The dataset includes a wide range of seed-head densities, from images with no visible seed heads to images with more than 1,000 seed heads.

---

## 2. Dataset Composition

### 2.1 Image split

| Split      |  Images |
| ---------- | ------: |
| Training   |     279 |
| Validation |      31 |
| **Total**  | **310** |

### 2.2 Image sources/settings

| Image type / setting          | Total images | Training images | Validation images | Notes                                                                                                                                                                                      |
| ----------------------------- | -----------: | --------------: | ----------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Semifield BenchBot images     |          123 |             111 |                12 | Zoysia grown in pots; images captured approximately 20 degrees off nadir.                                                                                                                  |
| 2026 farmer-mode field images |           89 |              81 |                 8 | Recently collected field images; captured at roughly 20 degrees.                                                                                                                           |
| 2025 farmer-mode field images |           98 |              87 |                11 | Field images captured at various angles, generally similar to the 2026 farmer-mode data. Some images include non-grass objects such as flags, PVC sampling collars, or numbered envelopes. |
| **Total**                     |      **310** |         **279** |            **31** |                                                                                                                                                                                            |

---

## 3. Annotation and Labeling History

The dataset was labeled through a combination of manual annotation and iterative model-assisted refinement.

The semifield data was initially labeled by hand using batches of images collected at different growth stages. Those labels were then used to train a preliminary student model. The student model generated rough seed-head labels, which were imported back into CVAT for manual correction and refinement.

This process was repeated across additional batches of semifield images and then extended to the farmer-mode field data. The 2025 farmer-mode data was labeled and used first, followed by the 2026 farmer-mode data.

All images used for this model were either labeled from scratch or manually refined from model-assisted labels. Each labeled instance was reviewed during the annotation and refinement process.

---

## 4. Training Procedure

### 4.1 Stage 1: Baseline training

The baseline model was initialized from `yolo26m.pt` and trained using stronger augmentation.

| Setting               |                  Value |
| --------------------- | ---------------------: |
| Run name              | `yolo26m_baseline_001` |
| Initial model         |           `yolo26m.pt` |
| Epochs completed      |                    205 |
| Image size            |                   2560 |
| Batch size            |                      4 |
| Optimizer             |                 `auto` |
| Initial learning rate |                  0.006 |
| Final LR factor       |                   0.01 |
| Mosaic                |                   0.75 |
| Max detections        |                   2000 |
| Single class          |                   True |

### 4.2 Stage 2: Fine-tuning

The fine-tuned model was initialized from the best weights of the baseline run. Fine-tuning used a lower learning rate and reduced augmentation, including no mosaic augmentation, to adapt the model more closely to realistic field images.

| Setting               |                                                                          Value |
| --------------------- | -----------------------------------------------------------------------------: |
| Run name              |                                                         `yolo26m_finetune_001` |
| Initial model         | `runs/detect/train/zoysia_mixed_2560_310/yolo26m_baseline_001/weights/best.pt` |
| Epochs completed      |                                                                             60 |
| Image size            |                                                                           2560 |
| Batch size            |                                                                              4 |
| Optimizer             |                                                                        `AdamW` |
| Initial learning rate |                                                                         0.0005 |
| Final LR factor       |                                                                           0.05 |
| Mosaic                |                                                                            0.0 |
| Max detections        |                                                                           2000 |
| Single class          |                                                                           True |

---

## 5. Performance Summary

The fine-tuned model improved over the baseline model on the available validation results.

| Metric    | Baseline best | Fine-tuned best | Change |
| --------- | ------------: | --------------: | -----: |
| Precision |         0.684 |           0.685 | +0.002 |
| Recall    |         0.614 |           0.628 | +0.014 |
| mAP50     |         0.648 |           0.666 | +0.018 |
| mAP50-95  |         0.353 |           0.382 | +0.029 |

Final-epoch validation metrics for the fine-tuned run:

| Metric    | Final value |
| --------- | ----------: |
| Precision |       0.664 |
| Recall    |       0.626 |
| mAP50     |       0.658 |
| mAP50-95  |       0.376 |

---

## 6. Version Information

| Item                            | Value                           |
| ------------------------------- | ------------------------------- |
| Model name                      | `zoysia-seedhead-yolov8m-v1.pt` |
| Model family                    | YOLO detection                  |
| Initial pretrained model        | `yolo26m.pt`                    |
| Task                            | Seed head detection             |
| Classes                         | Single class                    |
| Training/validation image pool  | 310 images                      |
| Training images                 | 279                             |
| Validation images               | 31                              |
| Capture settings                | 3 outdoor image types/settings  |
| Recommended image size          | 2560                            |
| Recommended starting confidence | 0.10                            |
| Recommended IoU/NMS setting     | 0.45                            |
| Recommended max detections      | 2000                            |
