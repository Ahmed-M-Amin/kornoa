# Krones Industrial Bottle Inspection AI - Final Implementation Plan

This implementation plan follows `.specify/memory/constitution.md`.

This implementation plan will later be divided into separate Spec-Kit specifications.

## 0. Final Project Goal

Build a full industrial-style system that receives bottle images and outputs:

- Reusable / Not Reusable
- confidence score
- defect reason when possible
- Grad-CAM explanation
- detector boxes or masks when available
- inference time
- `submission.csv`

The official challenge target is binary: `Reusable = 0` and `Not Reusable = 1`.
The dataset includes `train.csv`, `train_images`, `test_images`,
`sample_submission.csv`, and `train_annotations.json` with COCO annotations,
ROI, boxes, masks, and defect labels.

The final competition ranking depends on F1-score, runtime efficiency, and
technical insight, so the implementation must balance accuracy, speed, and
explainability.

## 1. Final Architecture

```text
Image / image folder
  -> Load image
  -> ROI crop / safe center crop
  -> Preprocessing
  -> Fast classifier
  -> Confidence gate
      -> If confident:
           -> final decision
      -> Else:
           -> detector / segmentation
           -> optional memory bank
           -> fusion logic
           -> final decision
  -> Explainability
  -> Dashboard + CSV + Kaggle submission
```

Important: ROI crop means crop the useful inspection region, not delete dark
pixels. Dark regions inside the bottle must stay because they can be real
defects.

## 1.1 System Architecture

```text
Data Layer
  - train.csv
  - train_images/
  - test_images/
  - train_annotations.json
  - sample_submission.csv

Configuration Layer
  - paths.yaml
  - classifier.yaml
  - detector.yaml
  - inference.yaml
  - dashboard.yaml

Processing Layer
  - COCO parser
  - dataset loader
  - ROI cropper
  - preprocessing and transforms

Training Layer
  - classifier training
  - detector/segmentation training
  - threshold search
  - hard-example mining
  - ensemble/distillation experiments

Inference Layer
  - fast classifier
  - confidence gate
  - optional detector
  - optional memory bank
  - fusion decision engine
  - benchmark
  - submission generator

Explainability Layer
  - Grad-CAM
  - detector boxes/masks
  - visualizer

Application Layer
  - Streamlit dashboard
  - single image test
  - folder workflow test
  - result table
  - speed report

Output Layer
  - model weights
  - predictions
  - metrics
  - figures
  - submission.csv
  - report-ready artifacts

## 2. Repository Structure

```text
krones-industrial-ai/
|-- .specify/
|   `-- memory/
|       `-- constitution.md
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- configs/
|   |-- paths.yaml
|   |-- classifier.yaml
|   |-- detector.yaml
|   |-- inference.yaml
|   `-- dashboard.yaml
|-- notebooks/
|   |-- 01_dataset_audit.ipynb
|   |-- 02_train_classifier.ipynb
|   |-- 03_train_detector.ipynb
|   |-- 04_distillation_memory.ipynb
|   `-- 05_kaggle_submission.ipynb
|-- src/
|   |-- data/
|   |   |-- dataset.py
|   |   |-- coco_parser.py
|   |   |-- roi.py
|   |   |-- preprocessing.py
|   |   `-- transforms.py
|   |-- models/
|   |   |-- classifier.py
|   |   |-- detector.py
|   |   |-- ensemble.py
|   |   |-- memory_bank.py
|   |   `-- distillation.py
|   |-- training/
|   |   |-- train_classifier.py
|   |   |-- train_detector.py
|   |   |-- losses.py
|   |   |-- metrics.py
|   |   |-- threshold_search.py
|   |   `-- hard_example_mining.py
|   |-- inference/
|   |   |-- predict.py
|   |   |-- fusion.py
|   |   |-- benchmark.py
|   |   `-- submission.py
|   |-- explainability/
|   |   |-- gradcam.py
|   |   `-- visualizer.py
|   |-- dashboard/
|   |   `-- app.py
|   `-- utils/
|       |-- config.py
|       |-- paths.py
|       |-- logger.py
|       `-- seed.py
|-- outputs/
|   |-- models/
|   |-- predictions/
|   |-- figures/
|   |-- reports/
|   |-- submissions/
|   `-- hard_examples/
`-- tests/
    |-- test_dataset.py
    |-- test_coco_parser.py
    |-- test_roi.py
    |-- test_inference.py
    `-- test_submission.py
```

```text
Local dataset path example for configs/paths.yaml:
E:/Bachelor Cyber Security/Semester 6/krones-vision-ai/korons/1st-krones-vision-ai-challenge
```

The local dataset path is an example for configuration only. Runtime modules
must load paths from config and must also support Kaggle and Colab paths.

## 2. Implementation Phases

### Phase 1 - Project Foundation

Goal: create the clean local project base.

Deliverables:

- `constitution.md`
- `requirements.txt`
- `configs/`
- `src/` base folders
- `outputs/` folders
- `tests/` folders

Install later from:

```text
requirements.txt
```

Main libraries:

- `torch`
- `torchvision`
- `timm`
- `ultralytics`
- `opencv-python`
- `albumentations`
- `pandas`
- `numpy`
- `scikit-learn`
- `pycocotools`
- `streamlit`
- `grad-cam`
- `onnxruntime`
- `pytest`

Quality gate:

- project opens locally
- requirements install
- config paths load
- no hardcoded paths in modules

### Phase 2 - Dataset Loading and Audit

Goal: understand the dataset before training.

Implement:

- `src/data/coco_parser.py`
- `src/data/dataset.py`
- `notebooks/01_dataset_audit.ipynb`

The audit must check:

- number of images
- train/test image matching
- class distribution
- defect-label distribution
- ROI availability
- missing files
- image sizes
- sample visualizations

Reason: the dataset has class imbalance, bottle-type imbalance, and possible
annotation quality issues.

Outputs:

- `outputs/reports/dataset_summary.csv`
- `outputs/figures/class_distribution.png`
- `outputs/figures/defect_distribution.png`
- `outputs/figures/sample_grid.png`

Quality gate:

- `train.csv` loads
- `train_annotations.json` loads
- `image_id` mapping works
- missing files are reported
- dataset figures are generated

### Phase 3 - ROI Crop and Preprocessing

Goal: focus the model on the useful bottle inspection region.

Implement:

- `src/data/roi.py`
- `src/data/preprocessing.py`
- `src/data/transforms.py`

ROI logic:

- Training images use annotation ROI when available and fall back to
  center/circular crop.
- Test images use safe center/circular crop and fall back to full resize.

Training transforms:

- resize
- brightness/contrast
- small rotation
- small shift/scale
- blur/noise
- normalize

Validation/test transforms:

- resize
- normalize only

Quality gate:

- ROI samples are saved
- internal dark defects are preserved
- crop size is consistent
- validation/test preprocessing is deterministic

### Phase 4 - V1 Fast Classifier

Goal: first working end-to-end pipeline.

Implement:

- `src/models/classifier.py`
- `src/training/train_classifier.py`
- `src/training/losses.py`
- `src/training/metrics.py`
- `src/training/threshold_search.py`

Model:

- EfficientNet-B0 first
- input size: `384x384`
- binary output

Training:

- loss: weighted BCE first
- optimizer: AdamW
- scheduler: cosine
- metric: F1-score
- validation split: stratified

F1-score is important because it balances false positives and false negatives,
and the competition uses F1 as the main performance metric.

Outputs:

- `outputs/models/classifier_effnet_b0_best.pth`
- `outputs/reports/classifier_metrics.json`
- `outputs/reports/best_threshold.json`
- `outputs/predictions/val_classifier_predictions.csv`

Quality gate:

- model trains
- validation F1 is calculated
- best threshold is saved
- model weights are saved
- validation predictions are saved

### Phase 5 - First Kaggle Submission

Goal: produce the first valid `submission.csv`.

Implement:

- `src/inference/predict.py`
- `src/inference/submission.py`
- `src/inference/benchmark.py`
- `notebooks/05_kaggle_submission.ipynb`

Output:

- `outputs/submissions/submission.csv`

The submission must follow:

```csv
image_id,target
image_001.png,0
image_002.png,1
```

Quality gate:

- `test_images` load
- predictions are generated
- `submission.csv` format matches `sample_submission.csv`
- inference time is measured

This is the end of V1.

## 3. Version Roadmap

### V1 - Working Baseline

Add ROI crop, EfficientNet-B0, F1, threshold search, and submission generation.

Purpose:

- prove the full pipeline works
- debug dataset paths
- debug labels
- generate first Kaggle score

### V2 - Strong Classifier

Add:

- focal loss
- weighted sampling
- stronger augmentation
- hard-example mining
- EfficientNet-B1/B2
- ConvNeXt-Tiny test

Purpose:

- increase F1
- reduce false positives
- reduce false negatives

Outputs:

- `outputs/hard_examples/false_positive.csv`
- `outputs/hard_examples/false_negative.csv`
- `outputs/reports/v2_comparison.json`

### V3 - Detector / Segmentation

Add:

- YOLOv8n or YOLO11n detector
- optional YOLO segmentation later
- defect boxes or masks
- defect category mapping
- conditional area logic

This matters because some labels are always good, some are conditionally faulty
above area thresholds, and some are always faulty.

Purpose:

- defect localization
- better explanation
- support hard cases
- presentation strength

### V4 - Hybrid Inference

Add:

- classifier confidence gate
- detector only for uncertain images
- fusion decision logic

Fusion logic:

```text
if classifier high confidence:
    return classifier decision

if classifier uncertain:
    run detector

if always-faulty defect detected:
    return Not Reusable

if conditional defect area above threshold:
    return Not Reusable

return classifier probability + tuned threshold decision
```

Purpose:

- keep speed high
- improve hard-case quality
- avoid running heavy detector always

### V5 - Explainability Dashboard

Add:

- Streamlit dashboard
- single image test
- folder workflow test
- prediction table
- Grad-CAM
- YOLO boxes or masks
- speed statistics
- CSV export

Dashboard output per image:

- image
- prediction
- confidence
- bad probability
- defect reason
- Grad-CAM
- detector box or mask
- inference time

Purpose:

- presentation demo
- technical insight
- explain industrial decision

### V6 - Memory and Ensemble Research

Add:

- feature memory bank for uncertain cases
- hard-example memory
- ensemble teacher predictions

Important rule:

- memory bank is optional
- memory bank runs only on uncertain cases
- memory bank must not slow normal inference

Purpose:

- improve rare/hard cases
- support water drop vs contamination
- support reflection vs real defect

### V7 - Final Optimized Model

Add:

- distilled fast student model
- optional ONNX export
- final threshold
- final benchmark
- final Kaggle notebook
- final report assets

Purpose:

- best final balance: F1 + speed + insight

## 4. Future Specification Division

This implementation plan will later be divided into separate Spec-Kit
specifications:

- SPEC-001: Project Foundation
- SPEC-002: Dataset Loading and Validation
- SPEC-003: COCO Annotation Parser
- SPEC-004: ROI Cropping and Preprocessing
- SPEC-005: Binary Classifier Training
- SPEC-006: Detector / Segmentation Training
- SPEC-007: Hybrid Inference Engine
- SPEC-008: Memory Bank and Hard Example Mining
- SPEC-009: Ensemble Teacher Models
- SPEC-010: Distilled Fast Student Model
- SPEC-011: Grad-CAM Explainability
- SPEC-012: Streamlit Dashboard
- SPEC-013: Kaggle Submission Generator
- SPEC-014: Experiment Tracking and Report Assets

Each future specification must follow `.specify/memory/constitution.md`.

## 

Dataset confidentiality must always be respected. The Krones competition data is
confidential and must not be shared outside the allowed competition environment.

## 5. Final Execution Order

1. Create repository structure
2. Add constitution.md
3. Add requirements.txt
4. Add configs
5. Implement config loader
6. Implement COCO parser
7. Implement dataset loader
8. Implement ROI cropper
9. Build dataset audit notebook
10. Train V1 EfficientNet-B0 classifier
11. Search best threshold
12. Generate first `submission.csv`
13. Benchmark inference speed
14. Improve classifier to V2
15. Add hard-example mining
16. Convert annotations for YOLO
17. Train detector / segmentation
18. Add hybrid fusion inference
19. Add Grad-CAM
20. Build Streamlit dashboard
21. Add optional memory bank
22. Add ensemble teacher models
23. Distill final fast model
24. Create final Kaggle notebook
25. Generate final report figures

## 6. Final Acceptance Criteria

The project is complete when:

- dataset audit works
- ROI crop works safely
- classifier trains
- validation F1 is reported
- best threshold is saved
- `submission.csv` is generated
- detector explains defects
- hybrid inference works
- Grad-CAM works
- Streamlit dashboard works
- speed benchmark is saved
- final Kaggle notebook runs
- report figures are ready

Final result:

One professional industrial-style bottle inspection system with training,
inference, dashboard, explainability, speed measurement, Kaggle submission, and
report-ready outputs.
