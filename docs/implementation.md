# Krones Industrial Bottle Inspection AI - Final Implementation Plan

This implementation plan follows `.specify/memory/constitution.md`.

This implementation plan will later be divided into separate Spec-Kit specifications.

## 0. Final Project Goal

Build a full industrial-style system that receives bottle images and outputs:

* Reusable / Not Reusable
* confidence score
* defect reason when possible
* Grad-CAM explanation
* detector boxes or masks when available
* inference time
* `submission.csv`

The official challenge target is binary: `Reusable = 0` and `Not Reusable = 1`.

The dataset includes `train.csv`, `train_images`, `test_images`,
`sample_submission.csv`, and `train_annotations.json` with COCO annotations,
ROI, boxes, masks, and defect labels.

The final competition ranking depends on F1-score, runtime efficiency, and
technical insight, so the implementation must balance accuracy, speed, and
explainability.

Important system goal:

```text
High F1-score
+ fast image-flow inference
+ explainable industrial decision
+ valid Kaggle submission
```

---

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
           -> optional feature memory bank
           -> fusion logic
           -> final decision
  -> Explainability
  -> Dashboard + CSV + Kaggle submission
```

Important: ROI crop means crop the useful inspection region, not delete dark
pixels. Dark regions inside the bottle must stay because they can be real
defects.

Normal fast inference must use:

```text
image folder
  -> batch load images
  -> ROI crop
  -> preprocessing
  -> classifier forward pass
  -> threshold decision
  -> result table + inference time
```

Heavy modules must not run for every image by default:

```text
detector
segmentation
Grad-CAM
feature memory bank
ensemble
```

They are only used for uncertain cases, selected demo images, or later research.

---

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
  - optional feature memory bank
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
  - hard examples
  - benchmarks
  - report-ready artifacts
```

---

## 1.2 Memory Strategy

The project uses three different memory concepts. They must not be mixed.

### 1.2.1 Model Weights

This is mandatory.

Model weights are the normal learned parameters after training.

Example V1 output:

```text
outputs/kaggle_v1/models/classifier_effnet_b0_best.pth
```

Purpose:

* store what the model learned
* load once during inference
* run one forward pass per image or per batch
* required for all predictions

Speed impact:

```text
Required
Fast
Used in every inference run
```

This is the only memory type that always runs during inference.

---

### 1.2.2 Hard-Example Memory

This must be implemented immediately after V1 training finishes.

Hard-example memory means storing difficult validation cases from the V1
prediction file.

Input:

```text
outputs/kaggle_v1/predictions/val_classifier_predictions.csv
```

Outputs:

```text
outputs/hard_examples/false_positives.csv
outputs/hard_examples/false_negatives.csv
outputs/hard_examples/uncertain.csv
outputs/hard_examples/high_loss_samples.csv
outputs/reports/hard_example_summary.json
```

Definitions:

```text
False positive:
Reusable bottle predicted as Not Reusable

False negative:
Not Reusable bottle predicted as Reusable

Uncertain:
Prediction probability near the best threshold

High-loss sample:
Image where the model is strongly wrong or confused
```

Purpose:

* understand V1 mistakes
* improve V2 training
* reduce false positives
* reduce false negatives
* oversample difficult cases
* tune augmentation
* detect ROI crop issues
* create report-ready insights

Important speed rule:

```text
Hard-example memory is training/evaluation-only.
It must not run during normal inference.
It must not slow Kaggle submission generation.
It must not slow the image-flow demo.
```

This is the memory that should be implemented now after V1.

---

### 1.2.3 Feature Memory Bank

This is optional and comes later.

Feature memory bank means storing embeddings/features of known hard cases and
comparing uncertain new images against them.

Purpose:

* support rare hard cases
* support water drop vs contamination
* support reflection vs real defect
* improve uncertain decisions

Important speed rule:

```text
Feature memory bank must be disabled by default.
It must not run for every image.
It may only run when the classifier is uncertain.
```

Activation condition:

```text
abs(probability_bad - best_threshold) <= uncertainty_margin
```

The feature memory bank is not part of V1 or V2. It belongs to later research.

---

## 2. Repository Structure

```text
krones-industrial-ai/
|-- .specify/
|   `-- memory/
|       `-- constitution.md
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- implementation.md
|-- configs/
|   |-- paths.yaml
|   |-- classifier.yaml
|   |-- detector.yaml
|   |-- inference.yaml
|   `-- dashboard.yaml
|-- notebooks/
|   |-- 01_dataset_audit.ipynb
|   |-- 02_train_classifier.ipynb
|   |-- 03_v1_evaluation_submission.ipynb
|   |-- 04_train_detector.ipynb
|   |-- 05_hybrid_inference.ipynb
|   `-- 06_final_kaggle_submission.ipynb
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
|   |-- hard_examples/
|   |-- benchmarks/
|   `-- gradcam/
`-- tests/
    |-- test_dataset.py
    |-- test_coco_parser.py
    |-- test_roi.py
    |-- test_classifier.py
    |-- test_training.py
    |-- test_threshold_search.py
    |-- test_hard_example_mining.py
    |-- test_inference.py
    |-- test_benchmark.py
    `-- test_submission.py
```

```text
Local dataset path example for configs/paths.yaml:
E:/Bachelor Cyber Security/Semester 6/krones-vision-ai/korons/1st-krones-vision-ai-challenge
```

The local dataset path is an example for configuration only. Runtime modules
must load paths from config and must also support Kaggle and Colab paths.

No dataset files must be committed to GitHub.

---

## 3. Implementation Phases

### Phase 1 - Project Foundation

Goal: create the clean local project base.

Deliverables:

* `constitution.md`
* `requirements.txt`
* `configs/`
* `src/` base folders
* `outputs/` folders
* `tests/` folders

Install later from:

```text
requirements.txt
```

Main libraries:

* `torch`
* `torchvision`
* `timm`
* `ultralytics`
* `opencv-python`
* `albumentations`
* `pandas`
* `numpy`
* `scikit-learn`
* `pycocotools`
* `streamlit`
* `grad-cam`
* `onnxruntime`
* `pytest`

Quality gate:

* project opens locally
* requirements install
* config paths load
* no hardcoded paths in modules

---

### Phase 2 - Dataset Loading and Audit

Goal: understand the dataset before training.

Implement:

* `src/data/coco_parser.py`
* `src/data/dataset.py`
* `notebooks/01_dataset_audit.ipynb`

The audit must check:

* number of images
* train/test image matching
* class distribution
* defect-label distribution
* ROI availability
* missing files
* extra/unreferenced files
* image sizes
* sample visualizations

Reason: the dataset has class imbalance, bottle-type imbalance, and possible
annotation quality issues.

Outputs:

* `outputs/reports/dataset_summary.csv`
* `outputs/reports/label_distribution.json`
* `outputs/figures/class_distribution.png`
* `outputs/figures/defect_distribution.png`
* `outputs/figures/sample_grid.png`

Quality gate:

* `train.csv` loads
* `train_annotations.json` loads
* `image_id` mapping works
* missing files are reported
* dataset figures are generated

---

### Phase 3 - ROI Crop and Preprocessing

Goal: focus the model on the useful bottle inspection region.

Implement:

* `src/data/roi.py`
* `src/data/preprocessing.py`
* `src/data/transforms.py`

ROI logic:

* Training images use annotation ROI when available and fall back to
  center/circular crop.
* Test images use safe center/circular crop and fall back to full resize.

Training transforms:

* resize
* brightness/contrast
* small rotation
* small shift/scale
* blur/noise
* normalize

Validation/test transforms:

* resize
* normalize only

Critical ROI rule:

```text
Do not delete internal dark regions.
Dark areas inside the bottle can be real defects.
```

Quality gate:

* ROI samples are saved
* internal dark defects are preserved
* crop size is consistent
* validation/test preprocessing is deterministic

---

### Phase 4 - V1 Fast Classifier

Goal: first working end-to-end pipeline.

Implement:

* `src/models/classifier.py`
* `src/training/train_classifier.py`
* `src/training/losses.py`
* `src/training/metrics.py`
* `src/training/threshold_search.py`

Model:

* EfficientNet-B0 first
* input size: `384x384`
* binary output

Training:

* loss: weighted BCE first
* optimizer: AdamW
* scheduler: cosine
* metric: F1-score
* validation split: stratified
* threshold search on validation set

F1-score is important because it balances false positives and false negatives,
and the competition uses F1 as the main performance metric.

Outputs:

* `outputs/kaggle_v1/models/classifier_effnet_b0_best.pth`
* `outputs/kaggle_v1/reports/classifier_metrics.json`
* `outputs/kaggle_v1/reports/best_threshold.json`
* `outputs/kaggle_v1/reports/label_distribution.json`
* `outputs/kaggle_v1/reports/split_distribution.json`
* `outputs/kaggle_v1/predictions/val_classifier_predictions.csv`

Quality gate:

* model trains
* validation F1 is calculated
* best threshold is saved
* model weights are saved
* validation predictions are saved
* no test data is used for training or threshold tuning

---

### Phase 5 - V1 Evaluation, Hard-Example Mining, First Submission, and Benchmark

Goal: evaluate V1 properly, create the first valid `submission.csv`, measure
speed, and prepare V2 improvement.

Implement:

* `src/training/hard_example_mining.py`
* `src/inference/predict.py`
* `src/inference/submission.py`
* `src/inference/benchmark.py`
* `notebooks/03_v1_evaluation_submission.ipynb`

Inputs:

* `outputs/kaggle_v1/models/classifier_effnet_b0_best.pth`
* `outputs/kaggle_v1/reports/classifier_metrics.json`
* `outputs/kaggle_v1/reports/best_threshold.json`
* `outputs/kaggle_v1/predictions/val_classifier_predictions.csv`
* `sample_submission.csv`
* `test_images/`

Hard-example outputs:

* `outputs/hard_examples/false_positives.csv`
* `outputs/hard_examples/false_negatives.csv`
* `outputs/hard_examples/uncertain.csv`
* `outputs/hard_examples/high_loss_samples.csv`
* `outputs/reports/hard_example_summary.json`

Submission output:

* `outputs/submissions/submission_v1.csv`

Benchmark output:

* `outputs/benchmarks/v1_inference_benchmark.json`

The submission must follow:

```csv
image_id,target
image_001.png,0
image_002.png,1
```

Hard-example mining rule:

```text
Use validation predictions only.
Do not use test images for hard-example mining.
```

Quality gate:

* V1 metrics load
* best threshold loads
* validation predictions load
* hard-example files are generated
* `test_images` load
* predictions are generated
* `submission.csv` format matches `sample_submission.csv`
* inference time is measured
* hard-example mining does not slow normal inference

This is the end of V1.

---

## 4. Version Roadmap

### V1 - Working Baseline

Add ROI crop, EfficientNet-B0, F1, threshold search, hard-example mining,
benchmarking, and submission generation.

Purpose:

* prove the full pipeline works
* debug dataset paths
* debug labels
* generate first Kaggle score
* create hard-example memory for V2

---

### V2 - Strong Classifier

Add:

* focal loss
* weighted sampling
* stronger augmentation
* hard-example oversampling
* EfficientNet-B1/B2
* ConvNeXt-Tiny test
* optional `448x448` input test if GPU allows

Purpose:

* increase F1
* reduce false positives
* reduce false negatives
* improve water drop vs real defect separation
* improve rare defect recognition

Inputs from V1:

* `false_positives.csv`
* `false_negatives.csv`
* `uncertain.csv`
* `high_loss_samples.csv`

Outputs:

* `outputs/kaggle_v2/models/classifier_best.pth`
* `outputs/kaggle_v2/reports/classifier_metrics.json`
* `outputs/kaggle_v2/reports/best_threshold.json`
* `outputs/kaggle_v2/reports/v1_vs_v2_comparison.json`
* `outputs/kaggle_v2/predictions/val_classifier_predictions.csv`
* `outputs/kaggle_v2/submissions/submission_v2.csv`

---

### V3 - Detector / Segmentation

Add:

* YOLOv8n or YOLO11n detector
* optional YOLO segmentation later
* defect boxes or masks
* defect category mapping
* conditional area logic

This matters because some labels are always good, some are conditionally faulty
above area thresholds, and some are always faulty.

Purpose:

* defect localization
* better explanation
* support hard cases
* presentation strength

---

### V4 - Hybrid Inference

Add:

* classifier confidence gate
* detector only for uncertain images
* fusion decision logic

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

* keep speed high
* improve hard-case quality
* avoid running heavy detector always

---

### V5 - Explainability Dashboard

Add:

* Streamlit dashboard
* single image test
* folder workflow test
* prediction table
* Grad-CAM
* YOLO boxes or masks
* speed statistics
* CSV export

Dashboard output per image:

* image
* prediction
* confidence
* bad probability
* defect reason
* Grad-CAM
* detector box or mask
* inference time

Purpose:

* presentation demo
* technical insight
* explain industrial decision

Speed rule:

```text
Grad-CAM runs only for selected images.
Grad-CAM must not run for every image in folder workflow.
```

---

### V6 - Feature Memory Bank and Ensemble Research

Add:

* optional feature memory bank for uncertain cases only
* ensemble teacher predictions
* optional distillation experiments

Important rule:

* feature memory bank is optional
* feature memory bank is disabled by default
* feature memory bank runs only on uncertain cases
* feature memory bank must not slow normal inference
* hard-example memory is not here because it was already created after V1

Purpose:

* improve rare/hard cases
* support water drop vs contamination
* support reflection vs real defect
* create technical insight

---

### V7 - Final Optimized Model

Add:

* distilled fast student model
* optional ONNX export
* final threshold
* final benchmark
* final Kaggle notebook
* final report assets

Purpose:

* best final balance: F1 + speed + insight

---

## 5. Future Specification Division

This implementation plan will later be divided into separate Spec-Kit
specifications:

* SPEC-001: Project Foundation
* SPEC-002: Dataset Loading and Validation
* SPEC-003: COCO Annotation Parser
* SPEC-004: ROI Cropping and Preprocessing
* SPEC-005: Binary Classifier Training
* SPEC-006: V1 Evaluation, Submission, Benchmark, and Hard-Example Mining
* SPEC-007: Strong Classifier V2
* SPEC-008: Detector / Segmentation Training
* SPEC-009: Hybrid Inference Engine
* SPEC-010: Grad-CAM Explainability
* SPEC-011: Streamlit Dashboard
* SPEC-012: Optional Feature Memory Bank
* SPEC-013: Ensemble Teacher Models and Distillation
* SPEC-014: Final Kaggle Notebook and Report Assets

Each future specification must follow `.specify/memory/constitution.md`.

Dataset confidentiality must always be respected. The Krones competition data is
confidential and must not be shared outside the allowed competition environment.

---

## 6. Final Execution Order

1. Create repository structure
2. Add `constitution.md`
3. Add `requirements.txt`
4. Add configs
5. Implement config loader
6. Implement COCO parser
7. Implement dataset loader
8. Implement ROI cropper
9. Build dataset audit notebook
10. Train V1 EfficientNet-B0 classifier
11. Search best threshold
12. Save V1 metrics and validation predictions
13. Generate first `submission.csv`
14. Benchmark V1 inference speed
15. Add hard-example mining from V1 validation predictions
16. Create `false_positives.csv`
17. Create `false_negatives.csv`
18. Create `uncertain.csv`
19. Create `high_loss_samples.csv`
20. Analyze V1 mistakes
21. Improve classifier to V2
22. Add focal loss
23. Add weighted sampler
24. Add stronger safe augmentation
25. Add hard-example oversampling
26. Test EfficientNet-B1/B2 and ConvNeXt-Tiny
27. Generate V2 submission
28. Benchmark V2 inference speed
29. Convert annotations for YOLO
30. Train detector / segmentation
31. Add hybrid fusion inference
32. Add Grad-CAM
33. Build Streamlit dashboard
34. Add optional feature memory bank only if useful
35. Add ensemble teacher models
36. Distill final fast model if useful
37. Create final Kaggle notebook
38. Generate final report figures

---

## 7. Immediate Next Step After SPEC-005 / V1 Training

When V1 training finishes, run:

```bash
find outputs/kaggle_v1 -type f
```

Expected files:

```text
outputs/kaggle_v1/models/classifier_effnet_b0_best.pth
outputs/kaggle_v1/reports/classifier_metrics.json
outputs/kaggle_v1/reports/best_threshold.json
outputs/kaggle_v1/reports/label_distribution.json
outputs/kaggle_v1/reports/split_distribution.json
outputs/kaggle_v1/predictions/val_classifier_predictions.csv
```

Then inspect metrics:

```python
import json

with open("outputs/kaggle_v1/reports/classifier_metrics.json") as f:
    metrics = json.load(f)

with open("outputs/kaggle_v1/reports/best_threshold.json") as f:
    threshold = json.load(f)

print(metrics)
print(threshold)
```

Then create hard-example memory:

```text
outputs/hard_examples/false_positives.csv
outputs/hard_examples/false_negatives.csv
outputs/hard_examples/uncertain.csv
outputs/hard_examples/high_loss_samples.csv
```

Then generate:

```text
outputs/submissions/submission_v1.csv
outputs/benchmarks/v1_inference_benchmark.json
```

Do not implement the feature memory bank now.

---

## 8. Final Acceptance Criteria

The project is complete when:

* dataset audit works
* ROI crop works safely
* classifier trains
* validation F1 is reported
* best threshold is saved
* model weights are saved
* validation predictions are saved
* `submission.csv` is generated
* inference benchmark is saved
* hard-example memory is generated
* V2 improvement is tested
* detector explains defects
* hybrid inference works
* Grad-CAM works
* Streamlit dashboard works
* speed benchmark is saved
* final Kaggle notebook runs
* report figures are ready
* dataset confidentiality is respected

Final result:

One professional industrial-style bottle inspection system with training,
inference, dashboard, explainability, speed measurement, hard-example mining,
Kaggle submission, and report-ready outputs.
