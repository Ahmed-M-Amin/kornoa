# Contract: Detector / Segmentation Training

## Scope

This contract defines SPEC-008 user-facing CLI workflows and artifact contracts. SPEC-008 prepares detector/segmentation training artifacts only. It must not implement SPEC-009 hybrid inference or change the accepted V2B classifier.

## Annotation Audit Contract

**Command shape**:

```text
python -m src.training.train_detector audit --config configs/detector.yaml
```

**Required inputs**:
- `train_annotations.json`
- Training image directory
- Detector output root

**Required behavior**:
- Load COCO images, annotations, and categories.
- Validate category and image relationships.
- Report annotation counts, category distribution, bbox size distribution, images without annotations, images with multiple defects, missing image references, invalid boxes, invalid masks, and segmentation availability.
- Save category mapping and audit JSON under `outputs/detector/reports/`.
- Do not read test images or sample submission rows.

**Required outputs**:
- `outputs/detector/reports/category_mapping.json`
- `outputs/detector/reports/annotation_audit.json`

## Dataset Conversion Contract

**Command shape**:

```text
python -m src.training.train_detector convert --config configs/detector.yaml
```

**Required behavior**:
- Create `outputs/detector/dataset/images/train`.
- Create `outputs/detector/dataset/images/val`.
- Create `outputs/detector/dataset/labels/train`.
- Create `outputs/detector/dataset/labels/val`.
- Clear only generated detector split folders before conversion: `dataset/images/train`, `dataset/images/val`, `dataset/labels/train`, and `dataset/labels/val`.
- Convert valid COCO bbox annotations to YOLO label files.
- Include no-annotation training images with empty YOLO label files.
- Skip and report invalid annotations.
- Attempt primary-defect-category stratification, including `no_annotation`, when each stratum has enough images.
- Use deterministic random fallback with saved seed when stratification is not practical.
- Preserve train/validation disjointness.
- Report included no-annotation image count.
- Report `output_dataset_cleaned`, `cleaned_paths`, `split_strategy_used`, and stratification fallback reason.
- Write `outputs/detector/dataset/data.yaml`.

**Required outputs**:
- `outputs/detector/dataset/data.yaml`
- `outputs/detector/reports/detector_split.json`
- `outputs/detector/reports/yolo_conversion_report.json`

## Visualization Contract

**Command shape**:

```text
python -m src.training.train_detector visualize --config configs/detector.yaml
```

**Required behavior**:
- Save category-colored box overlays for sample training or validation images.
- Save mask overlays if segmentation masks exist.
- Report mask visualization unavailable when masks are absent.
- Keep YOLO bbox detector training as the first training path; defer YOLO segmentation training unless masks are clean and sufficiently covered.

**Required outputs**:
- Files under `outputs/detector/figures/`
- `outputs/detector/reports/visualization_report.json`

## Training Contract

**Command shape**:

```text
python -m src.training.train_detector train --config configs/detector.yaml --model yolo11n
python -m src.training.train_detector train --config configs/detector.yaml --model yolov8n
```

**Required behavior**:
- Use the generated detector `data.yaml`.
- Save trained detector artifacts under `outputs/detector/models/`.
- Save training metadata under `outputs/detector/reports/`.
- Do not run automatically during tests.
- Do not change classifier artifacts.

## Evaluation Contract

**Command shape**:

```text
python -m src.training.train_detector evaluate --config configs/detector.yaml --model-path outputs/detector/models/best.pt
```

**Required behavior**:
- Evaluate the generated detector validation split.
- Save mAP, precision, recall, per-category metrics, and validation prediction examples when available.
- Mark unavailable metrics explicitly when the detector backend does not provide them.
- In dry-run mode, report `evaluation_mode=dry_run`, `per_category_metrics_available=false`, and `validation_prediction_examples_available=false`.

**Required output**:
- `outputs/detector/reports/detector_evaluation.json`
