# Quickstart: Detector / Segmentation Training

## Prerequisites

- SPEC-001 through SPEC-007 are complete.
- Current accepted fast classifier remains V2B EfficientNet-B1 `analysis_only`, public F1 `0.92121`.
- `train_annotations.json` and training images are available through configured dataset paths.
- `outputs/detector/` is an ignored generated-output path.

## Audit Annotations

```text
python -m src.training.train_detector audit --config configs/detector.yaml
```

Expected reports:

```text
outputs/detector/reports/category_mapping.json
outputs/detector/reports/annotation_audit.json
```

## Convert To Detector Dataset

```text
python -m src.training.train_detector convert --config configs/detector.yaml
```

Expected dataset:

```text
outputs/detector/dataset/images/train/
outputs/detector/dataset/images/val/
outputs/detector/dataset/labels/train/
outputs/detector/dataset/labels/val/
outputs/detector/dataset/data.yaml
outputs/detector/reports/detector_split.json
outputs/detector/reports/yolo_conversion_report.json
```

## Save Visual Samples

```text
python -m src.training.train_detector visualize --config configs/detector.yaml
```

Expected outputs:

```text
outputs/detector/figures/
outputs/detector/reports/visualization_report.json
```

## Train Detector Manually

Training is manual and must not run during tests. SPEC-008 trains the YOLO bounding-box detector first; YOLO segmentation training is deferred until segmentation masks are proven clean and sufficiently covered.

```text
python -m src.training.train_detector train --config configs/detector.yaml --model yolo11n
python -m src.training.train_detector train --config configs/detector.yaml --model yolov8n
```

Expected outputs:

```text
outputs/detector/models/
outputs/detector/reports/detector_training.json
```

## Evaluate Detector Manually

```text
python -m src.training.train_detector evaluate --config configs/detector.yaml --model-path outputs/detector/models/best.pt
```

Expected report:

```text
outputs/detector/reports/detector_evaluation.json
```

## Acceptance Checks

- `train_annotations.json` loads successfully.
- COCO image IDs map correctly to training image files.
- Defect categories are reported in stable detector class-index order.
- YOLO labels are generated for valid boxes only.
- No-annotation training images are included with empty YOLO label files and counted in conversion reports.
- Repeated conversion cleans only generated detector split folders and reports `output_dataset_cleaned=true`.
- Conversion reports include `split_strategy_used`; primary-defect-category stratification is used when practical, otherwise deterministic fallback is reported.
- `data.yaml` is valid and points to detector dataset folders.
- Train and validation detector splits are disjoint.
- Invalid boxes, masks, missing images, and unknown categories are reported.
- Sample box visualizations are saved.
- Mask validation/visualization is supported when masks exist; segmentation training remains deferred unless masks are clean.
- Detector training/evaluation commands are documented and runnable manually.
- Dry-run evaluation reports include `evaluation_mode=dry_run`, `per_category_metrics_available=false`, and `validation_prediction_examples_available=false`.
- Default classifier inference remains unchanged and detector-free.
