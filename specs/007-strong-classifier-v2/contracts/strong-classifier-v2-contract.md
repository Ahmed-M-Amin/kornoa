# Contract: Strong Classifier V2

## Scope

This contract defines the expected user-facing workflows and artifact contracts for SPEC-007. It is classifier-only and must not introduce detector, segmentation, Grad-CAM, dashboard, feature memory bank, default ensemble, distillation, or hybrid inference behavior.

## Training Workflow Contract

**Command shape**:

```text
python -m src.training.train_classifier --config configs/classifier_v2.yaml
```

**Required inputs**:
- Training labels and images from the configured dataset root.
- Existing ROI/preprocessing behavior from earlier specs.
- V1 baseline artifacts for comparison.
- Optional validation-derived hard-example files for offline analysis or explicitly enabled oversampling.
- Optional explicit V2 artifact source containing `val_classifier_predictions.csv` and `best_threshold.json` for V2-generated hard examples.

**Required behavior**:
- Report train and validation class counts before training.
- Train V2 candidates with the configured backbone, image size, imbalance strategy, augmentation recipe, and hard-example strategy.
- Treat `analysis_only` as the default hard-example strategy.
- Allow only `none`, `analysis_only`, and `oversample` hard-example strategies.
- Run hard-example oversampling only when `hard_example_strategy=oversample`.
- Verify before oversampling that every hard-example image ID used for training is absent from the current V2 validation split.
- Report hard-example rows loaded, eligible for training, excluded because they are in V2 validation, used for oversampling, and training/validation disjointness confirmation.
- Report `hard_example_source_used` and `hard_example_source_type`.
- When `hard_example_source` explicitly points to a V2 artifact folder with validation predictions and threshold, generate hard examples from those V2 outputs before using existing memory.
- Warn or fail if an explicit V2 artifact source resolves to nested `v1-artifacts`.
- Select best checkpoint by validation F1 after threshold search.
- When candidate validation F1 differs by `<= 0.002`, select the faster candidate.
- Save selected V2 artifacts under `outputs/kaggle_v2/`.
- Do not use test images, sample submission rows, or Kaggle public score for training, threshold tuning, or model selection.

**Required outputs**:
- `outputs/kaggle_v2/models/classifier_best.pth`
- `outputs/kaggle_v2/reports/classifier_metrics.json`
- `outputs/kaggle_v2/reports/best_threshold.json`
- `outputs/kaggle_v2/predictions/val_classifier_predictions.csv`

## Comparison Workflow Contract

**Command shape**:

```text
python -m src.training.train_classifier --config configs/classifier_v2.yaml --compare-v1
```

**Required inputs**:
- V1 classifier metrics.
- V1 best threshold.
- V1 benchmark report.
- Hard-example summary or loaded hard-example CSV counts.
- Selected V2 metrics, threshold, predictions, and benchmark.

**Required behavior**:
- Compare V2 against V1 for validation F1, Kaggle public score when available, threshold, false positives, false negatives, uncertain samples, inference speed, model size/backbone, and image size.
- Report whether V2 meets the +0.02 validation-F1 minimum target.
- Report whether selected V2 stays within the maximum 2x V1 inference-time ceiling.
- Report the close-F1 tolerance of `<= 0.002` and whether it affected candidate selection.
- Mark unavailable Kaggle V2 score explicitly when no score exists.

**Required output**:
- `outputs/kaggle_v2/reports/v1_vs_v2_comparison.json`

## Submission Workflow Contract

**Command shape**:

```text
python -m src.inference.submission --artifact-root outputs/kaggle_v2 --output-path outputs/kaggle_v2/submissions/submission_v2.csv
```

**Required behavior**:
- Load selected V2 classifier checkpoint and saved V2 threshold.
- Predict probability for `1 = Not Reusable`.
- Preserve sample submission image order.
- Write only `image_id,target` columns.
- Use binary target values only.
- Do not run hard-example memory, detector, feature memory bank, ensemble, Grad-CAM, or dashboard behavior.

**Required output**:
- `outputs/kaggle_v2/submissions/submission_v2.csv`

## Benchmark Workflow Contract

**Command shape**:

```text
python -m src.inference.benchmark --artifact-root outputs/kaggle_v2 --output-path outputs/kaggle_v2/benchmarks/v2_inference_benchmark.json
```

**Required fields**:
- `total_images`
- `total_time_seconds`
- `average_time_per_image`
- `images_per_second`
- `batch_size`
- `device`
- `model_name`
- `image_size`
- `artifact_root`
- `speed_multiplier_vs_v1`
- `within_v2_speed_ceiling`

**Required behavior**:
- Benchmark selected V2 classifier-only inference.
- Report speed cost relative to V1.
- Fail or mark non-selectable if selected V2 exceeds 2x V1 inference time.

## Artifact Confidentiality Contract

- Private dataset files and generated artifacts must stay local/private.
- V2 generated outputs must remain under ignored output paths.
- No test labels or test-derived pseudo-labels may appear in training, threshold, or comparison artifacts.
