# Contract: V1 Evaluation, Submission, Benchmark, and Hard-Example Mining

## V1 Prediction Contract

Inputs:

- Concrete V1 run artifact root or explicit model and threshold paths.
- One or more image paths or a test image directory plus sample image IDs.
- Optional device and batch-size configuration.

Outputs:

- Ordered prediction records with `image_id`, probability for `1 = Not Reusable`, saved threshold, and binary target.

Rules:

- The saved threshold is required.
- The default concrete run root is `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1`.
- If the parent root `artifacts/kaggle_v1_artifacts/outputs` is supplied, resolve its `kaggle_v1` child as the concrete run root.
- The concrete run root must contain `models/classifier_effnet_b0_best.pth`, `reports/classifier_metrics.json`, `reports/best_threshold.json`, and `predictions/val_classifier_predictions.csv`.
- ROI/preprocessing must match the V1 classifier path.
- No hard-example files are created by prediction.

## Submission Contract

Inputs:

- Dataset root or explicit `sample_submission.csv` and test image directory.
- Concrete V1 run artifact root, optional parent artifact root, or explicit model and threshold paths.
- Output path, defaulting to `outputs/submissions/submission_v1.csv`.

Outputs:

- Sample-aligned V1 submission CSV.

Rules:

- Preserve sample submission row order.
- Preserve the sample submission's binary label column name when it is `target` or `label`.
- Fail clearly when a sample image is missing.
- Do not run hard-example mining.
- Parent-root resolution follows the same `kaggle_v1` child detection rule as prediction.

## Benchmark Contract

Inputs:

- Image directory or sample submission plus test image directory.
- Concrete V1 run artifact root, optional parent artifact root, or explicit model and threshold paths.
- Output path, defaulting to `outputs/benchmarks/v1_inference_benchmark.json`.

Outputs:

- Benchmark JSON with image count, total runtime, average milliseconds per image, images per second, and artifact path references.

Rules:

- Measures the fast V1 classifier inference path.
- Does not run detector, Grad-CAM, feature memory bank, hard-example mining, or submission scoring.
- Parent-root resolution follows the same `kaggle_v1` child detection rule as prediction.

## Hard-Example Mining Contract

Inputs:

- V1 validation prediction CSV.
- Saved threshold record or threshold value.
- Output directory, defaulting to `outputs/hard_examples`.
- Summary path, defaulting to `outputs/reports/hard_example_summary.json`.
- Optional uncertainty margin and high-loss proxy row limit.

Outputs:

- `false_positives.csv`
- `false_negatives.csv`
- `uncertain.csv`
- `high_loss_samples.csv`
- `hard_example_summary.json`

Rules:

- Uses validation predictions only.
- Default validation predictions come from `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1/predictions/val_classifier_predictions.csv`.
- Existing manual hard-example files may exist under `artifacts/kaggle_v1_artifacts/outputs/hard_examples`, but they are not required inputs.
- Does not load model checkpoint.
- Does not read test images.
- Does not generate a Kaggle submission.
- High-loss proxy ranking is computed from saved probability severity: false positives with high probability, false negatives with low probability, then near-threshold uncertain cases.
