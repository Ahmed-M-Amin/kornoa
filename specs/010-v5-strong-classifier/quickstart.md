# Quickstart: V5 Strong Classifier

## Goal

Run a V5 strong classifier candidate that can challenge V4.2 public score `0.92181`, while preserving validation-only selection, probability exports, strict submission format, and runtime comparison against V2B.

## Preconditions

- Dataset paths are configured locally, on Kaggle, or on Colab.
- The V2B-compatible train/validation split can be resolved.
- Generated outputs are ignored under `outputs/kaggle_v5/v5_strong_classifier/`.
- V4.2 remains the current best public submission until V5 beats `0.92181`.

## Expected Output Files

```text
outputs/kaggle_v5/v5_strong_classifier/models/classifier_best.pth
outputs/kaggle_v5/v5_strong_classifier/predictions/val_classifier_predictions.csv
outputs/kaggle_v5/v5_strong_classifier/predictions/test_classifier_predictions.csv
outputs/kaggle_v5/v5_strong_classifier/reports/best_threshold.json
outputs/kaggle_v5/v5_strong_classifier/reports/classifier_metrics.json
outputs/kaggle_v5/v5_strong_classifier/submissions/submission_v5.csv
outputs/kaggle_v5/v5_strong_classifier/benchmarks/
```

## V5 Configuration Direction

Create or update:

```text
configs/v5_strong_classifier.yaml
```

Expected settings:

```yaml
experiment:
  name: v5_strong_classifier
  seed: 42

model:
  model_name: convnext_tiny
  fallback_model_name: efficientnet_b2
  image_size: 512
  pretrained: true

training:
  loss: focal
  sampler: class_balanced
  hard_example_strategy: analysis_only
  threshold_search: true

data:
  split_source: v2b_compatible

outputs:
  output_root: outputs/kaggle_v5/v5_strong_classifier
```

## Train V5

```powershell
python -u -m src.training.train_classifier `
  --config configs/v5_strong_classifier.yaml `
  --dataset-root $env:KRONES_DATASET_ROOT `
  --output-root outputs/kaggle_v5/v5_strong_classifier `
  --log-every-n-batches 25
```

Required checks after training:

- `classifier_best.pth` exists under the V5 model path.
- `best_threshold.json` includes selected threshold and validation F1.
- `classifier_metrics.json` includes validation distributions and V2B speed comparison.
- `val_classifier_predictions.csv` includes `image_id,true_label,prob_bad,classifier_prediction,target`.
- Hard-example oversampling is inactive unless explicitly enabled.
- V2B-compatible validation split is recorded.

## Generate Test Probabilities and Submission

```powershell
python -m src.inference.submission `
  --artifact-root outputs/kaggle_v5/v5_strong_classifier `
  --dataset-root $env:KRONES_DATASET_ROOT `
  --output-path outputs/kaggle_v5/v5_strong_classifier/submissions/submission_v5.csv `
  --save-test-predictions outputs/kaggle_v5/v5_strong_classifier/predictions/test_classifier_predictions.csv
```

Required checks after submission generation:

- `test_classifier_predictions.csv` includes `image_id,prob_bad,classifier_prediction,target`.
- `submission_v5.csv` includes exactly `image_id,target`.
- Test probabilities are in `[0, 1]`.
- Targets are binary integers.
- No test labels are used.
- Submission is generated but not submitted automatically.

## Run Tests

```powershell
python -m pytest tests -q
```

Focused tests expected during implementation:

```powershell
python -m pytest tests/test_classifier_v5.py tests/test_training_v5.py tests/test_inference_v5.py -q
```

## Acceptance Review

Report after implementation:

- Changed files summary.
- V5 config content.
- Test results.
- Validation F1.
- Selected threshold.
- Validation target distribution.
- Validation prediction distribution.
- Test prediction distribution.
- Submission row count.
- Inference time and V5/V2B speed ratio.
- Confirmation that test probabilities were generated.
- Confirmation that no generated outputs, artifacts, weights, predictions, reports, or submissions were committed.
- Manual public score comparison against V4.2 `0.92181`.

V5 becomes current best only if its public score is greater than `0.92181` and accepted inference is no more than 2x slower than V2B.
