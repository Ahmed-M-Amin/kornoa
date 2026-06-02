# Quickstart: Binary Classifier Training

## Automated Validation

Run the SPEC-005 test target once tasks are implemented:

```powershell
pytest tests/test_classifier.py tests/test_training.py -q
```

Expected behavior:

- Synthetic labels are validated as binary.
- An 80/20 stratified split is reproducible from a fixed seed.
- Weighted binary loss is used for V1 imbalance handling.
- Threshold search chooses the lowest threshold among tied best-F1 candidates.
- Synthetic training smoke completes in under one minute.
- Model, metrics, threshold, and validation prediction artifacts are generated
  under ignored output locations.

## Manual Synthetic Smoke Check

Use synthetic fixtures only:

```powershell
python -m src.training.train_classifier --config configs/classifier.yaml --dataset-root tests/fixtures/synthetic_dataset/classifier_cases --synthetic-smoke
```

The command should create:

```text
outputs/models/classifier_effnet_b0_best.pth
outputs/reports/classifier_metrics.json
outputs/reports/best_threshold.json
outputs/predictions/val_classifier_predictions.csv
```

## Private Dataset Training

When valid private dataset access is available, point `configs/paths.yaml` or
the training command at the private dataset root. Keep generated weights,
metrics, thresholds, and predictions under ignored output directories and do not
commit private-derived artifacts.

## Scope Guard

SPEC-005 validation should not generate Kaggle submissions, infer test labels,
train a detector, run hybrid inference, produce Grad-CAM, build dashboard code,
or create memory-bank artifacts.
