# V1R Repaired Baseline Plan

V1R exists as a cheap repaired baseline for testing classifier training logic before spending Kaggle GPU quota on larger experiments. It uses EfficientNet-B0, a 384 px input, four epochs, weighted BCE, no weighted sampler, mild safe augmentation, and the V2B-compatible split.

V1R is not the final model. V2B remains the stable base, and V4.2 hybrid remains the current best public-score reference. V1R is a diagnostic lab for confirming that repaired training choices improve V1-style behavior without repeating the V5A mistake of a slow, unstable run.

V1R relates to V1 and V2B by keeping the cheap V1-scale backbone while adopting safer V2B-compatible validation behavior: validation predictions are saved, the best threshold is selected only on validation F1 with `target=1`, and test predictions are disabled by default.

Useful V1R outcomes:

- Validation F1 improves over the original V1 baseline.
- Error audit shows fewer V1-only failures without creating many V2B regressions.
- Runtime remains cheap enough for repeated smoke checks.
- The validation threshold is stable across nearby thresholds.

Smoke training command:

```powershell
python -u -m src.training.train_classifier --config configs/v1r_repaired_baseline.yaml --dataset-root "$KRONES_DATASET_ROOT" --output-root outputs/kaggle_v1r/v1r_repaired_baseline_smoke --max-train-samples 64 --max-val-samples 64 --epochs 1 --log-every-n-batches 5
```

Full training command:

```powershell
python -u -m src.training.train_classifier --config configs/v1r_repaired_baseline.yaml --dataset-root "$KRONES_DATASET_ROOT" --output-root outputs/kaggle_v1r/v1r_repaired_baseline
```

Do not submit V1R automatically. Review validation metrics, threshold report, validation predictions, and the V1/V2B audit before creating any submission.
