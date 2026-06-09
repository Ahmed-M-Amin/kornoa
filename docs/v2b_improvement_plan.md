# V2B Improvement Plan

V2B remains the strongest stable base because it improved validation F1 over V1 while preserving the classifier-first workflow that already performs well. The V1 versus V2B audit showed real gains, including many errors that V2B fixed, but it also showed a bounded set of V2B regressions and a stable threshold range rather than a collapse in behavior.

V2B still gets some cases wrong. The right next step is targeted review, not aggressive retraining. Hard examples stay analysis-only because the previous hard-example path is exactly the kind of feedback loop that can distort the decision boundary and repeat V2B-HE behavior. That is why this stage forbids oversampling, focal loss, weighted sampler, and aggressive class weighting.

V5B is preparation-only. It defines a conservative fine-tune starting from the accepted V2B EfficientNet-B1 checkpoint, using a short three-epoch window, low learning rate, BCE loss, mild safe augmentation, and the V2B-compatible split. This is intentionally a smoke-test candidate only. No real training and no submission should happen yet.

V2B error review command:

```powershell
python -m src.analysis.v2b_error_review --config configs/v2b_error_review.yaml
```

Tiny smoke fine-tune command only:

```powershell
python -u -m src.training.train_classifier --config configs/v5b_safe_finetune.yaml --dataset-root "$KRONES_DATASET_ROOT" --output-root outputs/kaggle_v5b/v5b_safe_finetune_smoke --max-train-samples 64 --max-val-samples 64 --epochs 1 --log-every-n-batches 5
```

No real training and no submission yet.
