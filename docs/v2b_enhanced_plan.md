# V2B-Enhanced Aggressive Ladder Plan

## Scope

This ladder keeps the accepted V2B EfficientNet-B1 classifier as the base and tests controlled fine-tuning plus validation-selected TTA. It does not use detector fusion, V3 artifacts, test labels, focal loss, or a weighted sampler.

## Why V2B Remains The Accepted Base

V2B is the accepted classifier baseline for this project because it already has the best known public-score evidence among the classifier-only variants. The V2B-enhanced ladder starts from that checkpoint instead of replacing the decision source.

## Why This Starts From V2B Instead Of V5A

V5A is documented as a failed stronger-classifier attempt, so this ladder avoids promoting V5A behavior into the submission path. The work stays close to the known V2B operating point and changes only image size continuation, mild fine-tuning, and TTA.

## Why No Focal Loss Or Weighted Sampler

The ladder uses plain BCE because the goal is a conservative continuation from the accepted V2B base. Focal loss and weighted sampling are both disabled to avoid shifting the learned decision boundary aggressively during a short fine-tune.

## Why No V3 Fusion Is Included

V3 detector fusion is out of scope for this ladder. The TTA stage is classifier-only, which keeps the evaluation isolated from the hybrid inference work and avoids detector-driven test behavior.

## Fine-Tuning Order

The 384 stage starts from the accepted V2B EfficientNet-B1 checkpoint with a short low-learning-rate continuation. The 448 stage then starts from the 384 enhanced checkpoint and runs an even shorter continuation at lower learning rate so the larger image size is introduced after the base has been stabilized.

## Gated Horizontal Flip TTA

`no_tta` uses the base probability directly. `hflip_tta` averages base and horizontal-flip probabilities for every image. `gated_hflip_tta` averages base and horizontal-flip probabilities only when the base probability is within the configured margin of the candidate threshold; confident images keep the base probability.

## Commands

384 smoke training:

```bash
python -u -m src.training.train_classifier \
  --config configs/v2b_enhanced_384.yaml \
  --dataset-root "$KRONES_DATASET_ROOT" \
  --output-root outputs/kaggle_v2b_enhanced/384_smoke \
  --max-train-samples 64 \
  --max-val-samples 64 \
  --epochs 1 \
  --log-every-n-batches 5
```

384 Kaggle training:

```bash
python -u -m src.training.train_classifier \
  --config configs/v2b_enhanced_384.yaml \
  --dataset-root "$KRONES_DATASET_ROOT" \
  --output-root outputs/kaggle_v2b_enhanced/384 \
  --log-every-n-batches 25
```

448 Kaggle continuation:

```bash
python -u -m src.training.train_classifier \
  --config configs/v2b_enhanced_448.yaml \
  --dataset-root "$KRONES_DATASET_ROOT" \
  --output-root outputs/kaggle_v2b_enhanced/448 \
  --log-every-n-batches 25
```

TTA validation search:

```bash
python -m src.inference.v2b_enhanced_tta \
  --config configs/v2b_enhanced_tta.yaml
```

## Warnings

Do not submit automatically. Review the generated `outputs/kaggle_v2b_enhanced/tta/submissions/submission_v2b_enhanced_tta.csv` manually before any Kaggle action.

Do not use test labels. The TTA runner rejects label-like columns in test prediction input.

Do not train this ladder inside Codex. The commands above are for local smoke checks or Kaggle execution by the project owner.

Do not modify V3, retrain detectors, or include detector fusion in this ladder.
