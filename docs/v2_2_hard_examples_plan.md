# V2.2 Hard-Example Training Support Plan

## Why V2.2 Exists

V2.2 prepares conservative hard-example training support from the automatic V2.1 error triage. The goal is to give the next classifier more attention to known V2B validation mistakes without using FiftyOne, test labels, public leaderboard tuning, detector retraining, or submission generation.

This is preparation only. Real training requires later approval.

Runtime artifacts for this candidate are isolated under:

```text
outputs/kaggle_v2_2/v2_2_hard_examples/
```

## Baseline Context

- Original V2B validation F1: `0.9269170766830673`
- Original V2B public score: `0.92121`
- V4.2 hybrid public score: `0.92181`
- V5A rejected:
  - validation_f1: `0.75855`
  - threshold: `0.29`
  - runtime_seconds: `31110.57`

V2B-Enhanced and V5A show why validation-only gains, aggressive changes, and slow candidates must be treated as risk until precision, target distribution, runtime, and rollback evidence are checked.

## Automatic Triage Source

The automatic V2.1 triage files separate all `617` V2B validation mistakes:

- hard negatives: `523`
- hard positives: `94`
- uncertain examples: `424`

Definitions:

- hard negatives are FP rows: `true_label = 0`, `v2b_prediction = 1`
- hard positives are FN rows: `true_label = 1`, `v2b_prediction = 0`
- uncertain examples are near-threshold or ambiguous examples and are not strong hard examples by default

The visual cause tags are automatic suggestions only. They are not human ground truth.

Current hard-example strategy is explicitly `analysis_only`: files are validated, counts are logged, and train-image availability is checked before training starts, but no active oversampling/reweighting is applied yet. Conservative sampler integration is a separate future task and must be approved before real V2.2 training.

## Safety Boundaries

- No test labels.
- No submission.
- No FiftyOne dependency.
- No detector retraining.
- No V2B artifact overwrite.
- No public leaderboard threshold tuning.
- Binary validation F1 uses positive class `target = 1` and `zero_division = 0`.
- Threshold selection uses validation data only.

## Smoke Command

Use this only after approval to run a short smoke execution:

```bash
python -m src.training.train_classifier --config configs/v2_2_hard_examples.yaml --max-epochs 1 --limit-train-batches 10 --limit-val-batches 5
```

## Real Training Command For Later Approval

Do not run this yet:

```bash
python -m src.training.train_classifier --config configs/v2_2_hard_examples.yaml
```

## Safety Gates Before Submission

Before any submission is considered, V2.2 must produce complete validation predictions, best threshold, metrics, runtime report, target distribution report, and a comparison against locked V2B. Accept only if validation F1 improves, precision stays stable, target `1` distribution does not drift dangerously, runtime remains acceptable, and rollback to V2B/V4.2 is clear.
