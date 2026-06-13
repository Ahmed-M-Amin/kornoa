# Contract: Spec 018 V2B Cleaned Replay

## Dry-Run Command

```powershell
python -m src.training.train_classifier --config configs/v2b_cleaned_replay_training.yaml --dry-run-cleaned-replay
```

The command must:

- Validate the Spec 017 approved manifest exists and is non-empty.
- Validate excluded, deferred, and adjudication row files exist.
- Fail if approved manifest overlaps excluded, deferred, or adjudication rows.
- Validate original train labels and training images exist.
- Validate every approved row exists in labels and images.
- Validate locked V2B baseline checkpoint, predictions, threshold, metrics, and runtime report exist.
- Create only validation/report output directories.
- Write `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/cleaned_replay_dry_run_validation.json`.
- Not train.
- Not create a submission.
- Not use test labels.
- Not tune from leaderboard feedback.

## Training Command

```powershell
python -m src.training.train_classifier --config configs/v2b_cleaned_replay_training.yaml
```

The command must train only after dry-run validation conditions are satisfied.

Required outputs:

- `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/models/classifier_best.pth`
- `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/predictions/val_classifier_predictions.csv`
- `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/best_threshold.json`
- `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/classifier_metrics.json`
- `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/cleaned_replay_run_manifest.json`
- `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/cleaned_replay_comparison.json`
- `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/approved_manifest_snapshot.csv`
- `outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/benchmarks/runtime_metadata.json`

## Required Config Sections

- `experiment`
- `classifier`
- `data`
- `output`
- `cleaned_replay`
- `baseline`
- `safety`

## Required Safety Flags

- `safety.allow_test_labels: false`
- `safety.public_leaderboard_input: false`
- `safety.generate_submission: false`
- `safety.apply_relabels: false`

## Required Dry-Run Report Fields

- `approved_manifest_count`
- `auto_exclude_overlap_count`
- `needs_adjudication_overlap_count`
- `deferred_overlap_count`
- `missing_train_label_count`
- `missing_train_image_count`
- `baseline_artifacts_available`
- `validation_split_status`
- `training_started`
- `no_test_labels_used`
- `no_leaderboard_tuning`
- `no_submission_created`
- `warnings`

## Required Comparison Fields

- `baseline_name`
- `baseline_validation_f1`
- `cleaned_replay_validation_f1`
- `f1_delta`
- `baseline_threshold`
- `cleaned_replay_threshold`
- `validation_row_count`
- `decision`
- `decision_reason`
- `safety_gate_status`
- `artifact_paths`
