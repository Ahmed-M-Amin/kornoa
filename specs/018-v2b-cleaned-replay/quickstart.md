# Quickstart: Spec 018 V2B Cleaned Replay

## 1. Confirm Spec 017 outputs exist

```powershell
Test-Path outputs/analysis/data_quality_decision_lock/approved_cleaned_training_manifest.csv
Test-Path outputs/analysis/data_quality_decision_lock/auto_exclude_rows.csv
Test-Path outputs/analysis/data_quality_decision_lock/needs_adjudication_rows.csv
Test-Path outputs/analysis/data_quality_decision_lock/deferred_uncertain_rows.csv
Test-Path outputs/analysis/data_quality_decision_lock/reports/decision_lock_summary.json
```

Expected: all commands print `True`.

## 2. Run cleaned replay dry-run validation

```powershell
python -m src.training.train_classifier --config configs/v2b_cleaned_replay_training.yaml --dry-run-cleaned-replay
```

Expected output report:

```text
outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/cleaned_replay_dry_run_validation.json
```

Expected safety values:

```text
auto_exclude_overlap_count: 0
needs_adjudication_overlap_count: 0
deferred_overlap_count: 0
missing_train_label_count: 0
missing_train_image_count: 0
training_started: false
no_test_labels_used: true
no_leaderboard_tuning: true
no_submission_created: true
```

## 3. Run tests before training

```powershell
python -m pytest tests/test_v2b_cleaned_replay_training.py -q
python -m py_compile src/training/train_classifier.py
```

Expected: tests pass and compile check succeeds.

## 4. Train cleaned replay only after dry run passes

```powershell
python -m src.training.train_classifier --config configs/v2b_cleaned_replay_training.yaml
```

Expected outputs:

```text
outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/models/classifier_best.pth
outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/predictions/val_classifier_predictions.csv
outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/best_threshold.json
outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/classifier_metrics.json
outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/cleaned_replay_comparison.json
```

## 5. Read comparison before any submission work

```powershell
Get-Content outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/cleaned_replay_comparison.json
```

Decision rule:

```text
Use cleaned replay only if validation F1 beats locked V2B and all safety gates pass.
Reject cleaned replay if validation F1 is worse, equal without justified tie-break, or any safety gate fails.
```
