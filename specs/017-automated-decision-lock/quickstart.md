# Quickstart: Spec 017 Automated Decision Lock

## 1. Confirm Spec 016 outputs exist

```powershell
Test-Path outputs/analysis/data_quality_audit/data_quality_master.csv
Test-Path outputs/analysis/data_quality_audit/cleaned_training_manifest.csv
Test-Path outputs/analysis/data_quality_audit/manual_review_required_rows.csv
Test-Path outputs/analysis/data_quality_audit/reports/data_quality_summary.json
```

Expected: all commands print `True`.

## 2. Confirm Spec 014 blocked-row outputs exist

```powershell
Test-Path outputs/analysis/hard_row_visual_review/reports/phase3_allowed_hard_training_rows.csv
Test-Path outputs/analysis/hard_row_visual_review/reports/blocked_roi_pipeline_bug_rows.csv
Test-Path outputs/analysis/hard_row_visual_review/reports/blocked_suspected_mislabel_rows.csv
Test-Path outputs/analysis/hard_row_visual_review/reports/blocked_other_high_risk_rows.csv
```

Expected: all commands print `True`.

## 3. Run the decision-lock workflow

```powershell
python -m src.analysis.data_quality_decision_lock run --config configs/data_quality_decision_lock.yaml
```

Expected: command exits successfully and prints or writes a summary with safety flags:

```text
no_training_started: true
no_submission_created: true
no_test_labels_used: true
no_leaderboard_tuning: true
original_labels_unchanged: true
```

## 4. Verify required outputs

```powershell
Test-Path outputs/analysis/data_quality_decision_lock/decision_master.csv
Test-Path outputs/analysis/data_quality_decision_lock/approved_cleaned_training_manifest.csv
Test-Path outputs/analysis/data_quality_decision_lock/adjudication_queue.csv
Test-Path outputs/analysis/data_quality_decision_lock/reports/decision_lock_summary.json
```

Expected: all commands print `True`.

## 5. Run tests

```powershell
python -m pytest tests/test_data_quality_decision_lock.py -q
python -m py_compile src/analysis/data_quality_decision_lock.py
```

Expected: tests pass and compile check exits successfully.

## 6. Do not train yet

The next training spec must consume:

```text
outputs/analysis/data_quality_decision_lock/approved_cleaned_training_manifest.csv
```

No training, submission, test-label usage, leaderboard tuning, or original-label modification is allowed in Spec 017.
