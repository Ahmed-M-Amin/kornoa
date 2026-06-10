# Quickstart: V2.1 Error Intelligence

## Goal

Produce the V2B validation error audit required before any new candidate training.

## Inputs

Prepare or confirm a config at:

```text
configs/v2_1_error_intelligence.yaml
```

The config must point to:

- locked V2B validation predictions
- validation labels
- locked V2B threshold report
- locked V2B metrics report
- optional detector evidence
- optional image-quality evidence
- output root

## Run

```bash
python -m src.analysis.v2_1_error_intelligence --config configs/v2_1_error_intelligence.yaml
```

## Expected Outputs

```text
outputs/analysis/v2_1_error_intelligence/
|-- audit/
|   `-- v2b_validation_error_audit.csv
|-- review_groups/
|   |-- high_confidence_false_positives.csv
|   |-- high_confidence_false_negatives.csv
|   |-- near_threshold_false_positives.csv
|   |-- near_threshold_false_negatives.csv
|   `-- over_rejected_reusable.csv
`-- reports/
    |-- v2b_error_intelligence_summary.json
    |-- v2b_error_group_counts.csv
    |-- v2b_target_distribution_report.json
    |-- v2b_optional_evidence_report.json
    `-- v2b_audit_provenance.json
```

## Acceptance Checks

The run is accepted only when:

- every V2B validation prediction appears exactly once in the audit
- computed F1, precision, recall, TP, FP, FN, and TN match the locked V2B baseline
- near-threshold groups use `abs(probability - threshold) <= 0.05`
- high-confidence groups use `abs(probability - threshold) >= 0.30`
- missing detector or image-quality evidence is reported, not silently dropped
- `safety_flags` confirm no test labels, no sample-solution labels, no public leaderboard tuning, no training, and no submission generation

## Verification Commands

```bash
python -m py_compile src/analysis/v2_1_error_intelligence.py
python -m pytest tests/test_v2_1_error_intelligence.py -v
python -m pytest tests -q
```

## Out Of Scope

This quickstart must not run training, generate a Kaggle submission, tune thresholds on the public leaderboard, or change V3/V4 detector fusion behavior.
