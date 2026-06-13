# Quickstart: Hard-Row Technical Triage

## Goal

Run the governed hard-row technical-triage workflow, complete annotation and ROI evidence for the governed hard rows, and export the validated Phase 3 candidate package without starting any training.

## Inputs

Confirm the config at:

```text
configs/hard_row_visual_review.yaml
```

It must point to:

- the governed hard-row audit output
- the training images directory
- the training annotation JSON
- the hard-row review output root
- the governed hard-row count in `analysis.expected_hard_row_count`
- the expected Phase 3 package totals in `evidence_completion.expected_rows`, `expected_phase3_allowed_count`, and `expected_phase3_blocked_count`

## Run The Governed Review Workflow

```bash
python -m src.analysis.hard_row_visual_review --config configs/hard_row_visual_review.yaml
```

## Complete Evidence

```bash
python -m src.analysis.hard_row_visual_review --config configs/hard_row_visual_review.yaml --complete-evidence
```

## Export Phase 3 Candidate Package

```bash
python -m src.analysis.hard_row_visual_review --config configs/hard_row_visual_review.yaml --export-phase3-candidates
```

## Expected Outputs

```text
outputs/analysis/hard_row_visual_review/
|-- manual_review_template.csv
|-- review_manifest.csv
|-- evidence_assets/
|-- contact_sheets/
`-- reports/
    |-- final_training_action_plan_evidence_aware.csv
    |-- hard_row_annotation_evidence.csv
    |-- hard_row_crop_quality_evidence.csv
    |-- hard_row_evidence_summary.json
    |-- phase3_allowed_hard_training_rows.csv
    |-- phase3_artifact_focus_rows.csv
    |-- phase3_tiny_low_contrast_rows.csv
    |-- phase3_general_candidate_rows.csv
    |-- blocked_roi_pipeline_bug_rows.csv
    |-- blocked_suspected_mislabel_rows.csv
    |-- blocked_other_high_risk_rows.csv
    `-- phase3_candidate_package_summary.json
```

## Acceptance Checks

The workflow is accepted only when:

- the governed review outputs preserve one unique record per governed hard-row `image_id`
- evidence completion produces annotation and crop-quality records for the full governed pool or explicit missing statuses
- the evidence-aware action plan includes one recommended action and one `phase3_use_allowed` value per governed hard row
- the Phase 3 package export validates input totals before writing outputs
- no blocked rows appear in any allowed package output
- safety flags confirm no training, no submission creation, and no test-label use

## Verification Commands

```bash
python -m py_compile src/analysis/hard_row_visual_review.py
python -m pytest tests/test_hard_row_visual_review.py -q
```

## Out Of Scope

This quickstart must not train a Phase 3 classifier, create a Kaggle submission, tune thresholds on public leaderboard feedback, or modify original labels.
