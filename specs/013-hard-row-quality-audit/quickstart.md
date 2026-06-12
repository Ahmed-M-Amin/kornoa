# Quickstart: Hard Row Quality Audit

## Goal

Run one governed hard-row audit over the hardest validation-derived rows, assign exactly one primary audit category per row, and generate a summary that ranks the most fixable failure modes.

## Inputs

Prepare or confirm a config at:

```text
configs/hard_row_quality_audit.yaml
```

The config must point to:

- the governed hard-row source list
- one row-level context source with stable `image_id`
- an analysis output root

Optional later inputs:

- detector evidence
- crop-quality or image-quality evidence

## Run

```bash
python -m src.analysis.hard_row_quality_audit run --config configs/hard_row_quality_audit.yaml
```

## Expected Outputs

```text
outputs/analysis/hard_row_quality_audit/
|-- audit/
|   `-- hard_row_audit.csv
`-- reports/
    |-- hard_row_audit_category_counts.csv
    `-- hard_row_failure_mode_summary.json
```

## Acceptance Checks

The run is accepted only when:

- every governed hard row appears exactly once in the audit output
- every row receives exactly one primary audit category
- every row includes a non-empty audit rationale
- category counts sum to the full audited row count
- the failure-mode summary identifies ranked next-action groups
- safety flags confirm no training, no submission creation, no relabel application, and no test-label use

## Verification Commands

```bash
python -m py_compile src/analysis/hard_row_quality_audit.py
python -m pytest tests/test_hard_row_quality_audit.py -q
python -m pytest tests -q
```

## Out Of Scope

This quickstart must not train a model, create a Kaggle submission, tune thresholds on public leaderboard results, apply relabels automatically, or change detector or classifier behavior directly.
