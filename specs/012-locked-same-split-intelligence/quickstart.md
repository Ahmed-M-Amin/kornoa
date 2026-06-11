# Quickstart: Locked Same-Split Error Intelligence

## Goal

Run one locked same-split comparison between original V2B and one selected V2.x candidate, with V2.2 as the first required example, then update the rolling candidate comparison table.

## Inputs

Prepare or confirm a config at:

```text
configs/v2_2_hard_examples.yaml
```

The config must point to:

- locked original V2B validation predictions
- selected candidate checkpoint
- selected candidate threshold report
- dataset root for the locked validation images
- hard-negative, hard-positive, and uncertain-example files
- analysis output root

Optional later inputs:

- detector evidence
- image-quality evidence

## Run

```bash
python -m src.analysis.v2_2_same_split_eval run --config configs/v2_2_hard_examples.yaml
```

## Expected Outputs

```text
outputs/analysis/v2_2_same_split_eval/
|-- reports/
|   |-- v2_2_same_split_summary.json
|   |-- v2_2_same_split_metrics.csv
|   |-- v2_2_threshold_sweep_on_v2b_val.csv
|   `-- candidate_same_split_comparison.csv
`-- errors/
    |-- v2b_wrong_v22_correct.csv
    |-- v2b_correct_v22_wrong.csv
    |-- both_wrong.csv
    `-- both_correct.csv
```

## Acceptance Checks

The run is accepted only when:

- V2B and the selected candidate are evaluated on the exact same locked validation rows
- metrics exist for all three required sections
- every evaluated row appears in exactly one comparison bucket
- the rolling comparison table adds or updates one candidate record
- safety flags confirm no test labels, no training, and no submission creation
- a candidate marked `accepted` shows full locked-row improvement and no hard-example-only F1 regression beyond `0.01`

## Verification Commands

```bash
python -m py_compile src/analysis/v2_2_same_split_eval.py
python -m pytest tests/test_v2_2_same_split_eval.py -q
python -m pytest tests -q
```

## Out Of Scope

This quickstart must not train a model, create a Kaggle submission, tune thresholds on public leaderboard results, change hybrid detector/rule behavior, or compare multiple candidates in one evaluation run.
