# V2.2 Same-Split Validation Evaluation Design

## Objective

Add an offline evaluator that compares the trained V2.2 classifier against V2B on the original locked V2B validation split before any Kaggle submission work.

This design is validation-only. It must not train, create submissions, use test labels, overwrite V2B artifacts, or modify the trained V2.2 checkpoint.

## Scope

In scope:

- Load the original V2B validation prediction rows and recover the exact `image_id` set and binary targets.
- Run V2.2 inference on exactly those same images.
- Apply the trained V2.2 threshold from `best_threshold.json`.
- Compute V2B and V2.2 metrics on identical rows using binary F1 with positive class `1` and `zero_division=0`.
- Produce sectioned metrics for:
  - `all_original_v2b_validation_rows`
  - `hard_example_rows_only`
  - `original_v2b_validation_excluding_hard_examples`
- Write comparison error splits and a threshold sweep for V2.2 on the original V2B validation rows.

Out of scope:

- Training or fine-tuning.
- Submission creation.
- Threshold tuning for model selection.
- Test-label usage.
- Changes to V2B predictions or V2.2 model artifacts.

## Recommended Approach

Implement a dedicated module at `src/analysis/v2_2_same_split_eval.py` and keep it separate from training and inference entrypoints.

This is the cleanest fit for the repository because:

- `src/analysis/` already holds validation-only reporting workflows.
- V2.1 logic is close in shape, but this task compares two models on a shared row set instead of auditing one locked baseline.
- Keeping the evaluator separate avoids accidental coupling to training or submission flows.

## Alternatives Considered

### 1. Extend `src/analysis/v2_1_error_intelligence.py`

Rejected because the responsibilities diverge. V2.1 validates a locked V2B baseline and emits review bands; V2.2 same-split evaluation compares two models and runs fresh inference from a trained checkpoint.

### 2. Add the evaluator under `src/inference/`

Rejected because the task is not general prediction infrastructure. It is an offline evaluation gate with strict safety boundaries and report generation.

## Architecture

The implementation should have three layers inside one module.

### 1. Config and safety layer

Responsibilities:

- Load `configs/v2_2_hard_examples.yaml`.
- Resolve:
  - dataset root
  - V2B validation predictions CSV
  - V2.2 checkpoint path
  - V2.2 threshold JSON
  - hard-example CSV paths
  - analysis output root
- Enforce safety flags:
  - no training
  - no submission creation
  - no test labels
  - no V2B artifact overwrite

Failure mode:

- Raise a dedicated `V22SameSplitEvalError` with direct messages for missing checkpoint, missing V2B predictions, invalid threshold JSON, or forbidden config state.

### 2. Evaluation layer

Responsibilities:

- Load the original V2B validation rows.
- Normalize `image_id` values the same way V2.1 does.
- Resolve image paths from `data.dataset_root`.
- Run V2.2 inference on exactly those rows using the trained V2.2 checkpoint.
- Apply the persisted threshold, currently expected to be `0.42`.
- Compute metrics for both V2B and V2.2 on the same row set.
- Derive the hard-example subset by unioning all configured hard-example CSVs.
- Build the required three sections:
  - all rows
  - hard-example rows only
  - same rows excluding hard examples

Metrics to compute per section:

- F1
- precision
- recall
- TP
- FP
- TN
- FN
- prediction distribution
- target distribution

### 3. Reporting layer

Responsibilities:

- Write:
  - `outputs/analysis/v2_2_same_split_eval/reports/v2_2_same_split_summary.json`
  - `outputs/analysis/v2_2_same_split_eval/reports/v2_2_same_split_metrics.csv`
  - `outputs/analysis/v2_2_same_split_eval/reports/v2_2_threshold_sweep_on_v2b_val.csv`
  - `outputs/analysis/v2_2_same_split_eval/errors/v2b_wrong_v22_correct.csv`
  - `outputs/analysis/v2_2_same_split_eval/errors/v2b_correct_v22_wrong.csv`
  - `outputs/analysis/v2_2_same_split_eval/errors/both_wrong.csv`
  - `outputs/analysis/v2_2_same_split_eval/errors/both_correct.csv`
- Include safety flags in the summary JSON:
  - `used_test_labels: false`
  - `trained_model: false`
  - `submission_created: false`
- Add a `recommended_decision` field driven by same-split evidence rather than the optimistic training-run validation score.

## Data Model

Core row fields for the joined comparison frame:

- `image_id`
- `target`
- `v2b_probability`
- `v2b_prediction`
- `v22_probability`
- `v22_prediction`
- `v22_threshold`
- `is_hard_example`
- `hard_example_type`

Derived comparison partitions:

- `v2b_wrong_v22_correct`
- `v2b_correct_v22_wrong`
- `both_wrong`
- `both_correct`

## CLI

Add this command:

```bash
python -m src.analysis.v2_2_same_split_eval run --config configs/v2_2_hard_examples.yaml
```

The CLI should:

- parse a `run` subcommand
- default to `configs/v2_2_hard_examples.yaml`
- print a short success message with the summary path
- return a non-zero exit code without traceback for expected input errors

## Testing Strategy

Add `tests/test_v2_2_same_split_eval.py` with synthetic fixtures and inference stubbing.

Required coverage:

- evaluates exactly the same row set for V2B and V2.2
- computes binary F1 with `pos_label=1` and `zero_division=0`
- splits hard-example rows from the original validation rows
- fails clearly when the V2.2 checkpoint is missing
- fails clearly when V2B validation predictions are missing
- does not read test labels
- does not create submissions
- does not train

Recommended implementation detail:

- patch the V2.2 inference function in tests so synthetic probabilities can be injected without loading a real checkpoint
- keep one narrow integration-style test for path validation and report writing

## Verification

Required commands after implementation:

```bash
python -m py_compile src/analysis/v2_2_same_split_eval.py
python -m pytest tests -q
```

## Risks and Controls

Risk:
The dataset path structure may differ between local and Kaggle layouts.

Control:
Resolve image paths through config and validate every requested `image_id` before inference starts.

Risk:
V2B validation predictions may carry labels inline or require fallback to dataset labels.

Control:
Support both patterns, but reject any path or schema that points at test labels or submission labels.

Risk:
Hard-example CSVs may overlap.

Control:
Normalize `image_id`, union rows, and persist a stable `hard_example_type` summary instead of duplicating evaluation rows.

## Implementation Plan Boundary

This document is only the approved design for the evaluator. The next step is implementation in the analysis lane with tests first, then code, then compile and pytest verification.
