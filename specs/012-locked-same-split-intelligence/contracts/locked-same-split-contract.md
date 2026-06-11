# Contract: Locked Same-Split Error Intelligence

## Purpose

Define the expected inputs, outputs, decision logic, and safety guarantees for the Phase 1 locked same-split comparison workflow.

## Inputs

### Required Inputs

- Locked original V2B validation prediction file
- Selected V2.x candidate checkpoint
- Selected V2.x candidate threshold report
- Dataset root containing the locked validation image files
- Hard-negative file
- Hard-positive file
- Uncertain-example file

### Optional Inputs

- Detector evidence file
- Image-quality evidence file

## Input Guarantees

- The locked V2B validation prediction file is the authoritative same-row source.
- The compared candidate is one selected V2.x candidate per run, with V2.2 as the first required example.
- Optional evidence files may be absent without invalidating the core same-split comparison.
- Test labels, sample-submission labels, public leaderboard tuning inputs, training execution, and submission creation are forbidden.

## Core Workflow Contract

1. Load and normalize the locked V2B validation rows by `image_id`.
2. Run candidate inference on exactly those same rows.
3. Compute section metrics for:
   - `all_original_v2b_validation_rows`
   - `hard_example_rows_only`
   - `original_v2b_validation_excluding_hard_examples`
4. Partition every evaluated row into exactly one comparison bucket:
   - `v2b_wrong_candidate_correct`
   - `v2b_correct_candidate_wrong`
   - `both_wrong`
   - `both_correct`
5. Persist one rolling comparison-table record for the selected candidate.

## Output Contract

### Required Per-Run Outputs

- One summary JSON
- One metrics CSV
- One threshold sweep CSV
- Four row-level comparison CSVs

### Required Rolling Output

- One rolling comparison table containing one row per evaluated V2.x candidate

### Required Row-Level Fields

- `image_id`
- `target`
- `v2b_probability`
- `v2b_prediction`
- `candidate_probability`
- `candidate_prediction`
- `candidate_threshold`
- `is_hard_example`
- `hard_example_type`

## Decision Status Contract

Allowed statuses:

- `accepted`
- `rejected`
- `manual_review`

### `accepted`

The selected candidate is `accepted` only when:

- full locked-row F1 improves over V2B, and
- hard-example-only F1 does not regress by more than an absolute `0.01`

### `rejected`

The selected candidate is `rejected` when:

- full locked-row F1 is lower than V2B

### `manual_review`

The selected candidate is `manual_review` when:

- full locked-row F1 improves but hard-example-only F1 regresses beyond `0.01`, or
- the locked same-split evidence is mixed and not clearly stronger or weaker

## Safety Contract

The workflow must record:

```json
{
  "used_test_labels": false,
  "trained_model": false,
  "submission_created": false
}
```

It must also preserve:

- validation-only behavior
- locked-row identity
- one-candidate-per-run comparison
- compatibility with later optional evidence attachment

## Failure Contract

The workflow must fail clearly when:

- the locked V2B validation prediction file is missing
- the candidate checkpoint is missing
- the candidate threshold report is missing
- row identities do not align to the same locked row set
- any forbidden input source implies test-label or leaderboard leakage
