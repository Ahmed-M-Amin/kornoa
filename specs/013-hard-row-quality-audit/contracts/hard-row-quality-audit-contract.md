# Contract: Hard Row Quality Audit

## Purpose

Define the expected inputs, outputs, category rules, and safety guarantees for the Phase 2 hard-row quality audit workflow.

## Inputs

### Required Inputs

- One governed hard-row source set derived from prior validation-only analysis
- One row-level context source that preserves `image_id` and prior prediction context

### Optional Inputs

- Detector evidence source
- Crop-quality or image-quality evidence source

## Input Guarantees

- The governed hard-row source is the authoritative audit row set.
- The audit operates on validation-derived hard rows only.
- Optional evidence sources may be absent without invalidating the audit.
- Test labels, public leaderboard feedback, training execution, submission creation, and automatic relabel application are forbidden.

## Core Workflow Contract

1. Load and normalize the governed hard-row input set by `image_id`.
2. Reject duplicate hard-row identities before any audit output is written.
3. Join prior comparison context and optional evidence sources when available.
4. Produce exactly one row-level audit record for every governed hard row.
5. Assign exactly one primary audit category to each row from:
   - `likely_correct_but_hard`
   - `likely_ambiguous`
   - `likely_mislabeled`
   - `likely_preprocessing_issue`
   - `likely_detector_evidence_issue`
6. Persist one aggregated category summary and one ranked failure-mode summary.

## Output Contract

### Required Per-Run Outputs

- One row-level audit CSV
- One category-count summary CSV or JSON
- One ranked failure-mode summary JSON

### Required Row-Level Fields

- `image_id`
- `primary_audit_category`
- `audit_rationale`
- `hard_example_type`
- prior baseline or candidate context fields when available
- explicit optional-evidence status fields

## Safety Contract

The workflow must record:

```json
{
  "used_test_labels": false,
  "trained_model": false,
  "submission_created": false,
  "relabels_applied": false
}
```

It must also preserve:

- validation-derived audit behavior
- stable hard-row identity
- exact-one primary category per row
- optional-evidence compatibility

## Failure Contract

The workflow must fail clearly when:

- the governed hard-row source is missing
- duplicate hard-row identities remain after normalization
- required row-level context is unavailable and cannot be joined safely
- any forbidden input source implies test-label or leaderboard leakage
