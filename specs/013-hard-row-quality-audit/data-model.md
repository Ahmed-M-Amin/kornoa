# Data Model: Hard Row Quality Audit

## Hard Audit Input Row

Represents one governed hard row entering the Phase 2 audit.

**Fields**

- `image_id`: normalized hard-row image identifier.
- `target`: binary true label, `0` or `1`, when available from prior validation-only analysis.
- `baseline_probability`: original locked-baseline or governing baseline probability for `target = 1`, when available.
- `baseline_prediction`: original locked-baseline binary prediction, when available.
- `candidate_probability`: compared candidate probability for `target = 1`, when available.
- `candidate_prediction`: compared candidate binary prediction, when available.
- `hard_example_type`: one or more values from `hard_negative`, `hard_positive`, or `uncertain`.
- `source_paths`: provenance references for the hard-row source files and joined context files.

**Validation Rules**

- Each `image_id` appears exactly once after normalization.
- The input row must never originate from test labels or public leaderboard feedback.
- Missing optional comparison context is allowed only when explicitly recorded as absent.

## Audit Evidence Record

Represents the optional supporting evidence captured for one hard row.

**Fields**

- `image_id`
- `detector_evidence_status`: `present` or `missing`
- `detector_summary`: concise detector-evidence note or empty if absent
- `crop_quality_status`: `present` or `missing`
- `crop_quality_summary`: concise crop or image-quality note or empty if absent
- `secondary_notes`: optional supporting notes that do not override the primary category

**Validation Rules**

- Evidence fields may be absent, but absence must be explicit.
- Secondary notes cannot replace the required primary category.

## Audit Row Record

Represents the row-level audit result for one governed hard row.

**Fields**

- `image_id`
- `primary_audit_category`: one of `likely_correct_but_hard`, `likely_ambiguous`, `likely_mislabeled`, `likely_preprocessing_issue`, or `likely_detector_evidence_issue`
- `audit_rationale`: concise explanation of the assigned category
- `target`
- `baseline_probability`
- `baseline_prediction`
- `candidate_probability`
- `candidate_prediction`
- `hard_example_type`
- `detector_evidence_status`
- `crop_quality_status`
- `secondary_notes`

**Validation Rules**

- Every governed hard row must produce exactly one audit row record.
- `primary_audit_category` must be present and valid for every row.
- `audit_rationale` must be non-empty for every row.

## Audit Category Summary

Represents one aggregate count grouped by primary audit category.

**Fields**

- `primary_audit_category`
- `row_count`

**Validation Rules**

- The sum of all category counts must equal the total audited row count.
- Categories with zero rows may be omitted, but non-zero categories must appear exactly once.

## Failure-Mode Summary Record

Represents one ranked action-oriented summary item derived from the audit table.

**Fields**

- `rank`
- `failure_mode_group`
- `row_count`
- `recommended_next_action`
- `supporting_categories`

**Validation Rules**

- Ranked records must be ordered consistently from highest to lowest priority.
- Failure-mode groups must be traceable back to one or more row-level audit categories.
