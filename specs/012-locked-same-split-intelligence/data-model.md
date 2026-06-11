# Data Model: Locked Same-Split Error Intelligence

## Locked Validation Row

Represents one original V2B validation image used as the immutable comparison anchor.

**Fields**

- `image_id`: normalized locked validation image identifier.
- `target`: binary true label, `0` or `1`.
- `v2b_probability`: original V2B probability for `target = 1`.
- `v2b_prediction`: original V2B binary prediction after the locked threshold.
- `source_prediction_path`: provenance reference for the locked V2B prediction source.

**Validation Rules**

- Each `image_id` appears exactly once.
- `target` and `v2b_prediction` must be binary.
- `v2b_probability` must be within `[0, 1]`.
- Locked validation rows must never be sourced from test labels or sample submissions.

## Candidate Same-Split Row

Represents one selected V2.x candidate result on the same locked validation row.

**Fields**

- `image_id`: normalized locked validation image identifier.
- `candidate_name`: canonical compared candidate name, with V2.2 as the first required example.
- `candidate_probability`: candidate probability for `target = 1`.
- `candidate_prediction`: binary prediction after the candidate threshold.
- `candidate_threshold`: saved threshold used for same-split evaluation.
- `checkpoint_path`: provenance reference for the candidate checkpoint.
- `threshold_path`: provenance reference for the candidate threshold report.

**Validation Rules**

- Candidate rows must align 1:1 with locked validation rows after `image_id` normalization.
- `candidate_prediction` must be binary.
- `candidate_probability` and `candidate_threshold` must be within `[0, 1]`.

## Hard Example Tag

Represents whether a locked validation row belongs to one or more hard-example groups.

**Fields**

- `image_id`: normalized locked validation image identifier.
- `is_hard_example`: boolean summary flag.
- `hard_example_type`: one or more values from `hard_negative`, `hard_positive`, or `uncertain`.
- `hard_example_sources`: provenance references for the hard-example files used.

**Validation Rules**

- One row may belong to multiple hard-example types but must still remain one locked validation row.
- Missing hard-example tags default to non-hard-example, not error.

## Comparison Section

Represents one evaluation slice of the locked validation rows.

**Fields**

- `section_name`: one of `all_original_v2b_validation_rows`, `hard_example_rows_only`, or `original_v2b_validation_excluding_hard_examples`.
- `row_count`: total rows in the section.
- `model_name`: one of `v2b` or the selected candidate name.
- `f1`, `precision`, `recall`: binary classification metrics.
- `tp`, `fp`, `tn`, `fn`: confusion counts.
- `prediction_0_count`, `prediction_1_count`: prediction distribution counts.
- `target_0_count`, `target_1_count`: target distribution counts.
- `threshold`: threshold applied for the selected model within the section.

**Validation Rules**

- Sections must be computed from the same locked validation row source.
- Hard-example-only and excluding-hard-example sections must partition the locked rows correctly.

## Comparison Bucket Row

Represents one row-level correctness comparison between V2B and the selected candidate.

**Fields**

- `image_id`
- `target`
- `v2b_probability`
- `v2b_prediction`
- `candidate_probability`
- `candidate_prediction`
- `candidate_threshold`
- `is_hard_example`
- `hard_example_type`
- `bucket_name`: one of `v2b_wrong_candidate_correct`, `v2b_correct_candidate_wrong`, `both_wrong`, or `both_correct`

**Validation Rules**

- Every evaluated locked row must appear in exactly one comparison bucket.
- Bucket membership must be derived only from correctness comparisons on the locked row set.

## Candidate Comparison Record

Represents one persisted row in the rolling same-split comparison table.

**Fields**

- `candidate_name`
- `evaluation_timestamp`
- `full_locked_row_f1`
- `hard_example_only_f1`
- `excluding_hard_example_f1`
- `full_locked_row_precision`
- `full_locked_row_recall`
- `full_locked_row_fp`
- `full_locked_row_fn`
- `decision_status`: one of `accepted`, `rejected`, or `manual_review`
- `decision_reason`
- `hard_example_regression_delta`
- `report_root`

**Validation Rules**

- `decision_status` must be derived from locked same-split evidence only.
- `accepted` requires improved full locked-row F1 and no hard-example-only F1 regression beyond `0.01`.
- `rejected` requires lower full locked-row F1 than V2B.
- `manual_review` captures mixed cases that are neither `accepted` nor clearly `rejected`.

## Same-Split Summary

Represents the aggregate report for one selected candidate run.

**Fields**

- `evaluation_mode`
- `used_test_labels`
- `trained_model`
- `submission_created`
- `v2b_prediction_file`
- `candidate_checkpoint`
- `candidate_threshold`
- `hard_example_files`
- `sections`
- `recommended_decision`
- `comparison_table_path`

**Validation Rules**

- Safety flags must confirm the run remained analysis-only and validation-only.
- Summary paths must trace back to the locked V2B source and selected candidate source.
