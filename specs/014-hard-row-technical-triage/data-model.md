# Data Model: Hard-Row Technical Triage

## Governed Hard Row

Represents one validation-derived row selected into the governed hard-row pool.

**Fields**

- `image_id`: normalized unique row identifier
- `target`: binary true label, `0` or `1`
- `baseline_probability`
- `baseline_prediction`
- `candidate_probability`
- `candidate_prediction`
- `primary_audit_category`
- `hard_example_type`

**Validation Rules**

- Each `image_id` appears exactly once in the governed pool.
- The row must originate from validation-derived analysis only.
- Baseline and candidate prediction context must remain aligned to the same `image_id`.

## Review Manifest Row

Represents one governed hard row prepared for technical or human review.

**Fields**

- `image_id`
- `target`
- prediction context from baseline and candidate
- `bottle_type`
- `primary_audit_category`
- `hard_example_type`
- `suspicion_rank`
- `label_issue_score`
- `anomaly_score`
- `cluster_id`
- `cluster_description`
- review metadata fields

**Validation Rules**

- Every governed hard row must produce one review-manifest row.
- Ranking and cluster fields may be heuristic, but they must remain reproducible for a given input set.

## Technical Evidence Record

Represents the evidence attached to one governed hard row during evidence completion.

**Fields**

- `image_id`
- `annotation_count`
- `defect_categories`
- `largest_defect_area_ratio`
- `total_defect_area_ratio`
- `annotation_evidence_status`
- `detector_evidence_status`
- `roi_exists`
- `roi_coverage_ratio`
- `crop_quality_status`
- `brightness_score`
- `contrast_score`
- `blur_score_laplacian_variance`
- `edge_density`
- evidence asset paths

**Validation Rules**

- Evidence may be missing, but absence must be explicit through status fields.
- Detector fallback must not collapse all rows into a generic missing state when annotation evidence exists.
- ROI evidence may be explicit or derived, but derived evidence must still remain traceable to the row.

## Evidence-Aware Action Plan Row

Represents one final Spec 014 decision row before any training work begins.

**Fields**

- `image_id`
- `target`
- `prediction`
- `probability`
- `cluster_id`
- `label_issue_score`
- `anomaly_score`
- `recommended_action`
- `phase3_use_allowed`
- `reason`

**Validation Rules**

- Every governed hard row must produce exactly one action-plan row.
- Every row must have exactly one recommended action.
- `phase3_use_allowed` is a separate safety gate and may block trainable-looking action labels.

## Phase 3 Candidate Package

Represents the exported allowed and blocked outputs derived from the evidence-aware action plan.

**Fields**

- all required carry-through columns from the action plan
- package type: allowed general, allowed artifact-focus, allowed tiny/low-contrast, blocked ROI, blocked suspected mislabel, blocked other high risk

**Validation Rules**

- Allowed package rows must come only from `phase3_use_allowed = true`.
- Blocked package rows must come only from `phase3_use_allowed = false`.
- No row may appear in both an allowed package and a blocked package.
- Input row count and allowed/blocked totals must be validated against config-driven expectations before export writes any output files.
