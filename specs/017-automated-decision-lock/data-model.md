# Data Model: Spec 017 Automated Decision Lock

## DecisionRecord

Represents the final decision for one training row from the Spec 016 master audit table.

Fields:

- `image_id`: Stable training image identifier.
- `target`: Original binary label from training labels.
- `source_bucket`: Spec 016 source bucket, such as clean, hard-valid, manual-review-required, or blocked.
- `decision`: One of `auto_keep`, `auto_exclude`, `needs_adjudication`, or `defer`.
- `decision_reason`: Machine-readable reason for the decision.
- `decision_confidence`: One of `low`, `medium`, or `high`.
- `evidence_sources`: Pipe-separated evidence sources used by the decision.
- `blocked_flag`: Whether the row appears in any Spec 014 blocked set.
- `duplicate_group_id`: Stable duplicate or near-duplicate group identifier when available.
- `duplicate_conflict_flag`: Whether the duplicate group contains conflicting labels.
- `roi_quality_group`: ROI/crop group assigned during recalibration.
- `roi_severity`: One of `none`, `low`, `medium`, `high`, or `critical`.
- `prediction_risk_level`: One of `none`, `low`, `medium`, `high`, or `critical`.
- `embedding_status`: One of `missing`, `available`, or `partial`.
- `cluster_id`: Optional visual cluster identifier.
- `cluster_outlier_score`: Optional numeric outlier score.
- `cleanlab_label_quality_score`: Optional label-quality score.

Validation rules:

- Every Spec 016 master row must have exactly one DecisionRecord.
- `decision` must be one of the four allowed values.
- `auto_keep` rows must not have `blocked_flag = true`.
- `auto_keep` rows must not have unresolved duplicate conflicts.
- `auto_keep` rows must have an existing train label and image.

## ApprovedCleanedTrainingManifest

Represents the locked input for later cleaned-data training.

Fields:

- `image_id`: Approved training image identifier.
- `target`: Original binary label.
- `decision`: Must be `auto_keep`.
- `decision_reason`: Reason the row was approved.
- `decision_confidence`: Confidence for the keep decision.

Validation rules:

- Contains only `auto_keep` decisions.
- Contains no Spec 014 blocked image_ids.
- Contains no unresolved manual-review-required rows.
- Contains no unresolved duplicate-conflict rows.
- Does not modify original labels.

## AdjudicationQueueRow

Represents a high-impact uncertain row that needs focused review or stronger evidence.

Fields:

- `image_id`: Training image identifier.
- `target`: Original binary label.
- `risk_rank`: Numeric rank where lower means higher priority.
- `risk_bucket`: High-level risk category.
- `risk_group_id`: Group identifier for shared issue patterns.
- `primary_issue`: Main reason the row needs adjudication.
- `secondary_issue`: Additional issue when present.
- `suggested_decision`: Suggested keep, exclude, or defer action.
- `suggested_reason`: Evidence-backed reason for the suggestion.
- `evidence_sources`: Evidence used for the suggestion.
- `reviewer_decision`: Empty until a reviewer records a decision.
- `reviewer_confidence`: Empty until a reviewer records confidence.
- `reviewer_notes`: Empty until a reviewer records notes.

Validation rules:

- Rows in this entity must not appear in the approved manifest unless a later explicit approved decision is applied.
- Rows are sorted by risk rank.
- Group representatives appear before lower-impact members of the same group.

## EvidenceInventory

Represents evidence availability for the run.

Fields:

- `spec016_inputs_status`: Availability of required Spec 016 files.
- `spec014_inputs_status`: Availability of required Spec 014 files.
- `embedding_status`: `missing`, `partial`, or `available`.
- `prediction_status`: `missing`, `partial`, or `available`.
- `cleanlab_status`: `missing`, `partial`, or `available`.
- `contact_sheet_status`: Availability of visual review aids.
- `warnings`: List of missing or degraded evidence warnings.

Validation rules:

- Required Spec 016 and Spec 014 inputs must be available.
- Optional evidence may be missing, but missing status must be reported.

## ROIRecalibrationGroup

Represents a group of ROI/crop review rows with similar risk evidence.

Fields:

- `roi_quality_group`: Stable group name.
- `row_count`: Number of rows in the group.
- `target_0_count`: Count of reusable-label rows.
- `target_1_count`: Count of not-reusable-label rows.
- `representative_image_ids`: Pipe-separated representative image_ids.
- `group_risk_score`: Numeric risk score.
- `proposed_group_action`: One of `keep_candidate`, `exclude_candidate`, or `adjudicate_candidate`.

Validation rules:

- All ROI/crop review rows must map to one group.
- Severe group actions require corroborating evidence beyond one broad ROI flag.

## State Transitions

Allowed row decision flow:

```text
spec016_master_row
  -> auto_keep
  -> approved_cleaned_training_manifest

spec016_master_row
  -> auto_exclude
  -> auto_exclude_rows

spec016_master_row
  -> needs_adjudication
  -> adjudication_queue

spec016_master_row
  -> defer
  -> deferred_uncertain_rows
```

Blocked transitions:

- `auto_exclude` to approved manifest.
- `needs_adjudication` to approved manifest without later explicit approval.
- `defer` to approved manifest.
- Any Spec 014 blocked row to approved manifest.
