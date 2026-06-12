# Data Model: Controlled Phase 3 Hard-Example Training

## Locked Baseline Report

Represents the single comparison anchor for all controlled Phase 3 candidates.

**Fields**

- `baseline_name`
- `baseline_checkpoint_path`
- `validation_predictions_path`
- `best_threshold_path`
- `validation_f1`
- `precision`
- `recall`
- `tp`
- `fp`
- `fn`
- `tn`
- `validation_target_distribution`
- `test_target_distribution`
- `runtime_reference`

**Validation Rules**

- Exactly one locked baseline report is active for a controlled Phase 3 run.
- The baseline artifact paths must resolve to existing saved outputs.
- The locked baseline must not be overwritten by downstream candidate runs.
- The serialized report is saved under `outputs/analysis/phase3_controlled_training/locked_baseline_report.json`.

## Governed Phase 3 Training Input Set

Represents the hard-example rows that are eligible for controlled Phase 3 training.

**Fields**

- `image_id`
- `target`
- `approved_candidate_action`
- `phase3_use_allowed`
- `source_candidate_package_path`
- `baseline_membership_context`
- `hard_example_group`

**Validation Rules**

- Every row must trace back to an approved Spec 014 candidate-package artifact.
- Every included row must have `phase3_use_allowed = true`.
- No blocked or untraceable row may enter the governed training set.
- The persisted governed manifest is saved under `outputs/analysis/phase3_controlled_training/governed_phase3_training_manifest.csv`.

## Candidate Hypothesis Record

Represents the explicit reason for running one serious Phase 3 candidate.

**Fields**

- `candidate_name`
- `hypothesis`
- `candidate_family`
- `image_size`
- `hard_example_strategy`
- `baseline_reference`
- `intended_tradeoff`

**Validation Rules**

- Every serious candidate must declare exactly one explicit hypothesis.
- The hypothesis must be specific enough to explain what is being tested.

## Candidate Evaluation Record

Represents the complete controlled result set for one candidate.

**Fields**

- `candidate_name`
- `validation_f1`
- `hard_example_f1`
- `non_hard_example_f1`
- `precision`
- `recall`
- `fp_count`
- `fn_count`
- `best_threshold`
- `threshold_stability_summary`
- `changed_rows_vs_baseline`
- `validation_target_distribution`
- `prediction_target_distribution`
- `runtime_report_path`
- `artifact_paths`
- `ablation_status`
- `decision`
- `decision_reason`
- `artifact_completeness_status`
- `submission_path`
- `finalist_ready`
- `best_of_n_runtime`

**Validation Rules**

- A candidate evaluation record is incomplete if any required artifact path is missing.
- Improved candidates must include ablation or stability evidence.
- The decision field must be one of `accepted`, `rejected`, or `analysis_only`.
- The artifact package is considered complete only when validation predictions, test probabilities, threshold, metrics, runtime, target distribution, and submission files all exist.

## Candidate Comparison Table Row

Represents one row in the cross-candidate decision table.

**Fields**

- all key carry-through fields from the candidate evaluation record
- `best_of_n_runtime`
- `submission_path`
- `artifact_completeness_status`
- `finalist_ready`

**Validation Rules**

- Every serious candidate must appear exactly once in the comparison table.
- Comparison rows must use the same baseline reference and evaluation framing.

## Runtime Evidence Pack

Represents the repeated timing evidence for one candidate.

**Fields**

- `candidate_name`
- `timing_run_count`
- `best_runtime_seconds`
- `average_runtime_seconds`
- `seconds_per_image`
- `images_per_second`
- `device`
- `batch_size`
- `image_size`
- `benchmark_reference`

**Validation Rules**

- Runtime evidence must come from repeated benchmark runs.
- A finalist-ready candidate must have runtime evidence attached.
- The preferred runtime artifact path is `benchmarks/runtime_report.json`, with repeated runs aggregated into the runtime evidence pack.

## Insight Evidence Pack

Represents the competition-facing explanation package for the controlled Phase 3 program.

**Fields**

- `baseline_summary`
- `hard_example_failure_modes`
- `candidate_outcomes`
- `runtime_tradeoff_summary`
- `final_recommendation`
- `rejected_alternatives`

**Validation Rules**

- The insight pack must explain the final recommendation without requiring readers to reconstruct raw experiment history.
- The pack must stay consistent with the candidate comparison table and locked baseline report.
