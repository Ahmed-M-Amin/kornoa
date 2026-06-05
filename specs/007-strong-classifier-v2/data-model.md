# Data Model: Strong Classifier V2

## V1BaselineRecord

Represents the completed V1 baseline used for comparison.

**Fields**:
- `artifact_root`: Path to the concrete V1 artifact root used for comparison.
- `validation_f1`: V1 validation F1, expected around `0.91656`.
- `kaggle_public_score`: V1 Kaggle public score when available, expected around `0.91693`.
- `threshold`: V1 best threshold, expected `0.48`.
- `false_positives`: V1 false-positive count, expected `344` from the metrics artifact.
- `false_negatives`: V1 false-negative count, expected `344` from the metrics artifact.
- `uncertain_samples`: Count loaded from the hard-example summary or derived file.
- `inference_speed`: V1 benchmark timing summary.
- `model_name`: V1 backbone name, expected `efficientnet_b0`.
- `image_size`: V1 image size, expected `384`.

**Validation Rules**:
- V1 metrics and threshold must be loaded from saved artifacts, not hardcoded as the only source of truth.
- If prompt expectations and loaded artifact counts differ, the discrepancy is reported in the V2 comparison.

## V2ExperimentConfig

Represents one candidate V2 experiment.

**Fields**:
- `experiment_name`: Stable name such as `v2a_effnet_b0_recipe`, `v2b_effnet_b1`, or `v2c_effnet_b2`.
- `backbone`: Candidate classifier backbone.
- `image_size`: `384` baseline or optional `448`.
- `imbalance_strategy`: Focal loss, weighted sampler, weighted BCE compatibility, or combinations allowed by config.
- `augmentation_recipe`: Safe training augmentation profile.
- `hard_example_strategy`: `none`, `analysis_only`, or `oversample`; default is `analysis_only`.
- `batch_size`, `num_workers`, `epochs`, `seed`: Reproducibility and training-run controls.
- `speed_ceiling_multiplier`: Maximum allowed multiplier over V1 speed, fixed at `2.0`.

**Validation Rules**:
- `384` experiments must exist before any `448` candidate is considered selected.
- Large heavy backbones are invalid as default V2 candidates.
- Feature memory bank, detector, ensemble default, and distillation flags must be absent or disabled.
- Hard-example oversampling is invalid unless `hard_example_strategy` is explicitly `oversample`.

## HardExampleSourceSet

Represents validation-derived hard-example mistake files used for V2 training analysis or explicitly enabled oversampling.

**Fields**:
- `false_positives_path`
- `false_negatives_path`
- `uncertain_path`
- `high_loss_samples_path`
- `summary_path`
- `hard_example_source_used`
- `hard_example_source_type`: `v1_memory`, `v2_generated_memory`, or `explicit_existing_memory`
- `loaded_counts`
- `eligible_for_training_count`
- `excluded_validation_count`
- `used_for_oversampling_count`
- `train_validation_disjoint`
- `excluded_rows`

**Validation Rules**:
- Rows must come from validation predictions only.
- Rows not matching training or validation image IDs are excluded and reported.
- Rows may be used for V2 training oversampling only when their image IDs are absent from the current V2 validation split.
- Training and validation image ID sets must be disjoint before hard-example oversampling begins.
- If an explicit V2 artifact source contains `val_classifier_predictions.csv` and `best_threshold.json`, hard examples are generated from those V2 predictions before existing hard-example memory is used.
- Explicit V2 artifact sources must not silently resolve nested `v1-artifacts/outputs/hard_examples`; this must warn or fail.
- Hard-example sources are never used during normal V2 inference, submission generation, or benchmarking.

## V2CandidateResult

Represents the validation and speed outcome for one V2 candidate.

**Fields**:
- `experiment_name`
- `backbone`
- `image_size`
- `validation_f1`
- `best_threshold`
- `confusion_counts`
- `false_positives`
- `false_negatives`
- `uncertain_samples`
- `inference_speed`
- `speed_multiplier_vs_v1`
- `model_size`
- `selected`
- `selection_reason`

**Validation Rules**:
- `best_threshold` must be selected from validation predictions.
- `speed_multiplier_vs_v1` must be at most `2.0` for selected candidates.
- If candidates have close validation F1, the faster candidate wins.
- Close validation F1 means absolute validation F1 difference `<= 0.002`.

## ThresholdRecord

Represents the selected V2 threshold.

**Fields**:
- `threshold`
- `f1_score`
- `candidate_count`
- `tie_break`
- `v1_threshold`
- `threshold_delta_vs_v1`

**Validation Rules**:
- Threshold must be in `[0.0, 1.0]`.
- Tie handling must be deterministic.
- Threshold must not be chosen from test images or Kaggle public score.

## V1VsV2ComparisonReport

Represents the mandatory comparison artifact.

**Fields**:
- `v1`: `V1BaselineRecord`
- `v2_selected`: `V2CandidateResult`
- `candidate_results`: List of V2 candidate summaries.
- `validation_f1_delta`
- `kaggle_public_score_delta`: Present only when V2 Kaggle score is available.
- `threshold_delta`
- `false_positive_delta`
- `false_negative_delta`
- `uncertain_delta`
- `speed_delta`
- `model_size_delta`
- `image_size_delta`
- `recommendation`

**Validation Rules**:
- All required comparison fields must be present.
- Missing Kaggle V2 score must be represented as unavailable, not guessed.
- The report must state whether the selected V2 meets the +0.02 F1 target and 2x speed ceiling.
- The report must include the close-F1 tolerance used for candidate tie-breaking.

## V2OutputSet

Represents generated V2 artifacts.

**Fields**:
- `model`: `outputs/kaggle_v2/models/classifier_best.pth`
- `metrics`: `outputs/kaggle_v2/reports/classifier_metrics.json`
- `threshold`: `outputs/kaggle_v2/reports/best_threshold.json`
- `comparison`: `outputs/kaggle_v2/reports/v1_vs_v2_comparison.json`
- `validation_predictions`: `outputs/kaggle_v2/predictions/val_classifier_predictions.csv`
- `submission`: `outputs/kaggle_v2/submissions/submission_v2.csv`
- `benchmark`: `outputs/kaggle_v2/benchmarks/v2_inference_benchmark.json`

**Validation Rules**:
- Outputs must remain under ignored local output paths.
- V2 outputs must not overwrite V1 outputs.
