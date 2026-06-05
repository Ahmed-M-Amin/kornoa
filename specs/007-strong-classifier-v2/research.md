# Research: Strong Classifier V2

## Decision: Keep V2 as a single lightweight classifier by default

**Rationale**: The V2 goal is to close the gap between V1's F1 around 0.91656 and stronger leaderboard performance while preserving fast inference. A single classifier keeps the system simple, benchmarkable, and compatible with the existing V1 submission path.

**Alternatives considered**: Detector, hybrid inference, feature memory bank, ensemble, and distillation were rejected for SPEC-007 because they belong to later specs or optional future research and would blur the V2 classifier-only scope.

## Decision: Use validation F1 after threshold search as the model-selection metric

**Rationale**: The challenge target is binary F1. Selecting by validation F1 after threshold search aligns training, model selection, saved threshold, and submission behavior with the competition metric.

**Alternatives considered**: Raw validation loss was rejected because it may not map to the best F1 threshold. Kaggle public score was rejected as a selection signal because it would leak public-feedback tuning into model selection.

## Decision: Treat +0.02 validation F1 over V1 as the minimum V2 target

**Rationale**: A minimum +0.02 F1 target makes V2 more than a small recipe tweak while keeping it realistic for lightweight classifiers and free-tier GPU constraints. Stronger results toward 0.95+ remain desirable if speed stays acceptable.

**Alternatives considered**: Any positive delta was too weak for a V2 feature. A hard +0.04 gate was considered too risky before B1/B2 and augmentation experiments are validated.

## Decision: Enforce a maximum 2x V1 inference-time cost

**Rationale**: V2 can spend some runtime for better F1 but must remain a fast image-flow classifier. Two times V1 speed is a ceiling, not a target. Close F1 means absolute validation F1 difference `<= 0.002`, and candidates within that tolerance should choose the faster model.

**Alternatives considered**: No speed ceiling was rejected because it could allow a slow model to win on validation F1 alone. A no-slower-than-V1 rule was rejected because B1/B2 or 448x448 may deliver meaningful gains with acceptable extra cost.

## Decision: Evaluate V2A, V2B, and V2C in controlled order

**Rationale**: EfficientNet-B0 with an improved recipe isolates training improvements from backbone size. EfficientNet-B1 and B2 then test whether a slightly stronger backbone improves F1 within the 2x speed ceiling. ConvNeXt-Tiny remains optional and speed-checked.

**Alternatives considered**: Starting with larger or heavier models was rejected because the final model must remain lightweight by default. Skipping B0 was rejected because it would hide whether gains came from recipe changes or model capacity.

## Decision: Keep 384x384 as the baseline and allow 448x448 only after 384 experiments

**Rationale**: V1 used 384x384, so keeping it first gives a fair baseline and protects speed. 448x448 may help small defects, but it must be justified by an F1 gain and benchmarked speed cost.

**Alternatives considered**: Making 448x448 the default was rejected because it increases inference cost before proving value. Ignoring 448x448 entirely was rejected because it may help defect visibility.

## Decision: Improve imbalance handling with focal loss and weighted random sampling candidates

**Rationale**: V1 had hundreds of false positives and false negatives. Focal loss and weighted sampling directly target hard and minority patterns while class-count reporting keeps the effect measurable.

**Alternatives considered**: Weighted BCE alone was already used by V1 and is insufficient as the only V2 strategy. Oversampling without reporting was rejected because it can hide split imbalance and overfit.

## Decision: Use hard-example CSVs as offline analysis inputs by default and oversampling inputs only when explicitly enabled

**Rationale**: False positives, false negatives, uncertain samples, and high-loss samples from validation predictions are useful for error analysis by default. They may affect training composition only when `hard_example_strategy=oversample` is explicitly enabled. Allowed strategies are `none`, `analysis_only`, and `oversample`; the default is `analysis_only`. Any hard-example image ID used for oversampling must be absent from the current V2 validation split, and the training report must include loaded, eligible, validation-excluded, and oversampled counts plus train/validation disjointness confirmation.

**Alternatives considered**: Oversampling by default was rejected because it can silently change training composition and increase leakage risk. Feature memory bank and inference-time hard-example lookup were rejected for SPEC-007 because they are later optional research and would slow normal inference.

## Decision: Reject V2B-HE oversampling as a model-selection path

**Rationale**: V2B-HE oversampling produced high local validation F1 (`0.974077`) but public F1 fell to `0.90293`, despite good inference speed around `25.16` images/sec and confirmed train/validation disjointness. The run used `806` oversampled examples and resolved hard examples from nested `v1-artifacts` inside a V2B artifact package rather than true V2B-generated hard examples. This shows both overfitting risk and source-selection risk.

**Alternatives considered**: Keeping V2B-HE because of local validation F1 was rejected because public/Kaggle generalization is the relevant external check. Retrying hard-example oversampling immediately was rejected until source selection is fixed and overfitting controls exist.

## Decision: Prefer explicit V2 prediction-derived hard examples over nested existing memory

**Rationale**: When `hard_example_source` points at a V2 artifact folder and both `val_classifier_predictions.csv` and `best_threshold.json` exist, hard examples must be generated from those V2 outputs first. This prevents accidental reuse of nested V1 memories when the experiment explicitly asks for V2B artifacts. Reports must include `hard_example_source_type` with values such as `v1_memory`, `v2_generated_memory`, or `explicit_existing_memory`.

**Alternatives considered**: Recursive hard-example discovery alone was rejected because it can silently select stale nested memories. Using nested `v1-artifacts` from an explicit V2 source was rejected and must warn or fail.

## Decision: Use stronger safe training augmentation while keeping validation/test deterministic

**Rationale**: Brightness/contrast, gamma, light blur/noise, and small rotation/shift/scale variation can improve robustness without changing labels. Validation/test preprocessing must remain deterministic for reproducible F1 and threshold search.

**Alternatives considered**: Aggressive crop/cutout that could remove defects was rejected because defects and internal dark regions are critical evidence. Validation-time augmentation was rejected as a default because it complicates threshold comparability and speed.

## Decision: Save V2 outputs under `outputs/kaggle_v2/`

**Rationale**: A separate V2 output root keeps V1 artifacts immutable, supports V1-vs-V2 comparison, and keeps generated private artifacts under ignored paths.

**Alternatives considered**: Overwriting V1 outputs was rejected because it would break reproducibility and comparison. Scattering outputs across generic folders was rejected because V2 is a controlled experiment stage.
