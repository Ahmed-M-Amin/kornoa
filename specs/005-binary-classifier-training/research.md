# Research: Binary Classifier Training

## Decision: EfficientNet-B0-Compatible V1 Classifier

**Decision**: Use a lightweight image-classifier wrapper that can create an
EfficientNet-B0 baseline for private/full runs and a tiny synthetic-compatible
classifier path for automated smoke tests.

**Rationale**: The final implementation plan names EfficientNet-B0 as the first
V1 model and requires `384x384` input. Synthetic tests must finish under one
minute and should not depend on downloading pretrained weights, so the design
must allow a local tiny model variant for tests while preserving the V1 model
contract.

**Alternatives considered**:

- EfficientNet-B1/B2: deferred to stronger classifier specs because V1 needs a
  fast baseline first.
- Detector-first model: rejected because detector training belongs to SPEC-006.
- One-off notebook model: rejected because the constitution requires modular,
  reproducible code and saved artifacts.

## Decision: 80/20 Stratified Validation Split

**Decision**: Use an 80/20 stratified train/validation split with a fixed seed.

**Rationale**: This was clarified for SPEC-005 and is a conventional V1 baseline
that preserves class representation when data size allows. It is simple to test
and reproducible across local, Kaggle, and Colab runs.

**Alternatives considered**:

- 75/25 split: viable, but less aligned with the accepted clarification.
- Cross-validation: more robust but slower and more complex than needed for the
  first working V1 classifier.
- Random non-stratified split: rejected because it risks class leakage or a
  validation set missing rare classes.

## Decision: Weighted Binary Loss Only for V1

**Decision**: Account for class imbalance with weighted binary loss only.

**Rationale**: The final plan calls for weighted BCE first, and the clarification
deferred weighted sampling and focal loss to later classifier-improvement specs.
This keeps V1 interpretable and reduces moving parts while still addressing
binary imbalance.

**Alternatives considered**:

- Weighted sampling: deferred because it changes effective epoch composition and
  is better evaluated after the baseline works.
- Focal loss: deferred to V2 because it adds another hyperparameter and changes
  the baseline comparison.
- No weighting: rejected because imbalance handling is required by the
  constitution and spec.

## Decision: Validation F1 Drives Threshold and Checkpoint Selection

**Decision**: Compute validation probabilities, search thresholds for best
F1-score, choose the lowest threshold among tied best-F1 thresholds, and save the
checkpoint with the best post-threshold validation F1.

**Rationale**: The challenge metric is F1-score. Selecting by validation loss or
final epoch can choose a model that is less useful for the final binary decision.
The lowest-threshold tie-break is deterministic and slightly favors detecting
not-reusable bottles when F1 is equal.

**Alternatives considered**:

- Fixed `0.5` threshold: rejected because threshold tuning is explicitly in V1.
- Highest tied threshold: rejected because it may be less conservative for
  defect detection when F1 is equal.
- Validation loss checkpoint: rejected because it is not the target ranking
  metric.

## Decision: Generated Artifact Contract

**Decision**: Save best model weights, classifier metrics JSON, best threshold
JSON, and validation predictions CSV under ignored `outputs/` locations.

**Rationale**: Later inference and reporting specs need these files, and the
constitution requires reproducible experiment records while preserving dataset
confidentiality.

**Alternatives considered**:

- Commit generated artifacts: rejected because weights and predictions can be
  private-derived.
- Keep artifacts only in memory: rejected because downstream specs need saved
  thresholds and validation predictions.
- Save under ad hoc directories: rejected because SPEC-001 established output
  categories.
