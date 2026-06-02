# Research: ROI Cropping and Preprocessing

## Decision: Use annotation ROI when valid, expanded by 10%

**Rationale**: SPEC-003 exposes normalized annotation records, so training
images can use known region metadata when available. Expanding by 10% preserves
nearby bottle context while staying bounded and testable.

**Alternatives considered**:

- Exact annotation crop: rejected because it may cut away useful context around
  defect regions.
- Large margin such as 20%: rejected for V1 because it can reintroduce more
  irrelevant border/background.

## Decision: Treat ROIs under 10% of image area as unsafe

**Rationale**: Very small boxes are likely too narrow for a bottle-focused crop
and risk training on defect fragments without inspection context. A 10% area
threshold is simple to test and conservative for fallback behavior.

**Alternatives considered**:

- 5% threshold: rejected because it may admit overly small crops.
- Only reject non-positive dimensions: rejected because valid but tiny boxes can
  still be unsafe for classifier input preparation.

## Decision: Fallback to square center crop, then full-image resize

**Rationale**: A square center crop is deterministic, simple, and aligns with a
fixed classifier input size. Full-image resize prevents hard failures when a
center crop is unsafe for unusual image dimensions.

**Alternatives considered**:

- Circular crop fallback: rejected for V1 because masking can hide useful
  evidence and complicates deterministic tests.
- Fully configurable fallback modes: deferred until classifier performance shows
  a need for additional crop strategies.

## Decision: Default output size is 384x384

**Rationale**: The approved roadmap's next classifier phase uses `384x384`, so
SPEC-004 should produce the default image shape expected by SPEC-005.

**Alternatives considered**:

- 224x224: faster, but less aligned with the roadmap's first classifier target.
- No default size: rejected because it leaves tests and downstream specs
  underspecified.

## Decision: Validation/test preprocessing is deterministic

**Rationale**: Validation and test outputs must be stable across repeated runs
to avoid metric noise and leakage-like evaluation behavior. Random augmentation
belongs only to training preprocessing.

**Alternatives considered**:

- Applying light augmentation to validation: rejected because validation must
  simulate unseen production data consistently.
- Test-time augmentation: out of scope for SPEC-004 and better evaluated in a
  later inference or model optimization spec.

## Decision: Store ROI review samples under outputs/figures/roi_samples/

**Rationale**: ROI samples are visual quality-control artifacts and belong with
generated figures. The existing output ignore strategy already protects
generated figure artifacts from source control.

**Alternatives considered**:

- `outputs/reports/roi_samples/`: rejected because samples are images, not
  tabular reports.
- `outputs/hard_examples/roi_samples/`: rejected because hard-example mining is
  a later feature.
