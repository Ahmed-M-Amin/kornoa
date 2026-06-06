# Research: Hybrid Inference Engine

## Decision: Classifier-first, rejection-only fusion

**Rationale**: The roadmap's V4 logic returns classifier decisions for high-confidence images and uses the detector only when the classifier is uncertain. Clarification selected rejection-only detector authority, so detector evidence can reject uncertain images but cannot clear classifier rejections. This preserves the accepted classifier as the default decision source and avoids treating detector absence as proof of reusability.

**Alternatives considered**:
- Bidirectional detector override: rejected because it could turn classifier rejections into reusable decisions based on detector misses.
- Detector-first inference: rejected because it violates the speed and architecture constraints.
- Always run both models: rejected because SPEC-009 is intended to keep detector usage limited.

## Decision: Tune on validation overlap only

**Rationale**: Hybrid parameters depend on both classifier scores and detector evidence, so fair comparison requires image IDs that have both validation prediction sources and labels. Tuning outside the overlap would either fabricate missing detector evidence or compare metrics on mismatched populations.

**Alternatives considered**:
- Tune on full classifier validation set with missing detector rows treated as no defect: rejected because it biases detector behavior.
- Tune on Kaggle public score: rejected as leakage and not reproducible.
- Tune on test predictions: rejected because test labels are unavailable and must not be inferred manually.

## Decision: Prefer <= 30% detector usage

**Rationale**: The external V4 draft and clarification both set a 30% validation-overlap detector-usage preference. This keeps the detector as a fallback path and gives the selection logic a concrete speed-oriented tie-break.

**Alternatives considered**:
- No usage cap: rejected because it could turn hybrid inference into near-always-detector inference.
- 15% cap: rejected as unnecessarily strict before validation results show the useful uncertainty band.
- Runtime-only cap: deferred because reliable timing may not exist for every validation artifact.

## Decision: Per-category conditional-area thresholds with fallback

**Rationale**: Conditional defects can have different severity by category, so one global area threshold is too coarse. A default fallback keeps inference deterministic when SPEC-008 category mapping has a conditional category without a dedicated threshold, and fallback usage must be reported.

**Alternatives considered**:
- Single global threshold: rejected because category-specific defect semantics matter.
- Confidence/category only: rejected because the roadmap explicitly includes conditional area logic.
- Require every conditional category to be mapped before inference: rejected because it would block useful evaluation when sparse categories lack a calibrated threshold.

## Decision: Normalize prediction schemas at ingestion

**Rationale**: Accepted V2 and detector artifacts may have similar data with different column names. Normalizing at the boundary keeps fusion logic deterministic and reduces fragile path-specific code. Required normalized fields are image ID, classifier score, optional classifier target, optional label, detector confidence, optional defect category, and optional area.

**Alternatives considered**:
- Hard-code one artifact schema: rejected because artifact paths and notebooks have evolved across specs.
- Require manual preprocessing before hybrid inference: rejected because it increases reproducibility risk.

## Decision: Fail when test classifier confidence is missing

**Rationale**: Hybrid test inference cannot know which images are uncertain from a target-only classifier submission. If classifier probabilities are missing, any detector usage would be arbitrary or global, violating the classifier-gate design.

**Alternatives considered**:
- Treat all classifier submissions as confident: rejected because hybrid would do nothing without explaining why.
- Treat all test images as uncertain: rejected because it runs detector globally.
- Infer confidence from detector output: rejected because uncertainty must come from the classifier threshold distance.

## Decision: Keep implementation in `src/inference/`

**Rationale**: SPEC-009 is inference-only. Existing modules already separate inference, submission, benchmark, training, and detector conversion. Adding fusion and hybrid submission orchestration under `src/inference/` preserves that separation.

**Alternatives considered**:
- Add hybrid code under training: rejected because no training occurs.
- Add a new application package: rejected because the feature is a pipeline extension, not a separate app.
- Put hybrid code in notebooks: rejected because reproducibility requires tested modules.
