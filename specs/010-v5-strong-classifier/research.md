# Research: V5 Strong Classifier

## Decision: Primary Candidate Scope

Use ConvNeXt-Tiny at 512 image size as the primary V5 classifier candidate. Use EfficientNet-B2 only when resource limits block ConvNeXt-Tiny at 512.

**Rationale**: V4.2 hybrid improved public F1 only from `0.92121` to `0.92181`, so the next meaningful improvement must come from stronger classifier capacity. ConvNeXt-Tiny is the preferred stronger candidate from the V5 brief. A fallback is still necessary for Kaggle/Colab memory constraints.

**Alternatives considered**:

- Broad model sweep: rejected because it expands SPEC-010 beyond the first V5 implementation and complicates tasking.
- EfficientNet-B2 first: rejected because the clarified V5 direction prioritizes ConvNeXt-Tiny.

## Decision: Validation Split Policy

Reuse the existing V2B-compatible train/validation split for V5 model selection and threshold calibration.

**Rationale**: Keeping the validation split compatible with V2B/V4.2 makes V5 comparisons more meaningful and reduces the chance that a metric change comes from split composition rather than classifier quality.

**Alternatives considered**:

- New fixed V5 split: rejected because it weakens direct comparison with V2B/V4.2.
- Cross-validation: rejected for SPEC-010 because it increases training cost and scope before the first V5 candidate is established.

## Decision: Hard-Example Handling

Keep hard examples analysis-only by default. Hard-example oversampling requires explicit opt-in and split-safety reporting.

**Rationale**: V2B-HE oversampling produced high local validation F1 but weak public generalization, indicating overfitting risk. Analysis-only hard examples keep visibility into errors without changing the training distribution by default.

**Alternatives considered**:

- Oversample by default: rejected because the prior oversampling experiment overfit.
- Disable all hard-example use: rejected because hard-example reports remain useful for error analysis and reporting.

## Decision: Threshold and Acceptance Policy

Select thresholds using validation data only. V5 becomes accepted only after manual public-score review shows a score greater than V4.2 `0.92181`, with no public-score-derived retraining or threshold changes.

**Rationale**: Validation-only threshold search preserves leakage controls. Public score is an acceptance record, not a tuning signal.

**Alternatives considered**:

- Tune threshold from public leaderboard feedback: rejected as leakage and overfitting to public score.
- Accept based on validation F1 only: rejected because prior experiments showed local validation can fail to generalize publicly.

## Decision: Runtime Acceptance

Require the accepted V5 candidate to run no more than 2x slower than the accepted V2B classifier benchmark.

**Rationale**: The project constitution requires accuracy and speed together. ConvNeXt-Tiny at 512 may improve classifier quality but must remain practical for competition inference and future dashboard/hybrid use.

**Alternatives considered**:

- 3x slower if public score improves: rejected because it weakens the speed guard too early.
- Report-only runtime: rejected because it would leave the acceptance gate ambiguous.

## Decision: Prediction Export Schemas

Validation predictions use `image_id,true_label,prob_bad,classifier_prediction,target`. Test predictions use `image_id,prob_bad,classifier_prediction,target`. Final submissions use exactly `image_id,target`.

**Rationale**: Validation needs labels for metrics and threshold review; test predictions must expose probabilities for future hybrid consumption while excluding unavailable labels; submissions must remain strict Kaggle-compatible outputs.

**Alternatives considered**:

- Test-like validation schema without labels: rejected because validation metrics would need another join and increase error risk.
- Extended diagnostic columns in core exports: rejected because the core contract should remain stable; diagnostics can be separate reports.

## Decision: Artifact Layout

Write generated V5 artifacts under `outputs/kaggle_v5/v5_strong_classifier/` with subdirectories for `models`, `reports`, `predictions`, `submissions`, and `benchmarks`.

**Rationale**: This mirrors prior V2/V4 output organization, keeps generated artifacts ignored, and makes the V5 experiment easy to package for Kaggle/Colab.

**Alternatives considered**:

- Reusing `outputs/kaggle_v2/`: rejected because V5 must not overwrite accepted V2B artifacts.
- Writing artifacts inside `specs/`: rejected because generated weights and predictions must not be tracked.
