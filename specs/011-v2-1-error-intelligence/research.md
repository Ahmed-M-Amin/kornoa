# Research: V2.1 Error Intelligence

## Decision: Implement V2.1 as an offline analysis module

**Rationale**: The feature exists to understand V2B validation errors before changing training, thresholds, or inference. Keeping it offline prevents accidental impact on model runtime or submission behavior.

**Alternatives considered**: Adding the logic to training was rejected because the audit must be available before new training. Adding it to inference was rejected because no final prediction behavior should change.

## Decision: Extend the existing `src/analysis/` workflow family

**Rationale**: The repository already has `artifact_audit.py` and `v2b_error_review.py` with config-driven CSV/JSON analysis, pytest coverage, and safe leakage checks. V2.1 naturally fits this pattern.

**Alternatives considered**: A notebook-only audit was rejected because it would be harder to test and reproduce. A new top-level tool package was rejected because it would duplicate existing analysis conventions.

## Decision: Use locked-threshold probability distance for review groups

**Rationale**: The clarified spec defines near-threshold rows as `abs(probability - threshold) <= 0.05` and high-confidence rows as `abs(probability - threshold) >= 0.30`. This gives stable, repeatable groups across future candidates.

**Alternatives considered**: Quantile groups were rejected because group membership would shift with every dataset or prediction subset. Manual sorted lists were rejected because they are useful for review but not enough for objective tests.

## Decision: Validate audit metrics against the locked V2B baseline

**Rationale**: The audit is only useful if its TP, FP, FN, TN, precision, recall, and F1 reproduce the locked baseline. This prevents comparing later candidates against a misaligned validation set or threshold.

**Alternatives considered**: Trusting input metrics without recomputation was rejected because row alignment errors can silently invalidate the analysis. Recomputing without comparing to baseline was rejected because it would not prove the audit matches the locked record.

## Decision: Treat detector and image-quality evidence as optional left-joined evidence

**Rationale**: Optional evidence should improve analysis when present but must not remove validation rows. Missing evidence must be explicit so reviewers know which conclusions are unsupported.

**Alternatives considered**: Requiring detector and image-quality evidence for every row was rejected because current artifacts may be incomplete. Dropping rows without evidence was rejected because it would bias FP/FN counts.

## Decision: Output both machine-readable summaries and review-ready CSV groups

**Rationale**: JSON summaries support regression tests and candidate comparisons, while CSV audit and review groups support manual review and later evidence packs.

**Alternatives considered**: Only writing one aggregate report was rejected because it would not support manual FP/FN inspection. Only writing row-level CSV was rejected because it would make automated acceptance checks harder.
