# Research: Spec 018 V2B Cleaned Replay

## Decision: Extend the existing classifier trainer

**Rationale**: `src/training/train_classifier.py` already owns classifier config loading, split handling, threshold search, metrics, baseline reporting, Phase 3 dry-run validation, and artifact paths. Extending it keeps the cleaned replay comparable to previous controlled training work.

**Alternatives considered**:

- Create a separate training script: rejected because it would duplicate split, threshold, and reporting logic.
- Run from a notebook only: rejected because reproducibility requires config-driven execution and saved artifacts.

## Decision: Require dry-run validation before training

**Rationale**: The cleaned replay must prove that it uses only the Spec 017 approved manifest and does not include excluded, deferred, adjudication, test, submission, or leaderboard data before any training begins.

**Alternatives considered**:

- Validate only during training startup: rejected because dry-run validation should be runnable locally before GPU time.
- Trust Spec 017 outputs without rechecking overlaps: rejected because Spec 018 is the final gate before training.

## Decision: Use the locked V2B baseline as the comparison anchor

**Rationale**: The project already selected V2B as the reliable submitted baseline. A cleaned replay is meaningful only if it beats V2B under comparable validation rules.

**Alternatives considered**:

- Compare against Phase 3: rejected because Phase 3 trained successfully but underperformed and was rejected.
- Compare against public leaderboard only: rejected because leaderboard tuning is forbidden.

## Decision: No submission generation in Spec 018 dry-run

**Rationale**: Dry-run validation exists to prove safety and artifact readiness before training. Submission generation belongs after a trained candidate is accepted by validation comparison.

**Alternatives considered**:

- Generate a submission automatically after training: rejected because acceptance requires comparison review first.
- Include test predictions in dry-run validation: rejected because dry-run validation should not touch test labels or submission paths.

## Decision: Reject unless validation improves and safety gates pass

**Rationale**: A cleaned-data replay can still perform worse than the locked V2B baseline. The decision report must make rejection explicit when validation evidence does not justify promotion.

**Alternatives considered**:

- Accept any cleaned-data model because the data is cleaner: rejected because prior Phase 3 showed a cleaner-looking intervention can reduce validation F1.
- Mark inconclusive runs as accepted for further use: rejected because submission candidates need a clear validation win.
