# Feature Specification: Locked Same-Split Error Intelligence

**Feature Branch**: `[012-locked-same-split-intelligence]`

**Created**: 2026-06-11

**Status**: Draft

**Input**: User description: "Read `docs/new_implinmination.md` and make Phase 1: Locked Same-Split Error Intelligence according to the best practice of Spec Kit."

## Clarifications

### Session 2026-06-11

- Q: Should Phase 1 be scoped only to V2.2 vs V2B, or should it support future V2.x candidates too? -> A: Scope Phase 1 to any one compared V2.x candidate, with V2.2 as the first required example.
- Q: Should detector and image-quality evidence be mandatory in Phase 1? -> A: Keep detector and image-quality evidence optional in Phase 1, with explicit support for attaching them later.
- Q: Should Phase 1 output only one candidate artifact set or also maintain a rolling multi-candidate comparison table? -> A: Include a rolling comparison table across multiple candidates as part of Phase 1 output.
- Q: What decision statuses should the rolling comparison table use? -> A: Limit decision status to `accepted`, `rejected`, or `manual_review` based only on locked same-split evidence.
- Q: What locked same-split rule should decide whether a candidate becomes `accepted` instead of `manual_review`? -> A: Mark a candidate `accepted` if full locked-row performance improves and hard-example performance does not regress beyond a small tolerance; otherwise use `manual_review`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare Locked Validation Behavior (Priority: P1)

As the model developer, I need one locked same-split comparison between the original V2B baseline and one selected V2.x candidate, with V2.2 as the first required example, on exactly the same original V2B validation rows so I can trust whether a candidate truly improves the error profile before any submission decision.

**Why this priority**: Without a locked same-split comparison, local validation gains can be misleading because row composition changes. This is the minimum feature needed to prevent false optimism and to decide whether a candidate is genuinely better than V2B.

**Independent Test**: Can be fully tested by running the evaluator on the locked V2B validation row set and confirming that V2B and the compared candidate are both scored on the exact same `image_id` list with matching row counts.

**Acceptance Scenarios**:

1. **Given** the locked V2B validation prediction set exists, **When** the comparison is run, **Then** both V2B and the compared candidate are evaluated on the same original validation rows.
2. **Given** the comparison has completed, **When** metrics are reviewed, **Then** the output shows F1, precision, recall, TP, FP, TN, FN, prediction distribution, and target distribution for both models on the shared row set.
3. **Given** a candidate was trained with moved hard examples, **When** the same-split comparison is reviewed, **Then** the report distinguishes full locked-row performance from the optimistic in-training validation result.

---

### User Story 2 - Expose Hard-Example Impact (Priority: P2)

As the model developer, I need the locked same-split comparison divided into all rows, hard-example rows only, and locked validation rows excluding hard examples so I can see whether gains come from true boundary improvement or only from easier rows.

**Why this priority**: The roadmap’s target of `0.98+` is currently blocked by the hardest rows. This story isolates that bottleneck and shows whether a candidate is actually closing the hard-example gap.

**Independent Test**: Can be tested independently by comparing section row counts against the locked row set and the configured hard-example lists, then verifying that no row belongs to both the excluded and included sections incorrectly.

**Acceptance Scenarios**:

1. **Given** configured hard-negative, hard-positive, and uncertain example files, **When** the comparison is generated, **Then** the report includes sections for all locked rows, hard-example rows only, and locked rows excluding hard examples.
2. **Given** a row appears in any configured hard-example file, **When** the comparison output is reviewed, **Then** that row is marked as a hard example with its hard-example type visible.
3. **Given** a candidate improves only on easier rows, **When** sectioned metrics are reviewed, **Then** the hard-example section makes that weakness visible instead of hiding it inside the overall score.

---

### User Story 3 - Produce Review-Ready Error Intelligence (Priority: P3)

As the model developer, I need review-ready error splits and a rolling same-split comparison table across candidates so I can identify fixable failure modes, compare candidate outcomes consistently, and prepare evidence for future model decisions and the final insight package.

**Why this priority**: Improvement work and insight scoring both depend on understanding which exact decisions changed, not only on aggregate metrics.

**Independent Test**: Can be tested independently by verifying that every evaluated row falls into exactly one comparison bucket and that the saved review files preserve the expected per-row comparison columns.

**Acceptance Scenarios**:

1. **Given** V2B and candidate predictions are available on the locked row set, **When** comparison buckets are generated, **Then** separate outputs exist for V2B wrong / candidate correct, V2B correct / candidate wrong, both wrong, and both correct.
2. **Given** the error outputs are reviewed, **When** a row is inspected, **Then** the reviewer can see its target, both model probabilities and predictions, threshold used, and hard-example status.
3. **Given** future candidate training decisions depend on failure-mode evidence, **When** the outputs are consumed downstream, **Then** they provide enough information to rank fixable failure modes without re-running inference.
4. **Given** multiple V2.x candidates are evaluated over time, **When** Phase 1 outputs are reviewed, **Then** the same-split comparison table shows each candidate's locked-row metrics, hard-example metrics, and decision status in one consistent view.

### Edge Cases

- If the locked V2B validation prediction file is missing, the comparison must fail clearly before producing any report.
- If the compared candidate checkpoint or threshold record is missing, the comparison must fail clearly before scoring rows.
- If V2B and candidate predictions do not align to the same row set after normalization, the comparison is invalid.
- If hard-example files overlap, rows must still appear once in each sectioned evaluation while preserving all applicable hard-example tags.
- If optional evidence files are missing, the comparison must still preserve the locked row analysis and record that the optional evidence is absent.
- If any test labels, sample-submission labels, or public leaderboard feedback are used, the feature is invalid.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST evaluate the original V2B baseline and one selected V2.x candidate, with V2.2 as the first required comparison target, on exactly the original locked V2B validation rows.
- **FR-002**: The system MUST normalize row identity by `image_id` and reject duplicate or non-binary locked validation rows.
- **FR-003**: The system MUST compute binary F1, precision, recall, TP, FP, TN, FN, prediction distribution, and target distribution for both models on the same row set.
- **FR-004**: The system MUST produce separate metrics for all locked validation rows, hard-example rows only, and locked validation rows excluding hard examples.
- **FR-005**: The system MUST identify hard examples from the configured hard-negative, hard-positive, and uncertain example sources and preserve hard-example type in row-level outputs.
- **FR-006**: The system MUST produce row-level comparison outputs for V2B wrong and candidate correct, V2B correct and candidate wrong, both wrong, and both correct.
- **FR-007**: Each row-level comparison output MUST include `image_id`, target, V2B probability, V2B prediction, candidate probability, candidate prediction, candidate threshold, hard-example status, and hard-example type.
- **FR-008**: The system MUST produce a threshold sweep for the compared candidate on the locked V2B validation rows.
- **FR-009**: The feature MUST record the paths or identifiers of the V2B prediction source, candidate checkpoint, candidate threshold source, and hard-example files used for the comparison.
- **FR-010**: The feature MUST assign one decision status of `accepted`, `rejected`, or `manual_review` to the selected compared candidate based only on locked same-split evidence.
- **FR-011**: The feature MUST mark a candidate `accepted` only when full locked-row performance improves over V2B and hard-example performance does not regress beyond a defined small tolerance; otherwise the feature MUST use `manual_review` unless the candidate is clearly weaker overall.
- **FR-012**: The feature MUST prohibit test labels, sample-submission labels, public leaderboard tuning, training, and submission generation.
- **FR-013**: The feature MUST preserve enough row-level evidence to support future hard-row taxonomy, candidate comparison, and final insight reporting.
- **FR-014**: The feature MUST allow detector and image-quality evidence to be attached later without changing the locked same-split comparison contract or invalidating prior comparison outputs.
- **FR-015**: The feature MUST maintain a rolling comparison table that records locked same-split results for each evaluated V2.x candidate in a consistent candidate-to-V2B format.
- **FR-016**: The rolling comparison table MUST include, at minimum, candidate identity, locked full-row metrics, hard-example-only metrics, locked rows excluding hard examples metrics, and candidate decision status.

### Constitution Alignment *(mandatory)*

- **Binary Output**: Preserves the required operational labels `0 = Reusable` and `1 = Not Reusable` for targets, predictions, counts, and review outputs.
- **Accuracy/Speed**: Improves candidate selection quality without adding final inference cost because the feature is offline analysis only.
- **ROI/Hybrid Flow**: Uses existing classifier outputs as the comparison source and does not force detector execution or hybrid overrides as part of the comparison itself.
- **Annotation/Version Strategy**: Treats original V2B as the locked reference and supports controlled V2.x candidate comparison without implying a new final model.
- **Reproducibility/Leakage**: Requires saved source identifiers, validation-only locked rows, explicit thresholds, and zero use of test labels or public leaderboard tuning.
- **Confidentiality**: Keeps all evaluation artifacts, predictions, and candidate evidence inside local or competition-safe project paths.
- **Explainability**: Produces row-level evidence for changed decisions and preserved failures so later manual review and insight reporting can explain model behavior.
- **Bias/Imbalance**: Makes hard-example concentration, changed error balance, and target/prediction distributions visible rather than hiding them in one aggregate metric.

### Key Entities *(include if feature involves data)*

- **Locked Validation Row**: One original V2B validation image with stable `image_id`, binary target, and baseline prediction data used as the comparison anchor.
- **Candidate Same-Split Result**: One selected V2.x candidate model's probability and binary prediction for a locked validation row using the candidate's saved threshold, with V2.2 as the first required example.
- **Hard Example Tag**: A row-level marker showing whether the locked validation row belongs to hard-negative, hard-positive, uncertain, or multiple hard-example groups.
- **Comparison Section**: One evaluation slice over the locked rows, such as all rows, hard-example rows only, or locked rows excluding hard examples.
- **Comparison Bucket**: One row-level outcome group describing how V2B and the candidate differ on correctness for the same row.
- **Same-Split Summary**: The aggregate comparison artifact that records metrics, evidence paths, safety status, and the recommended decision.
- **Candidate Comparison Record**: One row in the rolling Phase 1 comparison table representing one evaluated V2.x candidate against the locked V2B baseline, including one of the statuses `accepted`, `rejected`, or `manual_review`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of locked V2B validation rows evaluated for the candidate match 100% of locked V2B baseline rows after `image_id` normalization.
- **SC-002**: The feature produces metrics for both models across all three required sections: all locked rows, hard-example rows only, and locked rows excluding hard examples.
- **SC-003**: 100% of evaluated rows appear in exactly one of the four row-level comparison buckets.
- **SC-004**: 100% of row-level comparison outputs include the required comparison columns for target, both model predictions, threshold, and hard-example status.
- **SC-005**: The feature records zero use of test labels, sample-submission labels, public leaderboard tuning, training, or submission creation.
- **SC-006**: A reviewer can determine from the generated outputs whether candidate gains come from the hardest rows, easier rows, or both without re-running inference.
- **SC-007**: The produced artifacts are sufficient to support the next hard-row failure-mode review and the candidate comparison table described in the roadmap.
- **SC-008**: After at least two candidate evaluations, the rolling comparison table shows each evaluated candidate in one consistent locked same-split summary format without manual reconciliation.
- **SC-009**: Any candidate marked `accepted` is traceably supported by an improved full locked-row result and no hard-example regression beyond the defined tolerance in the recorded comparison outputs.

## Assumptions

- The original V2B validation prediction file is the authoritative locked row source for same-split comparison.
- The feature compares one selected V2.x candidate at a time rather than comparing multiple candidates in a single evaluation run.
- Validation-only analysis on the locked rows is allowed within the competition workflow and does not violate leakage rules.
- Hard-example files are derived from prior validation-only analysis and are safe to use as row tags for locked-row comparison.
- Optional detector or image-quality evidence may be attached later, but the core Phase 1 feature remains valid without them.
- Phase 1 is an analysis and reporting feature only; it does not include new training, submission creation, or public-score-driven threshold tuning.
- The exact numeric tolerance for allowable hard-example regression will be defined during planning, but it must be small enough to preserve the roadmap’s focus on the hardest rows.
