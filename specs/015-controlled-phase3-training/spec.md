# Feature Specification: Controlled Phase 3 Hard-Example Training

**Feature Branch**: `[015-controlled-phase3-training]`

**Created**: 2026-06-12

**Status**: Draft

**Input**: User description: "Spec 015 for controlled Phase 3 hard-example from the plan docs/new_implinmination.md according to the best practice"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Lock Training Baseline And Candidate Inputs (Priority: P1)

As the model developer, I need one locked baseline report and one governed Phase 3 training input set so I can compare every new candidate against the same trusted reference instead of mixing incompatible artifacts.

**Why this priority**: Controlled training is invalid if the baseline, candidate pool, or training eligibility rules drift between runs. Baseline lock and governed inputs are the foundation for every later experiment.

**Independent Test**: Can be fully tested by creating the locked baseline record and governed Phase 3 training manifest, then confirming that only `phase3_use_allowed = true` hard rows are included and every comparison artifact points back to the same baseline anchor.

**Acceptance Scenarios**:

1. **Given** the approved Phase 3 candidate package exists, **When** the training preparation workflow runs, **Then** it produces one governed training input set containing only allowed Phase 3 rows and preserves stable row identity.
2. **Given** multiple prior model artifacts exist, **When** the baseline lock is created, **Then** it selects one explicit comparison anchor and records the metrics, threshold, prediction outputs, and runtime reference needed for later candidate comparisons.
3. **Given** a blocked or ambiguous hard row appears in an upstream artifact, **When** the governed training inputs are built, **Then** that row is excluded from training inputs and remains traceable as a rejected case.

---

### User Story 2 - Run Controlled Phase 3 Candidate Training (Priority: P2)

As the model developer, I need each Phase 3 candidate to be trained and evaluated under one controlled comparison workflow so I can tell whether the hard-example strategy improves the real decision boundary without collapsing precision or runtime standing.

**Why this priority**: The roadmap makes clear that broad recipe changes are not acceptable. Each candidate must answer a specific question and be measured against the same locked baseline and governed hard-example subset.

**Independent Test**: Can be fully tested by running one candidate through the controlled workflow and confirming that the resulting outputs include full-split evaluation, hard-example-only evaluation, excluded-hard-example evaluation, threshold search, target distribution, and changed-row comparison versus the baseline.

**Acceptance Scenarios**:

1. **Given** a candidate training hypothesis is approved, **When** the candidate workflow runs, **Then** it produces a complete comparison record covering full validation performance, hard-example performance, precision, recall, false positives, false negatives, and changed rows versus the baseline.
2. **Given** a candidate improves recall mainly by over-rejecting reusable bottles, **When** evaluation is reviewed, **Then** the candidate is marked as failing controlled acceptance rather than silently promoted.
3. **Given** a candidate claims improvement, **When** the workflow completes, **Then** it includes at least one targeted ablation or stability check showing whether the gain is robust rather than accidental.

---

### User Story 3 - Select Competition-Ready Finalists With Evidence (Priority: P3)

As the model developer, I need one comparison table, runtime evidence pack, and final-candidate decision record so I can choose a competition-ready Phase 3 path that balances model performance, runtime, and technical insight.

**Why this priority**: The competition score is not only model F1. The final selection must preserve efficiency standing and produce a defendable insight package, even if the most optimistic F1 target is not reached.

**Independent Test**: Can be fully tested by generating the candidate comparison table and final decision outputs, then confirming that every serious candidate has complete artifacts, runtime evidence, acceptance or rejection reasoning, and one final recommendation.

**Acceptance Scenarios**:

1. **Given** multiple serious candidates have been trained, **When** the comparison workflow is run, **Then** every candidate appears in one standardized comparison table with performance, runtime, and decision fields.
2. **Given** a candidate improves validation performance but incurs an unjustified runtime penalty, **When** finalist review is performed, **Then** the candidate is rejected or downgraded unless the tradeoff is explicitly justified.
3. **Given** the final recommendation is prepared, **When** the insight package is reviewed, **Then** it explains the locked baseline, the hard-example strategy, the most important failure modes, the runtime tradeoffs, and why the final path was chosen.

### Edge Cases

- If the locked baseline artifacts are missing, inconsistent, or point to multiple incompatible anchors, the Phase 3 training workflow must stop before training begins.
- If any Phase 3 training input row cannot be traced back to the approved candidate package, the workflow must reject that input set rather than continue.
- If blocked hard rows appear in a candidate training manifest, the workflow must fail before training or evaluation.
- If a candidate produces incomplete artifacts such as missing threshold reports, missing prediction outputs, or missing runtime evidence, that candidate must remain analysis-only and cannot be promoted.
- If a candidate improves clean-row behavior but the governed hard-example subset does not materially improve, the workflow must treat the candidate as insufficient for the `0.98+` pursuit.
- If a candidate gain exists only at a fragile threshold or a single unstable run, the candidate must be flagged as non-robust.
- If runtime evidence cannot be reproduced under repeated timing conditions, the candidate must not be treated as efficiency-ready.
- If public leaderboard observations conflict with locked validation evidence, the workflow must preserve the no-leakage policy and avoid public-score-only tuning decisions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST create one locked baseline report that serves as the only comparison anchor for controlled Phase 3 candidate evaluation.
- **FR-002**: The locked baseline report MUST include validation F1, precision, recall, confusion counts, best validation threshold, validation target distribution, test target distribution, and runtime reference for the active baseline.
- **FR-003**: The system MUST create one governed Phase 3 training input set derived only from approved Phase 3 candidate-package rows.
- **FR-004**: The governed Phase 3 training input set MUST exclude blocked hard rows and preserve stable row identity for every included example.
- **FR-005**: The system MUST require every Phase 3 training candidate to declare one explicit improvement hypothesis before candidate evaluation is accepted.
- **FR-006**: The system MUST evaluate each serious Phase 3 candidate against the locked baseline on the same governed validation split.
- **FR-007**: Candidate evaluation MUST include full validation performance, governed hard-example-only performance, excluded-hard-example performance, precision, recall, false-positive count, false-negative count, target distribution, and changed-row comparison versus the locked baseline.
- **FR-008**: The system MUST run validation-only threshold selection for each candidate and save the threshold sweep, selected threshold, and threshold-stability evidence.
- **FR-009**: The system MUST require at least one targeted ablation or stability check for any candidate that is treated as improved.
- **FR-010**: The system MUST save one complete output package for every serious candidate, including validation predictions, test probabilities, threshold report, metrics report, runtime report, target-distribution report, and submission artifact.
- **FR-011**: The system MUST maintain one standardized candidate comparison table covering performance, runtime, artifact completeness, and decision status for every serious candidate.
- **FR-012**: The system MUST classify every serious candidate as accepted, rejected, or analysis-only with an explicit reason.
- **FR-013**: The system MUST prevent promotion of candidates whose recall gains primarily come from unjustified reusable-bottle over-rejection.
- **FR-014**: The system MUST treat hard-example improvement as a separate acceptance gate for the `0.98+` pursuit rather than relying only on overall validation performance.
- **FR-015**: The system MUST include repeated runtime evidence suitable for competition-style efficiency review before a candidate can be treated as finalist-ready.
- **FR-016**: The system MUST produce one insight evidence pack that explains the baseline lock, hard-example failure modes, candidate outcomes, runtime tradeoffs, and final recommendation.
- **FR-017**: The system MUST preserve a rollback path by keeping candidate artifacts complete and comparable across runs.
- **FR-018**: The feature MUST remain compliant with no test-label use, no public-leaderboard threshold tuning, and no overwriting of locked baseline artifacts.

### Constitution Alignment *(mandatory)*

- **Binary Output**: Preserves the required `0 = Reusable` and `1 = Not Reusable` decision framing across baseline comparison, candidate evaluation, and final selection outputs.
- **Accuracy/Speed**: Explicitly balances hard-example F1 improvement with runtime competitiveness instead of optimizing validation performance in isolation.
- **ROI/Hybrid Flow**: Allows classifier-first or gated repair paths to be compared, but requires hard-example and precision evidence before any broader correction strategy is accepted.
- **Annotation/Version Strategy**: Treats Phase 3 as a controlled training program downstream of the approved hard-row candidate package rather than as uncontrolled experimentation.
- **Reproducibility/Leakage**: Requires locked comparison anchors, saved artifacts, validation-only thresholding, repeated runtime records, and zero reliance on test labels or leaderboard-only tuning.
- **Confidentiality**: Keeps training inputs, candidate outputs, comparison tables, and insight artifacts inside private project-controlled paths.
- **Explainability**: Requires changed-row comparisons, failure-mode evidence, candidate rationales, runtime tradeoffs, and final selection reasoning.
- **Bias/Imbalance**: Makes governed hard-example behavior a first-class acceptance gate so clean-row gains cannot hide failure on the hardest subset.

### Key Entities *(include if feature involves data)*

- **Locked Baseline Report**: The single comparison anchor containing the active baseline metrics, threshold, distribution, prediction outputs, and runtime reference.
- **Governed Phase 3 Training Input Set**: The approved training-eligible hard-example rows derived only from `phase3_use_allowed = true` candidate-package outputs.
- **Candidate Hypothesis Record**: The statement of what a specific Phase 3 candidate is trying to improve and why it is being run.
- **Candidate Evaluation Record**: The standardized result set for one candidate, including performance, threshold, distribution, changed-row, and runtime evidence.
- **Candidate Comparison Table**: The cross-candidate decision table used to accept, reject, or hold candidates for analysis only.
- **Runtime Evidence Pack**: The repeated timing and efficiency documentation used to judge finalist readiness.
- **Insight Evidence Pack**: The narrative and evidence set that explains what was learned from the controlled Phase 3 program and why the final path is justified.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of serious Phase 3 candidates are evaluated against one explicitly locked baseline anchor rather than mixed comparison references.
- **SC-002**: 100% of governed Phase 3 training inputs trace back to approved `phase3_use_allowed = true` candidate-package rows, with zero blocked rows admitted.
- **SC-003**: 100% of serious Phase 3 candidates produce complete artifact packages including predictions, threshold evidence, metrics, runtime evidence, distribution evidence, and submission output.
- **SC-004**: 100% of accepted or finalist-ready candidates include full validation performance, hard-example-only performance, excluded-hard-example performance, and changed-row comparison against the baseline.
- **SC-005**: 100% of candidates promoted as improved include at least one targeted ablation or stability check supporting that claim.
- **SC-006**: 100% of finalist-ready candidates include repeated runtime evidence suitable for competition-style efficiency review.
- **SC-007**: The final comparison table records an explicit decision and justification for every serious candidate.
- **SC-008**: The final insight package enables a reviewer to understand why a candidate was chosen or rejected without rerunning the full training history.
- **SC-009**: The controlled Phase 3 workflow records zero use of test labels, zero public-leaderboard-only threshold tuning, and zero overwriting of locked baseline artifacts.

## Assumptions

- The approved Phase 3 candidate package from Spec 014 already exists and is the only valid source of hard-example training eligibility.
- One original V2B-family baseline can be locked as the single comparison anchor for controlled Phase 3 work.
- Candidate training may involve multiple architectures or image sizes, but only serious candidates with explicit hypotheses enter the governed comparison workflow.
- Runtime evidence must reflect competition-style repeated measurement rather than one lucky local timing result.
- The insight deliverable is part of the feature scope because competition ranking depends on both model behavior and the quality of technical reasoning.
