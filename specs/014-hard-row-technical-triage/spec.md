# Feature Specification: Hard-Row Technical Triage

**Feature Branch**: `[014-hard-row-technical-triage]`

**Created**: 2026-06-12

**Status**: Draft

**Input**: User description: "Read `docs/new_implinmination.md` and make Spec 014 according to the best practice of Spec Kit."

## Clarifications

### Session 2026-06-12

- Q: Should `phase3_use_allowed` be determined only by action label, or should it remain a separate safety gate? → A: `phase3_use_allowed` remains a separate safety gate; trainable action labels may still be blocked by high label-risk, anomaly-risk, or weak evidence warnings.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build Hard-Row Review Set (Priority: P1)

As the model developer, I need one governed review set for the 442 hardest validation rows so I can inspect the exact failure cases that block higher locked-split performance without mixing in unrelated validation rows.

**Why this priority**: The roadmap identifies the hardest 442 rows as the main blocker to improving beyond strong clean-row performance. Without one governed set, later review, triage, and training decisions are inconsistent.

**Independent Test**: Can be fully tested by generating the governed hard-row review manifest from the quality-audit outputs and confirming that every row has one stable `image_id`, baseline/candidate prediction context, and hard-example classification.

**Acceptance Scenarios**:

1. **Given** the governed hard-row audit exists, **When** the review workflow is run, **Then** it produces one manifest containing exactly the governed hard rows and no duplicate `image_id` values.
2. **Given** the governed review manifest is generated, **When** a reviewer opens the output, **Then** each row shows target, prediction context, audit category, and hard-example type.
3. **Given** optional clustering or label-quality tooling is unavailable, **When** the review workflow runs, **Then** it still produces a usable governed review set with explicit fallback reporting.

---

### User Story 2 - Enrich Hard Rows With Technical Evidence (Priority: P2)

As the model developer, I need annotation, ROI, crop-quality, and evidence-overlay outputs for the governed hard rows so I can convert unknown model errors into evidence-aware technical decisions instead of relying on manual guessing.

**Why this priority**: The roadmap states that the hard rows must be separated into learnable cases, ambiguous cases, mislabeled cases, and preprocessing issues before any serious Phase 3 candidate can be trusted.

**Independent Test**: Can be fully tested by running evidence completion on the governed review set and confirming that each hard row receives annotation evidence where available, crop-quality metrics, detector-evidence fallback status, and evidence asset paths.

**Acceptance Scenarios**:

1. **Given** training annotations and images are available, **When** evidence completion runs, **Then** each governed hard row receives annotation-derived evidence or an explicit missing-evidence status.
2. **Given** ROI or crop quality is weak, **When** evidence-aware triage is produced, **Then** the row is marked with a preprocessing-oriented action instead of being treated as a safe training example.
3. **Given** detector outputs are absent, **When** evidence completion runs, **Then** annotation evidence is used as the fallback rather than marking all rows as detector-missing.

---

### User Story 3 - Export Safe Phase 3 Candidate Package (Priority: P3)

As the model developer, I need the evidence-aware action plan split into allowed and blocked Phase 3 candidate packages so I can start the next training program from rows that passed technical review while keeping high-risk rows out of training.

**Why this priority**: The roadmap requires strict candidate discipline. Training should only consume hard rows that passed technical triage, while blocked rows remain traceable for engineering or label follow-up.

**Independent Test**: Can be fully tested by validating the evidence-aware action plan counts, exporting the allowed and blocked package files, and confirming that no blocked row appears in any allowed package output.

**Acceptance Scenarios**:

1. **Given** the evidence-aware action plan exists, **When** the package export runs, **Then** it validates total row count, allowed count, and blocked count before writing any package files.
2. **Given** a row is marked `phase3_use_allowed = true`, **When** the package export is reviewed, **Then** that row appears only in the allowed Phase 3 outputs that match its recommended action.
3. **Given** a row is blocked because of ROI issues, suspected mislabeling, or other high-risk conditions, **When** the package export is reviewed, **Then** that row appears only in blocked outputs and never in allowed training files.

### Edge Cases

- If the governed hard-row audit input is missing, the workflow must fail clearly before producing any review, triage, or package outputs.
- If the hard-row manifest contains duplicate `image_id` values, the workflow must stop rather than silently dropping or duplicating rows.
- If training annotations exist but an image file is missing, the workflow must keep the row and mark image-dependent evidence as missing.
- If ROI is absent in annotations, the workflow may derive fallback ROI evidence from defect annotations but must record that the evidence was derived rather than explicit.
- If annotation evidence, crop-quality evidence, or detector fallback evidence is missing for a row, the workflow must preserve the row and assign a conservative action rather than excluding it silently.
- If the evidence-aware action plan fails row-count or allowed/blocked-count validation, the Phase 3 package export must stop without writing partial candidate files.
- If optional review tooling is unavailable, the governed review and evidence-completion outputs must remain valid and reproducible.
- If any test labels, public leaderboard feedback, training, or submission creation are attempted inside this feature, the feature is invalid.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST create one governed hard-row review set from the validation-only hard-row audit outputs and preserve one unique record per `image_id`.
- **FR-002**: The governed review set MUST include, at minimum, `image_id`, target, baseline probability, baseline prediction, candidate probability, candidate prediction, primary audit category, and hard-example type.
- **FR-003**: The system MUST produce a manual-review template and a review manifest for the governed hard rows.
- **FR-004**: The system MUST support optional label-quality scoring, clustering, anomaly scoring, and representative review outputs while preserving fallback behavior when optional tooling is unavailable.
- **FR-005**: The system MUST generate technical-triage outputs that expose representative hard-row clusters, suspicious rows, likely label issues, and likely anomaly rows for the governed hard-row set.
- **FR-006**: The system MUST enrich the governed hard rows with annotation evidence derived from the training annotation source, including annotation count, defect categories, defect-area evidence, and image dimensions where available.
- **FR-007**: The system MUST enrich the governed hard rows with ROI and crop-quality evidence, including ROI existence, ROI coverage, border-touching behavior, crop-size checks, brightness, contrast, blur, edge density, and crop-quality status.
- **FR-008**: The system MUST generate evidence-overlay assets for governed hard rows, including ROI overlays, annotation overlays, ROI crops, and enhanced review views.
- **FR-009**: The system MUST use annotation evidence as the fallback detector-evidence source when trained detector outputs are unavailable.
- **FR-010**: The system MUST produce one evidence-aware final action plan covering the full governed hard-row set, with one recommended action and one `phase3_use_allowed` decision per row.
- **FR-011**: The evidence-aware action plan MUST distinguish at least these action types when supported by row evidence: safe general Phase 3 candidates, artifact-focused training candidates, tiny or low-contrast defect candidates, ROI or preprocessing bugs, suspected mislabels, and other conservative blocked cases.
- **FR-011a**: The system MUST treat `phase3_use_allowed` as a separate safety gate rather than as a direct synonym for the recommended action label.
- **FR-011b**: The system MUST allow rows with otherwise trainable action labels to remain blocked when high label-risk, anomaly-risk, or weak evidence warnings indicate that Phase 3 use is not yet safe.
- **FR-012**: The system MUST export a Phase 3 candidate package only from rows marked `phase3_use_allowed = true`.
- **FR-013**: The Phase 3 candidate package export MUST validate the input action plan row count and allowed/blocked totals before writing any candidate package outputs.
- **FR-014**: The Phase 3 candidate package export MUST split allowed rows into action-specific outputs and split blocked rows into ROI-pipeline, suspected-mislabel, and other high-risk outputs.
- **FR-015**: The system MUST preserve enough row-level evidence in review, evidence-completion, and package-export outputs to support downstream training selection and insight reporting without rerunning the earlier analysis.
- **FR-016**: The feature MUST remain analysis-only and MUST prohibit training, submission creation, public leaderboard tuning, test-label use, and label overwriting.
- **FR-017**: The feature MUST keep output paths and summary reports stable enough for downstream planning and candidate-training work to reference them deterministically.

### Constitution Alignment *(mandatory)*

- **Binary Output**: Preserves `0 = Reusable` and `1 = Not Reusable` targets, predictions, and candidate-package labels across all outputs.
- **Accuracy/Speed**: Improves hard-row decision quality and candidate selection without changing deployed inference runtime because the feature is offline analysis and export only.
- **ROI/Hybrid Flow**: Strengthens ROI and hard-row evidence handling without forcing detector execution or changing the deployed classifier-plus-detector path.
- **Annotation/Version Strategy**: Uses existing training annotations and validation-derived hard-row evidence to prepare Phase 3 candidate inputs without declaring a new final model version.
- **Reproducibility/Leakage**: Requires governed validation-only inputs, stable row identity, saved reports, explicit counts, and zero use of test labels or public-score tuning.
- **Confidentiality**: Keeps dataset references, evidence assets, and candidate packages inside local project artifacts intended for competition-safe internal use.
- **Explainability**: Produces row-level audit context, evidence overlays, action reasoning, and allowed-versus-blocked package outputs that explain why a hard row is or is not safe for Phase 3.
- **Bias/Imbalance**: Makes hard-row concentrations, ambiguous groups, suspicious labels, and evidence-backed exclusions visible instead of letting them distort later training silently.

### Key Entities *(include if feature involves data)*

- **Governed Hard Row**: One validation row selected into the governed hard-row set, identified by stable `image_id`, with audit category and hard-example type.
- **Review Manifest**: The structured record used for human or technical review, combining governed hard-row prediction context with review metadata.
- **Technical Evidence Record**: The annotation, ROI, crop-quality, and detector-fallback evidence attached to one governed hard row.
- **Evidence Asset**: One generated visual artifact that helps explain ROI placement, annotation location, crop quality, or defect visibility for a governed hard row.
- **Evidence-Aware Action Plan Row**: One hard-row decision record containing evidence fields, recommended action, and the `phase3_use_allowed` gate.
- **Phase 3 Safety Gate**: The final allow-or-block decision that may override a trainable-looking action label when row risk remains too high for safe Phase 3 use.
- **Phase 3 Candidate Package**: The exported set of allowed and blocked CSV outputs derived from the evidence-aware action plan.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of governed hard-row records retain unique `image_id` values and trace back to the validation-only hard-row audit source.
- **SC-002**: 100% of governed hard rows receive one review-manifest record and one evidence-aware action-plan record.
- **SC-003**: 100% of governed hard rows receive either annotation evidence or an explicit missing-evidence status in the evidence outputs.
- **SC-004**: 100% of governed hard rows receive either crop-quality evidence or an explicit image/ROI-missing status in the crop-quality outputs.
- **SC-005**: 100% of governed hard rows receive exactly one recommended action and exactly one `phase3_use_allowed` value in the evidence-aware action plan.
- **SC-006**: The Phase 3 package export writes zero allowed rows that were marked blocked in the evidence-aware action plan.
- **SC-007**: The Phase 3 package export stops before writing outputs whenever the action-plan row count or allowed/blocked validation totals are incorrect.
- **SC-008**: A reviewer can determine from the produced outputs which hard rows are likely safe for Phase 3 training, which are blocked by ROI or preprocessing issues, and which require label or risk follow-up without rerunning the analysis.
- **SC-009**: The feature records zero use of training, submission creation, public leaderboard tuning, and test labels in all summary outputs.

## Assumptions

- The governed hard-row set remains the 442-row validation-only pool identified by the existing hard-row audit workflow.
- The original training annotations and training images are available locally for evidence completion when this feature runs in its full form.
- Optional detector outputs may be absent, so annotation evidence is the default fallback evidence source for localized defect reasoning.
- Evidence completion and candidate packaging are downstream of the governed review manifest and do not redefine the original hard-row audit membership.
- Phase 3 candidate packaging is a preparation step only and does not itself start model training.
