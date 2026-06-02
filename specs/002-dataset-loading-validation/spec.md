# Feature Specification: Dataset Loading and Validation

**Feature Branch**: `002-dataset-loading-validation`

**Created**: 2026-06-01

**Status**: Draft

**Input**: User description: "Read Phase 2 from docs/krones-final-implementation-plan.md and, according to GitHub Spec Kit best practices, create the 2nd spec."

## Clarifications

### Session 2026-06-01

- Q: For SPEC-002, how much COCO annotation handling should be included before SPEC-003? -> A: Audit-only annotation metadata handling for coverage, distribution, and ROI summaries; reusable COCO parser APIs remain in SPEC-003.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load Competition Dataset (Priority: P1)

A project contributor can point the project at the configured Krones competition
dataset and verify that the required dataset files and image folders are present
before any training or inference work begins.

**Why this priority**: Dataset loading is the first blocking step for all later
COCO parsing, ROI, training, validation, inference, and submission features.
Without reliable dataset discovery, later model work cannot be trusted.

**Independent Test**: Can be tested by using the configured dataset root and
confirming that `train.csv`, `train_images`, `test_images`,
`sample_submission.csv`, and `train_annotations.json` are discovered without
requiring hardcoded paths in core modules.

**Acceptance Scenarios**:

1. **Given** a configured dataset root containing the expected competition files,
   **When** a contributor runs the dataset loading check, **Then** the system
   reports all required files and folders as present.
2. **Given** a required dataset file or folder is missing, **When** a contributor
   runs the dataset loading check, **Then** the system reports the exact missing
   item without silently continuing.

---

### User Story 2 - Validate Dataset Integrity (Priority: P2)

A project contributor can audit the dataset and understand whether training
rows, image files, test images, and annotation records can be matched reliably.

**Why this priority**: The project depends on correct image-to-label and
image-to-annotation relationships. Broken mappings would create invalid training
data, misleading metrics, or unusable audit results.

**Independent Test**: Can be tested by running the dataset audit against the
configured dataset and verifying that train/test image matching, image IDs,
annotation availability, missing files, and image-size summaries are reported.

**Acceptance Scenarios**:

1. **Given** `train.csv` references training images, **When** the audit checks
   file matching, **Then** every referenced image is counted as matched or
   reported as missing.
2. **Given** COCO annotations are available, **When** the audit links annotations
   to images, **Then** annotation availability and unmatched annotation records
   are summarized for review.

---

### User Story 3 - Produce Dataset Audit Outputs (Priority: P3)

A project contributor can generate report-ready dataset audit outputs for class
distribution, defect-label distribution, ROI availability, image-size summary,
and visual sample review.

**Why this priority**: The constitution requires bias and imbalance handling.
The project must understand class imbalance, defect-label imbalance, ROI
availability, and possible annotation quality issues before training begins.

**Independent Test**: Can be tested by running the audit and confirming that the
expected summary tables and figures are produced in the approved generated-output
locations without exposing private dataset contents publicly.

**Acceptance Scenarios**:

1. **Given** the dataset loads successfully, **When** the audit completes,
   **Then** it produces dataset summary, class distribution, defect distribution,
   ROI availability, image-size summary, and sample visualization outputs.
2. **Given** audit outputs may derive from private competition data, **When** a
   contributor checks source control status, **Then** generated audit outputs are
   kept in ignored output locations unless explicitly approved for private team
   use.

---

### Edge Cases

- If the dataset root is unset or points to the wrong location, the audit must
  fail with a readable configuration error.
- If `train.csv` contains rows for missing images, the audit must report counts
  and identifiers for missing images.
- If image files exist without corresponding train rows, the audit must report
  unmatched files separately from missing files.
- If annotation records reference unknown images, the audit must report those
  orphaned annotations.
- If annotation categories or defect labels are empty, the audit must still
  produce class and availability summaries.
- If image dimensions vary, the audit must summarize sizes so later preprocessing
  specs can choose safe resizing and ROI behavior.
- If private dataset files are present in the working tree, the feature must not
  require committing or uploading them.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST load the dataset root from configuration rather
  than from hardcoded source-code paths.
- **FR-002**: The system MUST verify presence of `train.csv`, `train_images`,
  `test_images`, `sample_submission.csv`, and `train_annotations.json`.
- **FR-003**: The system MUST parse training labels from `train.csv` and report
  the number of labeled training rows.
- **FR-004**: The system MUST discover training image files and test image files
  under the configured dataset root.
- **FR-005**: The system MUST match training labels to training image files and
  report matched and missing image counts.
- **FR-006**: The system MUST identify image files that are present but not
  referenced by the expected metadata.
- **FR-007**: The system MUST read COCO annotation metadata sufficiently to
  summarize image-to-annotation coverage, category or defect-label distribution,
  and ROI availability.
- **FR-008**: The system MUST report orphaned annotations that reference unknown
  or unavailable images.
- **FR-009**: The system MUST generate a class distribution summary for the
  binary `0 = Reusable` and `1 = Not Reusable` labels.
- **FR-010**: The system MUST generate a defect-label distribution summary when
  defect labels are available in annotation metadata.
- **FR-011**: The system MUST summarize ROI availability for training images
  when ROI or annotation-derived region information is available.
- **FR-012**: The system MUST summarize image-size information for train and test
  images.
- **FR-013**: The system MUST produce dataset audit outputs in the approved
  generated-output locations for reports and figures.
- **FR-014**: The system MUST produce readable errors for missing files,
  malformed metadata, and configuration problems.
- **FR-015**: The system MUST NOT require private dataset files, annotations, or
  generated audit artifacts to be committed to public source control.
- **FR-016**: The system MUST provide a repeatable way for contributors to run
  the dataset audit from a clean project checkout with valid private dataset
  access.

### Constitution Alignment *(mandatory)*

- **Binary Output**: This feature audits the labels that later preserve the
  `0 = Reusable` and `1 = Not Reusable` binary decision.
- **Accuracy/Speed**: This feature does not train models, but it provides the
  dataset quality and imbalance facts needed for later F1-score and speed-aware
  model decisions.
- **ROI/Hybrid Flow**: This feature reports ROI availability and annotation
  coverage so later ROI and hybrid inference specs can use trustworthy inputs.
- **Annotation/Version Strategy**: This is `SPEC-002: Dataset Loading and
  Validation`, part of the V1 working pipeline. COCO parsing is summarized here;
  reusable parser APIs and full parser implementation details continue in
  `SPEC-003`.
- **Reproducibility/Leakage**: Audit outputs are generated from configured
  private data without using test labels or leaking test information into
  training.
- **Confidentiality**: Private images, annotations, labels, and generated audit
  artifacts remain local/private and are not uploaded or committed publicly.
- **Explainability**: Dataset sample visualizations support later technical
  review and explanation, but they must respect confidentiality.
- **Bias/Imbalance**: Class distribution, defect-label distribution, missing
  files, and ROI availability are mandatory outputs for bias and imbalance
  analysis.

### Key Entities *(include if feature involves data)*

- **Dataset Root**: Configured private directory containing the Krones
  competition files and folders.
- **Training Label Record**: A row from `train.csv` that maps an image identifier
  to the binary reusable/not-reusable target.
- **Image Asset**: A train or test image discovered under the configured dataset
  root.
- **Annotation Record**: Audit-only COCO-derived metadata linked to an image,
  including categories, boxes, masks, or ROI information when available.
- **Dataset Audit Report**: Generated summary of file presence, image matching,
  label distribution, annotation coverage, ROI availability, and image sizes.
- **Audit Figure**: Generated visual output such as class distribution, defect
  distribution, or sample image grid.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A contributor with valid private dataset access can run the audit
  and see required file/folder presence results in under five minutes.
- **SC-002**: The audit reports matched and missing training-image counts for
  100% of rows in `train.csv`.
- **SC-003**: The audit reports train/test image counts, class distribution,
  defect-label distribution when available, ROI availability, and image-size
  summary in generated report outputs.
- **SC-004**: Missing files, orphaned annotations, malformed metadata, and
  configuration errors produce readable diagnostics that identify the failing
  item.
- **SC-005**: Generated audit outputs are written only to approved output
  locations and do not require private dataset files or generated artifacts to be
  tracked in public source control.

## Assumptions

- Phase 2 covers dataset loading, validation, and audit-only annotation metadata
  summaries; reusable COCO parser APIs and full parser behavior are detailed in
  `SPEC-003`.
- The private Krones dataset remains local, Kaggle-private, Colab-private, or
  private-team-only according to `.specify/memory/constitution.md`.
- `train.csv` contains the authoritative training labels for binary
  reusable/not-reusable classification.
- `train_annotations.json` is COCO-style metadata with image, annotation,
  category, and possible ROI/region information.
- Test images are used only for discovery, counting, and submission-readiness
  checks; no test labels are inferred or manually assigned.
- Audit figures may use redacted or private-team-only visualizations and must
  not be published publicly without approval.
