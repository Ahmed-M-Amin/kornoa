# Feature Specification: COCO Annotation Parser

**Feature Branch**: `003-coco-annotation-parser`

**Created**: 2026-06-01

**Status**: Draft

**Input**: User description: "Read Phase 3 from docs/krones-final-implementation-plan.md and, according to GitHub Spec Kit best practices, create the 3rd spec. The approved future specification division defines SPEC-003 as COCO Annotation Parser; ROI Cropping and Preprocessing remains SPEC-004."

## Clarifications

### Session 2026-06-01

- Q: What parser result boundary should SPEC-003 define for downstream features? -> A: Parser returns normalized annotation records plus validation summaries; no detector-format export.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Parse COCO Annotation File (Priority: P1)

A project contributor can load the configured COCO-style annotation file and
receive normalized annotation records and a validated, inspectable annotation
summary before ROI, detector, or training features depend on it.

**Why this priority**: COCO annotation loading is a V1 prerequisite for ROI
availability checks, detector preparation, and later explanation outputs. If the
annotation file cannot be parsed reliably, later preprocessing and model work
will inherit invalid image-to-region mappings.

**Independent Test**: Can be tested with the synthetic dataset annotation file
by confirming images, annotations, and categories are read, counted, and linked
without using the private Krones dataset.

**Acceptance Scenarios**:

1. **Given** a valid COCO-style annotation file, **When** a contributor runs the
   annotation parser, **Then** the system reports image count, annotation count,
   category count, and successful image-to-annotation mappings.
2. **Given** an annotation file with missing required top-level collections,
   **When** a contributor runs the parser, **Then** the system returns a readable
   validation error naming the missing collection.

---

### User Story 2 - Validate Annotation Relationships (Priority: P2)

A project contributor can detect annotation quality problems such as orphaned
annotations, unknown categories, and malformed boxes before downstream features
use those annotations. Missing segmentation alone is not a warning diagnostic;
it is recorded as summary availability metadata.

**Why this priority**: The project must not silently train, crop, explain, or
convert detector data from broken annotation relationships.

**Independent Test**: Can be tested with synthetic annotation records that
include one valid annotation and controlled invalid records for orphaned image
references, unknown categories, malformed bounding boxes, and optional
segmentation fields.

**Acceptance Scenarios**:

1. **Given** an annotation references an unknown image, **When** validation runs,
   **Then** the annotation is reported as orphaned without becoming a training
   label.
2. **Given** an annotation references an unknown category, **When** validation
   runs, **Then** the unknown category is reported without stopping valid
   annotation summaries.
3. **Given** a bounding box is missing, negative, or has invalid dimensions,
   **When** validation runs, **Then** the malformed box is reported with its
   annotation identifier.

---

### User Story 3 - Provide Annotation-Derived Summaries (Priority: P3)

A project contributor can review annotation-derived category, ROI, box, mask,
and image coverage summaries that support later ROI preprocessing, detector
training, explainability, and report assets.

**Why this priority**: Later specs need stable annotation facts, but this spec
must stop at parsing and validation rather than implementing ROI crops or model
training.

**Independent Test**: Can be tested by running the parser on synthetic COCO
metadata and confirming coverage, category distribution, ROI availability,
bounding-box availability, and segmentation availability summaries are produced.

**Acceptance Scenarios**:

1. **Given** annotations include categories, boxes, and optional segmentation,
   **When** summaries are generated, **Then** the system reports category
   distribution, box availability, segmentation availability, and ROI
   availability.
2. **Given** some images have no annotations, **When** summaries are generated,
   **Then** the system reports annotated and unannotated image counts.

### Edge Cases

- If the annotation file is missing, the parser must report a readable missing
  file error.
- If the annotation file is malformed JSON, the parser must report a readable
  parse error without exposing private file contents.
- If duplicate image identifiers or duplicate annotation identifiers appear, the
  parser must report duplicate counts and identifiers.
- If an annotation has a valid image reference but no box or segmentation, the
  parser must keep the record and report that ROI information is unavailable in
  summaries only.
- If segmentation data exists in polygon or mask-style form, the parser must
  record `has_segmentation` and `segmentation_available_count` without requiring
  detector or segmentation training.
- If category labels are absent or empty, the parser must still report coverage
  and relationship validation results.
- If private annotation contents are used locally, generated diagnostics must
  remain local or ignored unless explicitly approved for private team sharing.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST load COCO-style annotation metadata from a
  configured annotation file path rather than a hardcoded private path.
- **FR-002**: The system MUST validate that the annotation metadata contains
  image, annotation, and category collections.
- **FR-003**: The system MUST parse image records with stable image identifiers,
  file names, and available width and height metadata.
- **FR-004**: The system MUST parse annotation records with annotation
  identifiers, image references, category references, bounding-box availability,
  segmentation availability, and ROI availability.
- **FR-005**: The system MUST parse category records and map category
  identifiers to readable category names when available.
- **FR-006**: The system MUST report image-to-annotation coverage, including
  annotated and unannotated image counts.
- **FR-007**: The system MUST report orphaned annotations that reference
  unknown images.
- **FR-008**: The system MUST report annotations that reference unknown
  categories.
- **FR-009**: The system MUST report malformed bounding boxes, including missing
  boxes, negative coordinates, and non-positive width or height values.
- **FR-010**: The system MUST summarize category or defect-label distribution
  from valid category-linked annotations.
- **FR-011**: The system MUST summarize ROI availability based on bounding-box
  or segmentation metadata.
- **FR-012**: The system MUST summarize segmentation availability without
  requiring segmentation model training.
- **FR-013**: The system MUST expose normalized annotation records and
  validation summaries that downstream ROI, detector, audit, and reporting
  features can consume.
- **FR-014**: The system MUST provide readable diagnostics for missing files,
  malformed JSON, missing collections, duplicate identifiers, orphaned
  annotations, unknown categories, and malformed boxes.
- **FR-015**: The system MUST NOT infer labels for test images or convert
  annotation metadata into test labels.
- **FR-016**: The system MUST NOT require private annotation files or generated
  diagnostics to be committed to public source control.
- **FR-017**: The system MUST be testable with synthetic COCO-style fixtures
  without requiring private Krones dataset access.

### Constitution Alignment *(mandatory)*

- **Binary Output**: This feature does not produce bottle decisions, but it
  preserves the later binary `0 = Reusable` and `1 = Not Reusable` pipeline by
  validating annotation evidence without changing label semantics.
- **Accuracy/Speed**: Reliable annotation parsing supports later F1-score and
  speed decisions by preventing invalid ROI, detector, or explanation inputs.
- **ROI/Hybrid Flow**: This feature provides ROI availability and region
  metadata for later ROI-first preprocessing and hybrid inference, but it does
  not implement cropping or detector fallback.
- **Annotation/Version Strategy**: This is `SPEC-003: COCO Annotation Parser`,
  part of the V1 working pipeline. It owns reusable annotation parsing behavior;
  ROI crop behavior continues in `SPEC-004`.
- **Reproducibility/Leakage**: Parser behavior must be repeatable from
  configuration and synthetic fixtures, and it must not infer labels or leak test
  information into training or validation.
- **Confidentiality**: Private annotation files and derived diagnostics remain
  local/private and are not committed publicly.
- **Explainability**: Parsed boxes, masks, categories, and ROI availability
  support later explanation outputs such as detector visualizations and defect
  reasons.
- **Bias/Imbalance**: Category and defect-label distribution summaries support
  later rare-defect and imbalance analysis.

### Key Entities *(include if feature involves data)*

- **Annotation Source**: Configured COCO-style metadata file containing image,
  annotation, and category collections.
- **COCO Image Record**: Metadata for an image, including identifier, file name,
  width, height, and whether annotations reference it.
- **COCO Annotation Record**: Metadata for a region or defect observation,
  including annotation identifier, linked image, category, box availability,
  segmentation availability, ROI availability, and validation status.
- **COCO Category Record**: Category identifier and readable defect/category
  name used to summarize annotation distribution.
- **Annotation Validation Report**: Summary of coverage, orphaned annotations,
  unknown categories, malformed boxes, duplicates, ROI availability, and
  segmentation availability.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A contributor can parse the synthetic COCO fixture and receive
  image, annotation, category, coverage, ROI, and segmentation summaries without
  private dataset access.
- **SC-002**: The parser reports 100% of orphaned annotations and unknown
  category references present in the provided annotation file.
- **SC-003**: The parser reports malformed bounding boxes with their annotation
  identifiers for every malformed box in the provided annotation file.
- **SC-004**: Annotation coverage summaries identify annotated and unannotated
  images for 100% of image records in the provided annotation file.
- **SC-005**: Missing files, malformed JSON, missing collections, duplicate
  identifiers, and relationship errors produce readable diagnostics that name
  the failing item.
- **SC-006**: Parser tests run successfully using only synthetic fixtures and do
  not require private Krones images, annotations, or labels.
- **SC-007**: The parser completes the synthetic COCO fixture parse in under one
  second and, when valid private dataset access is available, completes the
  private `train_annotations.json` parse in under one minute.

## Assumptions

- SPEC-003 follows `.specify/memory/constitution.md` and the approved future
  specification division in `docs/krones-final-implementation-plan.md`.
- The COCO-style annotation file contains image, annotation, and category
  collections compatible with the Krones challenge metadata.
- Full ROI cropping and preprocessing behavior is out of scope and will be
  specified in `SPEC-004`.
- Detector or segmentation dataset conversion and detector-format export are
  out of scope and will be specified with detector or segmentation training.
- Synthetic fixtures are sufficient for automated parser tests; private dataset
  access is optional for manual validation only.
- Generated diagnostics from private annotation files remain ignored or private
  unless explicitly approved for private team sharing.
