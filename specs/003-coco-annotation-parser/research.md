# Research: COCO Annotation Parser

## Decision: Use a Small Internal COCO Parser

**Rationale**: SPEC-003 needs predictable validation behavior, readable
diagnostics, and normalized records tailored to Krones downstream features. A
small internal parser can validate missing collections, duplicate identifiers,
orphaned annotations, unknown categories, malformed boxes, ROI availability, and
segmentation availability without introducing heavyweight behavior or external
format assumptions.

**Alternatives considered**:

- `pycocotools`: Useful later for mask-heavy workflows, but it is not ideal as
  the only validation layer because the project needs custom diagnostics and
  synthetic-fixture tests that run consistently on Windows, Kaggle, and Colab.
- Ad hoc parsing inside audit code: Rejected because it duplicates logic and
  weakens the SPEC-003 boundary clarified by the user.

## Decision: Normalize Records Before Summaries

**Rationale**: The clarified boundary requires normalized annotation records
plus validation summaries. Normalized records give SPEC-004 ROI work,
SPEC-006 detector work, audit reporting, and explainability a stable contract
without requiring detector-format export in this spec.

**Alternatives considered**:

- Summaries only: Rejected because downstream features would need to reparse
  raw COCO metadata.
- Detector-ready export: Rejected because it belongs to detector or
  segmentation training specs, not SPEC-003.

## Decision: Treat Boxes and Segmentations as ROI Evidence, Not Crop Commands

**Rationale**: SPEC-003 should report whether ROI evidence exists from bounding
boxes or segmentation data. Actual crop geometry, dark-defect preservation, and
fallback crop decisions belong to SPEC-004.

**Alternatives considered**:

- Compute final crops in the parser: Rejected as scope creep and a constitution
  risk because ROI logic has its own quality gate.
- Ignore segmentation availability: Rejected because future detector,
  segmentation, and explainability work need this signal.

## Decision: Synthetic Fixtures Are Mandatory for Automated Tests

**Rationale**: The constitution requires confidentiality. Parser tests must run
without private Krones data. Synthetic fixtures should include valid records and
controlled invalid examples for orphaned annotations, unknown categories,
duplicate identifiers, malformed boxes, missing segmentation, and ROI
availability.

**Alternatives considered**:

- Private dataset tests only: Rejected because public/source-control CI and
  local checkout validation would be blocked.
- Empty placeholder fixtures: Rejected because they cannot validate parser
  behavior or downstream contracts.

## Decision: Keep Diagnostics Structured and Redacted

**Rationale**: Diagnostics should identify failing item types and identifiers
without dumping private annotation contents. This supports debugging while
respecting dataset confidentiality.

**Alternatives considered**:

- Raw exception dumps: Rejected because they can expose private file paths or
  annotation content and are harder to test.
- Silent skipping: Rejected because malformed annotation relationships would
  propagate into ROI, detector, and explainability work.
