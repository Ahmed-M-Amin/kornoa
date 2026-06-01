# Feature Specification: ROI Cropping and Preprocessing

**Feature Branch**: `005-roi-cropping-preprocessing`

**Created**: 2026-06-01

**Status**: Draft

**Input**: User description: "Start SPEC-004 after completing SPEC-003. The approved future specification division defines SPEC-004 as ROI Cropping and Preprocessing. Phase 3 of the implementation plan says the goal is to focus the model on the useful bottle inspection region while preserving internal dark defects. Training images use annotation ROI when available and fall back to center/circular crop. Test images use safe center/circular crop and fall back to full resize. Training preprocessing includes resize, brightness/contrast, small rotation, small shift/scale, blur/noise, and normalize. Validation/test preprocessing is deterministic with resize and normalize only. Quality gate requires ROI samples, internal dark-defect preservation, consistent crop size, and deterministic validation/test preprocessing."

## Clarifications

### Session 2026-06-01

- Q: What default crop and preprocessing output size should SPEC-004 require? -> A: `384x384`.
- Q: What fallback crop strategy should SPEC-004 use when annotation ROI metadata is unavailable or invalid? -> A: square center crop first, then full-image resize if unsafe.
- Q: How much context should annotation-derived ROI crops keep around the ROI? -> A: expand ROI by 10% on each side, clipped to image bounds.
- Q: What minimum usable ROI size should SPEC-004 require before falling back? -> A: ROI is unsafe if its area is under 10% of image area.
- Q: Where should generated ROI review samples be saved? -> A: `outputs/figures/roi_samples/`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Produce Safe ROI Crops (Priority: P1)

A project contributor can crop bottle images to the useful inspection region so
later classifiers focus on bottle evidence rather than irrelevant camera border.
The crop must preserve possible defects inside the bottle, including dark
internal regions.

**Why this priority**: ROI-focused preprocessing is the next V1 prerequisite
after dataset and annotation parsing. If crops remove defect evidence or vary in
shape unexpectedly, downstream classifier and explainability work becomes
unreliable.

**Independent Test**: Can be tested with synthetic images and annotation records
by confirming annotation-backed crops, fallback crops, full-resize fallbacks,
and consistent output dimensions without private dataset access.

**Acceptance Scenarios**:

1. **Given** a training image with valid annotation ROI metadata, **When** ROI
   cropping runs, **Then** the crop expands the annotation region by 10% on each
   side, clips it to image bounds, and returns the configured output size.
2. **Given** a training image without usable ROI metadata, **When** ROI cropping
   runs, **Then** the system uses a safe square center fallback crop and returns
   the configured output size.
3. **Given** a fallback crop would remove internal dark bottle regions, **When**
   ROI cropping runs, **Then** internal dark evidence is preserved rather than
   blindly removed.

---

### User Story 2 - Apply Split-Specific Preprocessing (Priority: P2)

A project contributor can apply appropriate preprocessing for training,
validation, and test splits while preventing validation/test augmentation from
leaking training behavior into evaluation.

**Why this priority**: Training needs controlled augmentation for robustness,
but validation and test preprocessing must remain deterministic so metrics and
submission behavior are reproducible.

**Independent Test**: Can be tested by running preprocessing on synthetic
images multiple times and confirming training transforms may vary within safe
bounds while validation/test transforms are repeatable.

**Acceptance Scenarios**:

1. **Given** a training image, **When** training preprocessing runs, **Then** the
   image is resized, normalized, and may receive bounded brightness/contrast,
   rotation, shift/scale, blur, or noise changes.
2. **Given** a validation or test image, **When** preprocessing runs multiple
   times, **Then** the output is deterministic and limited to resize and
   normalization.

---

### User Story 3 - Generate Reviewable ROI Samples (Priority: P3)

A project contributor can generate safe visual samples of ROI crops and
preprocessed outputs to verify crop quality before classifier training starts.

**Why this priority**: Visual samples catch destructive crop behavior and border
artifacts early, but generated samples must remain private or ignored when based
on private data.

**Independent Test**: Can be tested with synthetic images by saving sample crop
outputs under approved generated-output locations and confirming generated files
are ignored and not tracked.

**Acceptance Scenarios**:

1. **Given** synthetic images, **When** sample generation runs, **Then** ROI
   sample images are saved under `outputs/figures/roi_samples/` for review with
   consistent dimensions.
2. **Given** generated sample outputs, **When** confidentiality validation runs,
   **Then** generated files are ignored and not tracked by source control.

### Edge Cases

- If annotation ROI metadata is missing, malformed, outside image bounds, or has
  area under 10% of the image area, the system must use the safe fallback crop.
- If fallback cropping cannot safely determine a bottle-focused region, the
  system must use full-image resize rather than failing silently.
- If an image is very small, unusually shaped, or grayscale-like, preprocessing
  must still produce the configured output size and channel format.
- If internal dark regions are present inside the bottle area, preprocessing must
  preserve them as possible defect evidence.
- If private image data is used locally, generated crop samples must remain
  local/private and ignored unless explicitly approved for private team sharing.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST accept image inputs and optional annotation ROI
  metadata without requiring hardcoded private dataset paths.
- **FR-002**: The system MUST produce ROI-focused crops for training images from
  annotation metadata when valid ROI metadata is available, expanding the ROI by
  10% on each side and clipping to image bounds.
- **FR-003**: The system MUST provide a safe square center fallback crop when
  annotation ROI metadata is unavailable, invalid, or smaller than 10% of image
  area.
- **FR-004**: The system MUST provide a full-image resize fallback when a safe
  square center crop cannot be produced.
- **FR-005**: The system MUST preserve internal dark bottle regions and MUST NOT
  remove black or dark pixels blindly.
- **FR-006**: The system MUST output crops and preprocessed images with
  consistent configured dimensions, defaulting to `384x384`.
- **FR-007**: The system MUST apply training preprocessing that supports resize,
  normalization, brightness/contrast adjustment, small rotation, small
  shift/scale, blur, and noise.
- **FR-008**: The system MUST apply deterministic validation and test
  preprocessing limited to resize and normalization.
- **FR-009**: The system MUST make preprocessing reproducible from configuration
  and deterministic seeds where randomness is used.
- **FR-010**: The system MUST generate reviewable ROI sample outputs for crop
  quality inspection under `outputs/figures/roi_samples/`.
- **FR-011**: The system MUST keep generated ROI samples and private-derived
  preprocessing artifacts ignored and untracked by default.
- **FR-012**: The system MUST be testable with synthetic images and synthetic
  annotation metadata without requiring private Krones dataset access.
- **FR-013**: The system MUST report readable errors or fallback reasons for
  invalid image inputs, invalid ROI metadata, and unsafe crop conditions.
- **FR-014**: The system MUST NOT perform classifier training, detector export,
  Kaggle submission generation, or test-label inference in SPEC-004.

### Constitution Alignment *(mandatory)*

- **Binary Output**: This feature does not produce bottle decisions, but it
  preserves the later `0 = Reusable` and `1 = Not Reusable` classifier path by
  preparing consistent image inputs.
- **Accuracy/Speed**: ROI-focused crops reduce irrelevant image area for later
  classifier speed and accuracy, while fallback behavior avoids failed inference.
- **ROI/Hybrid Flow**: This feature owns ROI preprocessing and internal
  dark-defect preservation; classifier, detector fallback, and fusion logic
  remain later specs.
- **Annotation/Version Strategy**: This is `SPEC-004: ROI Cropping and
  Preprocessing`, part of the V1 working pipeline after reusable COCO parsing.
- **Reproducibility/Leakage**: Validation and test preprocessing must be
  deterministic and must not infer or introduce test labels.
- **Confidentiality**: Private images and private-derived crop samples remain
  local/private and ignored unless explicitly approved for private team sharing.
- **Explainability**: Consistent ROI crops support later Grad-CAM and detector
  visualizations by preserving bottle evidence.
- **Bias/Imbalance**: Crop review must preserve rare and dark defect evidence so
  preprocessing does not erase minority defect patterns.

### Key Entities *(include if feature involves data)*

- **ROI Crop Request**: An image plus optional annotation-derived region metadata
  and target output settings.
- **ROI Crop Result**: A cropped or resized image with crop method, coordinates
  when available, output size, and fallback reason when used.
- **Preprocessing Profile**: Split-specific preprocessing rules for training,
  validation, and test behavior.
- **ROI Sample Report**: Generated review samples and metadata used to inspect
  crop quality and confidentiality status.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Synthetic images with valid ROI metadata produce crops at the
  default `384x384` output dimensions in 100% of covered test cases.
- **SC-002**: Synthetic images without valid ROI metadata use a documented
  fallback crop or full-image resize in 100% of covered test cases.
- **SC-003**: Validation and test preprocessing produce identical outputs across
  repeated runs for the same input and configuration.
- **SC-004**: Training preprocessing stays within configured augmentation bounds
  while preserving output dimensions for 100% of covered test cases.
- **SC-005**: Synthetic crop samples demonstrate that internal dark regions in
  the bottle area are preserved.
- **SC-006**: ROI sample outputs are generated under approved output locations
  and remain ignored and untracked by source control.
- **SC-007**: Automated tests run successfully using only synthetic images and
  synthetic annotation metadata.

## Assumptions

- SPEC-004 follows `.specify/memory/constitution.md` and the approved future
  specification division in `docs/krones-final-implementation-plan.md`.
- SPEC-003 already provides reusable annotation records and ROI availability
  metadata for training images when annotation data exists.
- Test images do not have trusted labels or annotation ROI metadata, so they use
  deterministic safe fallback preprocessing.
- Actual classifier training, threshold tuning, detector conversion, and
  submission generation remain out of scope for SPEC-004.
- Synthetic fixtures are sufficient for automated preprocessing tests; private
  dataset access is optional for manual visual validation only.
