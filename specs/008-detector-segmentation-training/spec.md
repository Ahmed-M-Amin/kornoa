# Feature Specification: Detector / Segmentation Training

**Feature Branch**: `008-detector-segmentation-training`

**Created**: 2026-06-05

**Status**: Draft

**Input**: User description: "Create SPEC-008: Detector / Segmentation Training for the Krones bottle inspection project. Use implementation.md as the authoritative master plan. Build the detector/segmentation foundation from train_annotations.json, preserve the current V2B EfficientNet-B1 analysis_only classifier as the accepted fast model, and do not implement SPEC-009 hybrid inference."

## Clarifications

### Session 2026-06-05

- Q: Should images with no defect annotations be included in the detector dataset? -> A: Include no-annotation training images with empty YOLO label files and report their count.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Audit Defect Annotations (Priority: P1)

The practitioner can load `train_annotations.json`, inspect its COCO-style image, annotation, and category records, and produce a defect-audit report that explains what detector training data is available.

**Why this priority**: SPEC-008 starts the defect-aware path. Before any detector training is useful, the project must prove that defect annotations are readable, mapped to real training images, and safe to use without leaking test images.

**Independent Test**: Can be tested by loading a synthetic COCO annotation file, validating image/category/annotation relationships, and checking the generated audit report for counts, distributions, missing references, invalid annotations, and multi-defect images.

**Acceptance Scenarios**:

1. **Given** a COCO-style annotation file with images, annotations, and categories, **When** the audit runs, **Then** the report includes image count, annotation count, category distribution, images with no annotations, images with multiple defects, missing image references, and invalid boxes or masks.
2. **Given** annotations that reference missing images or invalid boxes, **When** the audit runs, **Then** those records are reported and excluded from detector label conversion rather than crashing the workflow.
3. **Given** categories in the annotation file, **When** the category mapping report is generated, **Then** every category has a stable detector class index and readable defect name.

---

### User Story 2 - Generate Detector Dataset (Priority: P2)

The practitioner can convert valid COCO annotations into a deterministic detector dataset with disjoint train and validation splits, YOLO-format labels, and a valid dataset configuration file.

**Why this priority**: Detector training and evaluation require a clean dataset package. Split safety and label correctness are prerequisites for trustworthy mAP, precision, recall, and later hybrid inference experiments.

**Independent Test**: Can be tested by converting synthetic COCO annotations into `images/train`, `images/val`, `labels/train`, `labels/val`, and `data.yaml`, then verifying split disjointness and normalized label contents.

**Acceptance Scenarios**:

1. **Given** valid annotated training images, **When** conversion runs with a fixed seed, **Then** the same train/validation split is produced every time and no image appears in both splits.
2. **Given** valid bounding boxes, **When** YOLO labels are generated, **Then** each line contains a mapped class index and normalized center, width, and height values in the valid range.
3. **Given** detector dataset output is generated, **When** `data.yaml` is inspected, **Then** it points to the detector dataset folders and lists all defect categories in class-index order.

---

### User Story 3 - Visualize and Train Detector Candidates (Priority: P3)

The practitioner can save sample visualizations with category-colored defect boxes or masks, then run documented detector training and evaluation commands for lightweight YOLO candidates without changing the fast classifier inference path.

**Why this priority**: Visual samples are the quickest way to catch annotation conversion mistakes, and documented training/evaluation commands make SPEC-008 reproducible while keeping detector runtime out of default inference.

**Independent Test**: Can be tested by creating sample overlay figures from synthetic images and verifying that training/evaluation commands are documented and write reports under detector output paths without being invoked automatically by tests.

**Acceptance Scenarios**:

1. **Given** converted detector labels and source images, **When** visualization runs, **Then** sample images with category-colored boxes are saved under `outputs/detector/figures/`.
2. **Given** segmentation data exists, **When** visualization runs, **Then** mask overlays are saved; if masks do not exist, the report explicitly states that mask visualization is unavailable.
3. **Given** a detector candidate configuration, **When** the documented training or evaluation command is run manually, **Then** outputs are written under `outputs/detector/models/` and `outputs/detector/reports/` without modifying V2B classifier artifacts.

### Edge Cases

- `train_annotations.json` is missing, malformed, or missing required COCO collections.
- COCO images reference files that are not present under training images.
- Annotations reference unknown categories or image IDs.
- Bounding boxes have negative coordinates, zero or negative dimensions, or extend outside image bounds.
- Segmentation masks are absent, empty, malformed, or present only for some categories.
- Some images have no annotations, and some images have multiple defects.
- Images with no defect annotations are included in detector train/validation conversion with empty YOLO label files and counted explicitly in reports.
- Defect categories are too sparse for practical stratification; deterministic fallback splitting is required.
- Test images, sample-submission rows, public leaderboard feedback, and classifier hard-example memory must never enter detector training or validation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load `train_annotations.json` and support COCO-style `images`, `annotations`, and `categories` collections.
- **FR-002**: System MUST parse bounding boxes, segmentation masks when available, area fields when available, and ROI-related metadata when available.
- **FR-003**: System MUST map COCO image IDs to training image files and report missing image references.
- **FR-004**: System MUST generate a defect-category mapping report with stable detector class indices.
- **FR-005**: System MUST generate an annotation audit report covering annotation counts, category distribution, bounding-box size distribution, images without annotations, images with multiple defects, missing references, invalid boxes, and invalid masks.
- **FR-006**: System MUST safely skip invalid detector annotations while preserving them in audit diagnostics.
- **FR-007**: System MUST create detector dataset folders under `outputs/detector/dataset/`: `images/train`, `images/val`, `labels/train`, and `labels/val`.
- **FR-008**: System MUST create YOLO-format labels for valid bounding-box annotations.
- **FR-008a**: System MUST include no-annotation training images in detector train/validation conversion with empty YOLO label files and report their count.
- **FR-009**: System MUST create a valid `data.yaml` listing detector dataset paths and defect class names.
- **FR-010**: System MUST use deterministic train/validation splitting with a saved seed.
- **FR-011**: System MUST keep detector train and validation image IDs disjoint.
- **FR-012**: System SHOULD stratify detector splits by binary label or defect category when practical; if not practical, it MUST report the fallback.
- **FR-013**: System MUST save detector audit and split reports under `outputs/detector/reports/`.
- **FR-014**: System MUST save sample visualizations with category-colored boxes under `outputs/detector/figures/`.
- **FR-015**: System MUST save mask visualizations when segmentation masks exist and report unavailable masks when they do not.
- **FR-016**: System MUST document lightweight detector training command support for YOLOv8n or YOLO11n.
- **FR-017**: System MUST document detector evaluation output fields, including mAP, precision, recall, per-category metrics, and validation prediction examples when available.
- **FR-018**: System MUST save detector models under `outputs/detector/models/` and detector reports under `outputs/detector/reports/`.
- **FR-019**: System MUST NOT implement SPEC-009 hybrid inference, classifier-detector fusion, detector fallback routing, dashboard, Grad-CAM, feature memory bank, ensemble, or distillation in SPEC-008.
- **FR-020**: System MUST NOT modify the accepted V2B EfficientNet-B1 `analysis_only` classifier or its accepted public score record.
- **FR-021**: System MUST NOT use test images, sample submission rows, Kaggle public feedback, threshold tuning data, hard-example mining outputs, or pseudo-labels for detector training or validation.
- **FR-022**: System MUST keep the default fast classifier inference path unchanged: image, preprocessing/ROI, single classifier, threshold, target.

### Constitution Alignment *(mandatory)*

- **Binary Output**: SPEC-008 preserves `0 = Reusable` and `1 = Not Reusable`; detector classes explain defects and do not replace the binary target.
- **Accuracy/Speed**: Detector artifacts support future quality improvements, but the detector is not intended to run for every image by default and does not change current fast inference.
- **ROI/Hybrid Flow**: SPEC-008 prepares detector/segmentation outputs for later conditional use but does not implement hybrid fusion.
- **Annotation/Version Strategy**: This is the SPEC-008 detector/segmentation checkpoint using COCO annotations from `train_annotations.json`.
- **Reproducibility/Leakage**: Detector split seed, dataset conversion reports, audit reports, and output paths are saved; test images and public feedback are excluded.
- **Confidentiality**: Private annotations, images, labels, visualizations, models, and reports remain under local or competition-safe ignored output paths.
- **Explainability**: Detector boxes and masks provide defect localization and category evidence for later reporting.
- **Bias/Imbalance**: Category distributions, sparse categories, no-annotation images, and multi-defect cases are reported before training.

### Key Entities *(include if feature involves data)*

- **COCO Annotation Set**: The source `train_annotations.json` collections and relationship mappings.
- **Defect Category Map**: Stable mapping from COCO category IDs and names to detector class indices.
- **Detector Annotation**: A valid image/category/bounding-box or mask record eligible for detector conversion.
- **Detector Split**: Deterministic train/validation image assignment with disjointness and stratification metadata.
- **YOLO Dataset**: Converted detector dataset folders, label files, and `data.yaml`.
- **Detector Audit Report**: Counts, distributions, invalid records, missing references, and visualization availability.
- **Detector Candidate Result**: Training or evaluation report for a lightweight detector candidate.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `train_annotations.json` loads successfully and reports 100% of required COCO collections.
- **SC-002**: 100% of valid COCO image IDs used for detector conversion map to existing training image files.
- **SC-003**: 100% of detector train and validation image IDs are disjoint.
- **SC-004**: 100% of valid bounding-box annotations are converted to normalized detector labels, and 100% of invalid annotations are reported and skipped.
- **SC-004a**: 100% of included no-annotation images receive empty detector label files and are counted in conversion reports.
- **SC-005**: `data.yaml` is generated and lists all detector category names in class-index order.
- **SC-006**: Detector audit reports include annotation count, category distribution, bbox size distribution, no-annotation image count, multi-defect image count, missing image references, and invalid annotation counts.
- **SC-007**: At least one sample visualization with category-colored boxes is saved when valid annotated images exist.
- **SC-008**: Detector training and evaluation commands are documented and runnable manually without automatic training during tests.
- **SC-009**: Existing SPEC-005, SPEC-006, and SPEC-007 regression tests continue to pass.

## Assumptions

- `docs/implementation.md` is the authoritative master plan because `docs/krones_final_implementation_plan_complete.md` is not present in the repository.
- Current accepted fast classifier remains V2B EfficientNet-B1 `analysis_only` with Kaggle public F1 `0.92121`.
- V2A-remake, V2C EfficientNet-B2, and V2B-HE oversampling are rejected classifier-only paths for model selection.
- SPEC-008 may add detector/segmentation data preparation, reporting, visualization, and command wrappers, but SPEC-009 owns hybrid inference and fusion.
- Segmentation masks may be absent or incomplete; bounding-box detector conversion is still required.
- Detector training is documented and manually runnable but not executed automatically in tests.
