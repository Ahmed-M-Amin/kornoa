# Research: Detector / Segmentation Training

## Decision: Use COCO annotations as the detector source of truth

**Rationale**: `train_annotations.json` is the project's defect-aware annotation source and contains the image, annotation, category, bounding-box, segmentation, area, and optional metadata needed for detector training. SPEC-008 must build from these annotations rather than classifier predictions or hard-example files.

**Alternatives considered**: Using classifier validation errors as detector labels was rejected because they are not localization annotations. Using test images or public leaderboard feedback was rejected as leakage.

## Decision: Convert valid bounding boxes to YOLO labels first

**Rationale**: Bounding-box detection is the fastest reliable detector foundation and is the first training path for SPEC-008. YOLO labels are compact, easy to audit, and compatible with lightweight YOLOv8n or YOLO11n bbox training commands.

**Alternatives considered**: Full segmentation-first and YOLO segmentation training were deferred because segmentation masks may be absent, sparse, or inconsistent. COCO-native training was rejected for this project stage because the requested deliverable is YOLO bbox dataset conversion.

## Decision: Include no-annotation images as empty-label background examples

**Rationale**: No-annotation training images are useful negative/background examples for detector training and are compatible with YOLO when paired with empty label files. Keeping them in train/validation splits also preserves visibility into clean or unannotated examples while requiring explicit reporting of their count.

**Alternatives considered**: Excluding no-annotation images was rejected because it removes background examples and can bias detector validation toward defect-only images. Including them only in validation was rejected because training also benefits from background examples.

## Decision: Preserve segmentation metadata and visualize masks when present

**Rationale**: Segmentation masks may provide better defect localization later. SPEC-008 should not discard mask data, and mask validation/visualization is supported when clean masks exist, but segmentation training remains conditional on mask quality and coverage.

**Alternatives considered**: Rasterizing every mask into training targets immediately was rejected because it adds complexity before confirming mask quality and coverage.

## Decision: Use deterministic disjoint detector splits

**Rationale**: Detector metrics must be reproducible and leakage-free. A saved seed, stable image ordering, and explicit train/validation image lists make conversion repeatable and auditable.

**Alternatives considered**: Random unsaved splits were rejected because they prevent reproducibility. Reusing classifier splits blindly was rejected because detector eligibility depends on annotation/image availability.

## Decision: Stratify by defect category when practical, otherwise use deterministic fallback

**Rationale**: Defect category imbalance affects detector learning and mAP interpretation. Category-aware splitting is preferred when each category has enough images. If categories are too sparse, deterministic fallback is safer and must be reported.

**Alternatives considered**: Forcing stratification for sparse categories was rejected because it can produce empty or invalid splits. Stratifying by binary label only was rejected as insufficient when defect categories are available.

## Decision: Keep detector outside default inference

**Rationale**: The accepted fast public model remains V2B EfficientNet-B1 `analysis_only` with public F1 `0.92121`. SPEC-008 produces detector artifacts and reports only; SPEC-009 will decide conditional detector use for uncertain classifier cases.

**Alternatives considered**: Running a detector for every image was rejected for SPEC-008 because it would violate the speed-first strategy and duplicate SPEC-009 hybrid inference scope.

## Decision: Document YOLOv8n and YOLO11n training/evaluation commands without automatic training in tests

**Rationale**: Lightweight YOLO candidates match the project speed strategy, while tests must remain fast and deterministic. Manual commands can be run locally, on Kaggle, or on Colab when GPU resources are available.

**Alternatives considered**: Training detectors inside automated tests was rejected because it is slow, non-deterministic across environments, and unnecessary for validating dataset conversion.
