# Data Model: Detector / Segmentation Training

## CocoAnnotationSet

Represents the parsed `train_annotations.json` source.

**Fields**:
- `images`: COCO image records with ID, file name, width, height, and optional metadata.
- `annotations`: COCO annotation records with ID, image ID, category ID, bbox, segmentation, area, and optional metadata.
- `categories`: COCO category records with ID, name, and optional supercategory.
- `image_to_annotations`: Mapping from image ID to annotations.
- `category_to_annotations`: Mapping from category ID to annotations.

**Validation Rules**:
- Required collections must exist and be lists.
- Annotation image IDs must map to known images and existing training image files before detector conversion.
- Annotation category IDs must map to known categories.
- Invalid bbox or mask records are reported and skipped for detector label generation.

## DefectCategoryMap

Represents stable detector class assignment.

**Fields**:
- `category_id`: Original COCO category ID.
- `category_name`: Human-readable defect category.
- `class_index`: Zero-based detector class index.
- `annotation_count`: Number of annotations in the category.

**Validation Rules**:
- Class indices are deterministic and contiguous from `0`.
- Category names are saved in class-index order for `data.yaml`.
- Unknown categories are diagnostics, not label classes.

## DetectorAnnotation

Represents one valid detector training target.

**Fields**:
- `image_id`
- `image_path`
- `category_id`
- `class_index`
- `bbox_xywh`
- `normalized_yolo_bbox`
- `segmentation`
- `area`

**Validation Rules**:
- Bounding box width and height must be positive.
- Bounding box coordinates must be clipped or rejected according to audit policy; invalid boxes are skipped and reported.
- Normalized YOLO values must be in `[0.0, 1.0]`.
- Test images are never eligible.
- Images with no defect annotations are eligible detector examples and receive empty YOLO label files.

## DetectorSplit

Represents deterministic train/validation assignment.

**Fields**:
- `seed`
- `train_image_ids`
- `validation_image_ids`
- `stratification_key`
- `stratification_fallback_reason`
- `split_strategy_used`
- `train_validation_disjoint`

**Validation Rules**:
- Train and validation image IDs must be disjoint.
- Split must be reproducible for the same seed and inputs.
- Primary-defect-category stratification includes `no_annotation` and is used when practical; deterministic fallback is reported when categories are too sparse.

## YoloDataset

Represents converted detector dataset artifacts.

**Fields**:
- `dataset_root`
- `images_train_dir`
- `images_val_dir`
- `labels_train_dir`
- `labels_val_dir`
- `data_yaml_path`
- `class_names`
- `label_files`
- `no_annotation_image_count`

**Validation Rules**:
- Output folders must exist before conversion completes.
- Every copied or linked image has a matching label file, even if empty for no-annotation images when included.
- `data.yaml` must include train path, validation path, class count, and names.
- The conversion report must count included no-annotation images.
- Repeated conversion must clean only generated detector split folders and report `output_dataset_cleaned=true` plus `cleaned_paths`.

## DetectorAuditReport

Represents audit and conversion diagnostics.

**Fields**:
- `annotation_count`
- `category_distribution`
- `bbox_size_distribution`
- `images_without_annotations`
- `included_no_annotation_images`
- `images_with_multiple_defects`
- `missing_image_references`
- `invalid_boxes`
- `invalid_masks`
- `segmentation_available_count`
- `visualization_paths`

**Validation Rules**:
- Reports must be saved under `outputs/detector/reports/`.
- Invalid records must be traceable by image ID and annotation ID.
- Visualization availability must state whether masks existed.

## DetectorCandidateResult

Represents a manual detector training or evaluation result.

**Fields**:
- `model_name`
- `image_size`
- `dataset_yaml`
- `metrics`
- `map`
- `precision`
- `recall`
- `per_category_metrics`
- `validation_prediction_examples`
- `model_artifact_path`

**Validation Rules**:
- Missing optional YOLO metrics are marked unavailable, not fabricated.
- Results must not modify V2 classifier artifacts.
