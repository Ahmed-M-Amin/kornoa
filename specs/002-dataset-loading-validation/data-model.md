# Data Model: Dataset Loading and Validation

## Dataset Root

Represents the configured private competition dataset location.

**Fields**

- `root_path`: configured filesystem path
- `train_csv_path`: path to `train.csv`
- `train_images_dir`: path to `train_images/`
- `test_images_dir`: path to `test_images/`
- `sample_submission_path`: path to `sample_submission.csv`
- `annotations_path`: path to `train_annotations.json`

**Validation Rules**

- Root path must be configured before audit execution.
- Required files and folders must exist.
- Errors must name each missing item.

## Training Label Record

Represents a row from `train.csv`.

**Fields**

- `image_id`: stable image identifier or filename
- `target`: binary target where `0 = Reusable` and `1 = Not Reusable`
- `image_path`: resolved training image path
- `exists`: whether the referenced image file exists

**Validation Rules**

- Target must be one of `0` or `1`.
- Every row must be counted as matched or missing.
- Test labels must not be created or inferred.

## Image Asset

Represents a discovered train or test image.

**Fields**

- `image_id`: stable image identifier or filename
- `split`: `train` or `test`
- `path`: filesystem path
- `width`: image width when readable
- `height`: image height when readable
- `referenced_by_metadata`: whether metadata references the file

**Validation Rules**

- Images present without metadata references must be reported separately.
- Unreadable image metadata must produce a readable diagnostic.

## Audit Annotation Record

Represents audit-only metadata extracted from COCO-style annotations.

**Fields**

- `annotation_id`: annotation identifier when available
- `image_id`: linked image identifier
- `category_id`: category identifier when available
- `category_name`: defect/category label when available
- `has_bbox`: whether a bounding box is present
- `has_segmentation`: whether segmentation metadata is present
- `has_roi`: whether ROI or region information is available
- `orphaned`: whether the annotation references an unknown image

**Validation Rules**

- Annotation coverage must be summarized by image.
- Defect/category distribution must be summarized when labels exist.
- ROI availability must be summarized when region information exists.
- Reusable parser APIs are out of scope for this feature.

## Dataset Audit Report

Represents generated tabular audit outputs.

**Fields**

- `dataset_summary`: counts of required files, train/test images, labels,
  missing files, orphaned annotations, ROI availability, and image sizes
- `class_distribution`: binary target distribution
- `defect_distribution`: defect/category distribution when available
- `missing_files`: identifiers for missing referenced files
- `orphaned_annotations`: identifiers for annotations linked to unknown images

**Validation Rules**

- Reports must be written to approved generated-output locations.
- Reports must not require committing private data or generated artifacts.

## Audit Figure

Represents generated visual audit outputs.

**Fields**

- `class_distribution_figure`
- `defect_distribution_figure`
- `sample_grid_figure`

**Validation Rules**

- Figures must be written to approved generated-output locations.
- Sample visualizations are private-team artifacts unless explicitly approved for
  release.
