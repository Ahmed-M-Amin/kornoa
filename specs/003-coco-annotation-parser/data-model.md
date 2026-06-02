# Data Model: COCO Annotation Parser

## Annotation Source

Represents the configured COCO-style annotation file.

**Fields**

- `path`: filesystem path to the annotation JSON file
- `exists`: whether the file exists
- `parse_status`: `valid`, `missing`, or `malformed`
- `collection_status`: availability of `images`, `annotations`, and
  `categories`

**Validation Rules**

- The source path must be provided by configuration or caller input.
- Missing files must produce readable diagnostics.
- Malformed JSON must produce readable diagnostics without exposing private
  file contents.
- Required collections are `images`, `annotations`, and `categories`.

## Normalized COCO Image Record

Represents one image entry from COCO metadata.

**Fields**

- `image_id`: stable image identifier
- `file_name`: image filename from annotation metadata
- `width`: image width when available
- `height`: image height when available
- `annotation_count`: number of linked annotations
- `has_annotations`: whether one or more annotations reference the image
- `duplicate`: whether the image identifier appears more than once

**Validation Rules**

- Image identifiers must be tracked for duplicate detection.
- Images without annotations must remain valid but appear in coverage summaries.
- Width and height may be missing, but available values must be preserved.

## Normalized COCO Category Record

Represents one category entry from COCO metadata.

**Fields**

- `category_id`: stable category identifier
- `name`: readable category or defect label when available
- `annotation_count`: number of valid annotations linked to this category
- `duplicate`: whether the category identifier appears more than once

**Validation Rules**

- Category identifiers must support image annotation relationship validation.
- Missing or empty category names must not prevent relationship validation.
- Duplicate category identifiers must be reported.

## Normalized COCO Annotation Record

Represents one annotation entry from COCO metadata after validation.

**Fields**

- `annotation_id`: annotation identifier when available
- `image_id`: referenced image identifier
- `category_id`: referenced category identifier
- `category_name`: readable category name when available
- `bbox`: bounding box values when present
- `has_bbox`: whether a bounding box is present
- `bbox_valid`: whether the bounding box has valid dimensions and values
- `has_segmentation`: whether segmentation metadata is present
- `has_roi`: whether bounding-box or segmentation evidence is available
- `orphaned`: whether the image reference is unknown
- `unknown_category`: whether the category reference is unknown
- `duplicate`: whether the annotation identifier appears more than once
- `validation_errors`: structured validation issue names for this record

**Validation Rules**

- Orphaned annotations are reported but do not become training labels.
- Unknown category references are reported without stopping valid summaries.
- Bounding boxes with missing values, negative coordinates, or non-positive
  width or height are malformed.
- Segmentation availability is recorded but not converted into masks for this
  spec.
- Detector-format export is out of scope.

## Annotation Validation Summary

Represents aggregate parser and validation results.

**Fields**

- `image_count`
- `annotation_count`
- `category_count`
- `annotated_image_count`
- `unannotated_image_count`
- `orphaned_annotation_count`
- `unknown_category_count`
- `malformed_bbox_count`
- `duplicate_image_ids`
- `duplicate_annotation_ids`
- `duplicate_category_ids`
- `category_distribution`
- `roi_available_count`
- `segmentation_available_count`
- `diagnostics`

**Validation Rules**

- Coverage summaries must account for 100% of image records.
- Relationship summaries must account for 100% of annotation records.
- Diagnostics must identify failing items by type and identifier.
- Private annotation contents must not be emitted in diagnostics.
