# Contract: COCO Annotation Parser

## Purpose

Define the observable behavior for SPEC-003 COCO annotation parsing. This is a
parser and validation contract. It does not define detector-format export, ROI
cropping, model training, or dashboard behavior.

## Inputs

- Configured or caller-provided path to a COCO-style JSON annotation file
- Synthetic COCO fixture for automated tests
- Optional private Krones annotation file for local/private validation only

## Required Collections

The annotation file must contain:

- `images`
- `annotations`
- `categories`

Missing collections must produce readable validation diagnostics.

## Required Parser Results

The parser must return:

- normalized image records
- normalized annotation records
- normalized category records
- annotation validation summary
- structured diagnostics

## Required Validation Checks

- Missing annotation file
- Malformed JSON
- Missing required collections
- Duplicate image identifiers
- Duplicate annotation identifiers
- Duplicate category identifiers
- Image-to-annotation mapping
- Orphaned annotations that reference unknown images
- Annotations that reference unknown categories
- Bounding boxes with missing values, negative coordinates, or non-positive
  width or height
- Bounding-box availability
- Segmentation availability
- ROI availability from bounding-box or segmentation metadata
- Category or defect-label distribution

## Required Summaries

- image count
- annotation count
- category count
- annotated image count
- unannotated image count
- orphaned annotation count
- unknown category count
- malformed bounding-box count
- category distribution
- ROI availability count
- segmentation availability count

## Error Behavior

- Missing file errors must name the missing annotation path.
- Malformed JSON errors must identify the annotation file without exposing
  private annotation contents.
- Relationship errors must identify failing item identifiers.
- Valid records must remain available when unrelated invalid records exist.

## Confidentiality Rules

- Private annotation files must not be committed.
- Diagnostics from private annotations must remain local/private unless
  explicitly approved for private team sharing.
- Parser tests must not require private Krones images, annotations, or labels.

## Out of Scope

- Detector-format annotation export
- ROI crop generation
- Segmentation mask rasterization
- Model training
- Test-label inference or manual test labeling
