# Contract: Dataset Audit

## Purpose

Define the observable inputs and outputs for SPEC-002 dataset loading and
validation. This is a behavior contract, not a reusable COCO parser API.

## Inputs

- `configs/paths.yaml`
- private dataset root
- `train.csv`
- `train_images/`
- `test_images/`
- `sample_submission.csv`
- `train_annotations.json`

## Required Checks

- Required file/folder presence
- Training label row count
- Train image discovery
- Test image discovery
- Training label to image matching
- Present-but-unreferenced image reporting
- Audit-only annotation coverage
- Defect/category distribution when labels exist
- ROI availability when metadata exists
- Orphaned annotation reporting
- Image-size summary

## Outputs

Reports:

- `outputs/reports/dataset_summary.csv`
- `outputs/reports/missing_files.csv`
- `outputs/reports/orphaned_annotations.csv`
- `outputs/reports/class_distribution.csv`
- `outputs/reports/defect_distribution.csv`
- `outputs/reports/image_size_summary.csv`

Runtime:

- `outputs/reports/dataset_summary.csv` must include audit runtime information
  and whether the private dataset audit was under the five-minute target when
  valid private dataset access is available.

Figures:

- `outputs/figures/class_distribution.png`
- `outputs/figures/defect_distribution.png`
- `outputs/figures/sample_grid.png`

## Error Behavior

- Missing dataset root must produce a readable configuration error.
- Missing required files or folders must identify each missing item.
- Malformed metadata must identify the failing file.
- Missing referenced images must be reported without stopping unrelated summary
  generation.
- Orphaned annotations must be reported without becoming training labels.

## Confidentiality Rules

- Private dataset files must not be committed.
- Generated audit reports and figures must remain ignored unless explicitly
  approved for private team sharing.
- Files generated under `outputs/reports/` and `outputs/figures/` must be
  validated as ignored and not tracked.
- Test images may be counted and checked for submission readiness, but test
  labels must not be inferred or manually assigned.

## Out of Scope

- Reusable COCO parser APIs.
- ROI crop implementation.
- Model training or threshold tuning.
- Kaggle submission generation.
