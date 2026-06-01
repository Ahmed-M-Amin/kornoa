# Quickstart: COCO Annotation Parser

## Prerequisites

- Project dependencies from `requirements.txt` are installed.
- Synthetic annotation fixture exists at
  `tests/fixtures/synthetic_dataset/train_annotations.json`.
- No private Krones dataset access is required for automated tests.

## Validation Flow

1. Run parser tests:

   ```powershell
   pytest tests/test_coco_parser.py -q
   ```

   Expected: all parser tests pass using only synthetic fixtures.

2. Confirm parser behavior covers the required validation categories:

   - valid image, annotation, and category parsing
   - missing required collection diagnostics
   - malformed JSON diagnostics
   - duplicate identifier diagnostics
   - orphaned annotation diagnostics
   - unknown category diagnostics
   - malformed bounding-box diagnostics
   - ROI availability summary
   - segmentation availability summary
   - category distribution summary

3. Optional private validation:

   ```powershell
   python -m src.data.coco_parser --annotations configs/paths.yaml
   ```

   Run this only when valid private dataset access is available and the command
   has been implemented for local/private validation. Generated diagnostics must
   remain ignored or private.

4. Confirm source-control safety if diagnostics are generated:

   ```powershell
   git ls-files
   git status --short --ignored
   ```

   Expected: private annotation files and generated diagnostics are not tracked
   for public source control.

## Notes

- SPEC-003 returns normalized records and validation summaries.
- SPEC-003 does not export detector-ready annotations.
- SPEC-003 does not crop images or implement ROI preprocessing.
- SPEC-004 owns ROI cropping and preprocessing behavior.
