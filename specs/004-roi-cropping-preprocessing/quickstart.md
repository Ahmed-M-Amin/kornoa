# Quickstart: ROI Cropping and Preprocessing

## Automated Validation

Run the SPEC-004 test target once tasks are implemented:

```powershell
pytest tests/test_roi.py -q
```

Expected behavior:

- Synthetic annotation ROI crops produce `384x384` outputs.
- Invalid or missing ROI metadata uses square center crop or full-image resize.
- Internal dark synthetic regions are preserved.
- Validation/test preprocessing is deterministic across repeated runs.
- ROI crop and deterministic preprocessed validation/test samples are generated
  under `outputs/figures/roi_samples/` and ignored.

## Manual Synthetic Smoke Check

Use synthetic fixtures only:

```powershell
python -m src.data.roi --dataset-root tests/fixtures/synthetic_dataset --output-dir outputs/figures/roi_samples
```

The command should create reviewable ROI samples without requiring private
Krones dataset access.

## Private Dataset Validation

When valid private dataset access is available, run the same behavior against
the configured private dataset root. Keep all generated samples under ignored
output directories and do not commit private-derived images.

## Scope Guard

SPEC-004 validation should not train a classifier, export detector data,
generate Kaggle submissions, infer test labels, or run dashboard code.
