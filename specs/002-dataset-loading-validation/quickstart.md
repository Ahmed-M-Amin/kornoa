# Quickstart: Dataset Loading and Validation

## Goal

Verify that the configured Krones dataset can be discovered, validated, and
audited without hardcoded private paths or public dataset leakage.

## Prerequisites

- SPEC-001 foundation files are present.
- `configs/paths.yaml` contains a valid private dataset root for the active
  environment.
- Private Krones dataset access is available locally, on Kaggle, or on Colab.

## Steps

1. Confirm the active feature:

   ```powershell
   git branch --show-current
   ```

   Expected: `002-dataset-loading-validation`

2. Run automated tests with synthetic fixtures:

   ```powershell
   pytest tests/test_dataset.py tests/test_dataset_audit.py -q
   ```

   Expected: all tests pass without private dataset access.

3. Run the dataset audit against the configured private dataset only when valid
   private dataset access is available. If private access is unavailable, keep
   the validation limited to the synthetic fixture tests from step 2.

   ```powershell
   python -m src.data.audit --config configs/paths.yaml
   ```

   Expected: required files are checked, train/test images are counted,
   annotations are summarized, audit outputs are generated, and runtime is
   recorded with whether it is under the five-minute target.

4. Inspect generated audit outputs:

   ```powershell
   Get-ChildItem -Force outputs/reports
   Get-ChildItem -Force outputs/figures
   ```

   Expected: dataset summary and distribution artifacts exist in ignored output
   locations.

5. Confirm private/generated artifacts are not tracked:

   ```powershell
   git ls-files
   git status --short --ignored
   ```

   Expected: private dataset files and generated audit artifacts are not tracked
   for public source control. Files generated under `outputs/reports/` and
   `outputs/figures/` appear ignored unless explicitly approved for private team
   sharing.

## Success

The feature is ready when dataset presence, label matching, image discovery,
audit-only annotation coverage, class/defect distribution, ROI availability, and
image-size summaries are produced with readable diagnostics and confidentiality
preserved.
