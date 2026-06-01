# Data Model: Project Foundation

## Project Foundation

Represents the safe repository structure required before later features begin.

**Fields**

- `root_files`: `.gitignore`, `README.md`, `requirements.txt`, `AGENTS.md`
- `config_files`: `paths.yaml`, `classifier.yaml`, `detector.yaml`,
  `inference.yaml`, `dashboard.yaml`
- `source_packages`: `data`, `models`, `training`, `inference`,
  `explainability`, `dashboard`, `utils`
- `test_files`: foundation smoke tests plus future test placeholders
- `output_categories`: `models`, `predictions`, `figures`, `reports`,
  `submissions`, `hard_examples`

**Validation Rules**

- Safe foundation files may be tracked.
- Private dataset files must not be tracked.
- Generated artifact contents must not be tracked.
- Each package directory must be importable or ready for import.

## Configuration Set

Represents environment-specific and component-level configuration templates.

**Fields**

- `paths`: local, Kaggle, Colab, active environment, and output roots
- `classifier`: model name, image size, batch size, threshold defaults
- `detector`: model family and confidence defaults
- `inference`: confidence gate, device, batch mode, benchmark flag
- `dashboard`: upload mode, folder mode, explanation display toggles

**Validation Rules**

- Dataset paths must be configurable.
- The local dataset path may appear only as a config example.
- Core modules must not require machine-specific path constants.

## Generated Output Area

Represents future locations for reproducible experiment and demo outputs.

**Fields**

- `models`: model weights and exported models
- `predictions`: validation and inference prediction CSV files
- `figures`: charts, Grad-CAM images, and visual samples
- `reports`: metrics and benchmark reports
- `submissions`: Kaggle submission files
- `hard_examples`: false-positive, false-negative, and uncertain-case records

**Validation Rules**

- Directories may be tracked with `.gitkeep`.
- Generated contents must be ignored by `.gitignore`.

## Private Dataset Boundary

Represents the confidentiality boundary for competition data.

**Fields**

- `dataset_root`: local/private dataset directory
- `annotation_files`: private COCO annotation and labels
- `image_dirs`: private train and test images
- `derived_private_artifacts`: generated outputs that may reveal data

**Validation Rules**

- Dataset files, annotations, labels, and derived private artifacts must remain
  outside public source control.
- Ignore rules must cover the known local dataset directory and common generated
  artifact categories.
