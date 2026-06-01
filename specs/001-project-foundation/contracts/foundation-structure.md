# Contract: Project Foundation Structure

## Purpose

This contract defines the minimum safe filesystem structure that SPEC-001 must
provide for later Spec Kit features.

## Required Root Files

- `.gitignore`
- `README.md`
- `requirements.txt`
- `AGENTS.md`
- `.specify/memory/constitution.md`
- `.specify/feature.json`

`README.md` must reference `.specify/memory/constitution.md` and list the future
Spec Kit split from `SPEC-001` through `SPEC-014`.

## Required Config Files

- `configs/paths.yaml`
- `configs/classifier.yaml`
- `configs/detector.yaml`
- `configs/inference.yaml`
- `configs/dashboard.yaml`

## Required Source Package Directories

- `src/data/`
- `src/models/`
- `src/training/`
- `src/inference/`
- `src/explainability/`
- `src/dashboard/`
- `src/utils/`

Each source package directory must be safe to import in later Python work.

## Required Output Categories

- `outputs/models/`
- `outputs/predictions/`
- `outputs/figures/`
- `outputs/reports/`
- `outputs/submissions/`
- `outputs/hard_examples/`

Output directories may contain `.gitkeep` only. Generated contents must be
ignored.

## Required Test Files

- `tests/test_project_foundation.py`
- `tests/test_dataset.py`
- `tests/test_coco_parser.py`
- `tests/test_roi.py`
- `tests/test_inference.py`
- `tests/test_submission.py`

Only `tests/test_project_foundation.py` needs executable foundation checks in
SPEC-001. Other test files may be safe placeholders for later specs.

## Ignore Rule Contract

`.gitignore` must ignore:

- known private dataset directory: `1st-krones-vision-ai-challenge/`
- common private image/annotation data locations
- generated model weights and exported models
- generated predictions
- generated reports
- generated figures
- generated submissions
- generated hard-example artifacts
- Python caches and virtual environments

The ignore rules must allow `.gitkeep` files inside output directories.

Validation must check both tracked files with `git ls-files` and ignored working
tree files with `git status --short --ignored`.
