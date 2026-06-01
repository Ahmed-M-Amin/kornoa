# Krones Industrial Bottle Inspection AI

A computer vision system for automated bottle inspection using hybrid classifier-detector architecture with explainable AI capabilities.

## Project Governance

This project follows the principles defined in [`.specify/memory/constitution.md`](.specify/memory/constitution.md).

## Specification Roadmap

This project is organized into modular specifications following GitHub Spec Kit best practices:

- **SPEC-001: Project Foundation** - Repository structure, configuration, and dependency management (this spec)
- **SPEC-002: Dataset Audit** - Data quality analysis and class distribution investigation
- **SPEC-003: COCO Parser** - Annotation parsing and dataset loading utilities
- **SPEC-004: ROI Processing** - Region-of-interest cropping and preprocessing
- **SPEC-005: Classifier Training** - Binary classifier model training pipeline
- **SPEC-006: Detector Training** - Object detection / segmentation model training
- **SPEC-007: Hybrid Inference** - Classifier-first inference with detector fallback
- **SPEC-008: Memory Bank** - Feature memory bank for uncertain cases
- **SPEC-009: Threshold Search** - Optimal decision threshold optimization
- **SPEC-010: Explainability** - Grad-CAM and visualization components
- **SPEC-011: Benchmark Measurement** - Inference speed and resource benchmarking
- **SPEC-012: Kaggle Submission** - Submission generation and formatting
- **SPEC-013: Streamlit Dashboard** - Interactive inference and results dashboard
- **SPEC-014: Final Report** - Experiment reporting and documentation generation

## Project Structure

```
krones-vision-ai/
configs/          - Configuration files (paths, model settings)
notebooks/        - Jupyter notebooks for analysis and training
src/              - Source code modules (data, models, training, inference, etc.)
tests/            - Test suite for all modules
outputs/          - Generated artifacts (models, predictions, figures, reports)
```

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

## Development

This project is designed for local Windows development, Kaggle notebooks, and Google Colab execution.
