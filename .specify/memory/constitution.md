<!--
Sync Impact Report
Version change: 1.0.0 -> 1.1.0
Modified principles:
- Competition-First Output -> Core Challenge Rules
- Accuracy and Speed Together -> Accuracy and Speed Together
- Hybrid Industrial Architecture -> Hybrid Industrial Architecture
- ROI-First Processing -> ROI-First Processing with dark-defect preservation
- Dataset Confidentiality -> Dataset Confidentiality
- No Data Leakage -> No Data Leakage
- Reproducibility -> Reproducibility
- Bias and Imbalance Handling -> Bias and Imbalance Handling
- Explainability Is Required -> Explainability Is Required
- Professional Code Quality -> Professional Code Quality
- Kaggle and Colab Compatibility -> Kaggle, Colab, and Local Compatibility
Added sections:
- Model Version Strategy
- Model Memory Policy
- Annotation Gate
Removed sections:
- Final Model Strategy as a standalone principle; merged into Accuracy and Speed Together,
  Model Version Strategy, and Model Memory Policy
Templates requiring updates:
- .specify/templates/plan-template.md: updated
- .specify/templates/spec-template.md: updated
- .specify/templates/tasks-template.md: updated
- .specify/templates/commands/*.md: not present
Follow-up TODOs: none
-->

# Krones Industrial Bottle Inspection AI Constitution

## Core Principles

### I. Core Challenge Rules
The system MUST classify every bottle image with the binary operational decision
`0 = Reusable` or `1 = Not Reusable`. Extra outputs such as confidence score,
defect reason, Grad-CAM heatmap, detector boxes or masks, inference time, and
accepted/rejected counters MAY be produced, but they MUST NOT replace or weaken
the required binary prediction.

Rationale: The Krones Vision AI Challenge requires a stable binary decision;
all other outputs exist to explain, benchmark, or improve that decision.

### II. Accuracy and Speed Together
The project MUST optimize F1-score and inference speed together. Every model
decision MUST consider F1-score, inference time, memory usage, Kaggle free-tier
limits, Colab free-tier limits, and dashboard responsiveness. Large slow models
MAY be used as research, ensemble, or teacher models, but the final model MUST
be fast enough for efficient inference.

Rationale: Industrial inspection is only useful when accuracy and runtime are
both practical.

### III. Hybrid Industrial Architecture
The default architecture MUST be hybrid: ROI crop, fast binary classifier,
confidence gate, detector or segmentation model for uncertain cases, fusion
decision logic, and explanation output. The detector MUST NOT run on every image
unless experiments prove the accuracy or explanation gain is worth the runtime
cost.

Rationale: Conditional deeper analysis preserves speed while supporting hard
cases and explainability.

### IV. ROI-First Processing
The project MUST use ROI-focused preprocessing. ROI processing MUST crop or
focus on the bottle inspection region, reduce useless outer camera border, and
keep internal bottle details. The project MUST NOT remove black pixels blindly:
outer dark camera border MAY be ignored or cropped, but dark regions inside the
bottle area MUST stay because they may represent contamination, foreign object,
liquid, no-base-visible region, or another real defect.

Rationale: ROI processing improves signal quality, but blind dark-pixel removal
can erase the defect evidence the model must learn.

### V. Dataset Confidentiality
The Krones dataset, images, annotations, labels, and derived private artifacts
MUST remain confidential. Uploading dataset images, `train_annotations.json`,
train/test images, private competition notebooks, or private model outputs to
public repositories or public services is forbidden unless the competition rules
explicitly allow it. Local development, Kaggle competition execution, private
team work, and code generation without uploading private data are allowed.

Rationale: Competition data and private derived artifacts must remain controlled.

### VI. No Data Leakage
The project MUST NOT leak test information into training or validation.
Forbidden practices include training on test images, manually labeling test
images, choosing thresholds only from Kaggle public score, accidentally mixing
validation images into training, augmenting validation data like training data,
and using test distribution knowledge unfairly. Validation MUST simulate unseen
production data.

Rationale: Leakage creates misleading metrics and weak real-world behavior.

### VII. Reproducibility
Every serious experiment MUST be reproducible from code and configuration.
Required records include fixed random seed, saved config file, saved model
weights, saved validation predictions, saved best threshold, saved metric report,
documented train/validation split, clear experiment name, and no hidden manual
notebook changes.

Rationale: Reproducible experiments are required for comparison, debugging, and
technical reporting.

### VIII. Bias and Imbalance Handling
The project MUST explicitly handle dataset bias and imbalance. Required analysis
includes class distribution, bottle-type distribution when available,
defect-label distribution, rare defect cases, false positives, false negatives,
and uncertain predictions. Mitigation methods such as weighted loss, focal loss,
weighted sampling, rare-class augmentation, hard-example mining, and F1-threshold
tuning MUST be considered when relevant.

Rationale: A model that learns the dominant class is not acceptable for defect
inspection.

### IX. Explainability Is Required
The final system MUST explain decisions. Required explanation outputs include
prediction, confidence score, bad probability, Grad-CAM heatmap, detector boxes
or masks when available, defect reason when possible, and inference time. The
Streamlit dashboard MUST show why a bottle was accepted or rejected.

Rationale: Inspection outputs must be understandable for review, presentation,
and error analysis.

### X. Professional Code Quality
The codebase MUST be modular, readable, maintainable, and testable. Core modules
MUST avoid hardcoded dataset paths, load paths from config files, separate
training from inference, separate dashboard code from model code, use reusable
utilities, emit readable errors, use clear file names, and test critical
modules. Large all-in-one notebooks, duplicated preprocessing logic, magic
numbers without config, silent failures, and unclear experiment outputs are
forbidden.

Rationale: The project must support reliable iteration across training,
inference, dashboarding, and reporting.

### XI. Kaggle, Colab, and Local Compatibility
The system MUST support local Windows development, Kaggle free GPU training, and
Google Colab free-tier execution. Dataset paths MUST be configurable. The known
local dataset path is
`E:/Bachelor Cyber Security/Semester 6/krones-vision-ai/korons/1st-krones-vision-ai-challenge`,
but code MUST NOT depend only on this path and MUST support Kaggle and Colab
paths through configuration.

Rationale: Development and evaluation will occur across local and hosted
notebook environments.

## Model Version Strategy

The project MUST be developed through controlled versions. Versions are
checkpoints that add one serious capability at a time; they are not separate
projects.

- V1 First Working Pipeline: dataset loading, annotation loading, ROI crop, fast
  classifier, validation F1-score, best-threshold search, test inference,
  `submission.csv`, and inference speed measurement. Default models are
  EfficientNet-B0 or MobileNetV3.
- V2 Strong Classifier: stronger augmentation, weighted loss, focal loss,
  threshold tuning, hard-example mining, and stronger model testing. Candidate
  models include EfficientNet-B1/B2 and ConvNeXt-Tiny.
- V3 Detector / Segmentation: YOLO detector or segmentation model, defect boxes
  or masks, defect category mapping, and conditional defect-area support.
  Candidate models include YOLOv8n, YOLO11n, and YOLOv8n-seg when masks are
  clean.
- V4 Hybrid Inference: classifier-first inference, confidence-based detector
  fallback, fusion logic, final decision, and explanation.
- V5 Explainability Dashboard: Streamlit dashboard, single-image test, folder
  workflow test, prediction table, confidence score, Grad-CAM, detector boxes or
  masks, and speed statistics.
- V6 Memory and Ensemble Research: ensemble teacher models, feature memory bank
  for uncertain cases, hard-example memory, false-positive memory, and
  false-negative memory.
- V7 Final Optimized Model: distilled or optimized fast model, optional ONNX
  export, final threshold, final benchmark, final submission notebook, and final
  report figures.

## Model Memory Policy

The project MAY use model memory only when it improves quality without harming
speed too much. Allowed memory types include hard-example memory, feature memory
bank, saved false-positive and false-negative cases, and saved uncertain cases.
The classifier MUST run first, memory-bank use MUST be limited to uncertain
cases by default, nearest-neighbor search MUST be optimized, feature vectors
SHOULD stay small, and the memory bank MUST be optional in config.

LSTM, GRU, video temporal memory, and sequence transformer approaches are not
recommended for early versions because the challenge is image-based and each
bottle image requires its own binary decision.

## Required Implementation Modules

The project MUST be implemented as separate, independently testable modules for
project configuration, dataset loading, COCO annotation parsing, ROI cropping,
preprocessing and augmentation, binary classifier training, detector or
segmentation training, threshold search, hybrid fusion inference, memory-bank
support, benchmark measurement, Kaggle submission generation, Grad-CAM
explainability, visualization tools, Streamlit dashboard, and experiment
reporting.

## Required Outputs

The system MUST produce trained classifier weights, detector or segmentation
weights when available, best-threshold JSON, validation predictions CSV,
validation metrics JSON, inference results CSV, `submission.csv`, speed
benchmark report, Grad-CAM explanation images, detector visualization images,
Streamlit dashboard, and report-ready figures.

## Quality Gates

A feature is not complete unless its applicable quality gate passes:

- Dataset Gate: `train.csv` loads correctly; train images are matched correctly;
  test images are detected; annotations load correctly; missing files are
  reported; class distribution is generated.
- Annotation Gate: COCO JSON loads correctly; category mapping works;
  image-to-annotation mapping works; bounding boxes are parsed; segmentation
  polygons are parsed when available; ROI information is extracted when
  available.
- ROI Gate: ROI crop works on training images; fallback crop works on test
  images; internal dark defects are preserved; crop output size is consistent;
  visual crop samples are saved.
- Classifier Gate: model trains without crashing; validation F1-score is
  calculated; best threshold is saved; model weights are saved; validation
  predictions are saved; inference time is measured.
- Detector Gate: annotations convert to detector format correctly; detector
  trains successfully; sample detections are visualized; defect labels are
  mapped correctly; conditional defect areas can be estimated when possible.
- Inference Gate: single-image inference works; folder inference works;
  prediction CSV is saved; confidence score is produced; inference time is
  measured; Kaggle submission is generated.
- Dashboard Gate: image upload works; folder workflow works; predictions are
  shown; confidence is shown; Grad-CAM is shown; boxes or masks are shown when
  available; inference speed is shown; result table is exportable.

## Future Specification Division

Future specs SHOULD follow this planned division unless a later amendment
changes the roadmap: SPEC-001 Project Foundation, SPEC-002 Dataset Loading and
Validation, SPEC-003 COCO Annotation Parser, SPEC-004 ROI Cropping and
Preprocessing, SPEC-005 Binary Classifier Training, SPEC-006 Detector or
Segmentation Training, SPEC-007 Hybrid Inference Engine, SPEC-008 Memory Bank
and Hard Example Mining, SPEC-009 Ensemble Teacher Models, SPEC-010 Distilled
Fast Student Model, SPEC-011 Grad-CAM Explainability, SPEC-012 Streamlit
Dashboard, SPEC-013 Kaggle Submission Generator, and SPEC-014 Experiment
Tracking and Report Assets.

## Agent Usage Rules

Codex, Cursor, OpenCode, GLM, and Kimi may be used. All agents MUST follow this
constitution and MUST NOT introduce features that violate confidentiality,
reproducibility, speed, modularity, or no-data-leakage rules. Recommended
responsibilities are Codex for architecture, planning, complex logic, and
training-loop review; Cursor for local repository editing, refactoring, and
debugging; OpenCode for full module implementation, tests, and CLI scripts; GLM
for explanations, documentation drafts, and error analysis; and Kimi for
long-context review, notebook comparison, experiment summaries, and report
writing.

## Governance

This constitution is the highest-level rule document for the project. When quick
coding, model experiments, dashboard features, agent suggestions, or future specs
conflict with this constitution, this constitution wins.

Amendments require a clear reason, semantic version update, ISO date update, and
review before implementation continues. Version changes follow semantic
versioning: MAJOR for incompatible governance or principle removals/redefinitions,
MINOR for new principles or materially expanded guidance, and PATCH for wording,
typo, or non-semantic refinements. All plans, specs, tasks, and reviews MUST
verify applicable constitution gates before implementation proceeds. No
implementation SHOULD continue if it violates this constitution.

**Version**: 1.1.0 | **Ratified**: 2026-05-30 | **Last Amended**: 2026-05-30
