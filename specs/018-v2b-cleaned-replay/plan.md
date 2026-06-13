# Implementation Plan: Spec 018 V2B Cleaned Replay

**Branch**: `018-v2b-cleaned-replay` | **Date**: 2026-06-13 | **Spec**: `specs/018-v2b-cleaned-replay/spec.md`

**Input**: Feature specification from `specs/018-v2b-cleaned-replay/spec.md`

## Summary

Implement a controlled V2B-style cleaned-data replay that consumes only `outputs/analysis/data_quality_decision_lock/approved_cleaned_training_manifest.csv`, validates that no excluded, deferred, or adjudication rows enter training, runs through the existing classifier training stack, and writes a direct comparison against the locked V2B baseline. The plan extends `src/training/train_classifier.py` and adds a dedicated config and tests instead of creating a separate notebook or parallel trainer.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Windows and Kaggle training environment.

**Primary Dependencies**: Existing `torch`, `pandas`, `numpy`, `PyYAML`, `pytest`, `src.training.train_classifier`, `src.training.threshold_search`, `src.training.metrics`, and the Spec 017 decision-lock outputs.

**Storage**: Filesystem artifacts only. Inputs are CSV/JSON/checkpoint files; outputs are CSV, JSON, config snapshots, model checkpoints, and reports under `outputs/kaggle_v2b_cleaned_replay/`.

**Testing**: Pytest coverage in `tests/test_v2b_cleaned_replay_training.py`, plus compile check for `src/training/train_classifier.py`.

**Target Platform**: Local Windows first, with config-driven paths compatible with Kaggle and Colab execution.

**Project Type**: Single Python computer-vision training and evaluation project.

**Performance Goals**: Preserve V2B-style runtime practicality and report runtime metadata. The cleaned replay is only a candidate if validation F1 improves over the locked V2B baseline and all safety gates pass.

**Constraints**: Use only the Spec 017 approved manifest for training; exclude all `auto_exclude`, `needs_adjudication`, and `deferred_uncertain` rows; no test labels; no leaderboard tuning; no submission in dry-run validation; no original-label modification; no overwrite of locked V2B artifacts.

**Scale/Scope**: Current Spec 017 approved manifest has 35,211 rows. Current decision-lock outputs also include 20 auto-excluded rows and 111 needs-adjudication rows. Spec 018 validates these relationships and trains one controlled cleaned replay candidate.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The replay preserves `0 = Reusable` and `1 = Not Reusable`.
- Accuracy and speed: PASS. The replay is evaluated by validation F1 against the locked V2B baseline and records runtime evidence.
- Hybrid ROI-first architecture: PASS. The replay consumes Spec 017 cleaned rows after ROI/crop decisions; it does not alter ROI rules.
- Reproducibility and leakage control: PASS. The design requires dry-run validation, saved config, manifest snapshot, predictions, threshold, metrics, and no test-label or leaderboard use.
- Confidentiality: PASS. Dataset, model outputs, and derived reports remain inside private local/Kaggle/Colab paths.
- Explainability and reporting: PASS. Validation predictions, metrics, thresholds, and comparison reports are required.
- Bias and imbalance: PASS. Training and validation target counts are captured for cleaned-manifest comparison.
- Annotation and model-version strategy: PASS. This is a V2B-style classifier replay. Detector, segmentation, and annotation changes are out of scope.
- Modularity and compatibility: PASS. The implementation extends the existing classifier module and config structure.

## Project Structure

### Documentation (this feature)

```text
specs/018-v2b-cleaned-replay/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- v2b-cleaned-replay-contract.md
|-- checklists/
|   `-- requirements.md
`-- tasks.md
```

### Source Code (repository root)

```text
configs/
`-- v2b_cleaned_replay_training.yaml

src/
`-- training/
    `-- train_classifier.py

tests/
`-- test_v2b_cleaned_replay_training.py

outputs/
`-- kaggle_v2b_cleaned_replay/
    `-- cleaned_replay_v2b/
        |-- models/
        |-- predictions/
        |-- reports/
        `-- benchmarks/
```

**Structure Decision**: Extend the existing classifier training module because it already owns config loading, split handling, threshold search, metrics, Phase 3 dry-run validation, baseline loading, and artifact writing. Add Spec 018 as a governed replay mode with its own config and tests.

## Phase 0: Research

Research decisions are captured in `specs/018-v2b-cleaned-replay/research.md`.

Resolved decisions:

- Reuse `src/training/train_classifier.py` and add a cleaned replay mode.
- Add a dry-run command before training.
- Use Spec 017 approved manifest as the only training input.
- Compare against locked V2B baseline artifacts before any submission work.
- Reject the cleaned replay for submission use unless it beats baseline validation F1 and all safety gates pass.

## Phase 1: Design and Contracts

Design artifacts are generated:

- `specs/018-v2b-cleaned-replay/data-model.md`
- `specs/018-v2b-cleaned-replay/contracts/v2b-cleaned-replay-contract.md`
- `specs/018-v2b-cleaned-replay/quickstart.md`

The implementation must satisfy the dry-run and comparison contract before training results are trusted.

## Post-Design Constitution Check

- Binary output contract: PASS. Training and validation remain binary.
- Accuracy and speed: PASS. Baseline F1 and runtime evidence are explicit outputs.
- Hybrid ROI-first architecture: PASS. Spec 017 ROI/crop decisions are consumed as input.
- Reproducibility and leakage control: PASS. Dry-run validation and full saved artifacts are required.
- Confidentiality: PASS. No public artifact publication is introduced.
- Explainability and reporting: PASS. Comparison and validation reports are first-class outputs.
- Bias and imbalance: PASS. Target distributions are included in run and comparison artifacts.
- Annotation and model-version strategy: PASS. Scope remains V2B-style classifier replay.
- Modularity and compatibility: PASS. Paths are config-driven and the existing training module remains the integration point.

## Complexity Tracking

No constitution violations or complexity exceptions are required.
