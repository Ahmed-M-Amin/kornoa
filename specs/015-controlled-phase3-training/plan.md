# Implementation Plan: Controlled Phase 3 Hard-Example Training

**Branch**: `015-controlled-phase3-training` | **Date**: 2026-06-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/015-controlled-phase3-training/spec.md`

## Summary

Implement Spec 015 as a controlled Phase 3 training and evaluation workflow that starts from the approved Spec 014 candidate package, locks one baseline comparison anchor, runs hypothesis-driven candidate training on governed hard-example inputs, and produces a competition-aware comparison and selection package. The workflow must protect against leakage, preserve rollback artifacts, keep runtime evidence first-class, and produce a final insight pack that explains both candidate outcomes and the final recommendation.

The technical approach extends the current classifier-training stack instead of creating a separate experimental notebook path. It will build on the existing `src/training/train_classifier.py`, threshold search, hard-example loading, benchmark, and submission utilities while adding controlled comparison, candidate-governance, and reporting outputs under a dedicated Phase 3 experiment root.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Python training environment.

**Primary Dependencies**: Existing `torch`, `numpy`, `pandas`, `PyYAML`, `pytest`, classifier/backbone utilities in `src.models.classifier`, threshold helpers in `src.training.threshold_search`, and runtime/submission helpers in `src.inference`.

**Storage**: Filesystem only. Inputs are config files, approved Phase 3 candidate CSVs, locked validation predictions, checkpoints, and local dataset assets; outputs are CSV, JSON, model checkpoints, and submission artifacts under dedicated experiment output roots.

**Testing**: Pytest coverage for training contracts, hard-example loading, runtime benchmark outputs, and submission/report validation, centered on `tests/test_training_v5.py`, `tests/test_training_v2.py`, `tests/test_benchmark.py`, `tests/test_submission.py`, and new controlled-Phase-3 tests.

**Target Platform**: Local Windows development first, with config-driven execution compatible with Kaggle and Google Colab artifact movement.

**Project Type**: Single Python computer-vision training, evaluation, and reporting project.

**Performance Goals**: Candidate workflows should complete in reasonable local experiment time, produce validation-only threshold results, preserve or improve the active baseline F1 on the governed hard-example subset, and emit repeated runtime evidence suitable for competition-style efficiency comparison.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; use only approved `phase3_use_allowed = true` hard rows for governed Phase 3 eligibility; no test labels; no public-leaderboard threshold tuning; no overwriting of locked baseline artifacts; no uncontrolled recipe sweeps; every serious candidate must have a hypothesis, full artifact package, and rollback path.

**Scale/Scope**: One locked baseline report, one governed Phase 3 training eligibility set, a small number of serious classifier candidates, one rolling comparison table, repeated runtime reports, one final-candidate recommendation, and one insight evidence pack. Out of scope: detector retraining, broad hybrid-rule redesign, public-score-only model selection, and production deployment changes.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The plan keeps all training, validation, runtime, and submission outputs aligned to the required binary decision.
- Accuracy and speed: PASS. The feature explicitly balances hard-example F1 improvement against runtime evidence and competition efficiency scoring.
- Hybrid ROI-first architecture: PASS. The plan stays classifier-first and only permits gated repair comparisons when evidence justifies them; ROI preservation rules remain intact through the existing data pipeline.
- Reproducibility and leakage control: PASS. The plan requires one locked baseline anchor, validation-only thresholding, complete saved artifacts, and zero test-label or public-threshold leakage.
- Confidentiality: PASS. Dataset inputs, candidate packages, checkpoints, and reports remain local/private and under config-driven project paths.
- Explainability and reporting: PASS. The plan requires changed-row comparisons, candidate rationales, runtime tradeoffs, and a final insight evidence pack.
- Bias and imbalance: PASS. Governed hard-example performance is a first-class acceptance gate rather than being hidden behind aggregate validation metrics.
- Annotation and model-version strategy: PASS. This is a V5/V2.2-era controlled classifier program that consumes Spec 014 outputs without changing annotation governance or introducing memory-bank scope by default.
- Modularity and compatibility: PASS. The plan extends existing training, inference, and reporting modules with config-driven paths compatible with local, Kaggle, and Colab workflows.

## Project Structure

### Documentation (this feature)

```text
specs/015-controlled-phase3-training/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- controlled-phase3-training-contract.md
`-- tasks.md
```

### Source Code (repository root)

```text
configs/
|-- v2_2_hard_examples.yaml
|-- v5_strong_classifier.yaml
`-- phase3_controlled_training.yaml

src/
|-- data/
|   `-- hard_examples.py
|-- inference/
|   |-- benchmark.py
|   `-- submission.py
`-- training/
    |-- hard_example_mining.py
    |-- metrics.py
    |-- threshold_search.py
    `-- train_classifier.py

tests/
|-- test_benchmark.py
|-- test_submission.py
|-- test_training_v2.py
|-- test_training_v5.py
`-- test_phase3_controlled_training.py

outputs/
|-- analysis/
|   `-- phase3_controlled_training/
`-- kaggle_phase3/
    `-- controlled_phase3/
        |-- benchmarks/
        |-- models/
        |-- predictions/
        |-- reports/
        `-- submissions/
```

**Structure Decision**: Extend the existing classifier-training and reporting modules because the repository already has stable training, thresholding, benchmark, submission, and hard-example support. Spec 015 should add a controlled orchestration layer and reporting outputs rather than duplicating the training stack in notebooks or a parallel experiment runner.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/controlled-phase3-training-contract.md](./contracts/controlled-phase3-training-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: The data model separates the locked baseline, governed input set, candidate evaluation, runtime evidence, and final insight package without blurring training eligibility rules.
- PASS: The contract enforces validation-only thresholding, complete artifact packages, and blocked-row exclusion before candidate promotion.
- PASS: The design keeps hard-example improvement and runtime evidence as separate gates, which matches the competition scoring and the constitution.
- PASS: Outputs remain config-driven and modular, enabling local, Kaggle, and Colab-compatible execution without exposing private data.
