# Implementation Plan: Hard-Row Technical Triage

**Branch**: `014-hard-row-technical-triage` | **Date**: 2026-06-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/014-hard-row-technical-triage/spec.md`

## Summary

Implement Spec 014 as an offline hard-row technical-triage workflow that takes the governed 442-row validation-derived hard-example pool, generates review-ready manifests and cluster triage outputs, completes annotation/ROI/crop evidence for every governed row where possible, and exports a validated Phase 3 candidate package. The workflow must remain analysis-only, preserve row identity, provide evidence-backed reasons for allowed-versus-blocked rows, and keep all outputs under config-driven analysis paths.

The technical approach extends the existing `src/analysis/hard_row_visual_review.py` lane instead of creating a new training or inference module. It will keep the workflow in three connected stages: governed review generation, evidence completion, and candidate package export. Outputs remain local and reproducible, and the final `phase3_use_allowed` decision stays separate from the recommended action label.

## Technical Context

**Language/Version**: Python 3.11 target, compatible with the current local Python test environment.

**Primary Dependencies**: Existing `pandas`, `numpy`, `PyYAML`, `Pillow`, `scikit-learn`, `pytest`, and current analysis helpers already used by `src.analysis.hard_row_visual_review`.

**Storage**: Filesystem only. Inputs are CSV and JSON analysis artifacts plus local training images; outputs are CSV, JSON, and PNG/JPG evidence artifacts under `outputs/analysis/hard_row_visual_review/`.

**Testing**: Pytest with synthetic governed hard-row fixtures, evidence-completion assertions, candidate-package validation checks, and CLI coverage in `tests/test_hard_row_visual_review.py`.

**Target Platform**: Local Windows development first, with config-driven paths compatible with Kaggle and Google Colab when artifacts are copied there.

**Project Type**: Single Python computer-vision analysis and reporting project.

**Performance Goals**: Governed review generation and candidate export should complete in seconds for synthetic tests and low minutes for the real 442-row set; evidence completion may take longer because it renders per-row assets but must remain offline and non-interactive.

**Constraints**: Preserve `0 = Reusable` and `1 = Not Reusable`; operate only on validation-derived governed hard rows; do not use test labels or public leaderboard tuning; do not train; do not create submissions; do not overwrite labels; keep `phase3_use_allowed` as a separate safety gate that may still block trainable-looking action labels.

**Scale/Scope**: One governed 442-row hard-example pool, one manual-review template, one review manifest, one evidence-completed manifest, one evidence-aware action plan, one Phase 3 candidate package split, supporting contact sheets and evidence assets, and tests. Out of scope: actual Phase 3 training, classifier architecture changes, threshold search for production submission, detector retraining, and Kaggle submission generation.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The workflow preserves binary targets and predictions throughout review, evidence, and candidate packaging outputs.
- Accuracy and speed: PASS. The feature improves candidate selection quality and insight quality without changing deployed inference runtime because it remains offline analysis.
- Hybrid ROI-first architecture: PASS. The workflow evaluates ROI and detector-evidence quality but does not alter the deployed hybrid decision path.
- Reproducibility and leakage control: PASS. Inputs, counts, safety flags, evidence outputs, and package totals are saved; no test labels, training, or public-score tuning are allowed.
- Confidentiality: PASS. Images, annotations, evidence assets, and candidate packages remain local under competition-safe analysis paths.
- Explainability and reporting: PASS. Row-level rationale, overlays, evidence metrics, and blocked-versus-allowed outputs are central outputs of this feature.
- Bias and imbalance: PASS. The workflow isolates the highest-risk validation subset and exposes suspicious labels, anomalies, artifact confusion, and tiny-defect cases.
- Annotation and model-version strategy: PASS. This is a V2.x/Phase 3 preparation workflow and does not claim a new final model version.
- Modularity and compatibility: PASS. Analysis stays in `src/analysis`, paths remain config-driven, and the workflow is compatible with local, Kaggle, and Colab artifact movement.

## Project Structure

### Documentation (this feature)

```text
specs/014-hard-row-technical-triage/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- hard-row-technical-triage-contract.md
|-- checklists/
|   `-- requirements.md
`-- tasks.md
```

### Source Code (repository root)

```text
configs/
`-- hard_row_visual_review.yaml

src/
`-- analysis/
    `-- hard_row_visual_review.py

tests/
`-- test_hard_row_visual_review.py

outputs/
`-- analysis/
    `-- hard_row_visual_review/
        |-- contact_sheets/
        |-- evidence_assets/
        |-- fiftyone_export/
        |-- manus_review/
        `-- reports/
```

**Structure Decision**: Extend `src/analysis/hard_row_visual_review.py` because the repository already uses that module as the governed hard-row review lane. Keeping review generation, evidence completion, and candidate packaging in the same analysis module preserves one config surface and avoids leaking analysis behavior into training or submission code.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0: Research

See [research.md](./research.md).

## Phase 1: Design

See [data-model.md](./data-model.md), [contracts/hard-row-technical-triage-contract.md](./contracts/hard-row-technical-triage-contract.md), and [quickstart.md](./quickstart.md).

## Post-Design Constitution Check

- PASS: The data model preserves stable hard-row identity, governed review stages, evidence records, and a distinct Phase 3 safety gate.
- PASS: The contract keeps the workflow validation-derived, analysis-only, and reproducible while allowing optional tooling and optional detector outputs.
- PASS: The design explicitly separates trainable action labels from the final `phase3_use_allowed` safety gate.
- PASS: Outputs remain under ignored analysis paths and support downstream training selection and insight reporting without creating submissions or running training.
