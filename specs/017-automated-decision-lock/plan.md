# Implementation Plan: Spec 017 Automated Decision Lock

**Branch**: `017-automated-decision-lock` | **Date**: 2026-06-13 | **Spec**: `specs/017-automated-decision-lock/spec.md`

**Input**: Feature specification from `specs/017-automated-decision-lock/spec.md`

## Summary

Build a leakage-safe decision-lock workflow that consumes Spec 016 data-quality audit outputs and Spec 014 hard-row triage outputs, assigns exactly one controlled decision to every audited train row, and writes `approved_cleaned_training_manifest.csv` for later training. The workflow must reduce broad manual review into an evidence-ranked adjudication queue, conservatively handle missing optional evidence, recalibrate ROI/crop review groups, and enforce no training, no submission, no test-label usage, no leaderboard tuning, and no original-label modification.

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**: pandas, numpy, PyYAML, Pillow, pytest; optional cleanlab-style score input, optional embedding CSV input, optional prediction CSV input

**Storage**: Local filesystem CSV, JSON, and JPG outputs under `outputs/analysis/data_quality_decision_lock/`

**Testing**: pytest unit tests plus Python compile check

**Target Platform**: Local Windows development first, compatible with Kaggle and Colab path configuration

**Project Type**: Offline analysis CLI module

**Performance Goals**: Complete row-level decision assignment for the 35,342-row train audit table in under 5 minutes on local CPU when optional embedding generation is not performed inside this feature

**Constraints**: Must not train, create submissions, read test labels, tune to leaderboard feedback, overwrite original labels, or use rejected Phase 3 outputs as source-of-truth evidence

**Scale/Scope**: One decision record per Spec 016 master row; expected current scale is 35,342 train rows, 31,981 first-pass cleaned rows, 3,361 review-required rows, 422 Spec 014 approved hard rows, and 20 Spec 014 blocked rows

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Binary output contract: PASS. The feature preserves `0 = Reusable` and `1 = Not Reusable`; it only decides row eligibility for later training.
- Accuracy and speed: PASS. The feature targets future accuracy through safer training data and has no inference runtime impact.
- Hybrid ROI-first architecture: PASS. The feature audits ROI/crop evidence and preserves internal dark-defect caution by not excluding rows from broad ROI heuristics alone.
- Reproducibility and leakage control: PASS. The feature writes deterministic decision artifacts, records evidence availability, and explicitly blocks test labels, submissions, training, and leaderboard tuning.
- Confidentiality: PASS. All private dataset and derived artifacts remain inside local/project output paths.
- Explainability and reporting: PASS. Every decision has reason, confidence, and evidence source reporting.
- Bias and imbalance: PASS. Decision summaries include class/risk bucket counts so later training can inspect distribution shifts.
- Annotation and model-version strategy: PASS. The feature consumes V2B/Spec 014 evidence where available and prepares a locked manifest for a later V2B-style replay.
- Modularity and compatibility: PASS. The workflow is a separate analysis module with config-driven paths and dedicated tests.

## Project Structure

### Documentation (this feature)

```text
specs/017-automated-decision-lock/
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   `-- decision-lock-contract.md
|-- checklists/
|   `-- requirements.md
`-- spec.md
```

### Source Code (repository root)

```text
configs/
`-- data_quality_decision_lock.yaml

src/
`-- analysis/
    `-- data_quality_decision_lock.py

tests/
`-- test_data_quality_decision_lock.py

outputs/
`-- analysis/
    `-- data_quality_decision_lock/
        |-- decision_master.csv
        |-- approved_cleaned_training_manifest.csv
        |-- auto_keep_rows.csv
        |-- auto_exclude_rows.csv
        |-- needs_adjudication_rows.csv
        |-- deferred_uncertain_rows.csv
        |-- adjudication_queue.csv
        |-- reports/
        |   |-- decision_lock_summary.json
        |   |-- decision_rule_audit.csv
        |   |-- roi_recalibration_summary.csv
        |   `-- evidence_inventory.json
        `-- contact_sheets/
            |-- needs_adjudication_top_risk.jpg
            |-- auto_exclude_representatives.jpg
            `-- roi_group_representatives.jpg
```

**Structure Decision**: Use one focused analysis module, one config file, one test file, and Spec Kit design artifacts. Do not modify training modules for this feature.

## Phase 0: Research

Research decisions are captured in `specs/017-automated-decision-lock/research.md`.

Resolved decisions:

- Use conservative deterministic decision rules first, optional evidence second.
- Treat missing optional evidence as a reported limitation, not as failure.
- Recalibrate broad ROI/crop review rows by group evidence before exclusion.
- Keep adjudication separate from approved training manifest.
- Make every output traceable through decision reasons and rule audit rows.

## Phase 1: Design and Contracts

Design artifacts are generated:

- `specs/017-automated-decision-lock/data-model.md`
- `specs/017-automated-decision-lock/contracts/decision-lock-contract.md`
- `specs/017-automated-decision-lock/quickstart.md`

The implementation must satisfy the contract before `/speckit-implement` is considered complete.

## Post-Design Constitution Check

- Binary output contract: PASS. Decision-lock artifacts do not redefine target labels.
- Accuracy and speed: PASS. No model is trained; future training receives a safer manifest.
- Hybrid ROI-first architecture: PASS. ROI/crop recalibration is explicit and conservative.
- Reproducibility and leakage control: PASS. All required outputs are deterministic files with safety flags.
- Confidentiality: PASS. Outputs remain local/private.
- Explainability and reporting: PASS. Decision audit, evidence inventory, and contact sheets are planned.
- Bias and imbalance: PASS. Summary reporting includes decision counts by target and risk bucket.
- Annotation and model-version strategy: PASS. No new annotation format is required; optional embedding/prediction inputs are file-based.
- Modularity and compatibility: PASS. Config-driven CLI analysis module keeps training code separate.

## Complexity Tracking

No constitution violations require justification.
