# Contract: Controlled Data Quality Audit Pipeline

## Purpose

Define the stable contract for Spec 016 so the audit workflow produces reproducible, conservative, training-safe data outputs without crossing into model training or submission generation.

## Input Contract

The workflow consumes:

- one required training label file
- one required training image directory
- optional V2B validation prediction evidence
- optional precomputed OOF training prediction evidence
- optional Spec 014 allowed hard-row and blocked-row outputs
- optional bottle-type, annotation, embedding, and historical review evidence
- one config at `configs/data_quality_audit.yaml` that defines input paths, output root, thresholds, and safety toggles

The workflow must reject:

- missing required training labels
- missing required training image directory
- any attempt to use test labels
- any attempt to generate a submission
- any attempt to modify original source labels

## CLI Contract

The feature exposes:

- one primary audit command that runs the full audit and writes all core CSV/JSON outputs
- one optional contact-sheet command that builds visual review sheets from existing audit outputs

The primary audit command must:

- complete successfully when required inputs exist, even if optional evidence is missing
- mark reliability as limited when stronger optional evidence such as OOF predictions is missing
- produce one master audit row per source training row

## Master Table Contract

`data_quality_master.csv` must:

- contain exactly one row per source training row
- preserve the original binary target values
- include prediction evidence when available
- include Spec 014 governance evidence when available
- include image-quality and duplicate/conflict evidence
- include one final data-quality bucket and one evidence summary per row
- include traceability metadata linking rows to the audit run and configuration source

## Bucket Contract

The workflow must assign every row to exactly one final bucket:

- `clean_train`
- `hard_valid_train`
- `exclude_from_training`
- `manual_review_required`

The workflow must enforce:

- suspected-mislabel blocked rows never enter the cleaned training manifest
- duplicate label-conflict rows never enter the cleaned training manifest
- missing or unreadable image rows never enter the cleaned training manifest
- `manual_review_required` rows never enter the cleaned training manifest during this feature

## Cleaned Manifest Contract

`cleaned_training_manifest.csv` must:

- include only `clean_train` and `hard_valid_train` rows
- preserve the original binary target values
- include enough evidence columns to explain why a row is considered training-safe
- exclude all rows needing review, relabeling, or blocked-row adjudication

## Review Artifact Contract

The workflow must produce:

- one manual review template
- ranked suspicious-row outputs
- duplicate-conflict and ROI/image-quality issue reports
- optional contact sheets when images are available

Review artifacts must support later human decisions but must not change source labels or training eligibility automatically.

## Safety Contract

The full Spec 016 workflow must preserve:

- no training started
- no submission created
- no test-label use
- no leaderboard tuning
- no original label modification
- no dependence on optional third-party tooling for core workflow correctness
