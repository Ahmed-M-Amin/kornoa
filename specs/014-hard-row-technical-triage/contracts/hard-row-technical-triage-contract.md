# Contract: Hard-Row Technical Triage

## Purpose

Define the stable contract for Spec 014 outputs so later planning and task generation can rely on one governed hard-row triage workflow.

## Input Contract

The workflow consumes:

- one validation-derived governed hard-row audit source
- local train images
- local training annotations
- one config file that resolves all required paths from repository root

The workflow must reject:

- duplicate governed hard-row `image_id` values
- missing required governed-row inputs
- any attempt to use test labels or submission-only artifacts as governed hard-row sources

## Review Output Contract

The governed review stage must produce:

- one `manual_review_template.csv`
- one `review_manifest.csv`
- summary outputs for clustering, anomaly scoring, and triage reports when enabled

Every governed hard row must appear exactly once in the review manifest.

## Evidence Completion Contract

The evidence-completion stage must produce:

- one annotation evidence CSV
- one crop-quality evidence CSV
- one evidence-completed manifest
- one evidence-aware final action plan
- evidence assets for supported overlay and crop views

Every governed hard row must produce exactly one evidence-aware action-plan row.

`phase3_use_allowed` must remain a separate safety gate and must not be inferred only from the action label.

## Phase 3 Candidate Package Contract

The candidate package stage must:

- validate total input row count before writing outputs
- validate the configured governed-row total against the evidence-aware action plan before writing outputs
- validate allowed and blocked totals before writing outputs
- write allowed package files only from `phase3_use_allowed = true`
- write blocked package files only from `phase3_use_allowed = false`
- prevent any row from appearing in both allowed and blocked outputs

## Safety Contract

The full Spec 014 workflow must remain analysis-only:

- no training
- no submission generation
- no public leaderboard tuning
- no test-label use
- no label overwriting
