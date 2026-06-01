# Research: Project Foundation

## Decision: Use a single Python project layout

**Rationale**: The implementation plan is a single computer-vision project with
shared datasets, models, training, inference, explainability, and dashboard code.
A single `src/` package keeps imports simple for local Windows, Kaggle, and
Colab while still separating responsibilities by domain.

**Alternatives considered**:

- Separate backend/frontend projects: rejected because the dashboard is planned
  as Streamlit and does not require a separate web frontend.
- Notebook-only layout: rejected because the constitution forbids one huge
  notebook and requires modular, testable source code.

## Decision: Create only safe foundation files in SPEC-001

**Rationale**: The clarification requires actual directories/files, but only safe
foundation assets. Placeholder package files, config templates, test stubs,
README, requirements, and `.gitignore` are safe. Private datasets and generated
outputs are not safe to track.

**Alternatives considered**:

- Document-only foundation: rejected by clarification.
- Commit sample private data or generated artifacts: rejected by confidentiality
  and no-leakage principles.

## Decision: Track output directories with `.gitkeep` while ignoring contents

**Rationale**: Later features need predictable output categories, but generated
model weights, predictions, figures, reports, submissions, and hard examples may
be private. Keeping `.gitkeep` files documents the layout while ignoring actual
artifact contents.

**Alternatives considered**:

- Do not create `outputs/`: rejected because Phase 1 requires output folders.
- Track generated artifacts: rejected by confidentiality and reproducibility
  controls.

## Decision: Use YAML configuration templates

**Rationale**: The implementation plan names `paths.yaml`, `classifier.yaml`,
`detector.yaml`, `inference.yaml`, and `dashboard.yaml`. YAML is readable in
notebooks and scripts and supports environment-specific dataset paths without
hardcoding private paths inside modules.

**Alternatives considered**:

- Python constants: rejected because source-code path constants risk local-only
  assumptions.
- JSON only: acceptable but less friendly for comments and future manual edits.

## Decision: Add foundation smoke tests

**Rationale**: SPEC-001 is mostly structure, so tests should verify that package
imports and configuration files exist/read cleanly. This gives later specs a
minimal quality gate without pretending model behavior exists yet.

**Alternatives considered**:

- No tests until model code exists: rejected because Phase 1 quality gate
  requires config paths load and project opens locally.
