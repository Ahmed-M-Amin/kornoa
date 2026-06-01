# Research: Dataset Loading and Validation

## Decision: Use configuration-driven dataset roots

**Rationale**: The constitution requires local Windows, Kaggle, and Colab
compatibility and forbids hardcoded private dataset paths in core modules.
`configs/paths.yaml` is the single source for dataset root selection.

**Alternatives considered**:

- Hardcoded local path: rejected because it breaks Kaggle/Colab and violates the
  constitution.
- Environment variables only: acceptable as an override, but less discoverable
  for contributors than a checked-in config template.

## Decision: Use synthetic fixtures for automated tests

**Rationale**: Private Krones files must not be committed. Synthetic fixtures can
cover file presence, label matching, annotation coverage, missing files, and
error diagnostics without exposing private data.

**Alternatives considered**:

- Tests against the real dataset: rejected because private data access is not
  guaranteed and could leak file names/content.
- No automated tests: rejected because dataset loading is a blocking quality
  gate.

## Decision: Keep COCO handling audit-only in SPEC-002

**Rationale**: The clarification states that SPEC-002 reads enough annotation
metadata for coverage, distribution, and ROI summaries, while reusable COCO
parser APIs remain in SPEC-003. This avoids duplicating parser ownership.

**Alternatives considered**:

- Full parser now: rejected because it would make SPEC-003 redundant.
- Presence check only: rejected because Phase 2 requires annotation loading,
  defect distribution, and ROI availability summaries.

## Decision: Produce reports and figures as generated artifacts

**Rationale**: Phase 2 requires `dataset_summary.csv`, distribution figures, and
sample visualizations. These are derived from private data, so they belong in
ignored output locations and are not committed publicly.

**Alternatives considered**:

- Commit audit outputs: rejected by confidentiality requirements.
- Print-only audit: rejected because later reporting and review need repeatable
  artifacts.

## Decision: Fail loudly on missing or malformed inputs

**Rationale**: Silent failures would produce invalid training data and misleading
metrics. Missing files, malformed metadata, orphaned annotations, and wrong
dataset roots must return readable diagnostics.

**Alternatives considered**:

- Best-effort continuation: rejected for required files and metadata needed by
  later training and validation.
