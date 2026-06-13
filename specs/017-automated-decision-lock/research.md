# Research: Spec 017 Automated Decision Lock

## Decision: Use deterministic conservative decision rules as the primary gate

**Rationale**: The next training manifest must be reproducible and safe. Deterministic rules make every keep, exclude, adjudicate, and defer decision explainable and testable.

**Alternatives considered**:

- Manual review of all 3,361 review rows: rejected because it is slow and inconsistent.
- Fully automatic relabeling: rejected because changing labels is out of scope and risky.
- Training first and evaluating later: rejected because this feature is explicitly a data gate before training.

## Decision: Treat optional evidence as additive, not required

**Rationale**: Current Spec 016 evidence reported missing OOF predictions, missing embeddings, and fallback label-quality scoring. The workflow must still run safely without these artifacts while making its limitations visible.

**Alternatives considered**:

- Fail when embeddings or Cleanlab scores are missing: rejected because it would block the first useful decision lock.
- Ignore optional evidence entirely: rejected because embeddings, duplicate groups, and label-quality scores can materially improve decisions when available.

## Decision: Recalibrate ROI/crop review rows by evidence groups

**Rationale**: Spec 016 marked 3,230 rows for ROI/crop review, which is too broad to treat as automatic exclusion. Grouping by image quality, prediction risk, duplicate evidence, and optional visual clusters allows over-broad ROI flags to be downgraded safely.

**Alternatives considered**:

- Exclude every ROI/crop review row: rejected because it may remove valid training signal.
- Keep every ROI/crop review row: rejected because confirmed severe crop failures can harm training.

## Decision: Separate adjudication from approved training

**Rationale**: Rows needing human or higher-confidence evidence must not silently enter training. A separate adjudication queue preserves progress while preventing unresolved rows from contaminating the approved manifest.

**Alternatives considered**:

- Add reviewer columns directly to the approved manifest: rejected because it mixes unresolved and approved states.
- Require all adjudication before any output is useful: rejected because the automatic keep/exclude sets are useful immediately.

## Decision: Use row-level rule audit as the traceability backbone

**Rationale**: Later training, reporting, and debugging need to explain exactly why a row entered or did not enter the manifest. A rule audit is simpler and more reliable than reconstructing decisions from multiple bucket files.

**Alternatives considered**:

- Summary-only reporting: rejected because it cannot debug individual rows.
- Contact sheets only: rejected because images are useful for review but not sufficient for machine validation.
