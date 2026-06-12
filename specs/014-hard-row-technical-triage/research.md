# Research: Hard-Row Technical Triage

## Decision: Keep Spec 014 in the existing hard-row visual review module

**Rationale**: The repository already has a single governed hard-row analysis lane in `src/analysis/hard_row_visual_review.py`. Extending that lane keeps review generation, evidence completion, Manus merge support, and Phase 3 candidate packaging under one config and one test file.

**Alternatives considered**: Creating a separate evidence-completion module was rejected because it would duplicate config resolution, row normalization, and review-manifest logic. Embedding this feature into training code was rejected because Spec 014 must remain analysis-only.

## Decision: Use the governed hard-row pool as the only source of truth for this feature

**Rationale**: The roadmap and current outputs center on the governed 442-row validation-derived hard-example set. Using that pool as the single source of truth keeps row counts stable across review, evidence completion, and candidate packaging.

**Alternatives considered**: Recomputing membership from scratch at each stage was rejected because it risks row drift and weakens reproducibility. Mixing test or submission rows was rejected because it violates competition constraints.

## Decision: Treat `phase3_use_allowed` as a distinct safety gate

**Rationale**: Real outputs already show that some rows with trainable-looking action labels still need to stay blocked because of risk signals such as label issues, anomaly scores, or weak evidence. Keeping the gate separate from the action label makes package export deterministic and auditable.

**Alternatives considered**: Making every trainable action label automatically allowed was rejected because it would silently weaken the conservative candidate discipline required by the roadmap. Manual-review-only gating was rejected because it would not scale cleanly over the full governed pool.

## Decision: Use annotation evidence as the default detector fallback

**Rationale**: Optional detector outputs are not guaranteed to exist for every governed hard row, but local COCO-style annotations and train images are available. Using annotation evidence as the fallback avoids the previous failure mode where every row was treated as detector-missing.

**Alternatives considered**: Blocking the workflow when detector outputs are absent was rejected because it would make the feature unusable on the current artifact set. Ignoring localization evidence entirely was rejected because small or localized defects are central to the hard-row problem.

## Decision: Export blocked rows by risk class instead of collapsing them into one file

**Rationale**: Downstream work differs by blocked reason. ROI or preprocessing rows need engineering fixes, suspected mislabels need label verification, and the remaining high-risk rows need conservative review or stronger evidence. Separate blocked exports preserve that workflow.

**Alternatives considered**: A single blocked file was rejected because it weakens next-step prioritization. Hiding blocked rows entirely was rejected because those rows are still useful for diagnosis and reporting.
