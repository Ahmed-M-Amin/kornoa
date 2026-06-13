# Research: Locked Same-Split Error Intelligence

## Decision: Extend the existing `src/analysis/v2_2_same_split_eval.py` workflow

**Rationale**: The repository already has a same-split comparison module, tests, and output contract for V2.2. Extending that lane is lower risk than creating a second comparison workflow and keeps Phase 1 in one reproducible analysis path.

**Alternatives considered**: Creating a new top-level Phase 1 tool was rejected because it would duplicate row loading, metric computation, threshold sweep logic, and safety guards. Pushing the work into training or inference modules was rejected because this feature must remain analysis-only.

## Decision: Compare one selected V2.x candidate at a time

**Rationale**: The clarified spec requires Phase 1 to support future V2.x candidates but only one selected candidate per run. This produces a clean same-split artifact set that can be repeated for each candidate while avoiding multi-candidate inference complexity in one command.

**Alternatives considered**: Comparing multiple candidates in a single run was rejected because it would complicate inference orchestration, error handling, and test coverage without being required for the core value of the feature.

## Decision: Maintain a rolling candidate comparison table as a persisted report

**Rationale**: The roadmap requires consistent candidate comparison across time. Persisting one rolling table under the analysis report path gives a single evidence surface for future selection decisions and the final insight package.

**Alternatives considered**: Manual spreadsheet tracking was rejected because it is error-prone and non-reproducible. Keeping only per-candidate report files was rejected because it forces manual reconciliation later.

## Decision: Keep detector and image-quality evidence optional in Phase 1

**Rationale**: The locked same-split comparison must be valid even when optional evidence is absent. Phase 1’s primary value is the locked-row comparison itself; optional evidence should enrich later review, not block the core workflow.

**Alternatives considered**: Requiring optional evidence for all rows was rejected because current artifact coverage may be incomplete and would delay a valid same-split analysis. Excluding optional evidence entirely was rejected because later failure-mode review benefits from preserving an attachment point.

## Decision: Use `accepted`, `rejected`, and `manual_review` as the only candidate statuses

**Rationale**: The clarified spec limits decision status to locked same-split evidence only. Restricting statuses to these three keeps the analysis contract narrow and avoids mixing public-score or runtime-selection language into the core Phase 1 workflow.

**Alternatives considered**: Free-form decision text was rejected because it weakens downstream comparison-table consistency. Public-score-aware states were rejected because they belong to later selection stages, not Phase 1.

## Decision: Define the hard-example regression tolerance as absolute F1 drop greater than `0.01`

**Rationale**: The clarified spec requires a “small tolerance” for hard-example regression before a candidate loses `accepted` eligibility. An absolute `0.01` F1 tolerance on the hard-example-only section is small enough to preserve focus on the hardest rows while allowing minor noise from threshold ties or stochastic candidate differences.

**Alternatives considered**: Zero tolerance was rejected because it would over-penalize tiny fluctuations that do not materially harm the roadmap goal. Larger tolerances such as `0.02` or `0.03` were rejected because they would let the hard-row bottleneck weaken too much while still calling a candidate `accepted`.

## Decision: Auto-assign `rejected` only when both overall and hard-example evidence regress

**Rationale**: The clarified spec now makes all three decision states explicit. A candidate is `rejected` only when full locked-row F1 does not improve over V2B and hard-example-only F1 regresses by more than an absolute `0.01`, while mixed outcomes remain `manual_review`.

**Alternatives considered**: Rejecting any candidate with lower full locked-row F1 alone was rejected because it would collapse too many mixed-signal outcomes into `rejected` and weaken the distinction between `rejected` and `manual_review`. Requiring simultaneous precision and recall degradation was rejected because it makes the state logic harder to explain and test.
