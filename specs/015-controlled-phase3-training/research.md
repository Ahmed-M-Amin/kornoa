# Research: Controlled Phase 3 Hard-Example Training

## Decision: Use the approved Spec 014 candidate package as the only Phase 3 eligibility source

**Rationale**: Spec 014 already separates allowed and blocked hard rows with explicit evidence and safety gating. Recomputing eligibility inside Spec 015 would duplicate governance logic and weaken traceability.

**Alternatives considered**: Rebuilding the Phase 3 pool from raw hard-row audits was rejected because it risks admitting blocked rows. Allowing manual one-off row additions was rejected because it undermines reproducibility and rollback.

## Decision: Lock a single V2B-family baseline report before any Phase 3 training comparison

**Rationale**: The roadmap warns against mixing V2B, V2B-Enhanced, V4.2 hybrid, and V2.2 hard-example outputs as if they were equivalent anchors. One locked baseline is required so every candidate comparison measures the same reference behavior.

**Alternatives considered**: Using the latest best-looking model as an evolving baseline was rejected because it breaks cross-candidate comparability. Comparing every candidate to multiple baselines was rejected because it adds ambiguity to acceptance decisions.

## Decision: Build Spec 015 on the existing classifier training, threshold, benchmark, and submission modules

**Rationale**: The current repository already has config-driven training and artifact writing in `src/training/train_classifier.py`, thresholding in `src/training/threshold_search.py`, runtime evidence in `src/inference/benchmark.py`, and submission generation in `src/inference/submission.py`. Extending those modules preserves consistency and avoids a notebook-only side path.

**Alternatives considered**: A separate notebook-driven training workflow was rejected because it would weaken reproducibility. A brand-new orchestration module detached from the existing stack was rejected because the repository already contains the core primitives that Spec 015 needs.

## Decision: Treat hard-example improvement as a separate acceptance gate

**Rationale**: The roadmap makes clear that overall validation F1 can hide failure on the hard-example subset. A candidate that looks stronger on aggregate but does not materially improve the governed hard rows should not be treated as a serious `0.98+` branch.

**Alternatives considered**: Using only full-split validation F1 was rejected because it can reward clean-row stability while leaving the hardest rows unsolved. Using only hard-example-only F1 was rejected because it could ignore unacceptable precision collapse on the broader validation set.

## Decision: Require complete artifact packages and repeated runtime evidence for every serious candidate

**Rationale**: Candidate quality in this competition depends on more than one validation number. Threshold evidence, prediction exports, runtime reports, and submission artifacts are necessary for rollback, comparison, and final efficiency justification.

**Alternatives considered**: Saving only checkpoints and one metrics file was rejected because it leaves no stable evidence for changed-row analysis or runtime review. Treating a single local timing run as enough was rejected because the competition efficiency rubric is based on repeated notebook execution.

## Decision: Keep detector/rule logic out of the default Phase 3 training path

**Rationale**: The roadmap explicitly treats detector and rule logic as a gated repair path, not the default final shape. Spec 015 should therefore focus on controlled classifier candidates first and only allow gated repair comparisons as an evidence-backed branch.

**Alternatives considered**: Building detector retraining into Spec 015 was rejected because it expands scope beyond controlled hard-example classifier training. Broad rule overrides were rejected because earlier rule search did not justify widespread changes.
