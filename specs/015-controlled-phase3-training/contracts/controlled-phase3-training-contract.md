# Contract: Controlled Phase 3 Hard-Example Training

## Purpose

Define the stable contract for Spec 015 so controlled Phase 3 candidate training, comparison, and selection remain reproducible and competition-safe.

## Input Contract

The workflow consumes:

- one locked baseline report
- one approved Spec 014 candidate package
- one config at `configs/phase3_controlled_training.yaml` that defines candidate hypothesis, baseline references, artifact paths, and runtime/reporting outputs
- local training dataset and existing classifier-training dependencies

The workflow must reject:

- any training input row not traceable to the approved candidate package
- any blocked row entering the governed Phase 3 training set
- any candidate run that lacks an explicit baseline reference or candidate hypothesis
- any use of test labels or public-leaderboard-only threshold tuning

## Training Output Contract

Every serious candidate must produce:

- one validation predictions CSV at `predictions/val_classifier_predictions.csv`
- one test probabilities CSV
- one best-threshold JSON at `reports/best_threshold.json`
- one metrics JSON at `reports/classifier_metrics.json`
- one runtime report JSON at `benchmarks/runtime_report.json`
- one target-distribution report JSON at `reports/target_distribution_report.json`
- one submission CSV at `submissions/submission_phase3.csv`

Every candidate output package must remain tied to the same locked baseline anchor.

## Comparison Contract

The controlled comparison stage must:

- evaluate every serious candidate on the same governed validation framing
- save full validation, hard-example-only, and excluded-hard-example results
- save changed-row comparison evidence versus the locked baseline
- classify each candidate as `accepted`, `rejected`, or `analysis_only`
- record explicit decision reasons in one rolling comparison table at `reports/candidate_comparison_table.csv`

## Runtime Contract

Finalist-ready candidates must:

- include repeated runtime evidence
- save best-of-N timing results and summary timing fields
- preserve enough metadata to explain runtime tradeoffs against the baseline and other candidates

## Insight Contract

The final recommendation package must:

- explain the locked baseline
- explain the main hard-example failure modes
- summarize candidate wins and failures
- explain runtime tradeoffs
- justify the final recommendation in competition-aware terms
- remain serializable as `reports/insight_evidence_pack.json`

## Safety Contract

The full Spec 015 workflow must preserve:

- no test-label use
- no public-leaderboard-only threshold tuning
- no overwriting of locked baseline artifacts
- no admission of blocked hard rows into governed Phase 3 training
