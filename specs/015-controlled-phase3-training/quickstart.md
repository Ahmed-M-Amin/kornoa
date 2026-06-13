# Quickstart: Controlled Phase 3 Hard-Example Training

## Goal

Lock one baseline anchor, train one governed Phase 3 candidate, evaluate it against the approved hard-example set, and produce runtime and comparison artifacts without violating leakage or candidate-discipline rules.

## Inputs

Confirm that the workflow can access:

- the approved Spec 014 Phase 3 candidate package
- the locked baseline checkpoint, threshold report, and validation predictions
- the local training dataset
- the candidate training config for the controlled Phase 3 run

Representative inputs live under:

```text
configs/phase3_controlled_training.yaml
outputs/analysis/hard_row_visual_review/reports/
artifacts/kaggle_v2b_artifacts/
```

## Lock The Baseline Anchor

Prepare or refresh the locked baseline report using the saved V2B-family artifacts that will anchor every candidate comparison.

## Train One Controlled Candidate

```bash
python -m src.training.train_classifier --config configs/phase3_controlled_training.yaml
```

## Benchmark Candidate Runtime

```bash
python -m src.inference.benchmark --image-dir 1st-krones-vision-ai-challenge/test_images --artifact-root outputs/kaggle_phase3/controlled_phase3 --output-path outputs/kaggle_phase3/controlled_phase3/benchmarks/runtime_report.json --model-name efficientnet_b1
```

## Export Submission-Aligned Candidate Outputs

```bash
python -m src.inference.submission --dataset-root 1st-krones-vision-ai-challenge --artifact-root outputs/kaggle_phase3/controlled_phase3 --save-test-predictions outputs/kaggle_phase3/controlled_phase3/predictions/test_probabilities.csv --output-path outputs/kaggle_phase3/controlled_phase3/submissions/submission_phase3.csv
```

## Expected Outputs

```text
outputs/kaggle_phase3/controlled_phase3/
|-- models/
|-- predictions/
|   |-- val_classifier_predictions.csv
|   `-- test_probabilities.csv
|-- reports/
|   |-- locked_baseline_report.json
|   |-- candidate_comparison_table.csv
|   |-- classifier_metrics.json
|   |-- best_threshold.json
|   |-- target_distribution_report.json
|   `-- insight_evidence_pack.json
|-- benchmarks/
|   `-- runtime_report.json
`-- submissions/
    `-- submission_phase3.csv
```

```text
outputs/analysis/phase3_controlled_training/
|-- locked_baseline_report.json
`-- governed_phase3_training_manifest.csv
```

## Acceptance Checks

The workflow is accepted only when:

- the governed Phase 3 training set contains only approved `phase3_use_allowed = true` rows
- every serious candidate is evaluated against the same locked baseline anchor
- candidate outputs include full validation, hard-example-only, and excluded-hard-example evidence
- repeated runtime evidence exists for finalist-ready candidates
- every serious candidate appears in the comparison table with an explicit decision and reason
- safety rules confirm no test-label use, no public-score-only threshold tuning, and no baseline overwrite

## Verification Commands

```bash
python -m pytest tests/test_phase3_controlled_training.py tests/test_training_v2.py tests/test_training_v5.py tests/test_benchmark.py tests/test_submission.py -q
python -m py_compile src/training/train_classifier.py src/inference/benchmark.py src/inference/submission.py src/data/hard_examples.py
```

## Out Of Scope

This quickstart does not retrain detectors, redesign the hybrid inference path broadly, use public leaderboard feedback as a training signal, or bypass the approved Spec 014 hard-row candidate governance.
