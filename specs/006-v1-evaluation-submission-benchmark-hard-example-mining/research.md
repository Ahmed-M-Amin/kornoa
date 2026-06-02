# Research: V1 Evaluation, Submission, Benchmark, and Hard-Example Mining

## Decision: Default V1 Artifact Root

**Decision**: Use `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1` as the default concrete completed V1 run root, while also accepting the parent root `artifacts/kaggle_v1_artifacts/outputs` by resolving its `kaggle_v1` child.

**Rationale**: The user confirmed V1 training is complete and the concrete V1 run artifacts are under `artifacts/kaggle_v1_artifacts/outputs/kaggle_v1`. The parent `outputs` folder also contains hard-example files, so accepting the parent root as a convenience reduces path mistakes while keeping the concrete run root unambiguous.

**Alternatives considered**:

- Hardcode `outputs/kaggle_v1`: rejected because the current completed run is archived under `artifacts/kaggle_v1_artifacts/outputs`.
- Require every path individually: viable for advanced use, but less convenient than an artifact root with concrete-run-root and parent-root support.
- Move artifacts into `outputs/`: rejected because generated/private-derived artifacts should not be churned or recommitted.

## Decision: Saved Threshold Is the Only Decision Threshold

**Decision**: Load the saved V1 best-threshold JSON and use its threshold value for all probability-to-target conversion.

**Rationale**: SPEC-005 selected the threshold by validation F1. Reusing that saved value prevents accidental fallback to `0.5` and keeps Kaggle submission behavior tied to the validated V1 run.

**Alternatives considered**:

- Use `configs/inference.yaml` threshold: rejected for V1 submission because the saved run threshold is authoritative.
- Re-run threshold search: rejected because SPEC-006 is post-training evaluation, not training or tuning.
- Tune from Kaggle score: rejected because that risks leakage and irreproducibility.

## Decision: Sample Submission Drives Test Image Order

**Decision**: Generate submission rows in exactly the order specified by `sample_submission.csv`, using its image IDs and label column contract.

**Rationale**: Kaggle submission correctness depends on matching the expected rows. Filesystem ordering can differ across platforms, while the sample submission is the authoritative contract.

**Alternatives considered**:

- Sort `test_images/` directly: rejected because it can produce an order mismatch or include unsubmitted files.
- Join by discovered images only: rejected because missing sample rows should be surfaced as errors, not silently dropped.

## Decision: Benchmark the Fast V1 Classifier Path Only

**Decision**: Benchmark ROI/preprocessing plus classifier inference, and save total runtime, image count, average milliseconds per image, and images per second.

**Rationale**: The final plan requires speed measurement for V1. Normal V1 inference should not include hard-example mining, detector, Grad-CAM, feature memory bank, or other later modules because that would hide the classifier baseline cost.

**Alternatives considered**:

- Include submission CSV writing in benchmark timing: rejected because benchmark should focus on image-flow inference speed.
- Include hard-example mining: rejected because mining is validation-only and not part of normal inference.
- Include future detector or feature memory bank: rejected because they are out of scope for SPEC-006.

## Decision: High-Loss Proxy Ranking from Existing Validation CSV

**Decision**: Define high-loss proxy samples from saved validation CSV fields by ranking wrong or confused predictions: false positives with high probability, false negatives with low probability, then near-threshold uncertain cases.

**Rationale**: SPEC-005 did not save per-sample BCE loss. The clarified proxy is reproducible from existing validation predictions and avoids rerunning training or changing completed V1 artifacts.

**Alternatives considered**:

- Require an actual per-sample loss column: rejected because it would make current V1 artifacts unusable for mining.
- Include only misclassified rows: rejected because near-threshold uncertain cases are useful for V2 even when correctly classified.
- Rerun model over validation images to compute loss: rejected because SPEC-006 should consume saved V1 artifacts and avoid changing the training contract.

## Decision: Required Concrete Run Files

**Decision**: The concrete V1 run artifact root must contain `models/classifier_effnet_b0_best.pth`, `reports/classifier_metrics.json`, `reports/best_threshold.json`, and `predictions/val_classifier_predictions.csv`.

**Rationale**: Submission and benchmark workflows require the checkpoint and threshold, while hard-example mining requires validation predictions. The metrics JSON is part of the completed V1 record and should be validated with the run root so SPEC-006 does not operate on a partial artifact tree.

**Alternatives considered**:

- Validate only files needed by each individual workflow: rejected because it can make a partial V1 artifact root look valid until a later workflow fails.
- Require existing hard-example CSVs as inputs: rejected because SPEC-006 formalizes regeneration from validation predictions; manual hard-example files may exist under the parent output root but are not required.
