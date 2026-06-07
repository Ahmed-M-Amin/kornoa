# V5 Strong Classifier Closeout

## Status

SPEC-010 implementation is in place for the V5 strong classifier workflow:

- V5 config: `configs/v5_strong_classifier.yaml`
- V5 validation export: `image_id,true_label,prob_bad,classifier_prediction,target`
- V5 test export: `image_id,prob_bad,classifier_prediction,target`
- V5 strict submission: `image_id,target`
- V5 metrics/report wiring: implemented
- V5 fallback path: implemented for primary model creation failure
- V2/V4 regression coverage: passing

## Public Score

No V5 public score is recorded yet in this closeout.

Until a manual Kaggle submission beats `0.92181`, V4.2 remains the current best public submission and V5 remains analysis-only.

## Required Manual Review

- Run a real V5 training candidate on the intended dataset.
- Export test probabilities and submission from the selected V5 artifact root.
- Record the public score if a manual submission is made.
- Compare accepted V5 runtime against the V2B benchmark reference.
