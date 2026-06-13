# Final Model Decision

The final selected model is the V2B locked baseline.

- Public Kaggle score: 0.92121
- Internal validation F1: 0.926917
- Model family: EfficientNet-B1
- Final submission: outputs/submissions/v2b_locked_baseline_submission.csv

We tested a controlled Phase 3 hard-row training experiment using 422 approved hard rows from Spec 014. The pipeline worked technically, but the resulting model performed worse than the V2B baseline. Therefore, Phase 3 was rejected.

We also reviewed V2B_HE. Although it had a high internal validation F1, it gave a lower public Kaggle score, so it was rejected.

Final decision: use V2B locked baseline because it has the best verified public score and generalizes better.
