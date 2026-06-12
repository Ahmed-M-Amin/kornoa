# New Implementation Plan: Reach For 0.98 While Optimizing Final Score

## Objective

Maximize the probability of reaching `0.98+` hidden-test binary F1 while still optimizing for the real final competition score:

- `50%` model performance
- `30%` model efficiency
- `20%` most valuable insight

This document is not a guarantee that `0.98+` will happen. It is the strongest evidence-driven plan to pursue that level while preserving a realistic fallback path if the data ceiling is lower.

## Non-Negotiable Rules

- Do **not** use test labels.
- Do **not** tune thresholds on the public leaderboard.
- Do **not** pick a final model from one lucky validation number.
- Do **not** accept slower models unless the F1 gain is clearly meaningful.
- Do **not** overwrite locked baseline artifacts.

## Scoring Reality

The final ranking has three parts:

| Component | Weight | What it means for us |
| --- | ---: | --- |
| Model performance | `50%` | Hidden-test binary F1 for `target = 1`. |
| Model efficiency | `30%` | Official runtime in the organizer Kaggle notebook workflow. |
| Most valuable insight | `20%` | Quality of technical reasoning, analysis, and evidence. |

The correct target is therefore not:

```text
highest possible validation F1 at any cost
```

The correct target is:

```text
highest hidden-test F1 we can justify
+ strong runtime standing
+ strong technical evidence pack
```

## Current Reality Check

Known results and lessons:

| Candidate / approach | Result | Lesson |
| --- | ---: | --- |
| Original V2B | `0.92121` public | Strong original classifier baseline. |
| V4.2 hybrid | `0.92181` public | Best public score so far, but only a small gain. |
| V2B-Enhanced @ `0.51` | `0.89693` public | Validation-style changes can fail badly in public due to over-rejection. |
| V2B-Enhanced @ `0.61` | `0.91139` public | Calibration alone did not recover the lost behavior. |
| V3/V4 remake rule search | selected `0` rule changes | Detector rules were not yet strong enough to justify broad overrides. |
| V2.2 same-split full locked V2B validation | `0.9368` F1 | Real gain over V2B, but still far from `0.98`. |
| V2.2 same-split excluding hard examples | `0.9703` F1 | The clean subset is strong. |
| V2.2 same-split hard-example rows only | `0.3033` F1 | The hard examples remain the main blocker. |

## What `0.98+` Actually Requires

If we want `0.98+` on a locked split or hidden test, we should assume that one or more of these must become true:

1. The hard-example rows become much less noisy or much better modeled.
2. The label quality for the hardest rows improves.
3. The classifier gets stronger on defect-heavy or ambiguous rows without creating many new false positives.
4. The final decision path preserves V2B’s precision behavior while recovering more false negatives.

That means `0.98+` is **not** mainly a threshold problem.

It is mainly a:

- hard-example quality problem
- boundary-case modeling problem
- precision-preservation problem

## Final Strategy Shape

Use a two-track plan:

### Track A: Attack `0.98+`

This is the aggressive technical path. It focuses on closing the hard-example gap and improving the true decision boundary.

### Track B: Win The Actual Competition

This is the scoring-aware path. It ensures we also optimize runtime standing and insight quality even if hidden-test F1 does not reach `0.98+`.

Both tracks run together. We do not sacrifice final ranking just to chase one optimistic F1 target.

## Baseline Lock

Before any new training cycle, lock one original V2B baseline report and treat it as the only comparison anchor.

The baseline lock must include:

- validation F1
- precision for `target = 1`
- recall for `target = 1`
- TP / FP / FN / TN
- best validation threshold
- validation target distribution
- test submission target distribution
- runtime per image, if available
- checkpoint path
- validation prediction path
- submission path

This avoids mixing:

- original V2B
- V2B-Enhanced
- V4.2 hybrid
- V3/V4 remake
- V2.2 hard-example models

## Phase 1: Locked Same-Split Error Intelligence

Use the original locked V2B validation rows as the governing analysis set.

Build one joined audit table with:

- `image_id`
- `target`
- `v2b_probability`
- `v2b_prediction`
- `v22_probability`
- `v22_prediction`
- `error bucket`
- `is_hard_example`
- `hard_example_type`
- detector evidence, if available
- brightness / blur / crop quality fields, if available

Review these groups manually and quantitatively:

- V2B wrong, V2.2 correct
- V2B correct, V2.2 wrong
- both wrong
- high-confidence false positives
- high-confidence false negatives
- near-threshold disagreements
- hard negatives that are still predicted `1`
- hard positives that are still predicted `0`

The question is not “which rows are wrong?”

The real question is:

```text
which specific failure modes prevent 0.98?
```

Expected output:

- one hard-example taxonomy
- one clean-example taxonomy
- one ranked list of fixable failure modes

## Phase 2: Label And Data Quality Audit For The Hardest 442 Rows

This phase is the highest-value phase if the goal is truly `0.98+`.

For the hard-example pool, audit for:

- ambiguous labels
- incorrect labels
- weak ROI crops
- low-visibility defects
- detector misses
- visually reusable bottles that resemble defects
- visually defective bottles with weak texture cues

Split the 442 rows into:

- likely correct but hard
- likely ambiguous
- likely mislabeled
- likely preprocessing issue
- likely detector-evidence issue

If label correction is allowed within competition rules, this phase should become the top priority.

Reason:

If the hard subset is noisy, model capacity alone will not carry us to `0.98+`.

## Phase 3: Strong Classifier Program With Strict Candidate Discipline

Train only candidates with a clear hypothesis.

Priority order:

1. EfficientNet-B1 at `384`
2. EfficientNet-B1 at `448`
3. EfficientNet-B2 at `384`
4. ConvNeXt-Tiny at `384`
5. ConvNeXt-Tiny at `448` only if runtime remains competitive

Every candidate must answer one precise question, for example:

- does larger image size recover defect detail without precision collapse?
- does a stronger backbone reduce hard-example FN count?
- does a different architecture reduce over-rejection?

Do **not** launch broad recipe chaos.

Treat these as risky unless the error audit supports them:

- focal loss
- heavy oversampling
- aggressive hard-example mining
- broad threshold lowering
- broad detector overrides

## Phase 4: Hard-Example-Specific Improvement Plan

To reach `0.98+`, the hard-example subset must improve sharply.

Required experiments:

1. Conservative weighting vs no weighting
2. `384` vs `448` on the same backbone
3. EfficientNet vs ConvNeXt on the same hard rows
4. crop-quality filtered training vs unfiltered training
5. relabeled or cleaned hard subset vs original subset, if allowed

Measure for each candidate:

- full locked-split F1
- hard-example-only F1
- excluding-hard-example F1
- precision
- recall
- FP count
- FN count
- changed rows vs V2B

Success condition:

```text
the hard-example subset improves materially
without destroying clean-row precision
```

If the hard subset does not improve, `0.98+` should be treated as unlikely for that branch.

## Phase 5: Threshold, Stability, And Calibration

Thresholds must be searched on validation only using:

```python
from sklearn.metrics import f1_score

f1 = f1_score(
    y_true,
    y_pred,
    average="binary",
    pos_label=1,
    zero_division=0,
)
```

For every candidate, save:

- threshold sweep
- best threshold
- close-threshold range
- precision / recall curve summary
- changed rows vs V2B

Reject thresholds that:

- mainly gain recall by creating reusable-bottle over-rejection
- are only best at one fragile threshold
- destabilize target distribution sharply

Thresholds can polish a good model.
They will not rescue a weak hard-example strategy.

## Phase 6: Ensemble As Teacher, Not Default Final Model

Use ensembles to improve understanding first.

Purpose:

- identify stable hard negatives
- identify stable hard positives
- detect ambiguous rows
- detect likely label issues
- identify rows where multiple good models still disagree

Then use that information to:

- pick the best single fast model
- distill behavior into a deployable classifier
- build stronger insight evidence

Only allow an ensemble as the final model if:

- runtime stays competitive under the official efficiency scoring
- F1 gain is large enough to justify the runtime cost

## Phase 7: Detector And Rule Path Only As A Gated Repair

Detector/rule logic should not be broad.

Only use it as a cheap gated repair, mostly for uncertain classifier rows:

```text
abs(classifier_probability - threshold) <= uncertainty_margin
```

Accept detector/rule logic only if it:

- fixes real false negatives
- adds few false positives
- is cheap in runtime
- survives same-split comparison against V2B and the best classifier-only candidate

The default final shape remains:

```text
one strong fast classifier
+ calibrated threshold
+ optional cheap gated correction only if proven
```

## Phase 8: Efficiency Plan For The 30% Runtime Score

This phase must follow the organizer rule exactly.

### Official Efficiency Measurement

All finalists submit:

- the trained model
- a fully executable inference/evaluation notebook

Organizers run each finalist in a standardized Kaggle notebook environment with the same:

- hardware/runtime configuration
- preprocessing
- evaluation procedure
- timing cell

The timing cell is run multiple times per team, currently expected to be about `3` to `10` runs.

The **best (fastest) runtime** among those repeated runs becomes the team’s official runtime.

### Why This Changes Our Plan

We should optimize for:

- low cold-start variance
- low notebook setup overhead
- low per-image runtime
- stable preprocessing
- minimal unnecessary I/O

We should not optimize only for one lucky local timing run.

### Percentile-Based Normalization

Efficiency is not judged from raw runtime alone. Organizers normalize using percentile anchors:

- `t_p10` = 10th percentile runtime among finalist models
- `t_p90` = 90th percentile runtime among finalist models
- `t_model` = our official measured runtime, meaning the best runtime observed across repeated timing runs

Because the exact final formula depends on organizer evaluation, our practical objective is:

- be safely near the fast end of the finalist pool
- avoid being anywhere near the slow tail

### Concrete Runtime Requirements

Every candidate must save a runtime report containing:

- model name
- image size
- batch size
- device
- TTA mode
- total inference time
- seconds per image
- images per second
- checkpoint size
- whether detector/rules are used
- notebook startup/setup time if measurable

### Efficiency Decision Rules

- Prefer `384` over `448` unless `448` gives a clearly meaningful F1 gain.
- Prefer single-model inference over ensembles by default.
- Avoid TTA unless it gives a strong same-split gain relative to runtime cost.
- Keep preprocessing deterministic and minimal.
- Keep notebook cells short, direct, and repeatable.
- Benchmark every finalist candidate multiple times locally and in Kaggle-like conditions when possible.

## Phase 9: Insight Plan For The 20% Most Valuable Insight Score

This score is too large to treat as an afterthought.

We need a final insight evidence pack showing that we understood the problem better than other teams.

Required material:

- locked V2B baseline report
- same-split V2B vs V2.2 comparison
- why V2B-Enhanced failed
- why threshold-only tuning was insufficient
- what the 442 hard rows represent
- what failure modes were most important
- what experiments helped and which did not
- why detector binary rules were rejected or gated
- runtime comparison table
- final model decision rationale
- final F1 vs runtime vs insight tradeoff

The insight deliverable should answer:

```text
what did we learn about the bottle-classification problem
that was both technically correct and competition-useful?
```

## Candidate Gates

A candidate is accepted only if it passes all of these:

- full locked-split F1 improves over the active baseline
- precision does not collapse
- recall gain is not mostly over-rejection
- target distribution remains reasonable
- runtime remains competitive
- rollback path exists
- the candidate adds evidence, not just a number

Additional gate for `0.98+` pursuit:

- the hard-example subset trend must improve, not just the clean subset

## Candidate Comparison Table

Every candidate must be added to one comparison table before the next decision.

Required columns:

- candidate name
- model architecture
- image size
- checkpoint source
- training epochs
- learning rate
- full locked-split F1
- hard-example-only F1
- excluding-hard-example F1
- precision
- recall
- FP count
- FN count
- target `1` distribution
- changed rows vs V2B
- runtime per image
- best-of-N timing result
- public score, if submitted
- decision: accepted / rejected / analysis-only
- rejection reason

## Candidate Output Standard

Every candidate must save:

- `val_predictions.csv`
- `test_probabilities.csv`
- `best_threshold.json`
- `classifier_metrics.json`
- `runtime_report.json`
- `target_distribution_report.json`
- `submission.csv`

These are required for:

- reproducibility
- rollback
- candidate comparison
- insight reporting
- efficiency analysis

## Ablation Requirement

If a candidate improves, run at least one targeted ablation before trusting it.

Examples:

- no TTA vs TTA
- classifier only vs classifier plus rule
- `384` vs `448`
- EfficientNet-B1 vs EfficientNet-B2
- original hard-example set vs cleaned hard-example set

If we cannot explain the gain, we should treat the candidate as fragile.

## Stop Rules

Reject or stop a branch when:

- hard-example-only F1 does not improve after multiple targeted attempts
- precision collapses while recall rises
- target distribution drifts sharply without clear justification
- runtime penalty is too large for the observed gain
- public score drops clearly below V2B or V4.2
- the candidate repeats V2B-Enhanced behavior

## Recommended Priority Order

If the goal is maximum real upside, the next order should be:

1. Lock V2B baseline report
2. Deep audit of the 442 hard rows
3. Same-split comparison for every serious candidate
4. Strong classifier candidates at `384` and `448`
5. Hard-example-specific ablations
6. Runtime benchmarking under Kaggle-like conditions
7. Insight evidence pack assembly
8. Limited leaderboard sanity submissions only when justified

## Acceptance Criteria

This roadmap is being followed correctly only if:

- no test labels are used
- no public leaderboard threshold tuning is done
- one locked V2B baseline exists
- one same-split evaluation exists for every serious candidate
- one candidate comparison table exists
- candidate artifacts are complete
- runtime benchmarks are saved
- the hard-example audit exists
- target distribution reports exist
- ablations exist for claimed gains
- the final candidate has clear F1/runtime justification
- the final insight evidence pack is ready
- `0.98+` is treated as a goal to pursue, not as an assumed outcome

## Verification For This Document

After editing this document:

- confirm `docs/new_implinmination.md` exists
- confirm all sections above are present
- confirm numeric claims match existing project evidence
- confirm `0.98+` is described as a goal, not a guarantee
- confirm the efficiency section reflects repeated Kaggle timing runs and percentile-based normalization
- confirm the insight section is explicitly treated as `20%` of the final score
