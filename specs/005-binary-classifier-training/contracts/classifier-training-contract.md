# Contract: Binary Classifier Training

## Purpose

Define the observable behavior for SPEC-005 classifier training modules. This
contract is internal to the project and does not define a web API.

## Inputs

- Dataset configuration that resolves `train.csv` and `train_images/`.
- Classifier configuration containing model name, binary class count, image
  size `384`, split seed, loss weighting behavior, and output paths.
- SPEC-004 ROI/preprocessing outputs for normalized `3x384x384` classifier
  inputs.
- Optional synthetic smoke settings that keep automated training under one
  minute.

## Required Behaviors

- Validate that labels are binary `0` or `1`.
- Reject one-class or otherwise unusable training inputs with readable errors.
- Create an 80/20 stratified train/validation split with a fixed seed when data
  size allows.
- Use weighted binary loss only for V1 imbalance handling.
- Train a binary classifier that outputs probability for `1 = Not Reusable`.
- Evaluate validation probabilities and labels.
- Search thresholds for best validation F1-score.
- Pick the lowest threshold when multiple thresholds tie for best F1-score.
- Select the best checkpoint by validation F1 after threshold search.
- Save generated model, metrics, threshold, and validation prediction artifacts
  under ignored output directories.
- Keep generated artifacts untracked by default.

## Outputs

- `outputs/models/classifier_effnet_b0_best.pth`
- `outputs/reports/classifier_metrics.json`
- `outputs/reports/best_threshold.json`
- `outputs/predictions/val_classifier_predictions.csv`

## Error and Fallback Behavior

- Missing dataset files must produce readable dataset validation errors.
- Missing or unreadable labeled images must identify affected image IDs.
- Invalid labels must identify affected rows or image IDs.
- Insufficient class representation must stop training before producing
  misleading model artifacts.
- Synthetic smoke mode may use a tiny local classifier variant, but it must
  preserve the same artifact and metric contract.

## Confidentiality Rules

- Private images, model weights, validation predictions, and metric artifacts
  must not be committed publicly.
- Generated classifier artifacts must remain under ignored `outputs/`
  locations unless explicitly approved for private team sharing.
- Automated tests must use synthetic images and metadata only.

## Out of Scope

- Kaggle submission generation.
- Test-image inference.
- Detector or segmentation training.
- Hybrid inference or confidence-gated detector fallback.
- Grad-CAM or dashboard visualization.
- Memory-bank, ensemble teacher, distillation, or hard-example mining behavior.
