# Contract: ROI Cropping and Preprocessing

## Purpose

Define the observable behavior for SPEC-004 ROI crop and preprocessing modules.
This contract is internal to the project and does not define a web API.

## Inputs

- Image path or loaded image object.
- Optional SPEC-003 annotation bounding box metadata.
- Split name: `train`, `validation`, or `test`.
- Preprocessing profile with target size and augmentation bounds.
- Optional output directory for ROI samples.

## Required Behaviors

- Produce `384x384` outputs by default.
- Use annotation ROI for training images when valid ROI metadata is available.
- Expand annotation ROI by 10% on each side and clip to image bounds.
- Treat ROI area under 10% of image area as unsafe.
- Use square center crop when annotation ROI is unavailable or unsafe.
- Use full-image resize when a safe square center crop cannot be produced.
- Preserve internal dark bottle regions; never remove dark pixels blindly.
- Apply bounded random augmentations only for training preprocessing.
- Apply deterministic resize and normalization only for validation/test
  preprocessing.
- Save ROI crop and deterministic preprocessed validation/test review samples
  under `outputs/figures/roi_samples/`.
- Keep generated review samples ignored and untracked by default.

## Outputs

- ROI crop result with crop method, crop box, output size, and fallback reason
  when applicable.
- Preprocessed image with configured target size and channel format.
- ROI crop review sample images under `outputs/figures/roi_samples/`.
- Preprocessed validation/test review sample images under
  `outputs/figures/roi_samples/`.
- Optional metadata/report describing sample paths and crop methods.

## Error and Fallback Behavior

- Missing or unreadable images must produce readable errors.
- Malformed ROI metadata must produce a fallback reason and use square center
  crop where safe.
- Out-of-bounds ROI metadata must be clipped if still valid after clipping;
  otherwise fallback.
- Unsafe fallback crop conditions must use full-image resize rather than silent
  failure.

## Confidentiality Rules

- Private images and private-derived ROI samples must not be committed.
- Generated ROI samples must remain under ignored output locations.
- Automated tests must use synthetic images and metadata only.

## Out of Scope

- Classifier training.
- Detector-format export.
- Kaggle submission generation.
- Test-label inference.
- Grad-CAM or dashboard visualization.
