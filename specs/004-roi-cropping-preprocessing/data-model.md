# Data Model: ROI Cropping and Preprocessing

## ROI Crop Request

Represents one image prepared for ROI crop evaluation.

Fields:

- `image_id`: Stable image identifier.
- `image_path`: Source image path.
- `split`: One of `train`, `validation`, or `test`.
- `image_width`: Source image width in pixels.
- `image_height`: Source image height in pixels.
- `annotation_bbox`: Optional bounding box from SPEC-003 annotation metadata,
  represented as `x`, `y`, `width`, `height`.
- `target_size`: Output size, default `384x384`.

Validation rules:

- `image_path` must resolve to a readable image.
- `target_size` must contain positive width and height.
- `annotation_bbox`, when present, must have positive width and height before
  margin expansion.

## ROI Crop Result

Represents the crop selected for an image.

Fields:

- `image_id`: Matches the request image identifier.
- `method`: One of `annotation_roi`, `square_center`, or `full_resize`.
- `crop_box`: Pixel bounds used for crop when applicable.
- `output_size`: Final dimensions after resize.
- `fallback_reason`: Optional reason when annotation ROI or square center crop
  could not be used.
- `preserved_dark_regions`: Boolean or diagnostic flag used by synthetic tests
  and visual review.

Validation rules:

- `output_size` must equal the configured target size.
- `annotation_roi` crops must expand the source ROI by 10% on each side and clip
  to image bounds.
- Annotation ROI area under 10% of the image area is unsafe and must not produce
  an `annotation_roi` result.
- `full_resize` is only used when neither annotation ROI nor square center crop
  is safe.

## Preprocessing Profile

Represents split-specific preprocessing behavior.

Fields:

- `split`: One of `train`, `validation`, or `test`.
- `target_size`: Output size, default `384x384`.
- `normalize`: Required for every split.
- `brightness_contrast`: Training-only bounded augmentation.
- `rotation`: Training-only bounded augmentation.
- `shift_scale`: Training-only bounded augmentation.
- `blur_noise`: Training-only bounded augmentation.
- `seed`: Optional seed for reproducible random training transforms.

Validation rules:

- Validation and test profiles must not include random augmentations.
- Training profile augmentations must remain bounded by configuration.
- Repeated validation/test preprocessing for the same input and configuration
  must produce identical output.

## ROI Sample Report

Represents generated visual review artifacts.

Fields:

- `sample_dir`: Must be `outputs/figures/roi_samples/` by default.
- `sample_paths`: Generated review image paths.
- `source_split`: Split used for sample generation.
- `crop_methods`: Methods represented in the sample set.
- `ignored_by_git`: Whether generated samples are ignored.
- `tracked_by_git`: Whether generated samples are tracked.

Validation rules:

- Generated sample images must be ignored and untracked by default.
- Synthetic samples must include at least one annotation ROI case and one
  fallback case.
- Private-derived samples must not be committed unless explicitly approved for
  private team sharing.
