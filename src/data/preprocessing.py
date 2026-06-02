"""ROI plus split-specific preprocessing orchestration for SPEC-004."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

from src.data.roi import (
    DEFAULT_SAMPLE_DIR,
    PreprocessingProfile,
    RoiCropRequest,
    RoiCropResult,
    RoiSampleReport,
    crop_roi,
    load_image,
    validate_generated_output_confidentiality,
)
from src.data.transforms import (
    apply_preprocessing_transforms,
    create_preprocessing_profile,
    normalize_image,
)


@dataclass(frozen=True)
class PreprocessingResult:
    """Preprocessed image and the metadata that produced it."""

    image_id: str
    image: Image.Image
    normalized: np.ndarray
    profile: PreprocessingProfile
    crop_result: Optional[RoiCropResult] = None


def preprocess_image(
    request: RoiCropRequest,
    *,
    split: str,
    seed: Optional[int] = None,
    augmentation_recipe: str = "v1",
    profile: Optional[PreprocessingProfile] = None,
) -> PreprocessingResult:
    """Crop a request and apply split-specific preprocessing."""

    crop_result = crop_roi(request)
    active_profile = profile or create_preprocessing_profile(
        split,
        target_size=request.target_size,
        seed=seed,
        augmentation_recipe=augmentation_recipe,
    )
    image = apply_preprocessing_transforms(crop_result.image, active_profile)
    normalized = normalize_image(image, active_profile)
    return PreprocessingResult(
        image_id=request.image_id,
        image=image,
        normalized=normalized,
        profile=active_profile,
        crop_result=crop_result,
    )


def preprocess_pil_image(
    image: Image.Image | Path | str,
    *,
    split: str,
    seed: Optional[int] = None,
    augmentation_recipe: str = "v1",
    profile: Optional[PreprocessingProfile] = None,
) -> PreprocessingResult:
    """Apply preprocessing to a PIL image or path without annotation ROI metadata."""

    source = load_image(image)
    active_profile = profile or create_preprocessing_profile(split, seed=seed, augmentation_recipe=augmentation_recipe)
    output = apply_preprocessing_transforms(source, active_profile)
    normalized = normalize_image(output, active_profile)
    return PreprocessingResult(
        image_id="pil-image",
        image=output,
        normalized=normalized,
        profile=active_profile,
    )


def save_preprocessed_samples(
    requests: Iterable[RoiCropRequest],
    *,
    splits: Sequence[str] = ("validation", "test"),
    output_dir: Path | str = DEFAULT_SAMPLE_DIR,
) -> RoiSampleReport:
    """Save deterministic validation/test preprocessing samples for review."""

    sample_dir = Path(output_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_paths: list[Path] = []
    methods: dict[str, str] = {}

    for request in requests:
        for split in splits:
            if split == "train":
                raise ValueError("Review sample preprocessing is limited to validation/test splits")
            result = preprocess_image(request, split=split)
            path = sample_dir / f"{_safe_name(request.image_id)}_{split}_preprocessed.png"
            result.image.save(path)
            sample_paths.append(path)
            methods[f"{request.image_id}:{split}"] = result.profile.split

    confidentiality = validate_generated_output_confidentiality(sample_paths)
    return RoiSampleReport(
        sample_dir=sample_dir,
        sample_paths=sample_paths,
        crop_methods=methods,
        confidentiality=confidentiality,
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
