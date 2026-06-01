"""Synthetic tests for SPEC-004 ROI cropping and preprocessing."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.preprocessing import (
    preprocess_image,
    save_preprocessed_samples,
)
from src.data.roi import (
    DEFAULT_OUTPUT_SIZE,
    DEFAULT_SAMPLE_DIR,
    RoiCropRequest,
    crop_roi,
    save_roi_samples,
    validate_generated_output_confidentiality,
)
from src.data.transforms import create_preprocessing_profile


FIXTURE_DIR = Path("tests/fixtures/synthetic_dataset/roi_cases")


@pytest.fixture()
def roi_cases():
    return json.loads((FIXTURE_DIR / "roi_cases.json").read_text(encoding="utf-8"))["cases"]


def _request(case_name: str, roi_cases, **overrides):
    case = next(item for item in roi_cases if item["image_id"] == case_name)
    kwargs = {
        "image_id": case["image_id"],
        "image_path": FIXTURE_DIR / case["file_name"],
        "annotation_bbox": case["bbox"],
    }
    kwargs.update(overrides)
    return RoiCropRequest(**kwargs)


def _as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image)


def test_annotation_roi_expands_by_margin_and_clips_to_bounds(roi_cases):
    result = crop_roi(_request("dark-region", roi_cases))

    assert result.method == "annotation"
    assert result.crop_box == (40, 32, 160, 128)
    assert result.output_size == DEFAULT_OUTPUT_SIZE
    assert result.image.size == DEFAULT_OUTPUT_SIZE
    assert result.image.mode == "RGB"


def test_unsafe_small_annotation_roi_uses_square_center_fallback(roi_cases):
    result = crop_roi(_request("small-roi", roi_cases))

    assert result.method == "square_center"
    assert result.fallback_reason == "annotation_roi_too_small"
    assert result.crop_box == (20, 0, 180, 160)
    assert result.image.size == DEFAULT_OUTPUT_SIZE


def test_missing_roi_uses_square_center_fallback(roi_cases):
    result = crop_roi(_request("dark-region", roi_cases, annotation_bbox=None))

    assert result.method == "square_center"
    assert result.fallback_reason == "missing_annotation_roi"
    assert result.crop_box == (20, 0, 180, 160)
    assert result.image.size == DEFAULT_OUTPUT_SIZE


def test_full_image_resize_fallback_when_center_crop_is_unsafe(roi_cases):
    result = crop_roi(_request("wide-image", roi_cases))

    assert result.method == "full_resize"
    assert result.fallback_reason == "unsafe_center_crop"
    assert result.crop_box == (0, 0, 240, 1)
    assert result.image.size == DEFAULT_OUTPUT_SIZE


def test_internal_dark_region_is_preserved_in_roi_crop(roi_cases):
    result = crop_roi(_request("dark-region", roi_cases))
    pixels = _as_array(result.image)

    assert result.preserved_dark_regions is True
    assert pixels.min() < 20


def test_roi_sample_generation_uses_default_ignored_output_location(roi_cases):
    requests = [
        _request("dark-region", roi_cases),
        _request("small-roi", roi_cases),
    ]

    report = save_roi_samples(requests)

    assert report.sample_dir == DEFAULT_SAMPLE_DIR
    assert len(report.sample_paths) == 2
    assert {path.parent for path in report.sample_paths} == {DEFAULT_SAMPLE_DIR}
    assert set(report.crop_methods.values()) == {"annotation", "square_center"}
    assert all(path.exists() for path in report.sample_paths)


def test_generated_roi_samples_are_ignored_and_untracked(roi_cases):
    report = save_roi_samples([_request("dark-region", roi_cases)])

    confidentiality = validate_generated_output_confidentiality(report.sample_paths)

    assert confidentiality.tracked_paths == []
    assert sorted(confidentiality.ignored_paths) == sorted(report.sample_paths)


def test_optional_quickstart_cli_generates_synthetic_roi_samples(tmp_path):
    output_dir = tmp_path / "roi_samples"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data.roi",
            "--dataset-root",
            str(FIXTURE_DIR),
            "--output-dir",
            str(output_dir),
            "--limit",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(list(output_dir.glob("*.png"))) == 2


def test_preprocessed_validation_and_test_samples_are_reviewable(roi_cases):
    requests = [_request("dark-region", roi_cases)]

    report = save_preprocessed_samples(requests, splits=("validation", "test"))

    assert len(report.sample_paths) == 2
    assert all(path.parent == DEFAULT_SAMPLE_DIR for path in report.sample_paths)
    assert all(path.exists() for path in report.sample_paths)
    assert all(Image.open(path).size == DEFAULT_OUTPUT_SIZE for path in report.sample_paths)


def test_synthetic_roi_and_deterministic_preprocessing_batch_under_one_second(roi_cases):
    requests = [
        _request("dark-region", roi_cases),
        _request("small-roi", roi_cases),
        _request("wide-image", roi_cases),
    ] * 8

    started = time.perf_counter()
    for request in requests:
        preprocess_image(request, split="validation")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0


def test_preprocessing_profile_rejects_unknown_split():
    with pytest.raises(ValueError, match="Unsupported preprocessing split"):
        create_preprocessing_profile("holdout")


def test_training_preprocessing_preserves_size_and_pixel_bounds(roi_cases):
    result = preprocess_image(_request("dark-region", roi_cases), split="train", seed=7)
    pixels = _as_array(result.image)

    assert result.profile.split == "train"
    assert result.image.size == DEFAULT_OUTPUT_SIZE
    assert pixels.min() >= 0
    assert pixels.max() <= 255


def test_validation_preprocessing_is_deterministic(roi_cases):
    request = _request("dark-region", roi_cases)

    first = preprocess_image(request, split="validation")
    second = preprocess_image(request, split="validation")

    assert np.array_equal(_as_array(first.image), _as_array(second.image))


def test_test_preprocessing_is_deterministic(roi_cases):
    request = _request("small-roi", roi_cases)

    first = preprocess_image(request, split="test")
    second = preprocess_image(request, split="test")

    assert np.array_equal(_as_array(first.image), _as_array(second.image))


def test_seeded_training_preprocessing_is_reproducible(roi_cases):
    request = _request("dark-region", roi_cases)

    first = preprocess_image(request, split="train", seed=123)
    second = preprocess_image(request, split="train", seed=123)
    different = preprocess_image(request, split="train", seed=124)

    assert np.array_equal(_as_array(first.image), _as_array(second.image))
    assert not np.array_equal(_as_array(first.image), _as_array(different.image))
