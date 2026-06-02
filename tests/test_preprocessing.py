"""Focused tests for SPEC-004 split-specific preprocessing."""

import numpy as np
import yaml
from PIL import Image

from src.data.preprocessing import preprocess_image
from src.data.preprocessing import preprocess_pil_image
from src.data.roi import DEFAULT_OUTPUT_SIZE, RoiCropRequest
from src.data.transforms import create_preprocessing_profile


def _gradient_image() -> Image.Image:
    data = np.zeros((48, 64, 3), dtype=np.uint8)
    data[:, :, 0] = np.arange(64, dtype=np.uint8)
    data[:, :, 1] = np.arange(48, dtype=np.uint8)[:, None]
    data[:, :, 2] = 120
    return Image.fromarray(data, mode="RGB")


def test_validation_and_test_profiles_disable_augmentation():
    validation = create_preprocessing_profile("validation")
    test = create_preprocessing_profile("test")

    assert validation.augment is False
    assert test.augment is False
    assert validation.target_size == DEFAULT_OUTPUT_SIZE
    assert test.target_size == DEFAULT_OUTPUT_SIZE


def test_train_profile_enables_bounded_augmentation():
    profile = create_preprocessing_profile("train", seed=11)

    assert profile.augment is True
    assert profile.max_rotation_degrees <= 5
    assert profile.max_shift_ratio <= 0.05
    assert profile.noise_std <= 4


def test_preprocess_pil_image_keeps_rgb_384_output_for_all_splits():
    image = _gradient_image()

    for split in ("train", "validation", "test"):
        result = preprocess_pil_image(image, split=split, seed=5)
        pixels = np.asarray(result.image)
        assert result.image.size == DEFAULT_OUTPUT_SIZE
        assert result.image.mode == "RGB"
        assert pixels.shape == (384, 384, 3)
        assert pixels.dtype == np.uint8


def test_preprocess_pil_image_returns_normalized_chw_float_for_all_splits():
    image = _gradient_image()

    for split in ("train", "validation", "test"):
        result = preprocess_pil_image(image, split=split, seed=5)
        normalized = result.normalized

        assert normalized.shape == (3, 384, 384)
        assert normalized.dtype == np.float32
        assert -3.0 < float(normalized.min()) < 0.0
        assert 0.0 < float(normalized.max()) < 3.0


def test_validation_profile_repeats_exact_pixels():
    image = _gradient_image()

    first = preprocess_pil_image(image, split="validation")
    second = preprocess_pil_image(image, split="validation")

    assert np.array_equal(np.asarray(first.image), np.asarray(second.image))
    assert np.array_equal(first.normalized, second.normalized)


def test_roi_preprocessing_returns_normalized_output_without_removing_dark_regions():
    image = Image.new("RGB", (200, 160), "white")
    image.paste((0, 0, 0), (80, 60, 120, 100))
    request = RoiCropRequest(
        image_id="dark",
        image=image,
        annotation_bbox=[50, 40, 100, 80],
    )

    result = preprocess_image(request, split="validation")

    assert result.normalized.shape == (3, 384, 384)
    assert result.crop_result is not None
    assert result.crop_result.preserved_dark_regions is True
    assert np.asarray(result.image).min() < 20


def test_classifier_config_uses_spec_004_image_size():
    with open("configs/classifier.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["classifier"]["image_size"] == 384
