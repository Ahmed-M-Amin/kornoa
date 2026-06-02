"""Tests for SPEC-007 V2 safe augmentation."""

import numpy as np
from PIL import Image

from src.data.transforms import apply_preprocessing_transforms, create_preprocessing_profile


def test_v2_safe_training_profile_has_stronger_bounded_augmentation():
    profile = create_preprocessing_profile("train", augmentation_recipe="v2_safe", seed=42)

    assert profile.augment is True
    assert profile.brightness_delta <= 0.2
    assert profile.contrast_delta <= 0.2
    assert profile.gamma_delta <= 0.2
    assert profile.blur_probability <= 0.3
    assert profile.noise_std <= 6.0
    assert profile.max_rotation_degrees <= 8.0
    assert profile.max_shift_ratio <= 0.06


def test_v2_validation_and_test_preprocessing_are_deterministic():
    image = Image.new("RGB", (32, 32), (120, 90, 60))
    validation_profile = create_preprocessing_profile("validation", augmentation_recipe="v2_safe", seed=1)
    test_profile = create_preprocessing_profile("test", augmentation_recipe="v2_safe", seed=999)

    validation_a = np.asarray(apply_preprocessing_transforms(image, validation_profile))
    validation_b = np.asarray(apply_preprocessing_transforms(image, validation_profile))
    test_a = np.asarray(apply_preprocessing_transforms(image, test_profile))
    test_b = np.asarray(apply_preprocessing_transforms(image, test_profile))

    assert np.array_equal(validation_a, validation_b)
    assert np.array_equal(test_a, test_b)


def test_v2_safe_train_augmentation_preserves_image_size_and_rgb():
    image = Image.new("RGB", (64, 48), (180, 180, 180))
    profile = create_preprocessing_profile("train", augmentation_recipe="v2_safe", seed=10)

    augmented = apply_preprocessing_transforms(image, profile)

    assert augmented.size == profile.target_size
    assert augmented.mode == "RGB"
