"""Split-specific preprocessing transforms for SPEC-004."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from src.data.roi import DEFAULT_OUTPUT_SIZE, PreprocessingProfile


SUPPORTED_SPLITS = {"train", "validation", "test"}


def create_preprocessing_profile(
    split: str,
    *,
    target_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    seed: Optional[int] = None,
) -> PreprocessingProfile:
    """Create a split-specific preprocessing profile."""

    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported preprocessing split: {split}")

    if split == "train":
        return PreprocessingProfile(
            split=split,
            target_size=target_size,
            augment=True,
            seed=seed,
            max_rotation_degrees=5.0,
            max_shift_ratio=0.05,
            brightness_delta=0.12,
            contrast_delta=0.12,
            blur_probability=0.20,
            noise_std=4.0,
        )

    return PreprocessingProfile(split=split, target_size=target_size, augment=False, seed=seed)


def apply_preprocessing_transforms(
    image: Image.Image,
    profile: PreprocessingProfile,
) -> Image.Image:
    """Apply resize/normalization and optional bounded training augmentation."""

    result = image.convert("RGB").resize(profile.target_size, Image.Resampling.BILINEAR)
    if not profile.augment:
        return result

    rng = random.Random(profile.seed)
    result = _adjust_brightness_contrast(result, profile, rng)
    result = _rotate(result, profile, rng)
    result = _shift(result, profile, rng)
    result = _maybe_blur(result, profile, rng)
    result = _add_noise(result, profile, rng)
    return result.convert("RGB")


def normalize_image(image: Image.Image, profile: PreprocessingProfile) -> np.ndarray:
    """Return ImageNet-normalized CHW float32 data for model input."""

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mean = np.asarray(profile.normalize_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(profile.normalize_std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (pixels - mean) / std
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)


def _adjust_brightness_contrast(
    image: Image.Image,
    profile: PreprocessingProfile,
    rng: random.Random,
) -> Image.Image:
    brightness = 1.0 + rng.uniform(-profile.brightness_delta, profile.brightness_delta)
    contrast = 1.0 + rng.uniform(-profile.contrast_delta, profile.contrast_delta)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    return ImageEnhance.Contrast(image).enhance(contrast)


def _rotate(image: Image.Image, profile: PreprocessingProfile, rng: random.Random) -> Image.Image:
    degrees = rng.uniform(-profile.max_rotation_degrees, profile.max_rotation_degrees)
    return image.rotate(degrees, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))


def _shift(image: Image.Image, profile: PreprocessingProfile, rng: random.Random) -> Image.Image:
    max_x = int(image.size[0] * profile.max_shift_ratio)
    max_y = int(image.size[1] * profile.max_shift_ratio)
    if max_x == 0 and max_y == 0:
        return image
    offset_x = rng.randint(-max_x, max_x)
    offset_y = rng.randint(-max_y, max_y)
    shifted = ImageChops.offset(image, offset_x, offset_y)
    if offset_x > 0:
        shifted.paste((0, 0, 0), (0, 0, offset_x, image.size[1]))
    elif offset_x < 0:
        shifted.paste((0, 0, 0), (image.size[0] + offset_x, 0, image.size[0], image.size[1]))
    if offset_y > 0:
        shifted.paste((0, 0, 0), (0, 0, image.size[0], offset_y))
    elif offset_y < 0:
        shifted.paste((0, 0, 0), (0, image.size[1] + offset_y, image.size[0], image.size[1]))
    return shifted


def _maybe_blur(image: Image.Image, profile: PreprocessingProfile, rng: random.Random) -> Image.Image:
    if rng.random() > profile.blur_probability:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.6)))


def _add_noise(image: Image.Image, profile: PreprocessingProfile, rng: random.Random) -> Image.Image:
    if profile.noise_std <= 0:
        return image
    pixels = np.asarray(image).astype(np.int16)
    seed = rng.randint(0, 2**32 - 1)
    noise_rng = np.random.default_rng(seed)
    noise = noise_rng.normal(0, profile.noise_std, pixels.shape)
    noisy = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")
