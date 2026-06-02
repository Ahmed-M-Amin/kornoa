"""ROI cropping utilities for SPEC-004.

This module prepares image crops only. It does not train models or generate
predictions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


DEFAULT_OUTPUT_SIZE: Tuple[int, int] = (384, 384)
ROI_MARGIN_RATIO = 0.10
MIN_ROI_AREA_RATIO = 0.10
DEFAULT_SAMPLE_DIR = Path("outputs/figures/roi_samples")

BBox = Tuple[float, float, float, float]
CropBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class RoiCropRequest:
    """Input for ROI crop generation."""

    image_id: str
    image_path: Optional[Path | str] = None
    image: Optional[Image.Image] = None
    split: str = "train"
    annotation_bbox: Optional[Sequence[float]] = None
    target_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE


@dataclass(frozen=True)
class RoiCropResult:
    """Result of one ROI crop or fallback resize."""

    image_id: str
    image: Image.Image
    method: str
    crop_box: CropBox
    output_size: Tuple[int, int]
    fallback_reason: Optional[str] = None
    preserved_dark_regions: bool = False


@dataclass(frozen=True)
class PreprocessingProfile:
    """Split-specific preprocessing parameters."""

    split: str
    target_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE
    augment: bool = False
    seed: Optional[int] = None
    max_rotation_degrees: float = 0.0
    max_shift_ratio: float = 0.0
    brightness_delta: float = 0.0
    contrast_delta: float = 0.0
    gamma_delta: float = 0.0
    blur_probability: float = 0.0
    noise_std: float = 0.0
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class GeneratedOutputConfidentiality:
    """Git ignore/tracking status for generated files."""

    ignored_paths: list[Path] = field(default_factory=list)
    tracked_paths: list[Path] = field(default_factory=list)
    unchecked_paths: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class RoiSampleReport:
    """Review sample output metadata."""

    sample_dir: Path
    sample_paths: list[Path]
    crop_methods: Mapping[str, str]
    source_split: str = "synthetic"
    confidentiality: Optional[GeneratedOutputConfidentiality] = None


class RoiImageError(ValueError):
    """Raised when an image cannot be loaded for ROI preprocessing."""


def load_image(image_or_path: Image.Image | Path | str) -> Image.Image:
    """Load an image and normalize it to RGB."""

    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")

    path = Path(image_or_path)
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except FileNotFoundError as exc:
        raise RoiImageError(f"Image file does not exist: {path}") from exc
    except UnidentifiedImageError as exc:
        raise RoiImageError(f"Image file is not a readable image: {path}") from exc
    except OSError as exc:
        raise RoiImageError(f"Unable to read image file: {path}") from exc


def crop_roi(request: RoiCropRequest) -> RoiCropResult:
    """Create an annotation ROI crop, square center fallback, or full resize."""

    source = _load_request_image(request)
    width, height = source.size
    bbox = _normalize_bbox(request.annotation_bbox)
    fallback_reason: Optional[str] = None
    method = "annotation"

    if bbox is None:
        crop_box = _square_center_crop_box(width, height)
        method = "square_center"
        fallback_reason = "missing_annotation_roi"
    elif not _is_bbox_safe(bbox, width, height):
        crop_box = _square_center_crop_box(width, height)
        method = "square_center"
        fallback_reason = "annotation_roi_too_small"
    else:
        crop_box = _expanded_bbox_to_crop_box(bbox, width, height)

    if not _is_center_crop_safe(crop_box):
        crop_box = (0, 0, width, height)
        method = "full_resize"
        fallback_reason = "unsafe_center_crop"

    cropped = source.crop(crop_box)
    resized = cropped.resize(request.target_size, Image.Resampling.BILINEAR).convert("RGB")
    return RoiCropResult(
        image_id=str(request.image_id),
        image=resized,
        method=method,
        crop_box=crop_box,
        output_size=request.target_size,
        fallback_reason=fallback_reason,
        preserved_dark_regions=_contains_dark_pixels(resized),
    )


def save_roi_samples(
    requests: Iterable[RoiCropRequest],
    output_dir: Path | str = DEFAULT_SAMPLE_DIR,
) -> RoiSampleReport:
    """Save reviewable ROI crops for synthetic/manual inspection."""

    sample_dir = Path(output_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_paths: list[Path] = []
    crop_methods: dict[str, str] = {}

    for index, request in enumerate(requests):
        result = crop_roi(request)
        safe_id = _safe_name(result.image_id)
        path = sample_dir / f"{index:03d}_{safe_id}_{result.method}.png"
        result.image.save(path)
        sample_paths.append(path)
        crop_methods[result.image_id] = result.method

    confidentiality = validate_generated_output_confidentiality(sample_paths)
    return RoiSampleReport(
        sample_dir=sample_dir,
        sample_paths=sample_paths,
        crop_methods=crop_methods,
        confidentiality=confidentiality,
    )


def validate_generated_output_confidentiality(paths: Iterable[Path | str]) -> GeneratedOutputConfidentiality:
    """Return whether generated paths are ignored and untracked by git."""

    ignored: list[Path] = []
    tracked: list[Path] = []
    unchecked: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path)
        git_path = _normalize_git_path(path)
        if _git_path_is_tracked(git_path):
            tracked.append(path)
        if _git_path_is_ignored(git_path):
            ignored.append(path)
        else:
            unchecked.append(path)

    return GeneratedOutputConfidentiality(
        ignored_paths=ignored,
        tracked_paths=tracked,
        unchecked_paths=unchecked,
    )


def _load_request_image(request: RoiCropRequest) -> Image.Image:
    if request.image is not None:
        return load_image(request.image)
    if request.image_path is not None:
        return load_image(request.image_path)
    raise RoiImageError(f"ROI request {request.image_id!r} has no image or image_path")


def _normalize_bbox(value: Optional[Sequence[float]]) -> Optional[BBox]:
    if value is None or len(value) != 4:
        return None
    try:
        x, y, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _is_bbox_safe(bbox: BBox, image_width: int, image_height: int) -> bool:
    x, y, width, height = bbox
    if image_width <= 0 or image_height <= 0:
        return False
    left = max(0.0, x)
    top = max(0.0, y)
    right = min(float(image_width), x + width)
    bottom = min(float(image_height), y + height)
    clipped_width = max(0.0, right - left)
    clipped_height = max(0.0, bottom - top)
    if clipped_width <= 0 or clipped_height <= 0:
        return False
    return (clipped_width * clipped_height) >= (image_width * image_height * MIN_ROI_AREA_RATIO)


def _expanded_bbox_to_crop_box(bbox: BBox, image_width: int, image_height: int) -> CropBox:
    x, y, width, height = bbox
    margin_x = width * ROI_MARGIN_RATIO
    margin_y = height * ROI_MARGIN_RATIO
    left = max(0, floor(x - margin_x))
    top = max(0, floor(y - margin_y))
    right = min(image_width, ceil(x + width + margin_x))
    bottom = min(image_height, ceil(y + height + margin_y))
    return left, top, right, bottom


def _square_center_crop_box(image_width: int, image_height: int) -> CropBox:
    side = min(image_width, image_height)
    left = max(0, (image_width - side) // 2)
    top = max(0, (image_height - side) // 2)
    return left, top, left + side, top + side


def _is_center_crop_safe(crop_box: CropBox) -> bool:
    left, top, right, bottom = crop_box
    return (right - left) > 1 and (bottom - top) > 1


def _contains_dark_pixels(image: Image.Image) -> bool:
    pixels = np.asarray(image.convert("RGB"))
    return bool(np.any(pixels < 30))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _normalize_git_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _run_git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], check=False, capture_output=True, text=True)


def _git_path_is_tracked(path: str) -> bool:
    result = _run_git(["ls-files", "--error-unmatch", path])
    return result.returncode == 0


def _git_path_is_ignored(path: str) -> bool:
    result = _run_git(["check-ignore", path])
    return result.returncode == 0


def _load_cli_requests(dataset_root: Path, limit: int) -> list[RoiCropRequest]:
    metadata_path = dataset_root / "roi_cases.json"
    if metadata_path.exists():
        cases = json.loads(metadata_path.read_text(encoding="utf-8")).get("cases", [])
        return [
            RoiCropRequest(
                image_id=str(case["image_id"]),
                image_path=dataset_root / case["file_name"],
                annotation_bbox=case.get("bbox"),
            )
            for case in cases[:limit]
        ]

    images = sorted(dataset_root.glob("*.jpg"))[:limit]
    return [
        RoiCropRequest(image_id=image_path.stem, image_path=image_path, annotation_bbox=None)
        for image_path in images
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic ROI review samples.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args(argv)

    requests = _load_cli_requests(args.dataset_root, args.limit)
    if not requests:
        parser.error(f"No synthetic image fixtures found under {args.dataset_root}")
    report = save_roi_samples(requests, output_dir=args.output_dir)
    print(f"Generated {len(report.sample_paths)} ROI samples in {report.sample_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
