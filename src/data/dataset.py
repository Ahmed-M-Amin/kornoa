"""Dataset loading and validation helpers for the Krones Vision AI project."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class DatasetValidationError(Exception):
    """Raised when required dataset structure or labels are invalid."""

    def __init__(self, message: str, *, missing_items: list[str] | None = None):
        super().__init__(message)
        self.missing_items = missing_items or []


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved paths for the competition dataset layout."""

    root: Path
    train_csv: Path
    sample_submission: Path
    train_images: Path
    test_images: Path
    train_annotations: Path | None = None


@dataclass(frozen=True)
class ImageFile:
    """Discovered image file metadata."""

    image_id: str
    file_name: str
    path: Path


@dataclass
class DatasetValidationReport:
    """Integrity report for labels and image files."""

    paths: DatasetPaths
    train_rows: list[dict[str, str]] = field(default_factory=list)
    train_images: list[ImageFile] = field(default_factory=list)
    test_images: list[ImageFile] = field(default_factory=list)
    missing_train_images: list[str] = field(default_factory=list)
    unreferenced_train_images: list[str] = field(default_factory=list)


def resolve_dataset_paths(config: dict[str, Any] | str | Path) -> DatasetPaths:
    """Resolve dataset paths from a config dictionary or direct root path."""
    if isinstance(config, (str, Path)):
        dataset_config: dict[str, Any] = {"root": str(config)}
    else:
        dataset_config = config.get("dataset", config)

    root = Path(dataset_config.get("root", ".")).expanduser()
    return DatasetPaths(
        root=root,
        train_csv=root / dataset_config.get("train_csv", "train.csv"),
        sample_submission=root
        / dataset_config.get("sample_submission", "sample_submission.csv"),
        train_images=root / dataset_config.get("train_images", "train_images"),
        test_images=root / dataset_config.get("test_images", "test_images"),
        train_annotations=(
            root / dataset_config["train_annotations"]
            if dataset_config.get("train_annotations")
            else None
        ),
    )


def validate_required_paths(paths: DatasetPaths) -> None:
    """Validate that required files and folders exist."""
    required = {
        "dataset root": paths.root,
        "train.csv": paths.train_csv,
        "sample_submission.csv": paths.sample_submission,
        "train_images": paths.train_images,
        "test_images": paths.test_images,
    }
    if paths.train_annotations is not None:
        required["train_annotations.json"] = paths.train_annotations

    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise DatasetValidationError(
            "Missing required dataset items: " + ", ".join(missing),
            missing_items=missing,
        )


def load_train_labels(train_csv_path: str | Path) -> list[dict[str, str]]:
    """Load train.csv and validate binary labels."""
    train_csv_path = Path(train_csv_path)
    with train_csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise DatasetValidationError("train.csv contains no rows")
    fieldnames = set(rows[0].keys())
    if "image_id" not in fieldnames or not ({"target", "label"} & fieldnames):
        raise DatasetValidationError("train.csv must contain image_id and one of target/label columns")

    target_column = "target" if "target" in fieldnames else "label"
    normalized_rows = [
        {
            "image_id": row.get("image_id", "").strip(),
            "target": row.get(target_column, "").strip(),
        }
        for row in rows
    ]

    invalid = [row for row in normalized_rows if row["target"] not in {"0", "1"}]
    if invalid:
        invalid_ids = [row.get("image_id", "") for row in invalid]
        raise DatasetValidationError(
            "train.csv labels must be binary values 0 or 1: " + ", ".join(invalid_ids)
        )

    return normalized_rows


def discover_images(image_dir: str | Path) -> list[ImageFile]:
    """Discover supported image files in a directory."""
    image_dir = Path(image_dir)
    return [
        ImageFile(image_id=path.name, file_name=path.name, path=path)
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def audit_dataset(config: dict[str, Any] | str | Path) -> DatasetValidationReport:
    """Validate required paths and report label-to-image consistency."""
    paths = resolve_dataset_paths(config)
    validate_required_paths(paths)

    train_rows = load_train_labels(paths.train_csv)
    train_images = discover_images(paths.train_images)
    test_images = discover_images(paths.test_images)

    label_ids = {row["image_id"] for row in train_rows}
    train_image_ids = {image.image_id for image in train_images}

    return DatasetValidationReport(
        paths=paths,
        train_rows=train_rows,
        train_images=train_images,
        test_images=test_images,
        missing_train_images=sorted(label_ids - train_image_ids),
        unreferenced_train_images=sorted(train_image_ids - label_ids),
    )
