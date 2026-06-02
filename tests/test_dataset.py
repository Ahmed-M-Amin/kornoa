import csv
from pathlib import Path

import pytest

from src.data.dataset import (
    DatasetValidationError,
    audit_dataset,
    discover_images,
    load_train_labels,
    resolve_dataset_paths,
    validate_required_paths,
)
from src.utils.config import ConfigError, load_config


FIXTURE_ROOT = Path("tests/fixtures/synthetic_dataset")


def test_config_loader_reads_valid_paths_config():
    config = load_config("configs/paths.yaml")

    assert Path(config["dataset"]["root"]) == FIXTURE_ROOT


def test_config_loader_reports_missing_and_malformed_configs(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        load_config(tmp_path / "missing.yaml")

    malformed = tmp_path / "bad.yaml"
    malformed.write_text("dataset: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="Failed to parse YAML"):
        load_config(malformed)


def test_dataset_paths_resolve_required_items_from_config():
    paths = resolve_dataset_paths({"dataset": {"root": str(FIXTURE_ROOT)}})

    assert paths.root == FIXTURE_ROOT
    assert paths.train_csv == FIXTURE_ROOT / "train.csv"
    assert paths.sample_submission == FIXTURE_ROOT / "sample_submission.csv"
    assert paths.train_images == FIXTURE_ROOT / "train_images"
    assert paths.test_images == FIXTURE_ROOT / "test_images"


def test_required_path_validation_reports_exact_missing_items(tmp_path):
    paths = resolve_dataset_paths(tmp_path)

    with pytest.raises(DatasetValidationError) as exc_info:
        validate_required_paths(paths)

    assert exc_info.value.missing_items == [
        "train.csv",
        "sample_submission.csv",
        "train_images",
        "test_images",
    ]


def test_train_csv_loading_validates_binary_targets(tmp_path):
    rows = load_train_labels(FIXTURE_ROOT / "train.csv")

    assert [row["target"] for row in rows] == ["1", "0", "1", "0"]

    official_csv = tmp_path / "official_train.csv"
    with official_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "target"])
        writer.writeheader()
        writer.writerow({"image_id": "img_001.png", "target": "1"})
        writer.writerow({"image_id": "img_002.png", "target": "0"})

    official_rows = load_train_labels(official_csv)

    assert official_rows == [
        {"image_id": "img_001.png", "target": "1"},
        {"image_id": "img_002.png", "target": "0"},
    ]

    invalid_csv = tmp_path / "train.csv"
    with invalid_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "label"])
        writer.writeheader()
        writer.writerow({"image_id": "img_bad", "label": "2"})

    with pytest.raises(DatasetValidationError, match="binary"):
        load_train_labels(invalid_csv)


def test_train_csv_requires_image_id_and_target_or_label(tmp_path):
    missing_target_csv = tmp_path / "train.csv"
    missing_target_csv.write_text("image_id,status\nimg_001,1\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="image_id and one of target/label columns"):
        load_train_labels(missing_target_csv)


def test_train_and_test_image_discovery_uses_supported_suffixes():
    train_images = discover_images(FIXTURE_ROOT / "train_images")
    test_images = discover_images(FIXTURE_ROOT / "test_images")

    assert [image.image_id for image in train_images] == [
        "img_001.jpg",
        "img_002.jpg",
        "img_003.jpg",
        "img_004.jpg",
    ]
    assert [image.image_id for image in test_images] == ["img_005.jpg", "img_006.jpg"]


def test_dataset_audit_matches_real_style_full_png_filenames(tmp_path):
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "train_images"
    test_dir = dataset_root / "test_images"
    train_dir.mkdir(parents=True)
    test_dir.mkdir()
    image_id = "ad0f5a12-93fe-4ebb-9aae-ef12ef5246f8_000000000001.png"
    (dataset_root / "sample_submission.csv").write_text("image_id,target\nimg_005.png,0\n", encoding="utf-8")
    (dataset_root / "train.csv").write_text(f"image_id,target\n {image_id} ,1\n", encoding="utf-8")
    (train_dir / image_id).write_bytes(b"placeholder")
    (test_dir / "img_005.png").write_bytes(b"placeholder")

    report = audit_dataset(dataset_root)

    assert report.train_rows == [{"image_id": image_id, "target": "1"}]
    assert report.train_images[0].image_id == image_id
    assert report.train_images[0].path == train_dir / image_id
    assert report.missing_train_images == []
    assert report.unreferenced_train_images == []


def test_dataset_audit_matches_labels_to_images_and_reports_unreferenced(tmp_path):
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "train_images"
    test_dir = dataset_root / "test_images"
    train_dir.mkdir(parents=True)
    test_dir.mkdir()
    (dataset_root / "sample_submission.csv").write_text(
        "image_id,label\nimg_005,0\n", encoding="utf-8"
    )
    (dataset_root / "train.csv").write_text(
        "image_id,target\nimg_001.jpg,1\nimg_missing.jpg,0\n", encoding="utf-8"
    )
    for image_name in ["img_001.jpg", "img_extra.jpg"]:
        (train_dir / image_name).write_bytes(b"placeholder")
    (test_dir / "img_005.jpg").write_bytes(b"placeholder")

    report = audit_dataset(dataset_root)

    assert report.missing_train_images == ["img_missing.jpg"]
    assert report.unreferenced_train_images == ["img_extra.jpg"]
