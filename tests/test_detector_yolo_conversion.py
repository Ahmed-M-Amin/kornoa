import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.data.coco_parser import load_coco_annotations
from src.data.yolo_converter import (
    audit_detector_annotations,
    build_category_mapping,
    create_detector_visualizations,
    convert_coco_to_yolo_dataset,
)
from src.training.train_detector import DetectorConfigError, main as detector_main
from src.training.train_classifier import validate_generated_artifact_confidentiality


def _write_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(240, 240, 240)).save(path)


def _write_coco(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": "img_a", "file_name": "img_a.jpg", "width": 100, "height": 80},
                    {"id": "img_b", "file_name": "img_b.jpg", "width": 100, "height": 80},
                    {"id": "img_c", "file_name": "img_c.jpg", "width": 100, "height": 80},
                ],
                "annotations": [
                    {"id": 1, "image_id": "img_a", "category_id": 5, "bbox": [10, 8, 40, 20], "area": 800},
                    {"id": 2, "image_id": "img_a", "category_id": 9, "bbox": [50, 20, 20, 20], "area": 400},
                    {"id": 3, "image_id": "img_b", "category_id": 5, "bbox": [5, 5, 0, 10], "area": 0},
                    {"id": 4, "image_id": "img_missing", "category_id": 9, "bbox": [1, 1, 10, 10], "area": 100},
                ],
                "categories": [
                    {"id": 5, "name": "crack"},
                    {"id": 9, "name": "scratch"},
                ],
            }
        ),
        encoding="utf-8",
    )


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    image_dir = root / "train_images"
    for name in ("img_a.jpg", "img_b.jpg", "img_c.jpg"):
        _write_image(image_dir / name)
    annotation_path = root / "train_annotations.json"
    _write_coco(annotation_path)
    return annotation_path, image_dir


def _write_stratified_coco(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": "crack_1", "file_name": "crack_1.jpg", "width": 100, "height": 80},
                    {"id": "crack_2", "file_name": "crack_2.jpg", "width": 100, "height": 80},
                    {"id": "scratch_1", "file_name": "scratch_1.jpg", "width": 100, "height": 80},
                    {"id": "scratch_2", "file_name": "scratch_2.jpg", "width": 100, "height": 80},
                    {"id": "clean_1", "file_name": "clean_1.jpg", "width": 100, "height": 80},
                    {"id": "clean_2", "file_name": "clean_2.jpg", "width": 100, "height": 80},
                ],
                "annotations": [
                    {"id": 1, "image_id": "crack_1", "category_id": 5, "bbox": [10, 8, 40, 20]},
                    {"id": 2, "image_id": "crack_2", "category_id": 5, "bbox": [10, 8, 40, 20]},
                    {"id": 3, "image_id": "scratch_1", "category_id": 9, "bbox": [10, 8, 40, 20]},
                    {"id": 4, "image_id": "scratch_2", "category_id": 9, "bbox": [10, 8, 40, 20]},
                ],
                "categories": [
                    {"id": 5, "name": "crack"},
                    {"id": 9, "name": "scratch"},
                ],
            }
        ),
        encoding="utf-8",
    )


def _stratified_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "stratified_dataset"
    image_dir = root / "train_images"
    for name in ("crack_1", "crack_2", "scratch_1", "scratch_2", "clean_1", "clean_2"):
        _write_image(image_dir / f"{name}.jpg")
    annotation_path = root / "train_annotations.json"
    _write_stratified_coco(annotation_path)
    return annotation_path, image_dir


def test_category_mapping_and_detector_audit_report_invalid_records(tmp_path):
    annotation_path, image_dir = _dataset(tmp_path)
    coco = load_coco_annotations(annotation_path)

    mapping = build_category_mapping(coco)
    audit = audit_detector_annotations(coco, image_dir=image_dir)

    assert [(item.category_id, item.category_name, item.class_index) for item in mapping.items] == [
        (5, "crack", 0),
        (9, "scratch", 1),
    ]
    assert audit.annotation_count == 4
    assert audit.category_distribution == {"crack": 2, "scratch": 2}
    assert audit.images_without_annotations == ["img_c"]
    assert audit.images_with_multiple_defects == ["img_a"]
    assert audit.missing_image_references == ["img_missing"]
    assert [item.annotation_id for item in audit.invalid_boxes] == [3]
    assert audit.bbox_size_distribution["valid_count"] == 3


def test_yolo_dataset_conversion_creates_disjoint_split_labels_and_data_yaml(tmp_path):
    annotation_path, image_dir = _dataset(tmp_path)

    result = convert_coco_to_yolo_dataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        output_root=tmp_path / "outputs" / "detector",
        seed=17,
        val_fraction=0.5,
    )

    assert result.split.train_validation_disjoint is True
    assert set(result.split.train_image_ids).isdisjoint(result.split.validation_image_ids)
    assert result.data_yaml_path.exists()
    assert (result.dataset_root / "images" / "train").is_dir()
    assert (result.dataset_root / "images" / "val").is_dir()
    assert (result.dataset_root / "labels" / "train").is_dir()
    assert (result.dataset_root / "labels" / "val").is_dir()

    data_yaml = yaml.safe_load(result.data_yaml_path.read_text(encoding="utf-8"))
    assert data_yaml["nc"] == 2
    assert data_yaml["names"] == ["crack", "scratch"]

    label_text = "\n".join(path.read_text(encoding="utf-8") for path in result.label_files)
    assert "0 0.300000 0.225000 0.400000 0.250000" in label_text
    assert "1 0.600000 0.375000 0.200000 0.250000" in label_text
    assert result.conversion_report["skipped_invalid_annotations"] == 1
    assert result.conversion_report["skipped_missing_images"] == 1


def test_no_annotation_images_get_empty_same_stem_labels_and_report_counts(tmp_path):
    annotation_path, image_dir = _dataset(tmp_path)

    result = convert_coco_to_yolo_dataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        output_root=tmp_path / "outputs" / "detector",
        seed=17,
        val_fraction=0.5,
    )

    no_annotation_image_ids = {"img_c"}
    split_by_image_id = {
        image_id: "train" for image_id in result.split.train_image_ids
    } | {
        image_id: "val" for image_id in result.split.validation_image_ids
    }

    for image_id in no_annotation_image_ids:
        split_name = split_by_image_id[image_id]
        image_path = result.dataset_root / "images" / split_name / f"{image_id}.jpg"
        label_path = result.dataset_root / "labels" / split_name / f"{image_id}.txt"

        assert image_path.exists()
        assert label_path.exists()
        assert label_path.stem == image_path.stem
        assert label_path.read_text(encoding="utf-8") == ""

    expected_train_count = sum(
        1 for image_id in no_annotation_image_ids if split_by_image_id[image_id] == "train"
    )
    expected_val_count = sum(
        1 for image_id in no_annotation_image_ids if split_by_image_id[image_id] == "val"
    )

    assert result.conversion_report["no_annotation_train_count"] == expected_train_count
    assert result.conversion_report["no_annotation_val_count"] == expected_val_count
    assert result.conversion_report["total_empty_label_files"] == 2


def test_repeated_conversion_cleans_only_generated_split_outputs(tmp_path):
    annotation_path, image_dir = _stratified_dataset(tmp_path)
    output_root = tmp_path / "outputs" / "detector"

    first = convert_coco_to_yolo_dataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        output_root=output_root,
        seed=1,
        val_fraction=0.5,
    )
    stale_image = first.dataset_root / "images" / "train" / "stale.jpg"
    stale_label = first.dataset_root / "labels" / "val" / "stale.txt"
    stale_image.write_text("stale", encoding="utf-8")
    stale_label.write_text("stale", encoding="utf-8")

    second = convert_coco_to_yolo_dataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        output_root=output_root,
        seed=2,
        val_fraction=0.5,
    )

    train_images = {path.stem for path in (second.dataset_root / "images" / "train").glob("*.jpg")}
    val_images = {path.stem for path in (second.dataset_root / "images" / "val").glob("*.jpg")}
    train_labels = {path.stem for path in (second.dataset_root / "labels" / "train").glob("*.txt")}
    val_labels = {path.stem for path in (second.dataset_root / "labels" / "val").glob("*.txt")}

    assert train_images == set(second.split.train_image_ids)
    assert val_images == set(second.split.validation_image_ids)
    assert train_labels == set(second.split.train_image_ids)
    assert val_labels == set(second.split.validation_image_ids)
    assert not train_images.intersection(val_images)
    assert not train_labels.intersection(val_labels)
    assert second.conversion_report["output_dataset_cleaned"] is True
    assert second.conversion_report["cleaned_paths"] == [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val",
    ]


def test_split_stratifies_by_primary_defect_category_when_practical(tmp_path):
    annotation_path, image_dir = _stratified_dataset(tmp_path)

    result = convert_coco_to_yolo_dataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        output_root=tmp_path / "outputs" / "detector",
        seed=7,
        val_fraction=0.5,
    )

    assert result.split.split_strategy_used == "primary_defect_category"
    assert result.conversion_report["split_strategy_used"] == "primary_defect_category"
    assert result.split.stratification_fallback_reason == ""
    assert len(result.split.train_image_ids) == 3
    assert len(result.split.validation_image_ids) == 3
    assert any(image_id.startswith("clean_") for image_id in result.split.train_image_ids)
    assert any(image_id.startswith("clean_") for image_id in result.split.validation_image_ids)
    assert any(image_id.startswith("crack_") for image_id in result.split.train_image_ids)
    assert any(image_id.startswith("crack_") for image_id in result.split.validation_image_ids)
    assert any(image_id.startswith("scratch_") for image_id in result.split.train_image_ids)
    assert any(image_id.startswith("scratch_") for image_id in result.split.validation_image_ids)


def test_split_reports_deterministic_fallback_when_category_strata_are_too_rare(tmp_path):
    annotation_path, image_dir = _dataset(tmp_path)

    result = convert_coco_to_yolo_dataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        output_root=tmp_path / "outputs" / "detector",
        seed=17,
        val_fraction=0.5,
    )

    assert result.split.split_strategy_used == "deterministic_random"
    assert result.conversion_report["split_strategy_used"] == "deterministic_random"
    assert "too few images per primary defect category" in result.split.stratification_fallback_reason


def test_detector_visualization_saves_category_colored_overlay(tmp_path):
    annotation_path, image_dir = _dataset(tmp_path)
    conversion = convert_coco_to_yolo_dataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        output_root=tmp_path / "outputs" / "detector",
        seed=3,
        val_fraction=0.5,
    )

    report = create_detector_visualizations(
        annotation_path=annotation_path,
        image_dir=image_dir,
        output_dir=tmp_path / "outputs" / "detector" / "figures",
        max_samples=1,
    )

    assert len(report.figure_paths) == 1
    assert report.figure_paths[0].exists()
    assert report.figure_paths[0].stat().st_size > 0
    assert report.mask_visualizations_available is False
    assert conversion.dataset_root.exists()


def test_train_detector_cli_audit_convert_visualize_and_dry_run_commands(tmp_path):
    annotation_path, image_dir = _dataset(tmp_path)
    output_root = tmp_path / "outputs" / "detector"
    config = tmp_path / "detector.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "detector": {
                    "annotation_path": str(annotation_path),
                    "train_images": str(image_dir),
                    "output_root": str(output_root),
                    "seed": 11,
                    "val_fraction": 0.5,
                    "image_size": 640,
                }
            }
        ),
        encoding="utf-8",
    )

    assert detector_main(["audit", "--config", str(config)]) == 0
    assert detector_main(["convert", "--config", str(config)]) == 0
    assert detector_main(["visualize", "--config", str(config), "--max-samples", "1"]) == 0
    assert detector_main(["train", "--config", str(config), "--model", "yolo11n", "--dry-run"]) == 0
    assert detector_main(["evaluate", "--config", str(config), "--model-path", str(output_root / "models" / "best.pt"), "--dry-run"]) == 0

    assert (output_root / "reports" / "annotation_audit.json").exists()
    assert (output_root / "dataset" / "data.yaml").exists()
    assert (output_root / "reports" / "visualization_report.json").exists()
    assert (output_root / "reports" / "detector_training.json").exists()
    assert (output_root / "reports" / "detector_evaluation.json").exists()

    evaluation_report = yaml.safe_load((output_root / "reports" / "detector_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation_report["evaluation_mode"] == "dry_run"
    assert evaluation_report["per_category_metrics_available"] is False
    assert evaluation_report["validation_prediction_examples_available"] is False


def test_detector_config_rejects_test_images_for_training_source(tmp_path):
    annotation_path, image_dir = _dataset(tmp_path)
    test_image_dir = image_dir.parent / "test_images"
    test_image_dir.mkdir()
    config = tmp_path / "detector.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "detector": {
                    "annotation_path": str(annotation_path),
                    "train_images": str(test_image_dir),
                    "output_root": str(tmp_path / "outputs" / "detector"),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DetectorConfigError, match="test_images"):
        detector_main(["audit", "--config", str(config)])


def test_detector_generated_outputs_are_ignored():
    paths = [
        Path("outputs/detector/dataset/data.yaml"),
        Path("outputs/detector/reports/annotation_audit.json"),
        Path("outputs/detector/figures/sample.png"),
        Path("outputs/detector/models/best.pt"),
    ]

    confidentiality = validate_generated_artifact_confidentiality(paths)

    assert sorted(confidentiality.ignored_paths) == sorted(paths)
    assert confidentiality.tracked_paths == []
