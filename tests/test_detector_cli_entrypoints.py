import json
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(240, 240, 240)).save(path)


def _write_tiny_coco(root: Path) -> tuple[Path, Path]:
    image_dir = root / "train_images"
    for image_id in ("defect_a", "defect_b", "clean_a", "clean_b"):
        _write_image(image_dir / f"{image_id}.jpg")
    annotation_path = root / "train_annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": "defect_a", "file_name": "defect_a.jpg", "width": 100, "height": 80},
                    {"id": "defect_b", "file_name": "defect_b.jpg", "width": 100, "height": 80},
                    {"id": "clean_a", "file_name": "clean_a.jpg", "width": 100, "height": 80},
                    {"id": "clean_b", "file_name": "clean_b.jpg", "width": 100, "height": 80},
                ],
                "annotations": [
                    {"id": 1, "image_id": "defect_a", "category_id": 7, "bbox": [10, 8, 40, 20]},
                    {"id": 2, "image_id": "defect_b", "category_id": 7, "bbox": [20, 18, 20, 20]},
                ],
                "categories": [{"id": 7, "name": "scratch"}],
            }
        ),
        encoding="utf-8",
    )
    return annotation_path, image_dir


def test_spec008_module_help_entrypoints_print_useful_output():
    commands = [
        ("src.data.yolo_converter", "--help"),
        ("src.data.yolo_converter", "audit", "--help"),
        ("src.data.yolo_converter", "convert", "--help"),
        ("src.data.yolo_converter", "visualize", "--help"),
        ("src.training.train_detector", "--help"),
        ("src.training.train_detector", "train", "--help"),
        ("src.training.train_detector", "evaluate", "--help"),
    ]

    for command in commands:
        result = _run_module(*command)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip(), command

    yolo_help = _run_module("src.data.yolo_converter", "--help").stdout
    assert "audit" in yolo_help
    assert "convert" in yolo_help
    assert "visualize" in yolo_help

    detector_help = _run_module("src.training.train_detector", "--help").stdout
    assert "train" in detector_help
    assert "evaluate" in detector_help


def test_yolo_converter_audit_and_convert_subprocess_create_detector_outputs(tmp_path):
    annotation_path, image_dir = _write_tiny_coco(tmp_path / "dataset")
    output_root = tmp_path / "outputs" / "detector"

    audit = _run_module(
        "src.data.yolo_converter",
        "audit",
        "--annotation-path",
        str(annotation_path),
        "--image-dir",
        str(image_dir),
        "--output-root",
        str(output_root),
    )
    assert audit.returncode == 0, audit.stderr
    assert (output_root / "reports" / "annotation_audit.json").exists()

    convert = _run_module(
        "src.data.yolo_converter",
        "convert",
        "--annotation-path",
        str(annotation_path),
        "--image-dir",
        str(image_dir),
        "--output-root",
        str(output_root),
        "--seed",
        "7",
        "--val-fraction",
        "0.5",
    )
    assert convert.returncode == 0, convert.stderr

    dataset_root = output_root / "dataset"
    assert (dataset_root / "images" / "train").is_dir()
    assert (dataset_root / "images" / "val").is_dir()
    assert (dataset_root / "labels" / "train").is_dir()
    assert (dataset_root / "labels" / "val").is_dir()
    assert (dataset_root / "data.yaml").exists()

    data_yaml = yaml.safe_load((dataset_root / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["names"] == ["scratch"]

    image_files = list((dataset_root / "images").glob("*/*.jpg"))
    label_files = list((dataset_root / "labels").glob("*/*.txt"))
    assert {path.stem for path in image_files} == {path.stem for path in label_files}

    for clean_id in ("clean_a", "clean_b"):
        label_path = next(path for path in label_files if path.stem == clean_id)
        assert label_path.read_text(encoding="utf-8") == ""

    train_images = {path.name for path in (dataset_root / "images" / "train").glob("*.jpg")}
    val_images = {path.name for path in (dataset_root / "images" / "val").glob("*.jpg")}
    assert train_images.isdisjoint(val_images)


def test_yolo_converter_repeated_convert_removes_stale_generated_outputs(tmp_path):
    annotation_path, image_dir = _write_tiny_coco(tmp_path / "dataset")
    output_root = tmp_path / "outputs" / "detector"

    for seed in ("1", "2"):
        result = _run_module(
            "src.data.yolo_converter",
            "convert",
            "--annotation-path",
            str(annotation_path),
            "--image-dir",
            str(image_dir),
            "--output-root",
            str(output_root),
            "--seed",
            seed,
            "--val-fraction",
            "0.5",
        )
        assert result.returncode == 0, result.stderr
        if seed == "1":
            (output_root / "dataset" / "images" / "train" / "stale.jpg").write_text("stale", encoding="utf-8")
            (output_root / "dataset" / "labels" / "val" / "stale.txt").write_text("stale", encoding="utf-8")

    dataset_root = output_root / "dataset"
    assert not list(dataset_root.glob("images/*/stale.jpg"))
    assert not list(dataset_root.glob("labels/*/stale.txt"))

    report = json.loads((output_root / "reports" / "yolo_conversion_report.json").read_text(encoding="utf-8"))
    assert report["output_dataset_cleaned"] is True
    assert report["cleaned_paths"] == [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val",
    ]
