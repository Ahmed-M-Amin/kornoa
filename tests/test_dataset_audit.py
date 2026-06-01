import csv
from pathlib import Path

from PIL import Image

from src.data.audit import (
    run_dataset_audit,
    validate_generated_artifact_confidentiality,
)


FIXTURE_ROOT = Path("tests/fixtures/synthetic_dataset")


def read_csv_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_audit_only_annotation_coverage_and_roi_summary():
    result = run_dataset_audit(FIXTURE_ROOT, generate_outputs=False)

    assert result["annotation_coverage"]["annotation_image_count"] == 2
    assert result["annotation_coverage"]["annotated_image_count"] == 1
    assert result["annotation_coverage"]["unannotated_image_count"] == 1
    assert result["annotation_coverage"]["coverage_ratio"] == 0.5
    assert result["roi_availability"]["images_with_roi"] == 1
    assert result["roi_availability"]["images_without_roi"] == 1
    assert result["roi_availability"]["roi_ratio"] == 0.5


def test_orphaned_annotations_are_reported_without_training_labels():
    result = run_dataset_audit(FIXTURE_ROOT, generate_outputs=False)

    assert result["orphaned_annotations"] == [
        {
            "annotation_id": 2,
            "image_id": "img_missing",
            "category_name": "scratch",
            "has_roi": True,
        }
    ]


def test_image_size_summary_uses_synthetic_known_dimension_images():
    result = run_dataset_audit(FIXTURE_ROOT, generate_outputs=False)

    assert len(result["image_size_summary"]) == 6
    for row in result["image_size_summary"]:
        assert row["width"] == 100
        assert row["height"] == 100
        with Image.open(row["path"]) as image:
            assert image.size == (100, 100)


def test_dataset_summary_csv_generation(tmp_path):
    output_root = tmp_path / "outputs"

    result = run_dataset_audit(FIXTURE_ROOT, output_root=output_root)

    summary_rows = read_csv_rows(output_root / "reports" / "dataset_summary.csv")
    summary = {row["metric"]: row["value"] for row in summary_rows}
    assert summary["train_label_count"] == str(result["train_label_count"])
    assert summary["train_image_count"] == "4"
    assert summary["test_image_count"] == "2"
    assert summary["orphaned_annotation_count"] == "1"
    assert "runtime_seconds" in summary


def test_class_and_defect_distribution_outputs(tmp_path):
    output_root = tmp_path / "outputs"

    run_dataset_audit(FIXTURE_ROOT, output_root=output_root)

    class_rows = read_csv_rows(output_root / "reports" / "class_distribution.csv")
    defect_rows = read_csv_rows(output_root / "reports" / "defect_distribution.csv")

    assert {row["label"]: row["count"] for row in class_rows} == {"0": "2", "1": "2"}
    assert {row["defect_label"]: row["count"] for row in defect_rows} == {
        "defect": "1"
    }


def test_audit_figure_outputs(tmp_path):
    output_root = tmp_path / "outputs"

    run_dataset_audit(FIXTURE_ROOT, output_root=output_root)

    for figure_name in [
        "class_distribution.png",
        "defect_distribution.png",
        "sample_grid.png",
    ]:
        figure_path = output_root / "figures" / figure_name
        assert figure_path.exists()
        assert figure_path.stat().st_size > 0
        with Image.open(figure_path) as image:
            assert image.width > 0
            assert image.height > 0


def test_generated_artifact_confidentiality_for_reports_and_figures():
    output_root = Path("outputs")

    run_dataset_audit(FIXTURE_ROOT, output_root=output_root)

    artifact_paths = [
        output_root / "reports" / "dataset_summary.csv",
        output_root / "figures" / "class_distribution.png",
    ]
    confidentiality = validate_generated_artifact_confidentiality(artifact_paths)

    for artifact_path in artifact_paths:
        status = confidentiality[str(artifact_path)]
        assert status["ignored"] is True
        assert status["tracked"] is False
