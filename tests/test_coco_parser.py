from pathlib import Path

import pytest

from src.data.coco_parser import CocoParseError, load_coco_annotations


FIXTURE_ROOT = Path("tests/fixtures/synthetic_dataset")
COCO_CASES = FIXTURE_ROOT / "coco_cases"


def diagnostic_codes(result):
    return [diagnostic.code for diagnostic in result.summary.diagnostics]


def test_valid_coco_parsing_counts_and_known_image_mapping_are_exact():
    result = load_coco_annotations(FIXTURE_ROOT / "train_annotations.json")

    assert result.summary.image_count == 2
    assert result.summary.annotation_count == 2
    assert result.summary.category_count == 2
    assert result.summary.orphaned_annotations == 1
    assert result.summary.images_with_annotations == 1
    assert result.summary.images_without_annotations == 1
    assert set(result.image_to_annotations) == {"img_001"}


def test_missing_required_collections_emit_exact_diagnostics():
    result = load_coco_annotations(COCO_CASES / "missing_collections.json")

    assert result.summary.success is False
    assert diagnostic_codes(result) == ["missing_collection"]
    assert [d.context["collection"] for d in result.summary.diagnostics] == ["images"]


def test_malformed_json_and_missing_file_raise_readable_errors(tmp_path):
    with pytest.raises(CocoParseError, match="Malformed JSON"):
        load_coco_annotations(COCO_CASES / "malformed.json")

    missing = tmp_path / "missing_train_annotations.json"
    with pytest.raises(CocoParseError, match=missing.name):
        load_coco_annotations(missing)


def test_relationship_diagnostics_are_exact():
    result = load_coco_annotations(COCO_CASES / "relationship_errors.json")

    assert diagnostic_codes(result) == [
        "orphaned_annotation",
        "unknown_category",
        "malformed_bbox",
    ]
    assert result.summary.orphaned_annotations == 1
    assert result.summary.unknown_categories == 1
    assert result.summary.malformed_bboxes == 1
    assert set(result.image_to_annotations) == {"img_001"}


def test_malformed_bbox_diagnostics_are_exact_for_non_positive_and_missing_boxes():
    result = load_coco_annotations(COCO_CASES / "malformed_boxes.json")

    assert diagnostic_codes(result) == [
        "malformed_bbox",
        "malformed_bbox",
        "malformed_bbox",
    ]
    assert result.summary.malformed_bboxes == 3


def test_duplicate_identifier_diagnostics_are_exact(tmp_path):
    duplicate_file = tmp_path / "duplicates.json"
    duplicate_file.write_text(
        """
        {
          "images": [
            {"id": "img_001", "file_name": "a.jpg", "width": 1, "height": 1},
            {"id": "img_001", "file_name": "b.jpg", "width": 1, "height": 1}
          ],
          "annotations": [
            {"id": 1, "image_id": "img_001", "category_id": 1, "bbox": [0, 0, 1, 1]},
            {"id": 1, "image_id": "img_001", "category_id": 1, "bbox": [0, 0, 1, 1]}
          ],
          "categories": [
            {"id": 1, "name": "defect"},
            {"id": 1, "name": "duplicate"}
          ]
        }
        """,
        encoding="utf-8",
    )

    result = load_coco_annotations(duplicate_file)

    assert diagnostic_codes(result) == [
        "duplicate_image_id",
        "duplicate_annotation_id",
        "duplicate_category_id",
    ]
    assert result.summary.duplicate_ids == {
        "images": 1,
        "annotations": 1,
        "categories": 1,
    }


def test_annotation_summaries_do_not_infer_test_labels_or_rasterize_masks():
    result = load_coco_annotations(FIXTURE_ROOT / "train_annotations.json")

    assert result.category_distribution() == {1: 1, 2: 1}
    assert [image.image_id for image in result.annotated_images()] == ["img_001"]
    assert [image.image_id for image in result.unannotated_images()] == ["img_002"]
    assert result.roi_available_count() == 2
    assert result.segmentation_available_count() == 0
    assert not hasattr(result, "test_labels")
