"""Tests for COCO annotation parser (SPEC-003)."""

import json
from pathlib import Path

import pytest

from src.data.coco_parser import (
    CocoDiagnostic,
    CocoParseError,
    CocoParseResult,
    load_coco_annotations,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "synthetic_dataset"
COCO_CASES = FIXTURE_ROOT / "coco_cases"
VALID_ANNOTATIONS = FIXTURE_ROOT / "train_annotations.json"


# ---------------------------------------------------------------------------
# Phase 3 - User Story 1
# ---------------------------------------------------------------------------


def test_load_valid_coco_parsing():
    """T011: Valid COCO parsing produces correct counts and mappings."""
    result = load_coco_annotations(VALID_ANNOTATIONS, validate_relationships=False)

    assert result.summary.success
    assert result.summary.image_count == 2
    assert result.summary.annotation_count == 2
    assert result.summary.category_count == 2

    image_ids = {img.image_id for img in result.images}
    assert image_ids == {"img_001", "img_002"}

    assert "img_001" in result.image_to_annotations
    assert len(result.image_to_annotations["img_001"]) == 1
    assert "img_missing" not in image_ids
    assert result.summary.images_with_annotations == 1
    assert result.summary.images_without_annotations == 1


def test_missing_required_collection_diagnostic():
    """T012: Missing required collection produces readable diagnostic."""
    missing_file = COCO_CASES / "missing_collections.json"
    result = load_coco_annotations(missing_file, validate_relationships=False)

    assert not result.summary.success
    codes = [d.code for d in result.summary.diagnostics]
    assert "missing_collection" in codes
    messages = " ".join(d.message for d in result.summary.diagnostics)
    assert "images" in messages or "annotations" in messages or "categories" in messages


def test_malformed_json_diagnostic():
    """T013: Malformed JSON produces readable diagnostic."""
    malformed_file = COCO_CASES / "malformed.json"
    with pytest.raises(CocoParseError) as excinfo:
        load_coco_annotations(malformed_file)
    assert "malformed" in str(excinfo.value).lower() or "json" in str(excinfo.value).lower()


def test_missing_file_diagnostic():
    """T014: Missing annotation file produces readable diagnostic naming the file."""
    missing_path = FIXTURE_ROOT / "train_annotations_MISSING.json"
    with pytest.raises(CocoParseError) as excinfo:
        load_coco_annotations(missing_path)
    assert "train_annotations_MISSING.json" in str(excinfo.value)
