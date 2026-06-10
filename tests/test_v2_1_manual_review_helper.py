"""Tests for the V2.1 manual review helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image


REQUIRED_PREFILLED_COLUMNS = [
    "image_id",
    "error_type",
    "true_label",
    "v2b_prediction",
    "v2b_probability",
    "threshold_distance",
    "review_group",
    "visual_tag",
    "reviewer_note",
    "recommended_action",
    "suggested_visual_tag",
    "suggested_recommended_action",
    "final_visual_tag",
    "final_recommended_action",
]


def _write_manual_review_template(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "image_id": "fp_near.jpg",
                "error_type": "FP",
                "true_label": 0,
                "v2b_prediction": 1,
                "v2b_probability": 0.52,
                "threshold_distance": 0.02,
                "review_group": "near_threshold_false_positives",
                "visual_tag": "unknown",
                "reviewer_note": "",
                "recommended_action": "",
            },
            {
                "image_id": "fn_near.jpg",
                "error_type": "FN",
                "true_label": 1,
                "v2b_prediction": 0,
                "v2b_probability": 0.49,
                "threshold_distance": 0.01,
                "review_group": "near_threshold_false_negatives",
                "visual_tag": "unknown",
                "reviewer_note": "",
                "recommended_action": "",
            },
            {
                "image_id": "fp_far.jpg",
                "error_type": "FP",
                "true_label": 0,
                "v2b_prediction": 1,
                "v2b_probability": 0.88,
                "threshold_distance": 0.38,
                "review_group": "high_confidence_false_positives",
                "visual_tag": "unknown",
                "reviewer_note": "",
                "recommended_action": "",
            },
            {
                "image_id": "over_rejected.jpg",
                "error_type": "FP",
                "true_label": 0,
                "v2b_prediction": 1,
                "v2b_probability": 0.57,
                "threshold_distance": 0.07,
                "review_group": "over_rejected_reusable",
                "visual_tag": "unknown",
                "reviewer_note": "",
                "recommended_action": "",
            },
        ]
    ).to_csv(path, index=False)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    manual_review_template = tmp_path / "manual_review_template.csv"
    train_images = tmp_path / "train_images"
    output_root = tmp_path / "outputs" / "analysis" / "v2_1_error_intelligence" / "manual_review"
    train_images.mkdir(parents=True)
    _write_manual_review_template(manual_review_template)
    for image_id, color in {
        "fp_near.jpg": (200, 30, 30),
        "fn_near.jpg": (30, 200, 30),
        "fp_far.jpg": (30, 30, 200),
        "over_rejected.jpg": (200, 200, 40),
    }.items():
        Image.new("RGB", (80, 80), color).save(train_images / image_id)
    return {
        "manual_review_template": manual_review_template,
        "train_images": train_images,
        "output_root": output_root,
    }


def _write_config(tmp_path: Path, paths: dict[str, Path], **overrides: object) -> Path:
    payload = {
        "input": {
            "manual_review_template": str(paths["manual_review_template"]),
        },
        "dataset": {
            "train_images": str(paths["train_images"]),
        },
        "review": {
            "max_rows_per_group": 50,
            "near_threshold_distance": 0.05,
            "train_model": False,
            "generate_submission": False,
            "allow_test_labels": False,
        }
        | overrides,
        "output": {
            "root": str(paths["output_root"]),
        },
    }
    config_path = tmp_path / "v2_1_manual_review_helper.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_run_manual_review_helper_creates_html_file(tmp_path):
    from src.analysis.v2_1_manual_review_helper import run_manual_review_helper

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)

    outputs = run_manual_review_helper(config_path)

    assert outputs["html_preview"].exists()
    html = outputs["html_preview"].read_text(encoding="utf-8")
    assert "suggested_visual_tag" in html
    assert "near_threshold_false_positives" in html


def test_run_manual_review_helper_creates_prefilled_csv(tmp_path):
    from src.analysis.v2_1_manual_review_helper import run_manual_review_helper

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)

    outputs = run_manual_review_helper(config_path)

    prefilled = pd.read_csv(outputs["prefilled_csv"])
    assert list(prefilled.columns) == REQUIRED_PREFILLED_COLUMNS
    assert "near_threshold_ambiguous_check_manually" in prefilled["reviewer_note"].tolist()


def test_run_manual_review_helper_handles_missing_images_without_crashing(tmp_path):
    from src.analysis.v2_1_manual_review_helper import run_manual_review_helper

    paths = _write_inputs(tmp_path)
    (paths["train_images"] / "fp_far.jpg").unlink()
    config_path = _write_config(tmp_path, paths)

    outputs = run_manual_review_helper(config_path)

    prefilled = pd.read_csv(outputs["prefilled_csv"]).set_index("image_id")
    html = outputs["html_preview"].read_text(encoding="utf-8")
    assert prefilled.loc["fp_far.jpg", "suggested_visual_tag"] == "missing_image"
    assert prefilled.loc["fp_far.jpg", "suggested_recommended_action"] == "data_label_review"
    assert "missing_image" in html


def test_run_manual_review_helper_does_not_modify_original_template(tmp_path):
    from src.analysis.v2_1_manual_review_helper import run_manual_review_helper

    paths = _write_inputs(tmp_path)
    original = paths["manual_review_template"].read_text(encoding="utf-8")
    config_path = _write_config(tmp_path, paths)

    run_manual_review_helper(config_path)

    assert paths["manual_review_template"].read_text(encoding="utf-8") == original


def test_run_manual_review_helper_creates_required_supporting_outputs(tmp_path):
    from src.analysis.v2_1_manual_review_helper import run_manual_review_helper

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)

    outputs = run_manual_review_helper(config_path)

    assert outputs["instructions"].exists()
    assert outputs["tag_counts_template"].exists()
    instructions = outputs["instructions"].read_text(encoding="utf-8")
    tag_counts = pd.read_csv(outputs["tag_counts_template"])
    assert "not ground truth" in instructions
    assert list(tag_counts.columns) == [
        "review_group",
        "suggested_visual_tag",
        "final_visual_tag",
        "suggested_recommended_action",
        "final_recommended_action",
        "count",
        "review_status",
    ]


def test_config_rejects_training_or_submission_flags(tmp_path):
    from src.analysis.v2_1_manual_review_helper import V21ManualReviewHelperError, load_manual_review_helper_config

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths, train_model=True)

    with pytest.raises(V21ManualReviewHelperError, match="train_model"):
        load_manual_review_helper_config(config_path)
