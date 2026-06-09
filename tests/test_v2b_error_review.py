"""Tests for V2B error review and safe fine-tune preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image


ERROR_GROUPS = [
    "v1_correct_v2b_wrong",
    "both_wrong",
    "high_confidence_wrong",
    "uncertain",
]


def _write_error_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_review_inputs(root: Path, *, with_annotations: bool = True) -> dict[str, Path]:
    error_dir = root / "errors"
    image_dir = root / "train_images"
    error_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)

    rows_by_group = {
        "v1_correct_v2b_wrong": [
            {"image_id": "a.jpg", "y_true": 1, "v1_score": 0.20, "v1_target": 0, "v2b_score": 0.10, "v2b_target": 0},
            {"image_id": "b.jpg", "y_true": 0, "v1_score": 0.10, "v1_target": 0, "v2b_score": 0.80, "v2b_target": 1},
        ],
        "both_wrong": [
            {"image_id": "c.jpg", "y_true": 1, "v1_score": 0.20, "v1_target": 0, "v2b_score": 0.15, "v2b_target": 0},
        ],
        "high_confidence_wrong": [
            {"image_id": "d.jpg", "y_true": 0, "v1_score": 0.70, "v1_target": 1, "v2b_score": 0.95, "v2b_target": 1},
        ],
        "uncertain": [
            {"image_id": "e.jpg", "y_true": 1, "v1_score": 0.45, "v1_target": 0, "v2b_score": 0.33, "v2b_target": 1},
        ],
    }
    paths = {
        "train_csv": root / "train.csv",
        "train_annotations": root / "train_annotations.json",
        "train_images": image_dir,
    }
    for group, rows in rows_by_group.items():
        path = error_dir / f"{group}.csv"
        _write_error_csv(path, rows)
        paths[group] = path
    pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
            "target": [1, 0, 1, 0, 1],
        }
    ).to_csv(paths["train_csv"], index=False)
    if with_annotations:
        paths["train_annotations"].write_text(
            json.dumps(
                {
                    "annotations": [
                        {"image_id": "a.jpg", "category_name": "scratch"},
                        {"image_id": "b.jpg", "category_name": "dent"},
                        {"image_id": "c.jpg", "category_name": "bubble"},
                    ]
                }
            ),
            encoding="utf-8",
        )
    for image_id, color in {
        "a.jpg": (200, 20, 20),
        "b.jpg": (20, 200, 20),
        "c.jpg": (20, 20, 200),
        "d.jpg": (180, 180, 40),
        "e.jpg": (40, 180, 180),
    }.items():
        Image.new("RGB", (72, 72), color).save(image_dir / image_id)
    return paths


def _write_config(tmp_path: Path, paths: dict[str, Path]) -> Path:
    config_path = tmp_path / "v2b_error_review.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "train_csv": str(paths["train_csv"]),
                    "train_annotations": str(paths["train_annotations"]),
                    "train_images": str(paths["train_images"]),
                    "v1_correct_v2b_wrong": str(paths["v1_correct_v2b_wrong"]),
                    "both_wrong": str(paths["both_wrong"]),
                    "high_confidence_wrong": str(paths["high_confidence_wrong"]),
                    "uncertain": str(paths["uncertain"]),
                },
                "output": {"root": str(tmp_path / "outputs" / "analysis" / "v2b_error_review")},
                "review": {"contact_sheet_columns": 2, "contact_sheet_image_size": 64},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_synthetic_error_csvs_are_loaded_correctly(tmp_path):
    from src.analysis.v2b_error_review import load_review_config, load_error_groups

    config_path = _write_config(tmp_path, _make_review_inputs(tmp_path))
    config = load_review_config(config_path)

    grouped = load_error_groups(config["paths"])

    assert set(grouped) == set(ERROR_GROUPS)
    assert grouped["v1_correct_v2b_wrong"]["error_group"].unique().tolist() == ["v1_correct_v2b_wrong"]


def test_hard_example_csv_is_created_with_expected_groups(tmp_path):
    from src.analysis.v2b_error_review import run_review

    config_path = _write_config(tmp_path, _make_review_inputs(tmp_path))

    outputs = run_review(config_path)

    hard_examples = pd.read_csv(outputs["hard_examples"])
    assert set(hard_examples["error_group"]) == set(ERROR_GROUPS)
    assert {
        "image_id",
        "target",
        "error_group",
        "v1_prediction",
        "v2b_prediction",
        "v1_prob_bad",
        "v2b_prob_bad",
        "v2b_confidence",
        "annotation_categories",
        "source_csv",
        "review_priority",
    } <= set(hard_examples.columns)
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["safety_flags"]["hard_examples_mode"] == "analysis_only"
    assert summary["safety_flags"]["oversampling_enabled"] is False
    assert summary["safety_flags"]["focal_loss_enabled"] is False
    assert summary["safety_flags"]["weighted_sampler_enabled"] is False


def test_contact_sheet_generation_works_on_tiny_dummy_images(tmp_path):
    from src.analysis.v2b_error_review import run_review

    config_path = _write_config(tmp_path, _make_review_inputs(tmp_path))

    outputs = run_review(config_path)

    sheet_path = outputs["contact_sheets"]["v1_correct_v2b_wrong"][0]
    assert sheet_path.exists()
    image = Image.open(sheet_path)
    assert image.size[0] > 64
    assert image.size[1] > 64


def test_missing_optional_annotations_do_not_crash(tmp_path):
    from src.analysis.v2b_error_review import run_review

    config_path = _write_config(tmp_path, _make_review_inputs(tmp_path, with_annotations=False))

    outputs = run_review(config_path)

    breakdown = pd.read_csv(outputs["annotation_breakdown"])
    assert list(breakdown.columns) == ["annotation_category", "error_group", "count", "ratio"]
    assert breakdown.empty
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert "train_annotations" in summary["missing_optional_files"]


def test_test_labels_are_not_required(tmp_path):
    from src.analysis.v2b_error_review import run_review

    config_path = _write_config(tmp_path, _make_review_inputs(tmp_path))

    outputs = run_review(config_path)

    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["safety_flags"]["used_test_labels"] is False
    assert "test.csv" not in json.dumps(summary).lower()


def test_v5b_safe_finetune_config_preserves_analysis_only_guards():
    from src.training.train_classifier import load_classifier_config

    with open("configs/v5b_safe_finetune.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["model"]["model_name"] == "efficientnet_b1"
    assert config["model"]["start_checkpoint"].endswith("classifier_best.pth")
    assert config["model"]["image_size"] == 384
    assert config["training"]["epochs"] == 3
    assert config["training"]["learning_rate"] == pytest.approx(0.00002)
    assert config["training"]["loss"] == "bce"
    assert config["training"]["weighted_sampler"] is False
    assert config["training"]["focal_loss"] is False
    assert config["training"]["augmentation_recipe"] == "mild_safe"
    assert config["training"]["split_source"] == "v2b_compatible"
    assert config["training"]["hard_examples"] == "analysis_only"
    assert config["training"]["early_stopping"] is True

    loaded = load_classifier_config("configs/v5b_safe_finetune.yaml")
    assert loaded.model_name == "efficientnet_b1"
    assert loaded.image_size == 384
    assert loaded.imbalance_strategy == "bce"
    assert loaded.weighted_sampler is False
    assert loaded.augmentation_recipe == "mild_safe"
    assert loaded.hard_example_strategy == "analysis_only"
