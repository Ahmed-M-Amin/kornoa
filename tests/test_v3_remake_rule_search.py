"""Tests for V3/V4 remake rule search under binary F1 scoring."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_config(tmp_path: Path, *, max_target_shift: float = 0.50) -> Path:
    root = tmp_path / "inputs"
    val_v2b = _write_csv(
        root / "val_v2b.csv",
        [
            {"image_id": "a.jpg", "probability": 0.49, "true_label": 1},
            {"image_id": "b.jpg", "probability": 0.10, "true_label": 0},
            {"image_id": "c.jpg", "probability": 0.90, "true_label": 1},
            {"image_id": "d.jpg", "probability": 0.47, "true_label": 0},
            {"image_id": "e.jpg", "probability": 0.48, "true_label": 1},
            {"image_id": "f.jpg", "probability": 0.20, "true_label": 0},
        ],
    )
    threshold = root / "best_threshold.json"
    threshold.write_text(json.dumps({"threshold": 0.50}), encoding="utf-8")
    val_detector = _write_csv(
        root / "val_detector.csv",
        [
            {"image_id": "a.jpg", "confidence": 0.40, "category": "Break/Crack", "area": 0.01},
            {"image_id": "b.jpg", "confidence": 0.90, "category": "Break/Crack", "area": 0.10},
            {"image_id": "c.jpg", "confidence": 0.99, "category": "Break/Crack", "area": 0.10},
            {"image_id": "d.jpg", "confidence": 0.90, "category": "Water drop", "area": 0.30},
            {"image_id": "e.jpg", "confidence": 0.90, "category": "Chip", "area": 0.20},
            {"image_id": "f.jpg", "confidence": 0.90, "category": "Chip", "area": 0.01},
        ],
    )
    test_v2b = _write_csv(
        root / "submission_v2b.csv",
        [{"image_id": "ta.jpg", "target": 0}, {"image_id": "tb.jpg", "target": 1}],
    )
    test_detector = _write_csv(
        root / "test_detector.csv",
        [{"image_id": "ta.jpg", "confidence": 0.90, "category": "Break/Crack", "area": 0.10}],
    )
    sample_submission = _write_csv(
        root / "sample_submission.csv",
        [{"image_id": "ta.jpg", "target": 0}, {"image_id": "tb.jpg", "target": 0}],
    )
    train_csv = _write_csv(root / "train.csv", [{"image_id": "a.jpg", "target": 1}])
    config = {
        "paths": {
            "v2b_val_predictions": str(val_v2b),
            "v2b_test_submission": str(test_v2b),
            "v2b_threshold": str(threshold),
            "detector_val_predictions": str(val_detector),
            "detector_test_predictions": str(test_detector),
            "category_mapping": str(root / "category_mapping.json"),
            "train_csv": str(train_csv),
            "sample_submission": str(sample_submission),
            "output_root": str(tmp_path / "outputs" / "hybrid" / "v4_remake"),
        },
        "search": {
            "detector_conf_thresholds": [0.10, 0.35],
            "uncertainty_margins": [0.02, 0.04, 0.12],
            "conditional_area_threshold": 0.05,
            "f1_tie_tolerance": 0.002,
        },
        "safety": {
            "max_changed_ratio_without_f1_gain": 0.40,
            "max_target1_distribution_shift": max_target_shift,
            "max_precision_drop": 0.30,
            "minimum_clear_f1_gain": 0.002,
        },
    }
    path = tmp_path / "configs" / "v3_remake_rule_search.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_binary_f1_uses_positive_class_one_and_zero_division():
    from src.inference.v3_remake_rule_search import compute_positive_f1

    assert compute_positive_f1([0, 1, 1], [0, 0, 1]) == pytest.approx(2 / 3)
    assert compute_positive_f1([0, 0], [0, 0]) == 0.0


def test_detector_flips_uncertain_zero_to_one_but_not_high_confidence():
    from src.inference.v3_remake_rule_search import RemakeRuleConfig, apply_rules_to_validation

    classifier = pd.DataFrame(
        {
            "image_id": ["uncertain.jpg", "confident.jpg"],
            "v2b_probability": [0.49, 0.10],
            "v2b_target": [0, 0],
            "y_true": [1, 0],
        }
    )
    detector = pd.DataFrame(
        {
            "image_id": ["uncertain.jpg", "confident.jpg"],
            "detector_confidence": [0.90, 0.90],
            "category": ["Break/Crack", "Break/Crack"],
            "area": [0.1, 0.1],
        }
    )
    config = RemakeRuleConfig(v2b_threshold=0.50, detector_conf_threshold=0.35, uncertainty_margin=0.04)

    fused = apply_rules_to_validation(classifier, detector, config)

    assert fused.loc[fused["image_id"] == "uncertain.jpg", "v4_remake_target"].item() == 1
    assert fused.loc[fused["image_id"] == "confident.jpg", "v4_remake_target"].item() == 0


def test_never_always_and_conditional_category_rules():
    from src.inference.v3_remake_rule_search import RemakeRuleConfig, apply_rules_to_validation

    classifier = pd.DataFrame(
        {
            "image_id": ["never.jpg", "always.jpg", "small.jpg", "large.jpg"],
            "v2b_probability": [0.49, 0.49, 0.49, 0.49],
            "v2b_target": [0, 0, 0, 0],
            "y_true": [0, 1, 0, 1],
        }
    )
    detector = pd.DataFrame(
        {
            "image_id": ["never.jpg", "always.jpg", "small.jpg", "large.jpg"],
            "detector_confidence": [0.90, 0.90, 0.90, 0.90],
            "category": ["Water drop", "Break/Crack", "Chip", "Chip"],
            "area": [0.50, 0.01, 0.01, 0.20],
        }
    )
    config = RemakeRuleConfig(v2b_threshold=0.50, detector_conf_threshold=0.35, uncertainty_margin=0.04)

    fused = apply_rules_to_validation(classifier, detector, config)
    by_id = dict(zip(fused["image_id"], fused["v4_remake_target"]))

    assert by_id["never.jpg"] == 0
    assert by_id["always.jpg"] == 1
    assert by_id["small.jpg"] == 0
    assert by_id["large.jpg"] == 1


def test_binary_detector_artifact_columns_act_as_rejection_evidence(tmp_path):
    from src.inference.v3_remake_rule_search import RemakeRuleConfig, apply_rules_to_validation, read_detector_predictions_for_rules

    detector_path = _write_csv(
        tmp_path / "detector.csv",
        [
            {
                "image_id": "a.jpg",
                "image_path": "unused/a.jpg",
                "max_detector_conf": 0.90,
                "n_boxes": 5,
                "prediction": 1,
            }
        ],
    )
    detector = read_detector_predictions_for_rules(detector_path)
    classifier = pd.DataFrame(
        {"image_id": ["a.jpg"], "v2b_probability": [0.49], "v2b_target": [0], "y_true": [1]}
    )
    fused = apply_rules_to_validation(
        classifier,
        detector,
        RemakeRuleConfig(v2b_threshold=0.50, detector_conf_threshold=0.35, uncertainty_margin=0.04),
    )

    assert detector.loc[0, "category"] == "binary_detector_positive"
    assert fused.loc[0, "v4_remake_target"] == 1


def test_close_f1_tie_break_prefers_lower_usage_and_fewer_changes():
    from src.inference.v3_remake_rule_search import CandidateMetrics, select_best_candidate

    high_usage = CandidateMetrics(
        detector_conf_threshold=0.10,
        uncertainty_margin=0.12,
        validation_f1=0.8005,
        precision=0.80,
        recall=0.80,
        tp=8,
        fp=2,
        fn=2,
        changed_count_vs_v2b=4,
        zero_to_one_count=4,
        one_to_zero_count=0,
        detector_usage_percent=40.0,
        target1_count=10,
        target0_count=10,
        target1_distribution_diff=0.0,
        fixed_false_negatives=2,
        added_false_positives=1,
        rejected=False,
        rejected_reasons=[],
    )
    low_usage = CandidateMetrics(
        detector_conf_threshold=0.35,
        uncertainty_margin=0.04,
        validation_f1=0.8000,
        precision=0.80,
        recall=0.80,
        tp=8,
        fp=2,
        fn=2,
        changed_count_vs_v2b=1,
        zero_to_one_count=1,
        one_to_zero_count=0,
        detector_usage_percent=10.0,
        target1_count=7,
        target0_count=13,
        target1_distribution_diff=0.0,
        fixed_false_negatives=1,
        added_false_positives=0,
        rejected=False,
        rejected_reasons=[],
    )

    assert select_best_candidate([high_usage, low_usage], f1_tie_tolerance=0.002) == low_usage


def test_candidate_with_excessive_target_distribution_shift_is_rejected(tmp_path):
    from src.inference.v3_remake_rule_search import run_rule_search

    config_path = _write_config(tmp_path, max_target_shift=0.01)
    result = run_rule_search(config_path)
    grid = pd.read_csv(result.grid_path)

    assert grid["rejected"].any()
    assert grid["rejected_reasons"].str.contains("target1_distribution_shift").any()


def test_run_rule_search_writes_required_outputs_without_test_labels(tmp_path):
    from src.inference.v3_remake_rule_search import run_rule_search

    config_path = _write_config(tmp_path)
    result = run_rule_search(config_path)
    output_root = tmp_path / "outputs" / "hybrid" / "v4_remake"

    assert result.best.validation_f1 > 0.0
    assert result.best.one_to_zero_count == 0
    assert (output_root / "reports" / "v3_remake_grid.csv").exists()
    assert (output_root / "reports" / "v3_remake_best_config.json").exists()
    assert (output_root / "reports" / "v4_remake_validation_metrics.json").exists()
    assert (output_root / "reports" / "v2b_vs_v4_remake_diff.csv").exists()
    assert (output_root / "reports" / "target_distribution_report.json").exists()
    submission = output_root / "submissions" / "submission_v4_remake.csv"
    rows = list(csv.DictReader(submission.open(newline="", encoding="utf-8")))
    assert list(rows[0]) == ["image_id", "target"]
    assert "true_label" not in rows[0]
    assert result.uses_test_labels is False
