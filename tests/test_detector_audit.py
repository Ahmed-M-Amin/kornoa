"""Tests for V3B detector audit and calibration repair."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _val_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["a.png", "b.png", "c.png", "d.png"],
            "target": [1, 0, 1, 0],
            "max_detector_conf": [0.95, 0.90, 0.02, 0.01],
            "n_boxes": [3, 2, 1, 0],
            "prediction": [1, 1, 1, 1],
        }
    )


def test_all_positive_baseline_is_computed_correctly():
    from src.inference.detector_audit import build_calibration_report

    report = build_calibration_report(_val_predictions(), pd.DataFrame({"max_detector_conf": [0.9], "target": [1]}), 0.01, {})

    assert report["all_positive_f1_baseline"] == pytest.approx(2 / 3)
    assert report["current_detector_f1"] == pytest.approx(2 / 3)
    assert report["val_prediction_distribution"] == {"0": 0, "1": 4}
    assert "all-positive baseline" in report["reason_all_positive_selected"]


def test_threshold_grid_reports_all_positive_candidate():
    from src.inference.detector_audit import build_threshold_grid

    grid = build_threshold_grid(_val_predictions())
    row = grid[grid["threshold"] == 0.01].iloc[0]

    assert row["prediction_positive_count"] == 4
    assert row["prediction_positive_ratio"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(2 / 3)


def test_detailed_box_export_has_required_columns(tmp_path):
    from src.inference.detector_audit import DETAIL_COLUMNS, export_detailed_boxes

    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"
    status = export_detailed_boxes({}, _val_predictions(), _val_predictions(), {}, val_path, test_path)

    assert status["detailed_box_export_status"] == "skipped"
    assert list(pd.read_csv(val_path).columns) == DETAIL_COLUMNS
    assert list(pd.read_csv(test_path).columns) == DETAIL_COLUMNS


def test_category_area_rule_audit_writes_required_fields(tmp_path):
    from src.inference.detector_audit import build_category_area_rule_audit, build_threshold_grid

    details = pd.DataFrame(
        {
            "image_id": ["a.png", "b.png", "c.png"],
            "split": ["val", "val", "val"],
            "class_id": [1, 1, 2],
            "class_name": ["fault", "fault", "rare"],
            "box_conf": [0.9, 0.8, 0.7],
            "x1": [0, 0, 0],
            "y1": [0, 0, 0],
            "x2": [10, 10, 10],
            "y2": [10, 10, 10],
            "box_area": [100, 100, 100],
            "image_area": [1000, 1000, 1000],
            "box_area_ratio": [0.1, 0.1, 0.1],
        }
    )

    report = build_category_area_rule_audit(_val_predictions(), build_threshold_grid(_val_predictions()), details)

    assert {"best_conf_only_rule", "best_category_area_rule", "harmless_category_false_positive_rate"} <= set(report)
    assert {"faulty_category_true_positive_rate", "conditional_area_best_thresholds"} <= set(report)
    assert report["best_category_area_rule"]["f1"] >= 0


def test_detector_audit_cli_writes_required_reports(tmp_path):
    from src.inference.detector_audit import run_audit

    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"
    threshold_path = tmp_path / "threshold.json"
    _val_predictions().to_csv(val_path, index=False)
    pd.DataFrame(
        {"image_id": ["ta.png"], "max_detector_conf": [0.9], "n_boxes": [2], "target": [1]}
    ).to_csv(test_path, index=False)
    threshold_path.write_text(json.dumps({"best_threshold": 0.01, "best_f1": 2 / 3}), encoding="utf-8")
    config_path = tmp_path / "detector_v3b_audit.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "detector_audit": {
                    "val_predictions": str(val_path),
                    "test_predictions": str(test_path),
                    "threshold_report": str(threshold_path),
                    "output_dir": str(tmp_path / "outputs"),
                }
            }
        ),
        encoding="utf-8",
    )

    outputs = run_audit(config_path)

    for path in outputs.values():
        assert Path(path).exists()
    calibration = json.loads(outputs["calibration_report"].read_text(encoding="utf-8"))
    category = json.loads(outputs["category_area_rule_audit"].read_text(encoding="utf-8"))
    assert "all_positive_f1_baseline" in calibration
    assert "best_category_area_rule" in category
