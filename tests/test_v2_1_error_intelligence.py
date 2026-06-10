"""Tests for V2.1 error intelligence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_labels(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_quality(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_detector(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _base_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    labels = [
        {"image_id": "a.jpg", "target": 1},
        {"image_id": "b.jpg", "target": 0},
        {"image_id": "c.jpg", "target": 1},
        {"image_id": "d.jpg", "target": 0},
        {"image_id": "e.jpg", "target": 0},
        {"image_id": "f.jpg", "target": 1},
        {"image_id": "g.jpg", "target": 1},
    ]
    predictions = [
        {"image_id": "a.jpg", "prob_bad": 0.90},
        {"image_id": "b.jpg", "prob_bad": 0.54},
        {"image_id": "c.jpg", "prob_bad": 0.47},
        {"image_id": "d.jpg", "prob_bad": 0.49},
        {"image_id": "e.jpg", "prob_bad": 0.15},
        {"image_id": "f.jpg", "prob_bad": 0.10},
        {"image_id": "g.jpg", "prob_bad": 0.85},
    ]
    return labels, predictions


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    labels, predictions = _base_rows()
    paths = {
        "labels": tmp_path / "train.csv",
        "predictions": tmp_path / "val_predictions.csv",
        "threshold": tmp_path / "best_threshold.json",
        "metrics": tmp_path / "classifier_metrics.json",
        "submission": tmp_path / "submission_v2b.csv",
        "quality": tmp_path / "image_quality.csv",
        "detector": tmp_path / "detector.csv",
        "output_root": tmp_path / "outputs" / "analysis" / "v2_1_error_intelligence",
    }
    _write_labels(paths["labels"], labels)
    _write_predictions(paths["predictions"], predictions)
    paths["threshold"].write_text(json.dumps({"threshold": 0.50}), encoding="utf-8")
    paths["metrics"].write_text(
        json.dumps(
            {
                "f1_score": 4 / 7,
                "precision": 2 / 3,
                "recall": 0.5,
                "confusion_counts": {"tp": 2, "fp": 1, "fn": 2, "tn": 2},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"image_id": [f"test_{idx}.jpg" for idx in range(4)], "target": [0, 1, 1, 0]}).to_csv(
        paths["submission"], index=False
    )
    _write_quality(
        paths["quality"],
        [
            {"image_id": "a.jpg", "brightness": 0.4, "blur": 0.1, "crop_size": 120, "crop_confidence": 0.9},
            {"image_id": "c.jpg", "brightness": 0.6, "blur": 0.2, "crop_size": 100, "crop_confidence": 0.8},
            {"image_id": "f.jpg", "brightness": 0.3, "blur": 0.5, "crop_size": 95, "crop_confidence": 0.7},
            {"image_id": "g.jpg", "brightness": 0.5, "blur": 0.1, "crop_size": 110, "crop_confidence": 0.95},
        ],
    )
    _write_detector(
        paths["detector"],
        [
            {"image_id": "b.jpg", "category": "scratch", "confidence": 0.70, "bbox": "[1,2,3,4]", "area": 12.0},
            {"image_id": "b.jpg", "category": "dent", "confidence": 0.80, "bbox": "[1,2,4,6]", "area": 20.0},
            {"image_id": "f.jpg", "category": "bubble", "confidence": 0.90, "bbox": "[0,0,5,5]", "area": 25.0},
        ],
    )
    return paths


def _write_config(tmp_path: Path, paths: dict[str, Path], **analysis_overrides: object) -> Path:
    payload = {
        "baseline": {
            "validation_predictions": str(paths["predictions"]),
            "validation_labels": str(paths["labels"]),
            "threshold": str(paths["threshold"]),
            "metrics": str(paths["metrics"]),
            "checkpoint": str(tmp_path / "classifier_best.pth"),
            "submission": str(paths["submission"]),
        },
        "evidence": {
            "detector": str(paths["detector"]),
            "image_quality": str(paths["quality"]),
        },
        "analysis": {
            "positive_class": 1,
            "near_threshold_distance": 0.05,
            "high_confidence_distance": 0.30,
            "metric_tolerance": 0.000001,
            "allow_test_labels": False,
            "generate_submission": False,
            "train_model": False,
        }
        | analysis_overrides,
        "output": {"root": str(paths["output_root"])},
    }
    config_path = tmp_path / "v2_1_error_intelligence.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_load_config_requires_baseline_keys_and_defaults():
    from src.analysis.v2_1_error_intelligence import REQUIRED_BASELINE_KEYS, load_error_intelligence_config

    config = load_error_intelligence_config("configs/v2_1_error_intelligence.yaml")

    assert REQUIRED_BASELINE_KEYS <= set(config["baseline"])
    assert config["analysis"]["positive_class"] == 1
    assert config["analysis"]["near_threshold_distance"] == pytest.approx(0.05)
    assert config["analysis"]["high_confidence_distance"] == pytest.approx(0.30)


def test_config_rejects_forbidden_training_or_submission_flags(tmp_path):
    from src.analysis.v2_1_error_intelligence import V21ErrorIntelligenceError, load_error_intelligence_config

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths, train_model=True)

    with pytest.raises(V21ErrorIntelligenceError, match="train_model"):
        load_error_intelligence_config(config_path)


def test_config_rejects_test_labels_and_sample_submission_paths(tmp_path):
    from src.analysis.v2_1_error_intelligence import V21ErrorIntelligenceError, validate_config_safety

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["baseline"]["validation_labels"] = str(tmp_path / "test.csv")
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(V21ErrorIntelligenceError, match="validation_labels"):
        validate_config_safety(payload)


def test_run_error_intelligence_builds_audit_and_summary(tmp_path):
    from src.analysis.v2_1_error_intelligence import AUDIT_COLUMNS, run_error_intelligence

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)

    outputs = run_error_intelligence(config_path)

    audit = pd.read_csv(outputs["audit"])
    assert list(audit.columns) == AUDIT_COLUMNS
    assert len(audit) == 7
    assert set(audit["error_type"]) == {"TP", "FP", "TN", "FN"}
    assert outputs["summary"].exists()
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["baseline_match"] is True
    assert summary["computed_metrics"]["confusion_counts"] == {"tp": 2, "fp": 1, "fn": 2, "tn": 2}


def test_metric_mismatch_fails_before_accepting_outputs(tmp_path):
    from src.analysis.v2_1_error_intelligence import V21ErrorIntelligenceError, run_error_intelligence

    paths = _write_inputs(tmp_path)
    json_payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    json_payload["f1_score"] = 0.99
    paths["metrics"].write_text(json.dumps(json_payload), encoding="utf-8")
    config_path = _write_config(tmp_path, paths)

    with pytest.raises(V21ErrorIntelligenceError, match="locked baseline"):
        run_error_intelligence(config_path)


def test_review_groups_use_fixed_probability_bands(tmp_path):
    from src.analysis.v2_1_error_intelligence import run_error_intelligence

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)

    outputs = run_error_intelligence(config_path)

    near_fp = pd.read_csv(outputs["near_threshold_false_positives"])
    near_fn = pd.read_csv(outputs["near_threshold_false_negatives"])
    high_fp = pd.read_csv(outputs["high_confidence_false_positives"])
    high_fn = pd.read_csv(outputs["high_confidence_false_negatives"])

    assert near_fp["image_id"].tolist() == ["b.jpg"]
    assert near_fn["image_id"].tolist() == ["c.jpg"]
    assert high_fp.empty
    assert high_fn["image_id"].tolist() == ["f.jpg"]


def test_over_rejected_reusable_and_group_counts_are_reported(tmp_path):
    from src.analysis.v2_1_error_intelligence import run_error_intelligence

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)

    outputs = run_error_intelligence(config_path)

    over_rejected = pd.read_csv(outputs["over_rejected_reusable"])
    group_counts = pd.read_csv(outputs["group_counts"])
    assert over_rejected["image_id"].tolist() == ["b.jpg"]
    counts = dict(zip(group_counts["group_name"], group_counts["row_count"]))
    assert counts["near_threshold_false_positive"] == 1
    assert counts["near_threshold_false_negative"] == 1
    assert counts["high_confidence_false_negative"] == 1
    assert counts["over_rejected_reusable"] == 1


def test_detector_evidence_is_joined_with_strongest_row(tmp_path):
    from src.analysis.v2_1_error_intelligence import run_error_intelligence

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)

    outputs = run_error_intelligence(config_path)

    audit = pd.read_csv(outputs["audit"]).set_index("image_id")
    assert audit.loc["b.jpg", "detector_category"] == "dent"
    assert audit.loc["b.jpg", "detector_confidence"] == pytest.approx(0.80)
    assert audit.loc["b.jpg", "detector_evidence_status"] == "present"
    assert audit.loc["e.jpg", "detector_evidence_status"] == "missing"


def test_image_quality_is_joined_and_missing_rows_are_preserved(tmp_path):
    from src.analysis.v2_1_error_intelligence import run_error_intelligence

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)

    outputs = run_error_intelligence(config_path)

    audit = pd.read_csv(outputs["audit"]).set_index("image_id")
    assert audit.loc["a.jpg", "brightness"] == pytest.approx(0.4)
    assert audit.loc["a.jpg", "image_quality_status"] == "present"
    assert audit.loc["b.jpg", "image_quality_status"] == "missing"
    assert len(audit) == 7


def test_optional_evidence_report_and_provenance_are_written(tmp_path):
    from src.analysis.v2_1_error_intelligence import run_error_intelligence

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)

    outputs = run_error_intelligence(config_path)

    evidence_report = json.loads(outputs["optional_evidence_report"].read_text(encoding="utf-8"))
    provenance = json.loads(outputs["provenance"].read_text(encoding="utf-8"))
    target_distribution = json.loads(outputs["target_distribution"].read_text(encoding="utf-8"))

    assert evidence_report["missing_optional_evidence_counts"] == {"detector": 5, "image_quality": 3}
    assert "validation_predictions" in provenance["baseline"]
    assert provenance["optional_evidence"]["detector"] == str(paths["detector"])
    assert target_distribution["prediction_distribution"] == {"target_0": 4, "target_1": 3}


def test_cli_reports_expected_errors_without_traceback(tmp_path, capsys):
    from src.analysis.v2_1_error_intelligence import main

    paths = _write_inputs(tmp_path)
    paths["predictions"].unlink()
    config_path = _write_config(tmp_path, paths)

    exit_code = main(["--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "failed" in captured.err.lower()
    assert "traceback" not in captured.err.lower()
