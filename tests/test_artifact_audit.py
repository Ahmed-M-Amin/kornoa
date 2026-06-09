"""Tests for the V1/V2B artifact audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _write_prediction_csv(path: Path, *, probabilities: list[float]) -> None:
    pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "true_label": [1, 0, 1, 0],
            "probability": probabilities,
        }
    ).to_csv(path, index=False)


def _write_audit_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "train_csv": root / "train.csv",
        "train_annotations": root / "train_annotations.json",
        "bottletypes": root / "bottletypes.csv",
        "v1_predictions": root / "v1.csv",
        "v1_threshold": root / "v1_threshold.json",
        "v1_metrics": root / "v1_metrics.json",
        "v2b_predictions": root / "v2b.csv",
        "v2b_threshold": root / "v2b_threshold.json",
        "v2b_metrics": root / "v2b_metrics.json",
        "v3_detector_val_predictions": root / "missing_v3.csv",
        "v4_hybrid_metrics": root / "missing_v4_metrics.json",
        "v4_hybrid_diff": root / "missing_v4_diff.json",
    }
    pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "target": [1, 0, 1, 0],
            "bottle_type_id": [10, 10, 20, 20],
        }
    ).to_csv(paths["train_csv"], index=False)
    paths["train_annotations"].write_text(
        json.dumps(
            {
                "annotations": [
                    {"image_id": "a.jpg", "category_name": "scratch"},
                    {"image_id": "c.jpg", "category_name": "dent"},
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"bottle_type_id": [10, 20], "bottle_type": ["small", "large"]}).to_csv(paths["bottletypes"], index=False)
    _write_prediction_csv(paths["v1_predictions"], probabilities=[0.40, 0.20, 0.80, 0.70])
    _write_prediction_csv(paths["v2b_predictions"], probabilities=[0.90, 0.10, 0.30, 0.20])
    paths["v1_threshold"].write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    paths["v2b_threshold"].write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    paths["v1_metrics"].write_text(json.dumps({"f1_score": 0.5}), encoding="utf-8")
    paths["v2b_metrics"].write_text(json.dumps({"f1_score": 2 / 3}), encoding="utf-8")
    return paths


def _write_config(tmp_path: Path, paths: dict[str, Path]) -> Path:
    config_path = tmp_path / "artifact_audit.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {key: str(value) for key, value in paths.items()} | {"dataset_root": str(tmp_path)},
                "output": {"root": str(tmp_path / "outputs" / "analysis" / "v1_v2b_audit")},
                "analysis": {
                    "positive_class": 1,
                    "threshold_grid_start": 0.01,
                    "threshold_grid_end": 0.99,
                    "threshold_grid_step": 0.01,
                    "uncertainty_margin": 0.05,
                    "high_confidence_wrong_margin": 0.20,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_loads_audit_config_and_validates_required_path_keys():
    from src.analysis.artifact_audit import REQUIRED_PATH_KEYS, load_audit_config, validate_required_path_keys

    config = load_audit_config("configs/artifact_audit.yaml")

    validate_required_path_keys(config)
    assert REQUIRED_PATH_KEYS <= set(config["paths"])


def test_binary_f1_uses_positive_class_one_and_zero_division():
    from src.analysis.artifact_audit import compute_binary_f1

    assert compute_binary_f1([1, 0, 1, 0], [1, 1, 0, 0]) == pytest.approx(0.5)
    assert compute_binary_f1([0, 0], [0, 0]) == pytest.approx(0.0)


def test_run_audit_creates_threshold_sensitivity_error_splits_and_summary(tmp_path):
    from src.analysis.artifact_audit import ERROR_COLUMNS, ERROR_SPLIT_FILENAMES, run_audit

    config_path = _write_config(tmp_path, _write_audit_inputs(tmp_path))

    outputs = run_audit(config_path)

    assert outputs["threshold_sensitivity"].exists()
    sensitivity = pd.read_csv(outputs["threshold_sensitivity"])
    assert {"threshold", "v1_f1", "v2b_f1", "v2b_prediction_positive_count"} <= set(sensitivity.columns)
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["required_inputs"]["v1_predictions"]["status"] == "found"
    assert summary["required_inputs"]["v2b_predictions"]["status"] == "found"
    assert summary["v1_vs_v2b_counts"]["v1_wrong_v2b_correct"] == 2
    assert "v3_detector_val_predictions" in summary["optional_inputs"]

    for split_name, filename in ERROR_SPLIT_FILENAMES.items():
        path = outputs["errors_dir"] / filename
        assert path.exists(), split_name
        assert list(pd.read_csv(path).columns) == ERROR_COLUMNS


def test_missing_optional_v3_v4_artifacts_are_reported_not_fatal(tmp_path):
    from src.analysis.artifact_audit import run_audit

    config_path = _write_config(tmp_path, _write_audit_inputs(tmp_path))

    outputs = run_audit(config_path)
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))

    assert summary["optional_inputs"]["v3_detector_val_predictions"]["status"] == "missing"
    assert summary["optional_inputs"]["v4_hybrid_metrics"]["status"] == "missing"
    assert summary["optional_inputs"]["v4_hybrid_diff"]["status"] == "missing"


def test_missing_required_v1_or_v2b_predictions_fail_when_audit_runs(tmp_path):
    from src.analysis.artifact_audit import ArtifactAuditError, run_audit

    paths = _write_audit_inputs(tmp_path)
    paths["v1_predictions"].unlink()
    config_path = _write_config(tmp_path, paths)

    with pytest.raises(ArtifactAuditError, match="Missing required audit input"):
        run_audit(config_path)


def test_audit_cli_reports_missing_required_inputs_without_traceback(tmp_path, capsys):
    from src.analysis.artifact_audit import main

    paths = _write_audit_inputs(tmp_path)
    paths["v1_predictions"].unlink()
    config_path = _write_config(tmp_path, paths)

    exit_code = main(["--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Missing required audit input" in captured.err
    assert "Traceback" not in captured.err


def test_no_test_labels_are_used_by_audit_config():
    from src.analysis.artifact_audit import load_audit_config

    config = load_audit_config("configs/artifact_audit.yaml")
    flattened_values = "\n".join(str(value).lower() for value in config["paths"].values())

    assert "test.csv" not in flattened_values
    assert "sample_submission" not in flattened_values
