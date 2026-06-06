"""Tests for SPEC-009 hybrid submission orchestration."""

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


def _write_threshold(path: Path, threshold: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"threshold": threshold}), encoding="utf-8")
    return path


def _write_config(tmp_path: Path, *, classifier_test: Path | None = None, forbidden: Path | None = None) -> Path:
    root = tmp_path / "artifacts"
    val_classifier = _write_csv(
        root / "val_classifier.csv",
        [
            {"image_id": "a.jpg", "probability": 0.49, "label": 1},
            {"image_id": "b.jpg", "probability": 0.9, "label": 1},
            {"image_id": "c.jpg", "probability": 0.1, "label": 0},
        ],
    )
    test_classifier = classifier_test or _write_csv(
        root / "test_classifier_predictions.csv",
        [
            {"image_id": "ta.jpg", "probability": 0.49},
            {"image_id": "tb.jpg", "probability": 0.9},
        ],
    )
    val_detector = _write_csv(
        root / "val_detector.csv",
        [
            {"image_id": "a.jpg", "confidence": 0.9, "category": "crack", "area": 0.2},
            {"image_id": "b.jpg", "confidence": 0.0, "category": "none", "area": 0.0},
        ],
    )
    test_detector = _write_csv(
        root / "test_detector.csv",
        [
            {"image_id": "ta.jpg", "confidence": 0.9, "category": "crack", "area": 0.2},
            {"image_id": "tb.jpg", "confidence": 0.0, "category": "none", "area": 0.0},
        ],
    )
    threshold = _write_threshold(root / "best_threshold.json")
    config = {
        "classifier": {
            "model_id": "v2b_effnet_b1",
            "val_predictions": str(val_classifier),
            "test_predictions": str(test_classifier),
            "threshold_report": str(threshold),
        },
        "detector": {
            "model_id": "v3_detector",
            "val_predictions": str(val_detector),
            "test_predictions": str(test_detector),
        },
        "hybrid": {
            "output_dir": str(tmp_path / "outputs" / "hybrid" / "v4"),
            "detector_usage_preference": 0.3,
            "selection_metric": "f1",
            "always_faulty_categories": ["crack"],
            "conditional_categories": [],
            "default_conditional_area_threshold": 0.05,
            "uncertainty_margin_candidates": [0.05],
            "detector_conf_threshold_candidates": [0.25],
        },
    }
    if forbidden is not None:
        config["leakage_check"] = {"forbidden_paths": [str(forbidden)]}
    path = tmp_path / "configs" / "hybrid_inference.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_report_json_files_contain_required_contract_fields(tmp_path):
    from src.inference.hybrid_submission import run_search

    config_path = _write_config(tmp_path)
    report = run_search(config_path)

    config = json.loads((report.output_dir / "reports" / "hybrid_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((report.output_dir / "reports" / "hybrid_metrics.json").read_text(encoding="utf-8"))
    overlap = json.loads((report.output_dir / "reports" / "overlap_report.json").read_text(encoding="utf-8"))

    assert {"version", "classifier_model_id", "detector_model_id", "selection_rule", "uses_test_labels", "input_paths"} <= set(config)
    assert {"overlap_count", "classifier_baseline", "detector_baseline", "hybrid", "detector_usage"} <= set(metrics)
    assert {"classifier_val_count", "detector_val_count", "overlap_count", "used_for_parameter_search"} <= set(overlap)
    assert config["uses_test_labels"] is False


def test_search_command_writes_required_artifacts(tmp_path):
    from src.inference.hybrid_submission import main

    config_path = _write_config(tmp_path)

    assert main(["search", "--config", str(config_path)]) == 0

    output_dir = tmp_path / "outputs" / "hybrid" / "v4"
    assert (output_dir / "reports" / "hybrid_config.json").exists()
    assert (output_dir / "reports" / "hybrid_metrics.json").exists()
    assert (output_dir / "reports" / "overlap_report.json").exists()
    assert (output_dir / "predictions" / "val_hybrid_predictions.csv").exists()


def test_search_command_writes_v42_candidate_grid_and_summary(tmp_path):
    from src.inference.hybrid_submission import main

    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["hybrid"]["selection_mode"] = "best_nonzero_change"
    config["hybrid"]["uncertainty_margin_candidates"] = [0.0, 0.05]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["search", "--config", str(config_path)]) == 0

    output_dir = tmp_path / "outputs" / "hybrid" / "v4"
    grid = pd.read_csv(output_dir / "reports" / "hybrid_search_grid.csv")
    summary = json.loads((output_dir / "reports" / "hybrid_candidate_summary.json").read_text(encoding="utf-8"))

    assert {
        "classifier_threshold",
        "uncertainty_margin",
        "detector_conf_threshold",
        "detector_usage_count",
        "detector_usage_ratio",
        "changed_count_vs_classifier",
        "changed_ratio_vs_classifier",
        "validation_f1",
        "validation_accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
    } <= set(grid.columns)
    assert summary["selection_mode"] == "best_nonzero_change"
    assert "safe_nonzero_candidate_exists" in summary
    assert "selected_candidate" in summary


def test_run_command_writes_reports_and_strict_submission(tmp_path):
    from src.inference.hybrid_submission import main

    config_path = _write_config(tmp_path)

    assert main(["run", "--config", str(config_path)]) == 0

    output_dir = tmp_path / "outputs" / "hybrid" / "v4"
    rows = list(csv.DictReader((output_dir / "submissions" / "submission_v4_hybrid.csv").open("r", newline="", encoding="utf-8")))
    assert rows
    assert list(rows[0].keys()) == ["image_id", "target"]
    assert (output_dir / "reports" / "hybrid_metrics.json").exists()
    diff = json.loads((output_dir / "reports" / "v2b_vs_v4_diff.json").read_text(encoding="utf-8"))
    assert diff["row_count"] == len(rows)
    assert diff["uses_test_labels"] is False
    assert {"row_count", "changed_count", "changed_rows", "v2b_target_counts", "v4_target_counts"} <= set(diff)


def test_diagnostic_only_run_writes_reports_without_overwriting_submission(tmp_path):
    from src.inference.hybrid_submission import main

    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["hybrid"]["selection_mode"] = "diagnostic_only"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    submission = tmp_path / "outputs" / "hybrid" / "v4" / "submissions" / "submission_v4_hybrid.csv"
    submission.parent.mkdir(parents=True, exist_ok=True)
    submission.write_text("image_id,target\nexisting.jpg,0\n", encoding="utf-8")

    assert main(["run", "--config", str(config_path)]) == 0

    assert submission.read_text(encoding="utf-8") == "image_id,target\nexisting.jpg,0\n"
    assert (tmp_path / "outputs" / "hybrid" / "v4" / "reports" / "hybrid_search_grid.csv").exists()
    assert (tmp_path / "outputs" / "hybrid" / "v4" / "reports" / "hybrid_candidate_summary.json").exists()


def test_final_submission_schema_is_strict(tmp_path):
    from src.inference.hybrid_submission import write_submission

    predictions = pd.DataFrame(
        {"image_id": ["a.jpg"], "hybrid_target": [1], "debug": ["not allowed"], "decision_source": ["detector"]}
    )
    output_path = tmp_path / "submission.csv"

    write_submission(predictions, output_path)

    rows = list(csv.DictReader(output_path.open("r", newline="", encoding="utf-8")))
    assert rows == [{"image_id": "a.jpg", "target": "1"}]
    assert list(rows[0].keys()) == ["image_id", "target"]


def test_missing_classifier_test_confidence_fails_before_submission(tmp_path):
    from src.inference.hybrid_submission import HybridSubmissionError, main

    malformed = _write_csv(tmp_path / "artifacts" / "test_classifier_predictions.csv", [{"image_id": "ta.jpg", "model": "v2b"}])
    config_path = _write_config(tmp_path, classifier_test=malformed)
    assert main(["search", "--config", str(config_path)]) == 0

    with pytest.raises(HybridSubmissionError, match="Missing classifier test confidence"):
        main(["submit", "--config", str(config_path)])


def test_submit_requires_selected_config_by_default(tmp_path):
    from src.inference.hybrid_submission import HybridSubmissionError, main

    config_path = _write_config(tmp_path)

    with pytest.raises(HybridSubmissionError, match="selected validation-tuned hybrid_config.json"):
        main(["submit", "--config", str(config_path)])


def test_explicit_untuned_submit_allows_target_only_classifier_submission(tmp_path):
    from src.inference.hybrid_submission import main

    target_only = _write_csv(
        tmp_path / "artifacts" / "submission_v2b.csv",
        [
            {"image_id": "ta.jpg", "target": 0},
            {"image_id": "tb.jpg", "target": 1},
        ],
    )
    config_path = _write_config(tmp_path, classifier_test=target_only)

    assert main(["submit", "--config", str(config_path), "--allow-untuned-submit"]) == 0

    rows = list(
        csv.DictReader(
            (tmp_path / "outputs" / "hybrid" / "v4" / "submissions" / "submission_v4_hybrid.csv").open(
                "r",
                newline="",
                encoding="utf-8",
            )
        )
    )
    assert rows == [{"image_id": "ta.jpg", "target": "0"}, {"image_id": "tb.jpg", "target": "1"}]


def test_hybrid_uses_probability_input_not_hard_label_only_submission(tmp_path):
    from src.inference.hybrid_submission import main

    probability_input = _write_csv(
        tmp_path / "outputs" / "hybrid" / "v4" / "input" / "test_classifier_predictions_v2b.csv",
        [
            {"image_id": "ta.jpg", "prob_bad": 0.49, "classifier_prediction": 0, "target": 0},
            {"image_id": "tb.jpg", "prob_bad": 0.90, "classifier_prediction": 1, "target": 1},
        ],
    )
    config_path = _write_config(tmp_path, classifier_test=probability_input)

    assert main(["run", "--config", str(config_path)]) == 0

    output_dir = tmp_path / "outputs" / "hybrid" / "v4"
    hybrid_config = json.loads((output_dir / "reports" / "hybrid_config.json").read_text(encoding="utf-8"))
    diff = json.loads((output_dir / "reports" / "v2b_vs_v4_diff.json").read_text(encoding="utf-8"))

    assert hybrid_config["input_paths"]["classifier_test_predictions"].endswith("test_classifier_predictions_v2b.csv")
    assert diff["classifier_test_predictions"].endswith("test_classifier_predictions_v2b.csv")


def test_cli_rejects_sample_solution_or_test_label_paths(tmp_path):
    from src.inference.hybrid_submission import HybridSubmissionError, main

    forbidden = tmp_path / "sample_solution.csv"
    forbidden.write_text("image_id,target\nx.jpg,1\n", encoding="utf-8")
    config_path = _write_config(tmp_path, forbidden=forbidden)

    with pytest.raises(HybridSubmissionError, match="forbidden"):
        main(["search", "--config", str(config_path)])


def test_timing_report_is_written_when_timing_inputs_exist(tmp_path):
    from src.inference.hybrid_submission import main

    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["hybrid"]["timing"] = {
        "classifier_average_time_per_image": 0.1,
        "hybrid_average_time_per_image": 0.12,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["run", "--config", str(config_path)]) == 0

    report_path = tmp_path / "outputs" / "hybrid" / "v4" / "benchmarks" / "hybrid_timing_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["speed_multiplier_vs_classifier"] == pytest.approx(1.2)


def test_real_artifact_config_path_expectations():
    expected = {
        "test_classifier_predictions_v2b.csv": Path("outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv"),
        "submission_v2b.csv": Path("artifacts/kaggle_v2b_artifacts/kaggle_v2/v2b_effnet_b1/submissions/submission_v2b.csv"),
        "artifact category_mapping.json": Path("artifacts/krones_v3_yolov8n_50epoch_artifacts/reports/category_mapping.json"),
        "outputs detector category_mapping.json": Path("outputs/detector/reports/category_mapping.json"),
    }

    if expected["test_classifier_predictions_v2b.csv"].exists():
        rows = pd.read_csv(expected["test_classifier_predictions_v2b.csv"])
        assert len(rows) == 4418
        assert {"image_id", "prob_bad", "classifier_prediction", "target"} <= set(rows.columns)
        assert rows["prob_bad"].between(0, 1).all()
    assert expected["submission_v2b.csv"].exists() is True
    assert expected["artifact category_mapping.json"].exists() is True
    assert expected["outputs detector category_mapping.json"].exists() is False


def test_real_artifact_hybrid_run_creates_reports_and_submission(tmp_path, monkeypatch):
    from src.inference.hybrid_submission import main

    required = [
        Path("configs/hybrid_inference.yaml"),
        Path("outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv"),
        Path("artifacts/kaggle_v2b_artifacts/kaggle_v2/v2b_effnet_b1/kaggle_v2/predictions/val_classifier_predictions.csv"),
        Path("artifacts/kaggle_v2b_artifacts/kaggle_v2/v2b_effnet_b1/kaggle_v2/reports/best_threshold.json"),
        Path("artifacts/krones_v3_yolov8n_50epoch_artifacts/val_detector_binary_predictions.csv"),
        Path("artifacts/krones_v3_yolov8n_50epoch_artifacts/test_detector_predictions.csv"),
        Path("artifacts/krones_v3_yolov8n_50epoch_artifacts/reports/category_mapping.json"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip("real V4 artifacts not available: " + ", ".join(missing))

    config = yaml.safe_load(Path("configs/hybrid_inference.yaml").read_text(encoding="utf-8"))
    assert config["classifier"]["test_predictions"].endswith("outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv")
    assert config["detector"]["category_mapping"].endswith("artifacts/krones_v3_yolov8n_50epoch_artifacts/reports/category_mapping.json")
    config["hybrid"]["output_dir"] = str(tmp_path / "outputs" / "hybrid" / "v4")
    config["hybrid"]["uncertainty_margin_candidates"] = [0.05]
    config["hybrid"]["detector_conf_threshold_candidates"] = [0.25]
    config["hybrid"]["always_faulty_categories"] = ["binary_detector_positive"]
    config_path = tmp_path / "hybrid_inference.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.chdir(Path.cwd())
    assert main(["run", "--config", str(config_path)]) == 0

    output_dir = tmp_path / "outputs" / "hybrid" / "v4"
    metrics = json.loads((output_dir / "reports" / "hybrid_metrics.json").read_text(encoding="utf-8"))
    hybrid_config = json.loads((output_dir / "reports" / "hybrid_config.json").read_text(encoding="utf-8"))
    overlap = json.loads((output_dir / "reports" / "overlap_report.json").read_text(encoding="utf-8"))
    diff = json.loads((output_dir / "reports" / "v2b_vs_v4_diff.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((output_dir / "submissions" / "submission_v4_hybrid.csv").open("r", newline="", encoding="utf-8")))

    assert metrics["overlap_count"] > 0
    assert hybrid_config["input_paths"]["classifier_test_predictions"].endswith("test_classifier_predictions_v2b.csv")
    assert overlap["used_for_parameter_search"] == "validation_overlap_only"
    assert diff["row_count"] == 4418
    assert diff["classifier_test_predictions"].endswith("test_classifier_predictions_v2b.csv")
    assert rows
    assert list(rows[0].keys()) == ["image_id", "target"]
    assert {row["target"] for row in rows} <= {"0", "1"}
