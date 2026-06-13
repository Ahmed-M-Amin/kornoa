"""Tests for V2.2 same-split validation evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_setup(tmp_path: Path) -> dict[str, Path]:
    dataset_root = tmp_path / "1st-krones-vision-ai-challenge"
    train_images = dataset_root / "train_images"
    train_images.mkdir(parents=True, exist_ok=True)
    for image_id in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        (train_images / image_id).write_bytes(b"fake")

    v2b_predictions = _write_csv(
        tmp_path / "artifacts" / "v2b" / "predictions" / "val_classifier_predictions.csv",
        [
            {"image_id": "a.jpg", "true_label": 1, "probability": 0.90, "predicted_label": 1, "threshold": 0.42},
            {"image_id": "b.jpg", "true_label": 0, "probability": 0.70, "predicted_label": 1, "threshold": 0.42},
            {"image_id": "c.jpg", "true_label": 1, "probability": 0.40, "predicted_label": 0, "threshold": 0.42},
            {"image_id": "d.jpg", "true_label": 0, "probability": 0.20, "predicted_label": 0, "threshold": 0.42},
        ],
    )
    checkpoint = (tmp_path / "outputs" / "kaggle_v2_2" / "v2_2_hard_examples" / "models" / "classifier_best.pth")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fake-checkpoint")
    threshold = _write_json(
        tmp_path / "outputs" / "kaggle_v2_2" / "v2_2_hard_examples" / "reports" / "best_threshold.json",
        {"threshold": 0.42},
    )
    hard_negatives = _write_csv(
        tmp_path / "outputs" / "analysis" / "v2_1_error_intelligence" / "fiftyone_review" / "v2_2_candidate_hard_negatives_all.csv",
        [{"image_id": "b.jpg", "error_type": "FP", "true_label": 0, "v2b_prediction": 1}],
    )
    hard_positives = _write_csv(
        tmp_path / "outputs" / "analysis" / "v2_1_error_intelligence" / "fiftyone_review" / "v2_2_candidate_hard_positives_all.csv",
        [{"image_id": "c.jpg", "error_type": "FN", "true_label": 1, "v2b_prediction": 0}],
    )
    uncertain = _write_csv(
        tmp_path / "outputs" / "analysis" / "v2_1_error_intelligence" / "fiftyone_review" / "v2_2_uncertain_examples_all.csv",
        [{"image_id": "d.jpg"}],
    )

    config_path = tmp_path / "configs" / "v2_2_hard_examples.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "v2_2_hard_examples", "seed": 42},
                "data": {
                    "dataset_root": str(dataset_root),
                    "split_source": "v2b_compatible",
                    "validation_split": 0.2,
                    "original_v2b_validation_predictions": str(v2b_predictions),
                },
                "output": {
                    "root": str(tmp_path / "outputs" / "kaggle_v2_2" / "v2_2_hard_examples"),
                    "analysis_root": str(tmp_path / "outputs" / "analysis" / "v2_2_same_split_eval"),
                    "rolling_comparison_table": str(
                        tmp_path
                        / "outputs"
                        / "analysis"
                        / "v2_2_same_split_eval"
                        / "reports"
                        / "candidate_same_split_comparison.csv"
                    ),
                },
                "model": {
                    "candidate_name": "v2_2_hard_examples",
                    "model_name": "tiny_cnn",
                    "image_size": 384,
                    "num_classes": 2,
                    "checkpoint": str(checkpoint),
                    "threshold_report": str(threshold),
                },
                "training": {
                    "device": "cpu",
                    "batch_size": 4,
                },
                "hard_examples": {
                    "enabled": True,
                    "strategy": "conservative_loss_weighting",
                    "hard_negatives": str(hard_negatives),
                    "hard_positives": str(hard_positives),
                    "uncertain_examples": str(uncertain),
                },
                "metrics": {
                    "validation_f1": "binary",
                    "positive_class": 1,
                    "zero_division": 0,
                    "threshold_selection": "validation_only",
                },
                "safety": {
                    "allow_test_labels": False,
                    "generate_submission": False,
                    "train_detector": False,
                    "overwrite_v2b_artifacts": False,
                },
            }
        ),
        encoding="utf-8",
    )

    return {
        "config": config_path,
        "dataset_root": dataset_root,
        "v2b_predictions": v2b_predictions,
        "checkpoint": checkpoint,
        "threshold": threshold,
        "hard_negatives": hard_negatives,
        "hard_positives": hard_positives,
        "uncertain": uncertain,
        "analysis_output": tmp_path / "outputs" / "analysis" / "v2_2_same_split_eval",
    }


def _patch_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.analysis import v2_2_same_split_eval

    def _fake_infer(*, image_paths: list[Path], threshold: float, **_: object) -> pd.DataFrame:
        lookup = {
            "a.jpg": 0.80,
            "b.jpg": 0.30,
            "c.jpg": 0.60,
            "d.jpg": 0.10,
        }
        rows = []
        for image_path in image_paths:
            probability = lookup[image_path.name]
            rows.append(
                {
                    "image_id": image_path.name,
                    "v22_probability": probability,
                    "v22_prediction": int(probability >= threshold),
                    "v22_threshold": threshold,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(v2_2_same_split_eval, "_infer_v22_predictions", _fake_infer)


def test_same_split_eval_writes_required_outputs_and_same_row_metrics(tmp_path, monkeypatch):
    from src.analysis.v2_2_same_split_eval import run_same_split_evaluation

    paths = _base_setup(tmp_path)
    _patch_inference(monkeypatch)

    outputs = run_same_split_evaluation(paths["config"])

    assert outputs["summary"].exists()
    assert outputs["metrics"].exists()
    assert outputs["threshold_sweep"].exists()
    assert outputs["v2b_wrong_v22_correct"].exists()
    assert outputs["v2b_correct_v22_wrong"].exists()
    assert outputs["both_wrong"].exists()
    assert outputs["both_correct"].exists()

    metrics = pd.read_csv(outputs["metrics"])
    assert set(metrics["section"]) == {
        "all_original_v2b_validation_rows",
        "hard_example_rows_only",
        "original_v2b_validation_excluding_hard_examples",
    }
    assert set(metrics["model"]) == {"v2b", "v2_2"}
    assert len(metrics) == 6

    all_v2b = metrics.loc[
        (metrics["section"] == "all_original_v2b_validation_rows") & (metrics["model"] == "v2b")
    ].iloc[0]
    all_v22 = metrics.loc[
        (metrics["section"] == "all_original_v2b_validation_rows") & (metrics["model"] == "v2_2")
    ].iloc[0]
    assert all_v2b["row_count"] == 4
    assert all_v2b["tp"] == 1
    assert all_v2b["fp"] == 1
    assert all_v2b["tn"] == 1
    assert all_v2b["fn"] == 1
    assert all_v2b["prediction_1_count"] == 2
    assert all_v2b["target_1_count"] == 2
    assert all_v22["f1"] == pytest.approx(1.0)
    assert all_v22["precision"] == pytest.approx(1.0)
    assert all_v22["recall"] == pytest.approx(1.0)

    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["evaluation_mode"] == "validation_only_same_split"
    assert summary["used_test_labels"] is False
    assert summary["trained_model"] is False
    assert summary["submission_created"] is False
    assert summary["v22_threshold"] == pytest.approx(0.42)
    assert summary["recommended_decision"]
    assert summary["decision_status"] in {"accepted", "rejected", "manual_review"}
    assert summary["candidate_checkpoint"].endswith("classifier_best.pth")
    assert summary["candidate_threshold_report"].endswith("best_threshold.json")
    assert summary["comparison_table_path"].endswith("candidate_same_split_comparison.csv")


def test_same_split_eval_normalizes_locked_row_identity_before_alignment(tmp_path, monkeypatch):
    from src.analysis.v2_2_same_split_eval import run_same_split_evaluation

    paths = _base_setup(tmp_path)
    _write_csv(
        paths["v2b_predictions"],
        [
            {"image_id": "nested/a.jpg", "true_label": 1, "probability": 0.90, "predicted_label": 1, "threshold": 0.42},
            {"image_id": "nested\\b.jpg", "true_label": 0, "probability": 0.70, "predicted_label": 1, "threshold": 0.42},
            {"image_id": "./c.jpg", "true_label": 1, "probability": 0.40, "predicted_label": 0, "threshold": 0.42},
            {"image_id": "d.jpg", "true_label": 0, "probability": 0.20, "predicted_label": 0, "threshold": 0.42},
        ],
    )
    _patch_inference(monkeypatch)

    outputs = run_same_split_evaluation(paths["config"])
    metrics = pd.read_csv(outputs["metrics"])
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))

    all_v2b = metrics.loc[
        (metrics["section"] == "all_original_v2b_validation_rows") & (metrics["model"] == "v2b")
    ].iloc[0]
    assert all_v2b["row_count"] == 4
    assert summary["row_count"] == 4


def test_same_split_config_supports_phase1_setup_keys(tmp_path):
    from src.analysis.v2_2_same_split_eval import (
        DEFAULT_COMPARISON_TABLE,
        FIRST_REQUIRED_CANDIDATE_NAME,
        load_same_split_config,
    )

    paths = _base_setup(tmp_path)

    config = load_same_split_config(paths["config"])

    assert config["data"]["original_v2b_validation_predictions"] == str(paths["v2b_predictions"])
    assert str(config["output"]["analysis_root"]).replace("\\", "/").endswith("outputs/analysis/v2_2_same_split_eval")
    assert str(config["output"]["rolling_comparison_table"]).replace("\\", "/").endswith(
        "candidate_same_split_comparison.csv"
    )
    assert config["model"]["candidate_name"] == FIRST_REQUIRED_CANDIDATE_NAME
    assert str(DEFAULT_COMPARISON_TABLE).endswith("candidate_same_split_comparison.csv")


def test_same_split_eval_separates_hard_examples_and_error_buckets(tmp_path, monkeypatch):
    from src.analysis.v2_2_same_split_eval import run_same_split_evaluation

    paths = _base_setup(tmp_path)
    _patch_inference(monkeypatch)

    outputs = run_same_split_evaluation(paths["config"])

    metrics = pd.read_csv(outputs["metrics"])
    hard_rows = metrics.loc[metrics["section"] == "hard_example_rows_only"]
    clean_rows = metrics.loc[metrics["section"] == "original_v2b_validation_excluding_hard_examples"]
    assert set(hard_rows["row_count"]) == {3}
    assert set(clean_rows["row_count"]) == {1}

    improved = pd.read_csv(outputs["v2b_wrong_v22_correct"])
    regressed = pd.read_csv(outputs["v2b_correct_v22_wrong"])
    both_wrong = pd.read_csv(outputs["both_wrong"])
    both_correct = pd.read_csv(outputs["both_correct"])

    assert improved["image_id"].tolist() == ["b.jpg", "c.jpg"]
    assert regressed.empty
    assert both_wrong.empty
    assert both_correct["image_id"].tolist() == ["a.jpg", "d.jpg"]
    assert set(improved["is_hard_example"]) == {True}
    assert set(both_correct["hard_example_type"].fillna("")) == {"", "uncertain"}


def test_same_split_eval_preserves_overlapping_hard_example_types(tmp_path, monkeypatch):
    from src.analysis.v2_2_same_split_eval import run_same_split_evaluation

    paths = _base_setup(tmp_path)
    _write_csv(
        paths["uncertain"],
        [
            {"image_id": "c.jpg"},
        ],
    )
    _patch_inference(monkeypatch)

    outputs = run_same_split_evaluation(paths["config"])
    improved = pd.read_csv(outputs["v2b_wrong_v22_correct"])

    overlap_row = improved.loc[improved["image_id"] == "c.jpg"].iloc[0]
    assert overlap_row["hard_example_type"] == "hard_positive,uncertain"


def test_same_split_eval_sets_accepted_when_full_score_improves_and_hard_rows_stay_within_tolerance(tmp_path, monkeypatch):
    from src.analysis.v2_2_same_split_eval import run_same_split_evaluation

    paths = _base_setup(tmp_path)
    _patch_inference(monkeypatch)

    summary = json.loads(run_same_split_evaluation(paths["config"])["summary"].read_text(encoding="utf-8"))

    assert summary["decision_status"] == "accepted"
    assert "within tolerance" in summary["decision_reason"]


def test_same_split_eval_sets_manual_review_for_mixed_outcomes(tmp_path, monkeypatch):
    from src.analysis import v2_2_same_split_eval

    paths = _base_setup(tmp_path)
    _write_csv(
        paths["hard_negatives"],
        [{"image_id": "b.jpg", "error_type": "FP", "true_label": 0, "v2b_prediction": 1}],
    )
    paths["hard_positives"].write_text("image_id,error_type,true_label,v2b_prediction\n", encoding="utf-8")
    _write_csv(paths["uncertain"], [{"image_id": "a.jpg"}])

    def _mixed_infer(*, image_paths: list[Path], threshold: float, **_: object) -> pd.DataFrame:
        lookup = {
            "a.jpg": 0.10,
            "b.jpg": 0.30,
            "c.jpg": 0.60,
            "d.jpg": 0.10,
        }
        rows = []
        for image_path in image_paths:
            probability = lookup[image_path.name]
            rows.append(
                {
                    "image_id": image_path.name,
                    "v22_probability": probability,
                    "v22_prediction": int(probability >= threshold),
                    "v22_threshold": threshold,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(v2_2_same_split_eval, "_infer_v22_predictions", _mixed_infer)
    summary = json.loads(v2_2_same_split_eval.run_same_split_evaluation(paths["config"])["summary"].read_text(encoding="utf-8"))

    assert summary["decision_status"] == "manual_review"
    assert "mixed" in summary["decision_reason"]


def test_same_split_eval_sets_rejected_for_non_improved_full_score_and_hard_regression(tmp_path, monkeypatch):
    from src.analysis import v2_2_same_split_eval

    paths = _base_setup(tmp_path)
    _write_csv(
        paths["hard_negatives"],
        [{"image_id": "b.jpg", "error_type": "FP", "true_label": 0, "v2b_prediction": 1}],
    )
    paths["hard_positives"].write_text("image_id,error_type,true_label,v2b_prediction\n", encoding="utf-8")
    _write_csv(paths["uncertain"], [{"image_id": "a.jpg"}])

    def _rejected_infer(*, image_paths: list[Path], threshold: float, **_: object) -> pd.DataFrame:
        lookup = {
            "a.jpg": 0.10,
            "b.jpg": 0.30,
            "c.jpg": 0.20,
            "d.jpg": 0.10,
        }
        rows = []
        for image_path in image_paths:
            probability = lookup[image_path.name]
            rows.append(
                {
                    "image_id": image_path.name,
                    "v22_probability": probability,
                    "v22_prediction": int(probability >= threshold),
                    "v22_threshold": threshold,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(v2_2_same_split_eval, "_infer_v22_predictions", _rejected_infer)
    summary = json.loads(v2_2_same_split_eval.run_same_split_evaluation(paths["config"])["summary"].read_text(encoding="utf-8"))

    assert summary["decision_status"] == "rejected"
    assert "did not improve" in summary["decision_reason"]


def test_same_split_eval_uses_binary_f1_with_positive_class_1_and_zero_division_0(tmp_path, monkeypatch):
    from src.analysis import v2_2_same_split_eval

    paths = _base_setup(tmp_path)

    def _always_negative(*, image_paths: list[Path], threshold: float, **_: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image_id": [path.name for path in image_paths],
                "v22_probability": [0.05 for _ in image_paths],
                "v22_prediction": [0 for _ in image_paths],
                "v22_threshold": [threshold for _ in image_paths],
            }
        )

    monkeypatch.setattr(v2_2_same_split_eval, "_infer_v22_predictions", _always_negative)
    outputs = v2_2_same_split_eval.run_same_split_evaluation(paths["config"])

    metrics = pd.read_csv(outputs["metrics"])
    row = metrics.loc[
        (metrics["section"] == "all_original_v2b_validation_rows") & (metrics["model"] == "v2_2")
    ].iloc[0]
    assert row["f1"] == pytest.approx(0.0)
    assert row["precision"] == pytest.approx(0.0)
    assert row["recall"] == pytest.approx(0.0)
    assert row["prediction_1_count"] == 0


def test_same_split_eval_fails_clearly_if_checkpoint_missing(tmp_path):
    from src.analysis.v2_2_same_split_eval import V22SameSplitEvalError, run_same_split_evaluation

    paths = _base_setup(tmp_path)
    paths["checkpoint"].unlink()

    with pytest.raises(V22SameSplitEvalError, match="checkpoint"):
        run_same_split_evaluation(paths["config"])


def test_same_split_eval_fails_clearly_if_v2b_predictions_missing(tmp_path):
    from src.analysis.v2_2_same_split_eval import V22SameSplitEvalError, run_same_split_evaluation

    paths = _base_setup(tmp_path)
    paths["v2b_predictions"].unlink()

    with pytest.raises(V22SameSplitEvalError, match="V2B validation predictions"):
        run_same_split_evaluation(paths["config"])


def test_same_split_eval_rejects_training_submission_and_test_label_modes(tmp_path):
    from src.analysis.v2_2_same_split_eval import V22SameSplitEvalError, load_same_split_config

    paths = _base_setup(tmp_path)
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    payload["safety"]["generate_submission"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(V22SameSplitEvalError, match="generate_submission"):
        load_same_split_config(paths["config"])

    payload["safety"]["generate_submission"] = False
    payload["safety"]["allow_test_labels"] = True
    payload["data"]["original_v2b_validation_predictions"] = str(tmp_path / "sample_submission.csv")
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(V22SameSplitEvalError, match="test labels"):
        load_same_split_config(paths["config"])


def test_same_split_eval_rejects_training_and_non_analysis_output_roots(tmp_path):
    from src.analysis.v2_2_same_split_eval import V22SameSplitEvalError, load_same_split_config

    paths = _base_setup(tmp_path)
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    payload["safety"]["train_model"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(V22SameSplitEvalError, match="forbids training"):
        load_same_split_config(paths["config"])

    payload["safety"]["train_model"] = False
    payload["output"]["analysis_root"] = str(tmp_path / "outputs" / "submissions")
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(V22SameSplitEvalError, match="analysis output"):
        load_same_split_config(paths["config"])


def test_same_split_eval_rejects_duplicate_or_non_binary_locked_rows(tmp_path):
    from src.analysis.v2_2_same_split_eval import V22SameSplitEvalError, run_same_split_evaluation

    duplicate_paths = _base_setup(tmp_path / "duplicate_case")
    _write_csv(
        duplicate_paths["v2b_predictions"],
        [
            {"image_id": "a.jpg", "true_label": 1, "probability": 0.90, "predicted_label": 1, "threshold": 0.42},
            {"image_id": "a.jpg", "true_label": 1, "probability": 0.70, "predicted_label": 1, "threshold": 0.42},
        ],
    )

    with pytest.raises(V22SameSplitEvalError, match="duplicate image_id"):
        run_same_split_evaluation(duplicate_paths["config"])

    non_binary_paths = _base_setup(tmp_path / "non_binary_case")
    _write_csv(
        non_binary_paths["v2b_predictions"],
        [
            {"image_id": "a.jpg", "true_label": 2, "probability": 0.90, "predicted_label": 1, "threshold": 0.42},
            {"image_id": "b.jpg", "true_label": 0, "probability": 0.20, "predicted_label": 0, "threshold": 0.42},
        ],
    )

    with pytest.raises(V22SameSplitEvalError, match="targets must be binary"):
        run_same_split_evaluation(non_binary_paths["config"])


def test_same_split_eval_rejects_row_count_mismatch_after_normalization(tmp_path, monkeypatch):
    from src.analysis import v2_2_same_split_eval

    paths = _base_setup(tmp_path)

    def _mismatched_infer(*, image_paths: list[Path], threshold: float, **_: object) -> pd.DataFrame:
        rows = []
        for image_path in image_paths[:-1]:
            rows.append(
                {
                    "image_id": image_path.name,
                    "v22_probability": 0.5,
                    "v22_prediction": int(0.5 >= threshold),
                    "v22_threshold": threshold,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(v2_2_same_split_eval, "_infer_v22_predictions", _mismatched_infer)

    with pytest.raises(v2_2_same_split_eval.V22SameSplitEvalError, match="same row set"):
        v2_2_same_split_eval.run_same_split_evaluation(paths["config"])


def test_same_split_config_enforces_one_candidate_and_required_candidate_provenance(tmp_path):
    from src.analysis.v2_2_same_split_eval import FIRST_REQUIRED_CANDIDATE_NAME, V22SameSplitEvalError, load_same_split_config

    paths = _base_setup(tmp_path)
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))

    payload["model"]["candidate_name"] = ""
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(V22SameSplitEvalError, match="candidate_name"):
        load_same_split_config(paths["config"])

    payload["model"]["candidate_name"] = FIRST_REQUIRED_CANDIDATE_NAME
    payload["model"]["candidate_names"] = ["v2_2_hard_examples", "v2_3_candidate"]
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(V22SameSplitEvalError, match="one candidate per run"):
        load_same_split_config(paths["config"])

    payload.pop("model", None)
    payload["candidate"] = {"name": FIRST_REQUIRED_CANDIDATE_NAME}
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(V22SameSplitEvalError, match="checkpoint"):
        load_same_split_config(paths["config"])


def test_same_split_eval_cli_reports_expected_errors_without_traceback(tmp_path, capsys):
    from src.analysis.v2_2_same_split_eval import main

    paths = _base_setup(tmp_path)
    paths["v2b_predictions"].unlink()

    exit_code = main(["run", "--config", str(paths["config"])])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "failed" in captured.err.lower()
    assert "traceback" not in captured.err.lower()


def test_same_split_eval_writes_exhaustive_bucket_outputs_with_required_columns(tmp_path, monkeypatch):
    from src.analysis.v2_2_same_split_eval import ERROR_COLUMNS, run_same_split_evaluation

    paths = _base_setup(tmp_path)
    _patch_inference(monkeypatch)

    outputs = run_same_split_evaluation(paths["config"])
    bucket_frames = [
        pd.read_csv(outputs["v2b_wrong_v22_correct"]),
        pd.read_csv(outputs["v2b_correct_v22_wrong"]),
        pd.read_csv(outputs["both_wrong"]),
        pd.read_csv(outputs["both_correct"]),
    ]

    assert sum(len(frame) for frame in bucket_frames) == 4
    for frame in bucket_frames:
        assert list(frame.columns) == ERROR_COLUMNS


def test_same_split_eval_updates_rolling_comparison_table_per_candidate(tmp_path, monkeypatch):
    from src.analysis import v2_2_same_split_eval

    paths = _base_setup(tmp_path)
    _patch_inference(monkeypatch)

    first_outputs = v2_2_same_split_eval.run_same_split_evaluation(paths["config"])
    table_path = first_outputs["comparison_table"]
    first_table = pd.read_csv(table_path)
    assert first_table["candidate_name"].tolist() == ["v2_2_hard_examples"]
    assert "decision_reason" in first_table.columns

    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    payload["model"]["candidate_name"] = "v2_3_candidate"
    second_checkpoint = tmp_path / "outputs" / "kaggle_v2_3" / "v2_3_candidate" / "models" / "classifier_best.pth"
    second_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    second_checkpoint.write_bytes(b"fake-checkpoint")
    second_threshold = _write_json(
        tmp_path / "outputs" / "kaggle_v2_3" / "v2_3_candidate" / "reports" / "best_threshold.json",
        {"threshold": 0.42},
    )
    payload["model"]["checkpoint"] = str(second_checkpoint)
    payload["model"]["threshold_report"] = str(second_threshold)
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")

    second_outputs = v2_2_same_split_eval.run_same_split_evaluation(paths["config"])
    second_table = pd.read_csv(second_outputs["comparison_table"])
    assert set(second_table["candidate_name"]) == {"v2_2_hard_examples", "v2_3_candidate"}
    _write_csv(
        paths["hard_negatives"],
        [{"image_id": "b.jpg", "error_type": "FP", "true_label": 0, "v2b_prediction": 1}],
    )
    paths["hard_positives"].write_text("image_id,error_type,true_label,v2b_prediction\n", encoding="utf-8")
    _write_csv(paths["uncertain"], [{"image_id": "a.jpg"}])

    def _manual_review_infer(*, image_paths: list[Path], threshold: float, **_: object) -> pd.DataFrame:
        lookup = {
            "a.jpg": 0.10,
            "b.jpg": 0.30,
            "c.jpg": 0.60,
            "d.jpg": 0.10,
        }
        return pd.DataFrame(
            {
                "image_id": [path.name for path in image_paths],
                "v22_probability": [lookup[path.name] for path in image_paths],
                "v22_prediction": [int(lookup[path.name] >= threshold) for path in image_paths],
                "v22_threshold": [threshold for _ in image_paths],
            }
        )

    monkeypatch.setattr(v2_2_same_split_eval, "_infer_v22_predictions", _manual_review_infer)
    v2_2_same_split_eval.run_same_split_evaluation(paths["config"])
    updated_table = pd.read_csv(table_path)
    updated_row = updated_table.loc[updated_table["candidate_name"] == "v2_3_candidate"].iloc[0]
    assert updated_row["decision_status"] == "manual_review"
    assert len(updated_table) == 2


def test_same_split_eval_succeeds_without_optional_evidence_inputs(tmp_path, monkeypatch):
    from src.analysis.v2_2_same_split_eval import run_same_split_evaluation

    paths = _base_setup(tmp_path)
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    payload["evidence"] = {"detector": "", "image_quality": ""}
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    _patch_inference(monkeypatch)

    outputs = run_same_split_evaluation(paths["config"])
    assert outputs["summary"].exists()
