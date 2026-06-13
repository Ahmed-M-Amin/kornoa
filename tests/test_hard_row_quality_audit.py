"""Tests for the hard-row quality audit workflow."""

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


def _base_setup(tmp_path: Path) -> dict[str, Path]:
    (tmp_path / ".git").mkdir(parents=True, exist_ok=True)

    hard_negatives = _write_csv(
        tmp_path / "outputs" / "analysis" / "v2_1_error_intelligence" / "fiftyone_review" / "v2_2_candidate_hard_negatives_all.csv",
        [
            {
                "image_id": "nested/a.jpg",
                "filepath": "train_images/nested/a.jpg",
                "error_type": "FP",
                "true_label": 0,
                "v2b_prediction": 1,
                "v2b_probability": 0.88,
                "review_group": "high_confidence_false_positives",
                "reviewer_note": "Looks acceptable but is repeatedly over-rejected.",
                "auto_triage_reason": "good bottle rejected by model; useful hard negative",
            },
            {
                "image_id": "nested/a.jpg",
                "filepath": "train_images/nested/a.jpg",
                "error_type": "FP",
                "true_label": 0,
                "v2b_prediction": 1,
                "v2b_probability": 0.88,
                "review_group": "over_rejected_reusable",
                "reviewer_note": "Duplicate row should be deduplicated.",
                "auto_triage_reason": "good bottle rejected by model; useful hard negative",
            },
            {
                "image_id": "nested\\b.jpg",
                "filepath": "train_images/nested/b.jpg",
                "error_type": "FP",
                "true_label": 0,
                "v2b_prediction": 1,
                "v2b_probability": 0.82,
                "review_group": "near_threshold_false_positives",
                "reviewer_note": "Crop may be clipping the neck ring.",
                "auto_triage_reason": "good bottle rejected by model; useful hard negative",
            },
        ],
    )
    hard_positives = _write_csv(
        tmp_path / "outputs" / "analysis" / "v2_1_error_intelligence" / "fiftyone_review" / "v2_2_candidate_hard_positives_all.csv",
        [
            {
                "image_id": "./c.jpg",
                "filepath": "train_images/c.jpg",
                "error_type": "FN",
                "true_label": 1,
                "v2b_prediction": 0,
                "v2b_probability": 0.27,
                "review_group": "near_threshold_false_negatives",
                "reviewer_note": "Defect is faint and missed by the classifier.",
                "auto_triage_reason": "bad bottle accepted by model; useful hard positive",
            }
        ],
    )
    uncertain_examples = _write_csv(
        tmp_path / "outputs" / "analysis" / "v2_1_error_intelligence" / "fiftyone_review" / "v2_2_uncertain_examples_all.csv",
        [
            {
                "image_id": "d.jpg",
                "filepath": "train_images/d.jpg",
                "error_type": "FP",
                "true_label": 0,
                "v2b_prediction": 1,
                "v2b_probability": 0.34,
                "review_group": "near_threshold_false_positives",
                "reviewer_note": "Borderline row needs manual adjudication.",
                "auto_triage_reason": "near-threshold ambiguous",
            },
            {
                "image_id": "./c.jpg",
                "filepath": "train_images/c.jpg",
                "error_type": "FN",
                "true_label": 1,
                "v2b_prediction": 0,
                "v2b_probability": 0.27,
                "review_group": "near_threshold_false_negatives",
                "reviewer_note": "Overlap with hard positive should become ambiguous.",
                "auto_triage_reason": "near-threshold ambiguous",
            },
        ],
    )
    validation_predictions = _write_csv(
        tmp_path / "artifacts" / "kaggle_v2b_artifacts" / "kaggle_v2" / "v2b_effnet_b1" / "kaggle_v2" / "predictions" / "val_classifier_predictions.csv",
        [
            {"image_id": "a.jpg", "true_label": 0, "probability": 0.88, "threshold": 0.32, "predicted_label": 1},
            {"image_id": "b.jpg", "true_label": 0, "probability": 0.82, "threshold": 0.32, "predicted_label": 1},
            {"image_id": "c.jpg", "true_label": 1, "probability": 0.27, "threshold": 0.32, "predicted_label": 0},
            {"image_id": "d.jpg", "true_label": 0, "probability": 0.34, "threshold": 0.32, "predicted_label": 1},
            {"image_id": "e.jpg", "true_label": 1, "probability": 0.91, "threshold": 0.32, "predicted_label": 1},
        ],
    )
    detector_evidence = _write_csv(
        tmp_path / "evidence" / "detector_evidence.csv",
        [
            {"image_id": "a.jpg", "detector_summary": "low-confidence but present"},
            {"image_id": "d.jpg", "detector_evidence_status": "present", "detector_summary": "weak defect localization"},
        ],
    )
    image_quality = _write_csv(
        tmp_path / "evidence" / "image_quality.csv",
        [
            {"image_id": "b.jpg", "crop_quality_summary": "crop trims the cap edge"},
            {"image_id": "c.jpg", "crop_quality_status": "present", "crop_quality_summary": "slight blur in ROI"},
        ],
    )

    config_path = tmp_path / "configs" / "hard_row_quality_audit.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"name": "hard_row_quality_audit", "seed": 42},
                "data": {
                    "hard_negatives": "outputs/analysis/v2_1_error_intelligence/fiftyone_review/v2_2_candidate_hard_negatives_all.csv",
                    "hard_positives": "outputs/analysis/v2_1_error_intelligence/fiftyone_review/v2_2_candidate_hard_positives_all.csv",
                    "uncertain_examples": "outputs/analysis/v2_1_error_intelligence/fiftyone_review/v2_2_uncertain_examples_all.csv",
                    "validation_predictions": "artifacts/kaggle_v2b_artifacts/kaggle_v2/v2b_effnet_b1/kaggle_v2/predictions/val_classifier_predictions.csv",
                },
                "evidence": {
                    "detector": str(detector_evidence),
                    "image_quality": str(image_quality),
                },
                "output": {
                    "analysis_root": "outputs/analysis/hard_row_quality_audit",
                },
                "safety": {
                    "allow_test_labels": False,
                    "public_leaderboard_input": False,
                    "generate_submission": False,
                    "train_model": False,
                    "apply_relabels": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "config": config_path,
        "hard_negatives": hard_negatives,
        "hard_positives": hard_positives,
        "uncertain_examples": uncertain_examples,
        "validation_predictions": validation_predictions,
        "detector_evidence": detector_evidence,
        "image_quality": image_quality,
        "analysis_root": tmp_path / "outputs" / "analysis" / "hard_row_quality_audit",
        "repo_root": tmp_path,
    }


def test_hard_row_audit_resolves_relative_paths_from_repo_root(tmp_path):
    from src.analysis.hard_row_quality_audit import load_hard_row_audit_config, run_hard_row_quality_audit

    paths = _base_setup(tmp_path)
    config = load_hard_row_audit_config(paths["config"])

    assert "hard_negatives" in config["data"]
    outputs = run_hard_row_quality_audit(paths["config"])
    assert outputs["audit"].is_file()
    assert str(outputs["audit"]).startswith(str(paths["repo_root"]))


def test_hard_row_audit_rejects_test_labels_public_leaderboard_training_submission_and_relabels(tmp_path):
    from src.analysis.hard_row_quality_audit import HardRowQualityAuditError, run_hard_row_quality_audit

    paths = _base_setup(tmp_path)
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))

    payload["safety"]["allow_test_labels"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(HardRowQualityAuditError, match="test labels"):
        run_hard_row_quality_audit(paths["config"])

    payload["safety"]["allow_test_labels"] = False
    payload["safety"]["public_leaderboard_input"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(HardRowQualityAuditError, match="public leaderboard"):
        run_hard_row_quality_audit(paths["config"])

    payload["safety"]["public_leaderboard_input"] = False
    payload["safety"]["train_model"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(HardRowQualityAuditError, match="training"):
        run_hard_row_quality_audit(paths["config"])

    payload["safety"]["train_model"] = False
    payload["safety"]["generate_submission"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(HardRowQualityAuditError, match="submission"):
        run_hard_row_quality_audit(paths["config"])

    payload["safety"]["generate_submission"] = False
    payload["safety"]["apply_relabels"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(HardRowQualityAuditError, match="relabel"):
        run_hard_row_quality_audit(paths["config"])


def test_hard_row_audit_builds_governed_rows_from_real_csv_style_and_deduplicates_image_ids(tmp_path):
    from src.analysis.hard_row_quality_audit import run_hard_row_quality_audit

    paths = _base_setup(tmp_path)
    outputs = run_hard_row_quality_audit(paths["config"])

    audit = pd.read_csv(outputs["audit"], keep_default_na=False).set_index("image_id")
    assert audit.index.tolist() == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert audit.index.is_unique
    assert audit.loc["a.jpg", "hard_example_type"] == "hard_negative"
    assert audit.loc["c.jpg", "hard_example_type"] == "hard_positive,uncertain"
    assert audit.loc["d.jpg", "hard_example_type"] == "uncertain"
    assert audit.loc["a.jpg", "primary_audit_category"] == "likely_correct_but_hard"
    assert audit.loc["c.jpg", "primary_audit_category"] == "likely_ambiguous"
    assert audit.loc["d.jpg", "primary_audit_category"] == "likely_ambiguous"


def test_hard_row_audit_uses_validation_predictions_as_row_context_and_ignores_extra_rows(tmp_path):
    from src.analysis.hard_row_quality_audit import run_hard_row_quality_audit

    paths = _base_setup(tmp_path)
    outputs = run_hard_row_quality_audit(paths["config"])

    audit = pd.read_csv(outputs["audit"], keep_default_na=False)
    assert len(audit) == 4
    assert set(audit["image_id"]) == {"a.jpg", "b.jpg", "c.jpg", "d.jpg"}
    assert "e.jpg" not in set(audit["image_id"])
    assert set(audit["target"]) == {0, 1}
    assert set(audit["baseline_prediction"]) == {0, 1}
    assert set(audit["candidate_prediction"]) == {0, 1}


def test_hard_row_audit_fails_clearly_if_validation_predictions_do_not_cover_governed_rows(tmp_path):
    from src.analysis.hard_row_quality_audit import HardRowQualityAuditError, run_hard_row_quality_audit

    paths = _base_setup(tmp_path)
    rows = pd.read_csv(paths["validation_predictions"])
    rows = rows.loc[rows["image_id"] != "d.jpg"]
    rows.to_csv(paths["validation_predictions"], index=False)

    with pytest.raises(HardRowQualityAuditError, match="governed-row coverage"):
        run_hard_row_quality_audit(paths["config"])


def test_hard_row_audit_handles_missing_optional_evidence_files_safely(tmp_path):
    from src.analysis.hard_row_quality_audit import run_hard_row_quality_audit

    paths = _base_setup(tmp_path)
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    payload["evidence"]["detector"] = "missing/detector.csv"
    payload["evidence"]["image_quality"] = "missing/image_quality.csv"
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")

    outputs = run_hard_row_quality_audit(paths["config"])
    audit = pd.read_csv(outputs["audit"], keep_default_na=False)

    assert len(audit) == 4
    assert set(audit["detector_evidence_status"]) == {"missing"}
    assert set(audit["crop_quality_status"]) == {"missing"}


def test_hard_row_audit_attaches_optional_evidence_with_present_or_missing_status(tmp_path):
    from src.analysis.hard_row_quality_audit import run_hard_row_quality_audit

    paths = _base_setup(tmp_path)
    outputs = run_hard_row_quality_audit(paths["config"])

    audit = pd.read_csv(outputs["audit"], keep_default_na=False).set_index("image_id")
    assert audit.loc["a.jpg", "detector_evidence_status"] == "present"
    assert audit.loc["a.jpg", "detector_summary"] == "low-confidence but present"
    assert audit.loc["b.jpg", "detector_evidence_status"] == "missing"
    assert audit.loc["b.jpg", "crop_quality_status"] == "present"
    assert audit.loc["b.jpg", "crop_quality_summary"] == "crop trims the cap edge"


def test_hard_row_audit_preserves_category_count_and_summary_outputs(tmp_path):
    from src.analysis.hard_row_quality_audit import run_hard_row_quality_audit

    paths = _base_setup(tmp_path)
    outputs = run_hard_row_quality_audit(paths["config"])

    counts = pd.read_csv(outputs["category_counts"])
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    failure_modes = json.loads(outputs["failure_modes"].read_text(encoding="utf-8"))

    assert counts["row_count"].sum() == 4
    assert summary["total_audited_rows"] == 4
    assert summary["used_test_labels"] is False
    assert summary["trained_model"] is False
    assert summary["submission_created"] is False
    assert summary["leaderboard_tuning_used"] is False
    assert failure_modes["total_audited_rows"] == 4
    assert failure_modes["failure_modes"][0]["rank"] == 1
    assert failure_modes["failure_modes"][0]["recommended_next_action"]


def test_hard_row_audit_cli_reports_expected_errors_without_traceback(tmp_path, capsys):
    from src.analysis.hard_row_quality_audit import main

    paths = _base_setup(tmp_path)
    paths["validation_predictions"].unlink()

    exit_code = main(["run", "--config", str(paths["config"])])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "failed" in captured.err.lower()
    assert "traceback" not in captured.err.lower()
