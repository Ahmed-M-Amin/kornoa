"""Tests for the hard-row visual review workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image


REQUIRED_MANUAL_REVIEW_COLUMNS = [
    "image_id",
    "target",
    "baseline_probability",
    "baseline_prediction",
    "candidate_probability",
    "candidate_prediction",
    "bottle_type",
    "primary_audit_category",
    "hard_example_type",
    "suspicion_rank",
    "label_issue_score",
    "anomaly_score",
    "cluster_id",
    "cluster_description",
    "final_failure_mode",
    "label_quality_flag",
    "roi_preprocessing_flag",
    "detector_evidence_flag",
    "harmless_artifact_flag",
    "tiny_or_low_contrast_defect_flag",
    "reviewer_notes",
    "reviewer_id",
    "review_timestamp",
]
REQUIRED_ACTION_PLAN_COLUMNS = [
    "image_id",
    "target",
    "prediction",
    "probability",
    "bottle_type",
    "cluster_id",
    "suspicion_rank",
    "label_issue_score",
    "anomaly_score",
    "primary_audit_category",
    "hard_example_type",
    "recommended_action",
    "reason",
    "human_review_priority",
]
ALLOWED_RECOMMENDED_ACTIONS = {
    "safe_for_hard_training",
    "exclude_as_ambiguous",
    "suspected_mislabel",
    "roi_pipeline_bug",
    "needs_detector_evidence",
    "harmless_artifact_pattern",
    "rare_defect_cluster",
    "needs_expert_review",
}
MANUS_ALLOWED_ACTIONS = {
    "safe_for_hard_training",
    "safe_for_hard_training_with_artifact_focus",
    "safe_for_hard_training_with_high_resolution_focus",
    "exclude_as_ambiguous_until_verified",
    "needs_expert_review",
    "suspected_mislabel_exclude_from_training",
    "roi_pipeline_bug",
    "needs_detector_evidence",
    "rare_defect_cluster_or_outlier_review",
}
REQUIRED_MANUS_ACTION_COLUMNS = [
    "image_id",
    "target",
    "prediction",
    "probability",
    "bottle_type",
    "cluster_id",
    "suspicion_rank",
    "label_issue_score",
    "anomaly_score",
    "primary_audit_category",
    "hard_example_type",
    "final_failure_mode",
    "recommended_action",
    "human_review_priority",
    "phase3_use_allowed",
    "reason",
]
REQUIRED_EVIDENCE_ACTION_COLUMNS = [
    "image_id",
    "target",
    "prediction",
    "probability",
    "bottle_type",
    "cluster_id",
    "suspicion_rank",
    "label_issue_score",
    "anomaly_score",
    "primary_audit_category",
    "hard_example_type",
    "annotation_count",
    "defect_categories",
    "largest_defect_area_ratio",
    "total_defect_area_ratio",
    "annotation_evidence_status",
    "detector_evidence_status",
    "roi_exists",
    "roi_coverage_ratio",
    "crop_quality_status",
    "brightness_score",
    "contrast_score",
    "blur_score_laplacian_variance",
    "edge_density",
    "recommended_action",
    "phase3_use_allowed",
    "reason",
]
REQUIRED_PHASE3_PACKAGE_SUMMARY_KEYS = {
    "input_file",
    "row_count",
    "phase3_use_allowed_true_count",
    "phase3_use_allowed_false_count",
    "allowed_outputs",
    "blocked_outputs",
    "safety",
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_setup(tmp_path: Path) -> dict[str, Path]:
    (tmp_path / ".git").mkdir(parents=True, exist_ok=True)
    train_images = tmp_path / "1st-krones-vision-ai-challenge" / "train_images"
    train_images.mkdir(parents=True, exist_ok=True)
    for image_id, color in {
        "a.png": (220, 40, 40),
        "b.png": (40, 220, 40),
        "c.png": (40, 40, 220),
    }.items():
        Image.new("RGB", (64, 96), color).save(train_images / image_id)

    audit_csv = _write_csv(
        tmp_path / "outputs" / "analysis" / "hard_row_quality_audit" / "audit" / "hard_row_audit.csv",
        [
            {
                "image_id": "a.png",
                "primary_audit_category": "likely_correct_but_hard",
                "audit_rationale": "Hard false positive with stable label.",
                "target": 1,
                "baseline_probability": 0.82,
                "baseline_prediction": 1,
                "candidate_probability": 0.76,
                "candidate_prediction": 1,
                "hard_example_type": "hard_positive",
                "detector_evidence_status": "missing",
                "detector_summary": "",
                "crop_quality_status": "missing",
                "crop_quality_summary": "",
                "secondary_notes": "",
            },
            {
                "image_id": "b.png",
                "primary_audit_category": "likely_ambiguous",
                "audit_rationale": "Near-threshold uncertain positive.",
                "target": 1,
                "baseline_probability": 0.31,
                "baseline_prediction": 0,
                "candidate_probability": 0.35,
                "candidate_prediction": 0,
                "hard_example_type": "hard_positive,uncertain",
                "detector_evidence_status": "present",
                "detector_summary": "weak localization",
                "crop_quality_status": "missing",
                "crop_quality_summary": "",
                "secondary_notes": "",
            },
            {
                "image_id": "c.png",
                "primary_audit_category": "likely_correct_but_hard",
                "audit_rationale": "Hard negative with crop concern.",
                "target": 0,
                "baseline_probability": 0.62,
                "baseline_prediction": 1,
                "candidate_probability": 0.55,
                "candidate_prediction": 1,
                "hard_example_type": "hard_negative",
                "detector_evidence_status": "missing",
                "detector_summary": "",
                "crop_quality_status": "present",
                "crop_quality_summary": "crop clips shoulder",
                "secondary_notes": "",
            },
        ],
    )
    train_csv = _write_csv(
        tmp_path / "1st-krones-vision-ai-challenge" / "train.csv",
        [
            {"image_id": "a.png", "target": 1, "bottle_type_id": 1},
            {"image_id": "b.png", "target": 1, "bottle_type_id": 2},
            {"image_id": "c.png", "target": 0, "bottle_type_id": 1},
        ],
    )
    bottletypes_csv = _write_csv(
        tmp_path / "1st-krones-vision-ai-challenge" / "bottletypes.csv",
        [
            {"id": 1, "name": "Amber crown"},
            {"id": 2, "name": "Euro brown"},
        ],
    )
    annotations_json = _write_json(
        tmp_path / "1st-krones-vision-ai-challenge" / "train_annotations.json",
        {
            "images": [
                {"id": 1, "file_name": "a.png"},
                {"id": 2, "file_name": "b.png"},
                {"id": 3, "file_name": "c.png"},
            ],
            "categories": [
                {"id": 1, "name": "spot"},
                {"id": 2, "name": "scratch"},
            ],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [8, 10, 20, 30]},
                {"image_id": 2, "category_id": 2, "bbox": [12, 16, 18, 22]},
            ],
        },
    )
    config_path = tmp_path / "configs" / "hard_row_visual_review.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "hard_row_visual_review",
                    "spec": "014",
                    "output_root": "outputs/analysis/hard_row_visual_review",
                },
                "inputs": {
                    "hard_row_audit_csv": "outputs/analysis/hard_row_quality_audit/audit/hard_row_audit.csv",
                    "train_images_dir": "1st-krones-vision-ai-challenge/train_images",
                    "train_csv": "1st-krones-vision-ai-challenge/train.csv",
                    "bottletypes_csv": "1st-krones-vision-ai-challenge/bottletypes.csv",
                    "train_annotations_json": "1st-krones-vision-ai-challenge/train_annotations.json",
                    "baseline_predictions_csv": None,
                    "candidate_predictions_csv": None,
                    "embeddings_path": None,
                    "detector_or_roi_artifacts": None,
                },
                "analysis": {
                    "expected_hard_row_count": 3,
                    "image_id_column": "image_id",
                    "target_column": "target",
                    "enable_cleanlab": True,
                    "cleanlab_optional": True,
                    "enable_fiftyone": True,
                    "fiftyone_optional": True,
                    "enable_clustering": True,
                    "enable_anomaly_detection": True,
                    "enable_contact_sheets": True,
                    "enable_enhanced_views": True,
                },
                "visuals": {
                    "thumbnail_size": 96,
                    "max_images_per_contact_sheet": 2,
                    "generate_original_view": True,
                    "generate_roi_crop_view": True,
                    "generate_contrast_enhanced_view": True,
                    "generate_edge_enhanced_view": True,
                    "generate_heatmap_view": False,
                },
                "clustering": {"method": "kmeans", "n_clusters": 2, "random_state": 42},
                "anomaly": {"method": "isolation_forest", "contamination": 0.2, "random_state": 42},
                "review": {
                    "default_final_failure_mode": "unclear_needs_annotation_or_expert_review",
                    "reviewer_id_default": "",
                    "preserve_audit_trail": True,
                },
                "triage": {
                    "enabled": True,
                    "output_root": "outputs/analysis/hard_row_visual_review",
                    "cluster_representatives_per_cluster": 2,
                    "top_n_suspicious": 2,
                    "top_n_label_issues": 2,
                    "top_n_anomalies": 2,
                    "allowed_recommended_actions": sorted(ALLOWED_RECOMMENDED_ACTIONS),
                    "thresholds": {
                        "high_label_issue_score": 0.75,
                        "high_anomaly_score": 0.75,
                        "near_boundary_margin": 0.10,
                        "high_priority_score": 0.75,
                    },
                    "safety": {
                        "training_enabled": False,
                        "submission_enabled": False,
                        "test_label_usage_enabled": False,
                        "label_modification_enabled": False,
                    },
                },
                "safety": {
                    "used_test_labels": False,
                    "training_enabled": False,
                    "submission_enabled": False,
                    "leaderboard_tuning_enabled": False,
                },
                "manus_review": {
                    "enabled": True,
                    "expected_rows": 3,
                    "input_dir": "outputs/analysis/hard_row_visual_review/manus_review",
                    "filled_manual_review_template": "outputs/analysis/hard_row_visual_review/manus_review/filled_manual_review_template.csv",
                    "review_summary_json": "outputs/analysis/hard_row_visual_review/manus_review/review_summary.json",
                    "human_verification_priority_list": "outputs/analysis/hard_row_visual_review/manus_review/human_verification_priority_list.csv",
                    "original_manual_review_template": "outputs/analysis/hard_row_visual_review/manual_review_template.csv",
                    "review_manifest": "outputs/analysis/hard_row_visual_review/review_manifest.csv",
                    "cluster_triage_summary": "outputs/analysis/hard_row_visual_review/reports/cluster_triage_summary.json",
                    "cluster_representatives": "outputs/analysis/hard_row_visual_review/reports/cluster_representatives.csv",
                    "top_suspicious_rows": "outputs/analysis/hard_row_visual_review/reports/top_suspicious_rows.csv",
                    "top_label_issue_rows": "outputs/analysis/hard_row_visual_review/reports/top_label_issue_rows.csv",
                    "top_anomaly_rows": "outputs/analysis/hard_row_visual_review/reports/top_anomaly_rows.csv",
                    "high_label_issue_threshold": 0.70,
                    "high_anomaly_threshold": 0.95,
                },
                "evidence_completion": {
                    "enabled": True,
                    "expected_rows": 3,
                    "expected_phase3_allowed_count": None,
                    "expected_phase3_blocked_count": None,
                    "review_manifest": "outputs/analysis/hard_row_visual_review/review_manifest.csv",
                    "manual_review_template": "outputs/analysis/hard_row_visual_review/manual_review_template.csv",
                    "hard_row_audit_csv": "outputs/analysis/hard_row_quality_audit/audit/hard_row_audit.csv",
                    "train_images_dir": "1st-krones-vision-ai-challenge/train_images",
                    "train_csv": "1st-krones-vision-ai-challenge/train.csv",
                    "train_annotations_json": "1st-krones-vision-ai-challenge/train_annotations.json",
                    "bottletypes_csv": "1st-krones-vision-ai-challenge/bottletypes.csv",
                    "detector_outputs_dir": None,
                    "output_root": "outputs/analysis/hard_row_visual_review",
                    "evidence_assets_dir": "outputs/analysis/hard_row_visual_review/evidence_assets",
                    "tiny_defect_area_ratio_threshold": 0.08,
                    "min_roi_coverage_ratio": 0.10,
                    "max_roi_coverage_ratio": 0.95,
                    "min_crop_width": 24,
                    "min_crop_height": 24,
                    "blur_laplacian_issue_threshold": 40.0,
                    "low_contrast_issue_threshold": 15.0,
                    "edge_density_low_threshold": 0.01,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "repo_root": tmp_path,
        "config": config_path,
        "audit_csv": audit_csv,
        "train_images": train_images,
        "train_csv": train_csv,
        "bottletypes_csv": bottletypes_csv,
        "annotations_json": annotations_json,
        "output_root": tmp_path / "outputs" / "analysis" / "hard_row_visual_review",
    }


def test_visual_review_workflow_creates_required_outputs_and_manual_review_template(tmp_path):
    from src.analysis.hard_row_visual_review import run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    outputs = run_hard_row_visual_review(paths["config"])

    manual_review = pd.read_csv(outputs["manual_review_template"], keep_default_na=False)
    manifest = pd.read_csv(outputs["review_manifest"], keep_default_na=False)
    summary = json.loads(outputs["visual_review_summary"].read_text(encoding="utf-8"))

    assert outputs["manual_review_template"].is_file()
    assert outputs["review_manifest"].is_file()
    assert outputs["clustering_summary"].is_file()
    assert outputs["anomaly_summary"].is_file()
    assert outputs["label_quality_summary"].is_file()
    assert outputs["cluster_representatives"].is_file()
    assert outputs["top_suspicious_rows"].is_file()
    assert outputs["top_label_issue_rows"].is_file()
    assert outputs["top_anomaly_rows"].is_file()
    assert outputs["cluster_triage_summary"].is_file()
    assert outputs["final_training_action_plan"].is_file()
    assert outputs["contact_sheets_dir"].is_dir()
    assert outputs["cluster_galleries_dir"].is_dir()
    assert outputs["top_suspicious_gallery_dir"].is_dir()
    assert list(manual_review.columns) == REQUIRED_MANUAL_REVIEW_COLUMNS
    assert set(manifest["image_id"]) == {"a.png", "b.png", "c.png"}
    assert manifest["suspicion_rank"].tolist() == [1, 2, 3]
    assert summary["trained_model"] is False
    assert summary["submission_created"] is False
    assert summary["used_test_labels"] is False


def test_visual_review_workflow_reports_missing_images_without_crashing(tmp_path):
    from src.analysis.hard_row_visual_review import run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    (paths["train_images"] / "c.png").unlink()

    outputs = run_hard_row_visual_review(paths["config"])

    missing_images = pd.read_csv(outputs["missing_images"], keep_default_na=False)
    assert missing_images["image_id"].tolist() == ["c.png"]
    assert outputs["manual_review_template"].is_file()


def test_visual_review_workflow_rejects_duplicate_image_ids(tmp_path):
    from src.analysis.hard_row_visual_review import HardRowVisualReviewError, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    audit = pd.read_csv(paths["audit_csv"], keep_default_na=False)
    pd.concat([audit, audit.iloc[[0]]], ignore_index=True).to_csv(paths["audit_csv"], index=False)

    with pytest.raises(HardRowVisualReviewError, match="duplicate image_id"):
        run_hard_row_visual_review(paths["config"])


def test_visual_review_workflow_rejects_unexpected_hard_row_count(tmp_path):
    from src.analysis.hard_row_visual_review import HardRowVisualReviewError, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    payload["analysis"]["expected_hard_row_count"] = 4
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(HardRowVisualReviewError, match="expected hard-row count"):
        run_hard_row_visual_review(paths["config"])


def test_visual_review_workflow_fails_when_audit_input_is_missing(tmp_path):
    from src.analysis.hard_row_visual_review import HardRowVisualReviewError, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    paths["audit_csv"].unlink()

    with pytest.raises(HardRowVisualReviewError, match="Missing required visual review input"):
        run_hard_row_visual_review(paths["config"])


def test_visual_review_workflow_enforces_allowed_final_failure_modes(tmp_path):
    from src.analysis.hard_row_visual_review import HardRowVisualReviewError, validate_manual_review_frame

    frame = pd.DataFrame(
        [
            {
                "image_id": "a.png",
                "target": 0,
                "baseline_probability": 0.9,
                "baseline_prediction": 1,
                "candidate_probability": 0.8,
                "candidate_prediction": 1,
                "bottle_type": "Amber crown",
                "primary_audit_category": "likely_correct_but_hard",
                "hard_example_type": "hard_negative",
                "suspicion_rank": 1,
                "label_issue_score": 0.2,
                "anomaly_score": 0.1,
                "cluster_id": 0,
                "cluster_description": "test cluster",
                "final_failure_mode": "not_allowed",
                "label_quality_flag": "",
                "roi_preprocessing_flag": "",
                "detector_evidence_flag": "",
                "harmless_artifact_flag": "",
                "tiny_or_low_contrast_defect_flag": "",
                "reviewer_notes": "",
                "reviewer_id": "",
                "review_timestamp": "",
            }
        ]
    )

    with pytest.raises(HardRowVisualReviewError, match="Invalid final_failure_mode"):
        validate_manual_review_frame(frame)


def test_visual_review_workflow_handles_missing_cleanlab_and_fiftyone_safely(tmp_path, monkeypatch):
    import src.analysis.hard_row_visual_review as visual_review

    paths = _base_setup(tmp_path)

    def fake_optional_import(name: str):
        if name in {"cleanlab", "fiftyone"}:
            return None
        raise AssertionError(f"unexpected optional import: {name}")

    monkeypatch.setattr(visual_review, "_import_optional_dependency", fake_optional_import)
    outputs = visual_review.run_hard_row_visual_review(paths["config"])

    label_quality = json.loads(outputs["label_quality_summary"].read_text(encoding="utf-8"))
    summary = json.loads(outputs["visual_review_summary"].read_text(encoding="utf-8"))
    assert label_quality["status"] in {"heuristic_fallback", "disabled"}
    assert summary["cleanlab_integration"]["available"] is False
    assert summary["fiftyone_integration"]["available"] is False


def test_visual_review_workflow_writes_clustering_anomaly_and_label_quality_summaries(tmp_path):
    from src.analysis.hard_row_visual_review import run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    outputs = run_hard_row_visual_review(paths["config"])

    clustering = json.loads(outputs["clustering_summary"].read_text(encoding="utf-8"))
    anomaly = json.loads(outputs["anomaly_summary"].read_text(encoding="utf-8"))
    label_quality = json.loads(outputs["label_quality_summary"].read_text(encoding="utf-8"))

    assert clustering["row_count"] == 3
    assert clustering["cluster_count"] >= 1
    assert "clusters" in clustering
    assert anomaly["row_count"] == 3
    assert "top_anomalies" in anomaly
    assert label_quality["row_count"] == 3
    assert "score_summary" in label_quality


def test_visual_review_workflow_creates_triage_reports_and_allowed_recommended_actions(tmp_path):
    from src.analysis.hard_row_visual_review import run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    outputs = run_hard_row_visual_review(paths["config"])

    action_plan = pd.read_csv(outputs["final_training_action_plan"], keep_default_na=False)
    cluster_representatives = pd.read_csv(outputs["cluster_representatives"], keep_default_na=False)
    top_suspicious = pd.read_csv(outputs["top_suspicious_rows"], keep_default_na=False)
    top_label_issues = pd.read_csv(outputs["top_label_issue_rows"], keep_default_na=False)
    top_anomalies = pd.read_csv(outputs["top_anomaly_rows"], keep_default_na=False)
    triage_summary = json.loads(outputs["cluster_triage_summary"].read_text(encoding="utf-8"))

    assert list(action_plan.columns) == REQUIRED_ACTION_PLAN_COLUMNS
    assert set(action_plan["recommended_action"]).issubset(ALLOWED_RECOMMENDED_ACTIONS)
    assert set(cluster_representatives["recommended_action"]).issubset(ALLOWED_RECOMMENDED_ACTIONS)
    assert (cluster_representatives.groupby("cluster_id").size() <= 2).all()
    assert len(top_suspicious) == 2
    assert len(top_label_issues) == 2
    assert len(top_anomalies) == 2
    assert triage_summary["top_suspicious_rows_count"] == 2
    assert triage_summary["top_label_issue_rows_count"] == 2
    assert triage_summary["top_anomaly_rows_count"] == 2
    assert triage_summary["top_suspicious_gallery_created"] is True


def test_visual_review_workflow_handles_missing_embeddings_features_without_crashing(tmp_path, monkeypatch):
    import src.analysis.hard_row_visual_review as visual_review

    paths = _base_setup(tmp_path)

    def fake_embeddings(review, *, images_root):
        return np.zeros((len(review), 3), dtype=float), ["fallback_1", "fallback_2", "fallback_3"]

    monkeypatch.setattr(visual_review, "_compute_embeddings", fake_embeddings)
    outputs = visual_review.run_hard_row_visual_review(paths["config"])

    triage_summary = json.loads(outputs["cluster_triage_summary"].read_text(encoding="utf-8"))
    assert outputs["final_training_action_plan"].is_file()
    assert "missing_embeddings_fallback_used" in triage_summary


def test_visual_review_config_rejects_training_submission_and_test_label_usage(tmp_path):
    from src.analysis.hard_row_visual_review import HardRowVisualReviewError, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))

    payload["safety"]["training_enabled"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(HardRowVisualReviewError, match="training"):
        run_hard_row_visual_review(paths["config"])

    payload["safety"]["training_enabled"] = False
    payload["safety"]["submission_enabled"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(HardRowVisualReviewError, match="submission"):
        run_hard_row_visual_review(paths["config"])

    payload["safety"]["submission_enabled"] = False
    payload["safety"]["used_test_labels"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(HardRowVisualReviewError, match="test labels"):
        run_hard_row_visual_review(paths["config"])

    payload["safety"]["used_test_labels"] = False
    payload["safety"]["leaderboard_tuning_enabled"] = True
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(HardRowVisualReviewError, match="leaderboard tuning"):
        run_hard_row_visual_review(paths["config"])


def _write_manus_review_files(paths: dict[str, Path], *, metadata_mutation: dict[str, object] | None = None) -> None:
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    payload["manus_review"]["high_anomaly_threshold"] = 1.10
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")

    manual_review = pd.read_csv(paths["output_root"] / "manual_review_template.csv", keep_default_na=False)
    manual_review.loc[manual_review["image_id"] == "a.png", "final_failure_mode"] = "correct_but_visually_hard"
    manual_review.loc[manual_review["image_id"] == "a.png", "label_quality_flag"] = "label_ok"
    manual_review.loc[manual_review["image_id"] == "a.png", "roi_preprocessing_flag"] = "no"
    manual_review.loc[manual_review["image_id"] == "a.png", "detector_evidence_flag"] = "not_applicable"
    manual_review.loc[manual_review["image_id"] == "a.png", "harmless_artifact_flag"] = "no"
    manual_review.loc[manual_review["image_id"] == "a.png", "tiny_or_low_contrast_defect_flag"] = "no"
    manual_review.loc[manual_review["image_id"] == "b.png", "final_failure_mode"] = "likely_mislabeled"
    manual_review.loc[manual_review["image_id"] == "b.png", "label_quality_flag"] = "label_suspicious"
    manual_review.loc[manual_review["image_id"] == "b.png", "roi_preprocessing_flag"] = "needs_review"
    manual_review.loc[manual_review["image_id"] == "b.png", "detector_evidence_flag"] = "needs_review"
    manual_review.loc[manual_review["image_id"] == "b.png", "harmless_artifact_flag"] = "unknown"
    manual_review.loc[manual_review["image_id"] == "b.png", "tiny_or_low_contrast_defect_flag"] = "no"
    manual_review.loc[manual_review["image_id"] == "c.png", "final_failure_mode"] = "roi_or_crop_problem"
    manual_review.loc[manual_review["image_id"] == "c.png", "label_quality_flag"] = "label_ok"
    manual_review.loc[manual_review["image_id"] == "c.png", "roi_preprocessing_flag"] = "yes"
    manual_review.loc[manual_review["image_id"] == "c.png", "detector_evidence_flag"] = "needs_review"
    manual_review.loc[manual_review["image_id"] == "c.png", "harmless_artifact_flag"] = "no"
    manual_review.loc[manual_review["image_id"] == "c.png", "tiny_or_low_contrast_defect_flag"] = "no"
    manual_review["reviewer_id"] = "Manus_AI_Reviewer"
    manual_review["review_timestamp"] = "2026-06-12"
    manual_review["reviewer_notes"] = "synthetic manus review"
    if metadata_mutation:
        for column, value in metadata_mutation.items():
            manual_review.loc[0, column] = value

    manus_dir = paths["output_root"] / "manus_review"
    manus_dir.mkdir(parents=True, exist_ok=True)
    manual_review.to_csv(manus_dir / "filled_manual_review_template.csv", index=False)
    _write_json(manus_dir / "review_summary.json", {"status": "ok", "reviewed_rows": int(len(manual_review))})
    _write_csv(
        manus_dir / "human_verification_priority_list.csv",
        [{"image_id": "b.png", "priority": "high"}, {"image_id": "c.png", "priority": "high"}],
    )


def test_merge_manus_review_generates_final_action_files_and_expected_actions(tmp_path):
    from src.analysis.hard_row_visual_review import merge_manus_review, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    run_hard_row_visual_review(paths["config"])
    _write_manus_review_files(paths)

    outputs = merge_manus_review(paths["config"])

    validation = json.loads(outputs["manus_review_validation"].read_text(encoding="utf-8"))
    action_plan = pd.read_csv(outputs["final_hard_row_action_plan"], keep_default_na=False)
    phase3 = pd.read_csv(outputs["phase3_training_candidates"], keep_default_na=False)
    excluded = pd.read_csv(outputs["excluded_rows"], keep_default_na=False)
    engineering = pd.read_csv(outputs["engineering_fix_candidates"], keep_default_na=False)
    queue = pd.read_csv(outputs["human_verification_queue"], keep_default_na=False)

    assert validation["valid"] is True
    assert list(action_plan.columns) == REQUIRED_MANUS_ACTION_COLUMNS
    assert set(action_plan["recommended_action"]).issubset(MANUS_ALLOWED_ACTIONS)
    assert phase3["image_id"].tolist() == ["a.png"]
    assert excluded["image_id"].tolist() == ["b.png"]
    assert engineering["image_id"].tolist() == ["b.png", "c.png"]
    assert set(queue["image_id"]) == {"b.png", "c.png"}


def test_merge_manus_review_reports_row_count_and_metadata_mismatch(tmp_path):
    from src.analysis.hard_row_visual_review import merge_manus_review, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    run_hard_row_visual_review(paths["config"])
    _write_manus_review_files(paths, metadata_mutation={"candidate_prediction": 99})
    manus_path = paths["output_root"] / "manus_review" / "filled_manual_review_template.csv"
    manus = pd.read_csv(manus_path, keep_default_na=False).iloc[:2]
    manus.to_csv(manus_path, index=False)

    outputs = merge_manus_review(paths["config"])
    validation = json.loads(outputs["manus_review_validation"].read_text(encoding="utf-8"))
    queue = pd.read_csv(outputs["human_verification_queue"], keep_default_na=False)

    assert validation["valid"] is False
    assert validation["row_count_matches_expected"] is False
    assert validation["metadata_mismatch_count"] >= 1
    assert len(queue) >= 1
    assert set(queue["image_id"]).issubset({"a.png", "b.png", "c.png"})


def test_merge_manus_review_reports_invalid_failure_mode_and_flag_values(tmp_path):
    from src.analysis.hard_row_visual_review import merge_manus_review, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    run_hard_row_visual_review(paths["config"])
    _write_manus_review_files(paths)

    manus_path = paths["output_root"] / "manus_review" / "filled_manual_review_template.csv"
    manus = pd.read_csv(manus_path, keep_default_na=False)
    manus.loc[0, "final_failure_mode"] = "bad_value"
    manus.loc[1, "label_quality_flag"] = "invalid_flag"
    manus.to_csv(manus_path, index=False)

    outputs = merge_manus_review(paths["config"])
    validation = json.loads(outputs["manus_review_validation"].read_text(encoding="utf-8"))
    queue = pd.read_csv(outputs["human_verification_queue"], keep_default_na=False)

    assert validation["invalid_final_failure_mode_count"] == 1
    assert validation["invalid_flag_value_count"] >= 1
    assert len(queue) >= 2
    assert set(queue["image_id"]).issubset({"a.png", "b.png", "c.png"})


def test_merge_manus_review_cli_supports_flag(tmp_path, capsys):
    from src.analysis.hard_row_visual_review import main, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    run_hard_row_visual_review(paths["config"])
    _write_manus_review_files(paths)

    exit_code = main(["--config", str(paths["config"]), "--merge-manus-review"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "manus" in captured.out.lower()


def test_complete_evidence_generates_expected_outputs_and_detector_fallback(tmp_path):
    from src.analysis.hard_row_visual_review import complete_hard_row_evidence, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    run_hard_row_visual_review(paths["config"])

    outputs = complete_hard_row_evidence(paths["config"])

    annotation = pd.read_csv(outputs["annotation_evidence_csv"], keep_default_na=False)
    crop = pd.read_csv(outputs["crop_quality_evidence_csv"], keep_default_na=False)
    manifest = pd.read_csv(outputs["evidence_completed_manifest"], keep_default_na=False)
    action_plan = pd.read_csv(outputs["final_training_action_plan_evidence_aware"], keep_default_na=False)
    summary = json.loads(outputs["hard_row_evidence_summary"].read_text(encoding="utf-8"))

    assert outputs["annotation_evidence_csv"].is_file()
    assert outputs["crop_quality_evidence_csv"].is_file()
    assert outputs["evidence_completed_manifest"].is_file()
    assert outputs["final_training_action_plan_evidence_aware"].is_file()
    assert outputs["evidence_assets_dir"].is_dir()
    assert set(annotation["annotation_evidence_status"]) == {"annotation_evidence_available", "annotation_evidence_missing"}
    assert "annotation_evidence_available" in set(action_plan["detector_evidence_status"])
    assert "detector_missing" not in set(action_plan["detector_evidence_status"])
    assert list(action_plan.columns) == REQUIRED_EVIDENCE_ACTION_COLUMNS
    assert summary["trained_model"] is False
    assert summary["submission_created"] is False
    assert summary["used_test_labels"] is False
    assert len(annotation) == 3
    assert len(crop) == 3
    assert len(manifest) == 3


def test_complete_evidence_maps_roi_issue_and_tiny_defect_actions(tmp_path):
    from src.analysis.hard_row_visual_review import complete_hard_row_evidence, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    run_hard_row_visual_review(paths["config"])

    outputs = complete_hard_row_evidence(paths["config"])
    action_plan = pd.read_csv(outputs["final_training_action_plan_evidence_aware"], keep_default_na=False)

    by_image = action_plan.set_index("image_id")
    assert by_image.loc["a.png", "recommended_action"] == "possible_phase3_candidate"
    assert by_image.loc["b.png", "recommended_action"] == "tiny_or_low_contrast_defect_candidate"
    assert by_image.loc["c.png", "recommended_action"] == "roi_pipeline_bug"
    assert bool(by_image.loc["a.png", "phase3_use_allowed"]) is True
    assert bool(by_image.loc["c.png", "phase3_use_allowed"]) is False


def test_complete_evidence_handles_missing_images_and_missing_roi_safely(tmp_path):
    from src.analysis.hard_row_visual_review import complete_hard_row_evidence, run_hard_row_visual_review

    paths = _base_setup(tmp_path)
    (paths["train_images"] / "c.png").unlink()
    run_hard_row_visual_review(paths["config"])

    outputs = complete_hard_row_evidence(paths["config"])
    crop = pd.read_csv(outputs["crop_quality_evidence_csv"], keep_default_na=False)

    by_image = crop.set_index("image_id")
    assert by_image.loc["c.png", "image_exists"] in {False, "False", "false", 0}
    assert by_image.loc["c.png", "crop_quality_status"] == "missing"
    assert by_image.loc["c.png", "roi_exists"] in {False, "False", "false", 0}


def _write_phase3_candidate_input(paths: dict[str, Path]) -> Path:
    phase3_input = paths["output_root"] / "reports" / "final_training_action_plan_evidence_aware.csv"
    payload = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    payload["evidence_completion"]["expected_rows"] = 6
    payload["evidence_completion"]["expected_phase3_allowed_count"] = 3
    payload["evidence_completion"]["expected_phase3_blocked_count"] = 3
    paths["config"].write_text(yaml.safe_dump(payload), encoding="utf-8")
    rows = [
        {
            "image_id": "a.png",
            "target": 1,
            "prediction": 1,
            "probability": 0.81,
            "bottle_type": "Amber crown",
            "cluster_id": 0,
            "suspicion_rank": 1,
            "label_issue_score": 0.11,
            "anomaly_score": 0.12,
            "primary_audit_category": "likely_correct_but_hard",
            "hard_example_type": "hard_positive",
            "annotation_count": 1,
            "defect_categories": "Water drop",
            "largest_defect_area_ratio": 0.02,
            "total_defect_area_ratio": 0.03,
            "annotation_evidence_status": "annotation_evidence_available",
            "detector_evidence_status": "annotation_evidence_available",
            "roi_exists": True,
            "roi_coverage_ratio": 0.20,
            "crop_quality_status": "ok",
            "brightness_score": 120.0,
            "contrast_score": 25.0,
            "blur_score_laplacian_variance": 100.0,
            "edge_density": 0.15,
            "recommended_action": "safe_for_hard_training_with_artifact_focus",
            "phase3_use_allowed": True,
            "reason": "artifact",
        },
        {
            "image_id": "b.png",
            "target": 1,
            "prediction": 1,
            "probability": 0.74,
            "bottle_type": "Euro brown",
            "cluster_id": 1,
            "suspicion_rank": 2,
            "label_issue_score": 0.18,
            "anomaly_score": 0.14,
            "primary_audit_category": "likely_correct_but_hard",
            "hard_example_type": "hard_positive",
            "annotation_count": 1,
            "defect_categories": "Chip",
            "largest_defect_area_ratio": 0.004,
            "total_defect_area_ratio": 0.004,
            "annotation_evidence_status": "annotation_evidence_available",
            "detector_evidence_status": "annotation_evidence_available",
            "roi_exists": True,
            "roi_coverage_ratio": 0.18,
            "crop_quality_status": "ok",
            "brightness_score": 110.0,
            "contrast_score": 18.0,
            "blur_score_laplacian_variance": 80.0,
            "edge_density": 0.08,
            "recommended_action": "tiny_or_low_contrast_defect_candidate",
            "phase3_use_allowed": True,
            "reason": "tiny",
        },
        {
            "image_id": "c.png",
            "target": 0,
            "prediction": 1,
            "probability": 0.62,
            "bottle_type": "Amber crown",
            "cluster_id": 2,
            "suspicion_rank": 3,
            "label_issue_score": 0.22,
            "anomaly_score": 0.20,
            "primary_audit_category": "likely_correct_but_hard",
            "hard_example_type": "hard_negative",
            "annotation_count": 2,
            "defect_categories": "Scuffing",
            "largest_defect_area_ratio": 0.03,
            "total_defect_area_ratio": 0.05,
            "annotation_evidence_status": "annotation_evidence_available",
            "detector_evidence_status": "annotation_evidence_available",
            "roi_exists": True,
            "roi_coverage_ratio": 0.22,
            "crop_quality_status": "ok",
            "brightness_score": 130.0,
            "contrast_score": 32.0,
            "blur_score_laplacian_variance": 120.0,
            "edge_density": 0.21,
            "recommended_action": "possible_phase3_candidate",
            "phase3_use_allowed": True,
            "reason": "general",
        },
        {
            "image_id": "d.png",
            "target": 1,
            "prediction": 0,
            "probability": 0.29,
            "bottle_type": "Amber crown",
            "cluster_id": 3,
            "suspicion_rank": 4,
            "label_issue_score": 0.71,
            "anomaly_score": 0.74,
            "primary_audit_category": "likely_ambiguous",
            "hard_example_type": "hard_positive,uncertain",
            "annotation_count": 1,
            "defect_categories": "Chip",
            "largest_defect_area_ratio": 0.01,
            "total_defect_area_ratio": 0.01,
            "annotation_evidence_status": "annotation_evidence_available",
            "detector_evidence_status": "annotation_evidence_available",
            "roi_exists": True,
            "roi_coverage_ratio": 0.19,
            "crop_quality_status": "likely_issue",
            "brightness_score": 100.0,
            "contrast_score": 20.0,
            "blur_score_laplacian_variance": 70.0,
            "edge_density": 0.07,
            "recommended_action": "roi_pipeline_bug",
            "phase3_use_allowed": False,
            "reason": "roi bug",
        },
        {
            "image_id": "e.png",
            "target": 1,
            "prediction": 0,
            "probability": 0.31,
            "bottle_type": "Amber crown",
            "cluster_id": 4,
            "suspicion_rank": 5,
            "label_issue_score": 0.82,
            "anomaly_score": 0.25,
            "primary_audit_category": "likely_ambiguous",
            "hard_example_type": "hard_positive,uncertain",
            "annotation_count": 1,
            "defect_categories": "Chip",
            "largest_defect_area_ratio": 0.01,
            "total_defect_area_ratio": 0.01,
            "annotation_evidence_status": "annotation_evidence_available",
            "detector_evidence_status": "annotation_evidence_available",
            "roi_exists": True,
            "roi_coverage_ratio": 0.20,
            "crop_quality_status": "ok",
            "brightness_score": 102.0,
            "contrast_score": 22.0,
            "blur_score_laplacian_variance": 75.0,
            "edge_density": 0.09,
            "recommended_action": "suspected_mislabel_exclude_from_training",
            "phase3_use_allowed": False,
            "reason": "mislabel",
        },
        {
            "image_id": "f.png",
            "target": 1,
            "prediction": 0,
            "probability": 0.28,
            "bottle_type": "Amber crown",
            "cluster_id": 5,
            "suspicion_rank": 6,
            "label_issue_score": 0.33,
            "anomaly_score": 0.66,
            "primary_audit_category": "likely_ambiguous",
            "hard_example_type": "hard_positive,uncertain",
            "annotation_count": 1,
            "defect_categories": "Water drop",
            "largest_defect_area_ratio": 0.02,
            "total_defect_area_ratio": 0.02,
            "annotation_evidence_status": "annotation_evidence_available",
            "detector_evidence_status": "annotation_evidence_available",
            "roi_exists": True,
            "roi_coverage_ratio": 0.20,
            "crop_quality_status": "ok",
            "brightness_score": 105.0,
            "contrast_score": 24.0,
            "blur_score_laplacian_variance": 90.0,
            "edge_density": 0.13,
            "recommended_action": "safe_for_hard_training_with_artifact_focus",
            "phase3_use_allowed": False,
            "reason": "blocked artifact",
        },
    ]
    phase3_input.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(phase3_input, index=False)
    return phase3_input


def test_export_phase3_candidates_writes_required_outputs_and_excludes_blocked_rows(tmp_path):
    from src.analysis.hard_row_visual_review import export_phase3_candidates

    paths = _base_setup(tmp_path)
    _write_phase3_candidate_input(paths)

    outputs = export_phase3_candidates(
        paths["config"],
        expected_row_count=6,
        expected_allowed_count=3,
        expected_blocked_count=3,
    )

    allowed = pd.read_csv(outputs["phase3_allowed_hard_training_rows"], keep_default_na=False)
    artifact = pd.read_csv(outputs["phase3_artifact_focus_rows"], keep_default_na=False)
    tiny = pd.read_csv(outputs["phase3_tiny_low_contrast_rows"], keep_default_na=False)
    general = pd.read_csv(outputs["phase3_general_candidate_rows"], keep_default_na=False)
    roi_blocked = pd.read_csv(outputs["blocked_roi_pipeline_bug_rows"], keep_default_na=False)
    mislabel_blocked = pd.read_csv(outputs["blocked_suspected_mislabel_rows"], keep_default_na=False)
    other_blocked = pd.read_csv(outputs["blocked_other_high_risk_rows"], keep_default_na=False)
    summary = json.loads(outputs["phase3_candidate_package_summary"].read_text(encoding="utf-8"))

    assert set(allowed["image_id"]) == {"a.png", "b.png", "c.png"}
    assert artifact["image_id"].tolist() == ["a.png"]
    assert tiny["image_id"].tolist() == ["b.png"]
    assert general["image_id"].tolist() == ["c.png"]
    assert roi_blocked["image_id"].tolist() == ["d.png"]
    assert mislabel_blocked["image_id"].tolist() == ["e.png"]
    assert other_blocked["image_id"].tolist() == ["f.png"]
    assert not set(allowed["image_id"]) & set(roi_blocked["image_id"])
    assert not set(allowed["image_id"]) & set(mislabel_blocked["image_id"])
    assert not set(allowed["image_id"]) & set(other_blocked["image_id"])
    assert REQUIRED_PHASE3_PACKAGE_SUMMARY_KEYS.issubset(summary)
    assert summary["phase3_use_allowed_true_count"] == 3
    assert summary["phase3_use_allowed_false_count"] == 3
    assert summary["safety"]["training_started"] is False
    assert summary["safety"]["submission_created"] is False
    assert summary["safety"]["used_test_labels"] is False


def test_export_phase3_candidates_fails_when_validation_counts_do_not_match(tmp_path):
    from src.analysis.hard_row_visual_review import HardRowVisualReviewError, export_phase3_candidates

    paths = _base_setup(tmp_path)
    _write_phase3_candidate_input(paths)

    with pytest.raises(HardRowVisualReviewError, match="allowed row count"):
        export_phase3_candidates(
            paths["config"],
            expected_row_count=6,
            expected_allowed_count=4,
            expected_blocked_count=2,
        )


def test_export_phase3_candidates_cli_supports_flag(tmp_path, capsys):
    from src.analysis.hard_row_visual_review import main

    paths = _base_setup(tmp_path)
    _write_phase3_candidate_input(paths)

    exit_code = main(["--config", str(paths["config"]), "--export-phase3-candidates"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "candidate package" in captured.out.lower()
