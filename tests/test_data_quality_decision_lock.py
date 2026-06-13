"""Tests for Spec 017 automated decision-lock workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.analysis.data_quality_decision_lock import (
    DECISION_COLUMNS,
    DecisionLockConfig,
    DecisionLockError,
    load_config,
    run_decision_lock,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_empty_image_id_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["image_id"]).to_csv(path, index=False)
    return path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_image(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)
    return path


def _build_paths(tmp_path: Path) -> dict[str, Path]:
    dataset_dir = tmp_path / "dataset"
    analysis_dir = tmp_path / "analysis"
    outputs_dir = tmp_path / "outputs"
    return {
        "dataset_dir": dataset_dir,
        "train_csv": dataset_dir / "train.csv",
        "train_images_dir": dataset_dir / "train_images",
        "master": analysis_dir / "data_quality_master.csv",
        "cleaned": analysis_dir / "cleaned_training_manifest.csv",
        "manual_review": analysis_dir / "manual_review_required_rows.csv",
        "review_template": analysis_dir / "review_decision_template.csv",
        "summary": analysis_dir / "reports" / "data_quality_summary.json",
        "exclusion_counts": analysis_dir / "reports" / "exclusion_reason_counts.csv",
        "allowed": tmp_path / "spec014" / "allowed.csv",
        "blocked_roi": tmp_path / "spec014" / "blocked_roi.csv",
        "blocked_mislabel": tmp_path / "spec014" / "blocked_mislabel.csv",
        "blocked_other": tmp_path / "spec014" / "blocked_other.csv",
        "output_root": outputs_dir / "decision_lock",
        "config": tmp_path / "decision_lock.yaml",
        "oof": tmp_path / "optional" / "oof.csv",
        "cleanlab": tmp_path / "optional" / "cleanlab.csv",
        "clusters": tmp_path / "optional" / "clusters.csv",
    }


def _write_base_fixture(tmp_path: Path) -> dict[str, Path]:
    paths = _build_paths(tmp_path)
    _write_csv(
        paths["train_csv"],
        [
            {"image_id": "clean.jpg", "target": 0},
            {"image_id": "hard.jpg", "target": 1},
            {"image_id": "manual_low.jpg", "target": 0},
            {"image_id": "manual_high.jpg", "target": 1},
            {"image_id": "blocked.jpg", "target": 0},
            {"image_id": "missing.jpg", "target": 1},
            {"image_id": "dup_a.jpg", "target": 0},
        ],
    )
    for image_id, color in {
        "clean.jpg": (120, 120, 120),
        "hard.jpg": (220, 60, 60),
        "manual_low.jpg": (128, 128, 128),
        "manual_high.jpg": (30, 30, 30),
        "blocked.jpg": (60, 180, 60),
        "dup_a.jpg": (90, 90, 200),
    }.items():
        _write_image(paths["train_images_dir"] / image_id, color)

    _write_csv(
        paths["master"],
        [
            {
                "image_id": "clean.jpg",
                "target": 0,
                "data_quality_bucket": "clean_train",
                "exclusion_reason": "",
                "blocked_reason": "",
                "roi_quality_flag": "ok",
                "label_issue_score": 0.05,
                "duplicate_conflict_flag": False,
                "duplicate_group_id": "",
                "review_priority": 99,
                "v2b_probability": 0.05,
                "oof_probability": "",
                "cluster_id": "",
                "cluster_outlier_score": "",
            },
            {
                "image_id": "hard.jpg",
                "target": 1,
                "data_quality_bucket": "hard_valid_train",
                "exclusion_reason": "",
                "blocked_reason": "",
                "roi_quality_flag": "ok",
                "label_issue_score": 0.10,
                "duplicate_conflict_flag": False,
                "duplicate_group_id": "",
                "review_priority": 99,
                "v2b_probability": 0.92,
                "oof_probability": "",
                "cluster_id": "",
                "cluster_outlier_score": "",
            },
            {
                "image_id": "manual_low.jpg",
                "target": 0,
                "data_quality_bucket": "manual_review_required",
                "exclusion_reason": "roi_or_crop_problem_requires_review",
                "blocked_reason": "",
                "roi_quality_flag": "low_contrast_or_weak_texture",
                "label_issue_score": 0.10,
                "duplicate_conflict_flag": False,
                "duplicate_group_id": "",
                "review_priority": 6,
                "v2b_probability": 0.12,
                "oof_probability": "",
                "cluster_id": "",
                "cluster_outlier_score": "",
            },
            {
                "image_id": "manual_high.jpg",
                "target": 1,
                "data_quality_bucket": "manual_review_required",
                "exclusion_reason": "high_label_issue_score",
                "blocked_reason": "",
                "roi_quality_flag": "ok",
                "label_issue_score": 0.96,
                "duplicate_conflict_flag": False,
                "duplicate_group_id": "",
                "review_priority": 1,
                "v2b_probability": 0.02,
                "oof_probability": "",
                "cluster_id": "",
                "cluster_outlier_score": "",
            },
            {
                "image_id": "blocked.jpg",
                "target": 0,
                "data_quality_bucket": "exclude_from_training",
                "exclusion_reason": "spec014_suspected_mislabel",
                "blocked_reason": "spec014_suspected_mislabel",
                "roi_quality_flag": "ok",
                "label_issue_score": 0.25,
                "duplicate_conflict_flag": False,
                "duplicate_group_id": "",
                "review_priority": 3,
                "v2b_probability": 0.60,
                "oof_probability": "",
                "cluster_id": "",
                "cluster_outlier_score": "",
            },
            {
                "image_id": "missing.jpg",
                "target": 1,
                "data_quality_bucket": "exclude_from_training",
                "exclusion_reason": "missing_or_unreadable_image",
                "blocked_reason": "",
                "roi_quality_flag": "missing_image",
                "label_issue_score": 0.20,
                "duplicate_conflict_flag": False,
                "duplicate_group_id": "",
                "review_priority": 7,
                "v2b_probability": 0.55,
                "oof_probability": "",
                "cluster_id": "",
                "cluster_outlier_score": "",
            },
            {
                "image_id": "dup_a.jpg",
                "target": 0,
                "data_quality_bucket": "clean_train",
                "exclusion_reason": "",
                "blocked_reason": "",
                "roi_quality_flag": "ok",
                "label_issue_score": 0.15,
                "duplicate_conflict_flag": False,
                "duplicate_group_id": "",
                "review_priority": 99,
                "v2b_probability": 0.20,
                "oof_probability": "",
                "cluster_id": "",
                "cluster_outlier_score": "",
            },
        ],
    )
    _write_csv(
        paths["cleaned"],
        [
            {"image_id": "clean.jpg", "target": 0},
            {"image_id": "hard.jpg", "target": 1},
            {"image_id": "dup_a.jpg", "target": 0},
        ],
    )
    _write_csv(
        paths["manual_review"],
        [
            {"image_id": "manual_low.jpg", "target": 0},
            {"image_id": "manual_high.jpg", "target": 1},
        ],
    )
    _write_csv(paths["review_template"], [{"image_id": "manual_high.jpg", "target": 1}])
    _write_json(paths["summary"], {"manual_review_required_count": 2, "no_training_started": True})
    _write_csv(
        paths["exclusion_counts"],
        [
            {"exclusion_reason": "roi_or_crop_problem_requires_review", "count": 1},
            {"exclusion_reason": "high_label_issue_score", "count": 1},
        ],
    )
    _write_csv(paths["allowed"], [{"image_id": "hard.jpg"}])
    _write_empty_image_id_csv(paths["blocked_roi"])
    _write_csv(paths["blocked_mislabel"], [{"image_id": "blocked.jpg"}])
    _write_empty_image_id_csv(paths["blocked_other"])
    return paths


def _write_config(paths: dict[str, Path], include_optional: bool = False) -> Path:
    optional_block = f"""
optional_evidence:
  oof_predictions_path: {paths["oof"].as_posix() if include_optional else ""}
  v2b_predictions_path: ""
  image_embeddings_path: ""
  cleanlab_scores_path: {paths["cleanlab"].as_posix() if include_optional else ""}
  cluster_assignments_path: {paths["clusters"].as_posix() if include_optional else ""}
"""
    config_text = f"""
inputs:
  spec016_master_path: {paths["master"].as_posix()}
  spec016_cleaned_manifest_path: {paths["cleaned"].as_posix()}
  spec016_manual_review_required_path: {paths["manual_review"].as_posix()}
  spec016_review_template_path: {paths["review_template"].as_posix()}
  spec016_summary_path: {paths["summary"].as_posix()}
  spec016_exclusion_reasons_path: {paths["exclusion_counts"].as_posix()}
  spec014_allowed_hard_rows_path: {paths["allowed"].as_posix()}
  spec014_blocked_roi_pipeline_bug_rows_path: {paths["blocked_roi"].as_posix()}
  spec014_blocked_suspected_mislabel_rows_path: {paths["blocked_mislabel"].as_posix()}
  spec014_blocked_other_high_risk_rows_path: {paths["blocked_other"].as_posix()}
{optional_block}
dataset:
  train_csv_path: {paths["train_csv"].as_posix()}
  train_images_dir: {paths["train_images_dir"].as_posix()}
output:
  analysis_root: {paths["output_root"].as_posix()}
quality:
  label_issue_threshold: 0.80
  prediction_high_risk_threshold: 0.75
  prediction_critical_risk_threshold: 0.90
  contact_sheet_limit: 16
  adjudication_queue_limit: 50
safety:
  allow_test_labels: false
  public_leaderboard_input: false
  generate_submission: false
  train_model: false
  apply_relabels: false
"""
    paths["config"].write_text(config_text, encoding="utf-8")
    return paths["config"]


def test_load_config_rejects_unsafe_flags(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
inputs:
  spec016_master_path: a.csv
  spec016_cleaned_manifest_path: b.csv
  spec016_manual_review_required_path: c.csv
  spec016_review_template_path: d.csv
  spec016_summary_path: e.json
  spec016_exclusion_reasons_path: f.csv
  spec014_allowed_hard_rows_path: g.csv
  spec014_blocked_roi_pipeline_bug_rows_path: h.csv
  spec014_blocked_suspected_mislabel_rows_path: i.csv
  spec014_blocked_other_high_risk_rows_path: j.csv
dataset:
  train_csv_path: train.csv
  train_images_dir: train_images
safety:
  train_model: true
""",
        encoding="utf-8",
    )

    with pytest.raises(DecisionLockError, match="Safety flag must remain false"):
        load_config(config_path)


def test_run_decision_lock_creates_expected_mvp_outputs(tmp_path: Path) -> None:
    paths = _write_base_fixture(tmp_path)
    config = load_config(_write_config(paths))

    summary = run_decision_lock(config)

    decision_master = pd.read_csv(paths["output_root"] / "decision_master.csv")
    manifest = pd.read_csv(paths["output_root"] / "approved_cleaned_training_manifest.csv")
    adjudication = pd.read_csv(paths["output_root"] / "adjudication_queue.csv")

    assert set(DECISION_COLUMNS).issubset(set(decision_master.columns))
    assert len(decision_master) == 7
    assert set(manifest["image_id"]) == {"clean.jpg", "hard.jpg", "manual_low.jpg", "dup_a.jpg"}
    assert "blocked.jpg" not in set(manifest["image_id"])
    assert "manual_high.jpg" not in set(manifest["image_id"])
    assert set(adjudication["image_id"]) == {"manual_high.jpg"}
    assert summary["blocked_overlap_count"] == 0
    assert summary["duplicate_conflict_approved_count"] == 0
    assert summary["unresolved_review_approved_count"] == 0
    assert summary["no_training_started"] is True
    assert summary["no_submission_created"] is True


def test_run_decision_lock_excludes_duplicate_conflicts_from_optional_evidence(tmp_path: Path) -> None:
    paths = _write_base_fixture(tmp_path)
    _write_csv(paths["oof"], [{"image_id": "dup_a.jpg", "probability": 0.25}])
    _write_csv(paths["cleanlab"], [{"image_id": "dup_a.jpg", "score": 0.10}])
    _write_csv(
        paths["clusters"],
        [{"image_id": "dup_a.jpg", "duplicate_group_id": "dup-1", "cluster_id": "c1", "cluster_outlier_score": 0.8}],
    )

    master = pd.read_csv(paths["master"])
    master["duplicate_group_id"] = master["duplicate_group_id"].astype("object")
    master.loc[master["image_id"] == "dup_a.jpg", "duplicate_conflict_flag"] = True
    master.loc[master["image_id"] == "dup_a.jpg", "duplicate_group_id"] = "dup-1"
    master.to_csv(paths["master"], index=False)

    config = load_config(_write_config(paths, include_optional=True))
    run_decision_lock(config)

    manifest = pd.read_csv(paths["output_root"] / "approved_cleaned_training_manifest.csv")
    excluded = pd.read_csv(paths["output_root"] / "auto_exclude_rows.csv")
    assert "dup_a.jpg" not in set(manifest["image_id"])
    assert "dup_a.jpg" in set(excluded["image_id"])


def test_run_decision_lock_reports_missing_optional_evidence_without_failing(tmp_path: Path) -> None:
    paths = _write_base_fixture(tmp_path)
    config = load_config(_write_config(paths))

    run_decision_lock(config)

    evidence = json.loads((paths["output_root"] / "reports" / "evidence_inventory.json").read_text(encoding="utf-8"))
    assert evidence["prediction_status"] == "missing"
    assert evidence["embedding_status"] == "missing"
    assert evidence["cleanlab_status"] == "missing"
    assert any("not configured" in warning or "missing" in warning for warning in evidence["warnings"])


def test_run_decision_lock_creates_roi_summary_and_contact_sheets(tmp_path: Path) -> None:
    paths = _write_base_fixture(tmp_path)
    config = load_config(_write_config(paths))

    run_decision_lock(config)

    roi_summary = pd.read_csv(paths["output_root"] / "reports" / "roi_recalibration_summary.csv")
    assert "roi_quality_group" in roi_summary.columns
    assert "low_contrast_or_weak_texture" in set(roi_summary["roi_quality_group"])

    for name in (
        "needs_adjudication_top_risk.jpg",
        "auto_exclude_representatives.jpg",
        "roi_group_representatives.jpg",
    ):
        assert (paths["output_root"] / "contact_sheets" / name).exists()
