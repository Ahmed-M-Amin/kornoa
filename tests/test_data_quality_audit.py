"""Tests for the controlled data quality audit workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.analysis.data_quality_audit import (
    DataQualityConfig,
    apply_duplicate_audit,
    apply_label_issue_scoring,
    apply_optional_cleanlab_scoring,
    apply_roi_quality_audit,
    apply_spec014_evidence,
    assign_data_quality_buckets,
    build_cleaned_training_manifest,
    build_contact_sheet,
    build_input_file_inventory,
    build_master_table,
    build_validation_alignment_summary,
    load_config,
    run_audit,
    write_reports,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


def _base_config(tmp_path: Path, train_csv: Path, images_dir: Path, predictions_csv: Path | None = None) -> DataQualityConfig:
    return DataQualityConfig(
        train_csv_path=train_csv,
        train_images_dir=images_dir,
        output_root=tmp_path / "out",
        v2b_validation_predictions_path=predictions_csv,
        oof_train_predictions_path=None,
        image_embeddings_path=None,
        bottletypes_path=None,
        train_annotations_path=None,
        hard_row_quality_audit_path=None,
        hard_row_review_manifest_path=None,
        final_hard_row_action_plan_path=None,
        phase3_training_candidates_path=None,
        excluded_ambiguous_or_noisy_rows_path=None,
        engineering_fix_candidates_path=None,
        human_verification_queue_path=None,
        allowed_hard_rows_path=None,
        blocked_suspected_mislabel_rows_path=None,
        blocked_roi_pipeline_bug_rows_path=None,
        blocked_other_high_risk_rows_path=None,
        confident_wrong_probability_threshold=0.90,
        label_issue_threshold=0.80,
        dark_brightness_threshold=20.0,
        bright_brightness_threshold=235.0,
        low_contrast_threshold=8.0,
        weak_edge_threshold=2.0,
        min_image_width=32,
        min_image_height=32,
        review_template_limit=800,
        contact_sheet_limit=64,
        use_cleanlab_if_available=False,
        use_fiftyone_if_available=False,
        use_imagehash_if_available=False,
    )


def test_load_data_quality_config_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "data_quality_audit.yaml"
    config_path.write_text(
        """
dataset:
  train_csv_path: train.csv
  train_images_dir: train_images
output:
  analysis_root: outputs/analysis/data_quality_audit
predictions:
  v2b_validation_predictions_path: ""
  oof_train_predictions_path: ""
spec014:
  allowed_hard_rows_path: ""
  blocked_suspected_mislabel_rows_path: ""
  blocked_roi_pipeline_bug_rows_path: ""
  blocked_other_high_risk_rows_path: ""
quality:
  confident_wrong_probability_threshold: 0.90
  label_issue_threshold: 0.80
  dark_brightness_threshold: 20.0
  bright_brightness_threshold: 235.0
  low_contrast_threshold: 8.0
  weak_edge_threshold: 2.0
optional_tools:
  use_cleanlab_if_available: true
  use_fiftyone_if_available: false
safety:
  allow_test_labels: false
  public_leaderboard_input: false
  generate_submission: false
  train_model: false
  apply_relabels: false
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, DataQualityConfig)
    assert config.train_csv_path == (tmp_path / "train.csv").resolve()
    assert config.train_images_dir == (tmp_path / "train_images").resolve()
    assert config.output_root == (tmp_path / "outputs" / "analysis" / "data_quality_audit").resolve()
    assert config.confident_wrong_probability_threshold == 0.90
    assert config.use_cleanlab_if_available is True


def test_build_master_table_from_train_and_v2b_predictions(tmp_path: Path) -> None:
    train_csv = _write_csv(
        tmp_path / "train.csv",
        [
            {"image_id": "a.jpg", "target": 0},
            {"image_id": "b.jpg", "target": 1},
            {"image_id": "c.jpg", "target": 0},
        ],
    )
    predictions_csv = _write_csv(
        tmp_path / "val_predictions.csv",
        [
            {"image_id": "a.jpg", "target": 0, "probability": 0.10, "prediction": 0},
            {"image_id": "b.jpg", "target": 1, "probability": 0.95, "prediction": 1},
        ],
    )
    images_dir = tmp_path / "train_images"
    images_dir.mkdir()

    config = _base_config(tmp_path, train_csv, images_dir, predictions_csv)
    master, warnings = build_master_table(config)

    assert len(master) == 3
    assert set(master["image_id"]) == {"a.jpg", "b.jpg", "c.jpg"}
    assert master.loc[master["image_id"] == "a.jpg", "v2b_probability"].item() == 0.10
    assert master.loc[master["image_id"] == "b.jpg", "v2b_correct"].item() is True
    assert pd.isna(master.loc[master["image_id"] == "c.jpg", "v2b_probability"]).item()
    assert any("OOF train predictions missing" in warning for warning in warnings)


def test_spec014_blocked_rows_map_to_safe_buckets(tmp_path: Path) -> None:
    master = pd.DataFrame(
        [
            {"image_id": "allowed.jpg", "target": 1},
            {"image_id": "mislabel.jpg", "target": 0},
            {"image_id": "roi.jpg", "target": 1},
            {"image_id": "risk.jpg", "target": 0},
        ]
    )
    allowed = _write_csv(tmp_path / "allowed.csv", [{"image_id": "allowed.jpg", "hard_example_type": "correct_but_visually_hard"}])
    mislabel = _write_csv(tmp_path / "mislabel.csv", [{"image_id": "mislabel.jpg"}])
    roi = _write_csv(tmp_path / "roi.csv", [{"image_id": "roi.jpg"}])
    risk = _write_csv(tmp_path / "risk.csv", [{"image_id": "risk.jpg"}])

    config = _base_config(tmp_path, tmp_path / "train.csv", tmp_path / "images")
    object.__setattr__(config, "allowed_hard_rows_path", allowed)
    object.__setattr__(config, "blocked_suspected_mislabel_rows_path", mislabel)
    object.__setattr__(config, "blocked_roi_pipeline_bug_rows_path", roi)
    object.__setattr__(config, "blocked_other_high_risk_rows_path", risk)

    updated, warnings = apply_spec014_evidence(master, config)

    buckets = dict(zip(updated["image_id"], updated["hard_row_status"]))
    reasons = dict(zip(updated["image_id"], updated["blocked_reason"]))
    assert buckets["allowed.jpg"] == "spec014_allowed_hard"
    assert buckets["mislabel.jpg"] == "spec014_blocked"
    assert reasons["mislabel.jpg"] == "spec014_suspected_mislabel"
    assert reasons["roi.jpg"] == "spec014_roi_pipeline_bug"
    assert reasons["risk.jpg"] == "spec014_other_high_risk"
    assert warnings == []


def test_roi_quality_audit_marks_missing_and_low_contrast_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_image(images_dir / "flat.jpg", color=(128, 128, 128))
    master = pd.DataFrame(
        [
            {"image_id": "flat.jpg", "target": 0},
            {"image_id": "missing.jpg", "target": 1},
        ]
    )

    updated = apply_roi_quality_audit(
        master,
        images_dir=images_dir,
        dark_threshold=20.0,
        bright_threshold=235.0,
        low_contrast_threshold=8.0,
        weak_edge_threshold=2.0,
    )

    flags = dict(zip(updated["image_id"], updated["roi_quality_flag"]))
    assert flags["flat.jpg"] == "low_contrast_or_weak_texture"
    assert flags["missing.jpg"] == "missing_image"
    assert updated.loc[updated["image_id"] == "flat.jpg", "image_width"].item() == 64


def test_duplicate_audit_flags_same_file_hash_with_label_conflict(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    content = b"same-bytes"
    (images_dir / "a.jpg").write_bytes(content)
    (images_dir / "b.jpg").write_bytes(content)
    master = pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0},
            {"image_id": "b.jpg", "target": 1},
        ]
    )

    updated, conflicts = apply_duplicate_audit(master, images_dir)
    assert updated["duplicate_conflict_flag"].tolist() == [True, True]
    assert len(conflicts) == 2
    assert conflicts["duplicate_group_id"].nunique() == 1


def test_confident_wrong_rows_receive_high_label_issue_score() -> None:
    master = pd.DataFrame(
        [
            {"image_id": "wrong.jpg", "target": 0, "oof_probability": 0.96, "v2b_probability": pd.NA, "blocked_reason": pd.NA, "duplicate_conflict_flag": False, "roi_quality_flag": "ok"},
            {"image_id": "right.jpg", "target": 1, "oof_probability": 0.97, "v2b_probability": pd.NA, "blocked_reason": pd.NA, "duplicate_conflict_flag": False, "roi_quality_flag": "ok"},
            {"image_id": "boundary.jpg", "target": 1, "oof_probability": 0.51, "v2b_probability": pd.NA, "blocked_reason": pd.NA, "duplicate_conflict_flag": False, "roi_quality_flag": "ok"},
        ]
    )

    scored = apply_label_issue_scoring(master)
    scores = dict(zip(scored["image_id"], scored["label_issue_score"]))
    priorities = dict(zip(scored["image_id"], scored["review_priority"]))
    assert scores["wrong.jpg"] > scores["right.jpg"]
    assert scores["wrong.jpg"] >= 0.90
    assert priorities["wrong.jpg"] == 1
    assert priorities["boundary.jpg"] == 7


def test_cleanlab_missing_keeps_fallback_scores_and_reports_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    master = pd.DataFrame([{"image_id": "a.jpg", "target": 0, "label_issue_score": 0.2}])

    def fake_import(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", fake_import)
    updated, warnings = apply_optional_cleanlab_scoring(master, enabled=True)

    assert updated["label_issue_score"].tolist() == [0.2]
    assert any("Cleanlab not installed" in warning for warning in warnings)


def test_cleaned_manifest_excludes_blocked_and_manual_review_rows() -> None:
    master = pd.DataFrame(
        [
            {"image_id": "clean.jpg", "target": 0, "hard_row_status": pd.NA, "blocked_reason": pd.NA, "roi_quality_flag": "ok", "duplicate_conflict_flag": False, "label_issue_score": 0.1},
            {"image_id": "hard.jpg", "target": 1, "hard_row_status": "spec014_allowed_hard", "blocked_reason": pd.NA, "roi_quality_flag": "ok", "duplicate_conflict_flag": False, "label_issue_score": 0.2},
            {"image_id": "blocked.jpg", "target": 0, "hard_row_status": "spec014_blocked", "blocked_reason": "spec014_suspected_mislabel", "roi_quality_flag": "ok", "duplicate_conflict_flag": False, "label_issue_score": 0.2},
            {"image_id": "manual.jpg", "target": 1, "hard_row_status": pd.NA, "blocked_reason": pd.NA, "roi_quality_flag": "low_contrast_or_weak_texture", "duplicate_conflict_flag": False, "label_issue_score": 0.2},
        ]
    )

    bucketed = assign_data_quality_buckets(master, label_issue_threshold=0.80)
    manifest = build_cleaned_training_manifest(bucketed)

    assert set(manifest["image_id"]) == {"clean.jpg", "hard.jpg"}
    assert dict(zip(bucketed["image_id"], bucketed["data_quality_bucket"]))["blocked.jpg"] == "exclude_from_training"
    assert dict(zip(bucketed["image_id"], bucketed["data_quality_bucket"]))["manual.jpg"] == "manual_review_required"


def test_write_reports_creates_summary_and_ranked_outputs(tmp_path: Path) -> None:
    master = pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0, "data_quality_bucket": "clean_train", "exclude_from_training": False, "exclusion_reason": "", "label_issue_score": 0.1, "review_priority": 99, "roi_quality_flag": "ok", "duplicate_conflict_flag": False, "oof_probability": pd.NA, "anomaly_score": 0.1},
            {"image_id": "b.jpg", "target": 1, "data_quality_bucket": "manual_review_required", "exclude_from_training": True, "exclusion_reason": "high_label_issue_score", "label_issue_score": 0.95, "review_priority": 1, "roi_quality_flag": "ok", "duplicate_conflict_flag": False, "oof_probability": pd.NA, "anomaly_score": 0.8},
        ]
    )
    warnings = ["OOF train predictions missing; label-quality scoring is limited"]

    write_reports(master, output_root=tmp_path, warnings=warnings, duplicate_conflicts=pd.DataFrame())

    assert (tmp_path / "reports" / "data_quality_summary.json").exists()
    assert (tmp_path / "reports" / "bucket_counts.csv").exists()
    assert (tmp_path / "reports" / "top_suspicious_rows.csv").exists()
    assert (tmp_path / "review_decision_template.csv").exists()

    summary = json.loads((tmp_path / "reports" / "data_quality_summary.json").read_text(encoding="utf-8"))
    assert summary["total_rows"] == 2
    assert summary["manual_review_required_count"] == 1
    assert summary["no_training_started"] is True
    assert summary["no_submission_created"] is True
    assert summary["no_test_labels_used"] is True


def test_build_contact_sheet_creates_image_grid(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_image(images_dir / "a.jpg", color=(255, 0, 0), size=(32, 32))
    _write_image(images_dir / "b.jpg", color=(0, 255, 0), size=(32, 32))
    rows = pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0, "data_quality_bucket": "clean_train"},
            {"image_id": "b.jpg", "target": 1, "data_quality_bucket": "manual_review_required"},
        ]
    )

    output_path = tmp_path / "sheet.jpg"
    build_contact_sheet(rows, images_dir=images_dir, output_path=output_path, max_rows=2)
    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.width > 32
        assert image.height > 32


def test_input_file_inventory_records_missing_and_present_files(tmp_path: Path) -> None:
    present = tmp_path / "present.csv"
    missing = tmp_path / "missing.csv"
    present.write_text("image_id,target\nx.jpg,0\n", encoding="utf-8")
    inventory = build_input_file_inventory({"present": present, "missing": missing})
    assert dict(zip(inventory["input_name"], inventory["exists"])) == {"present": True, "missing": False}
    assert inventory.loc[inventory["input_name"] == "present", "size_bytes"].item() > 0


def test_validation_alignment_summary_reports_missing_split_metadata() -> None:
    master = pd.DataFrame([{"image_id": "a.jpg", "target": 0, "data_quality_bucket": "clean_train"}])
    summary = build_validation_alignment_summary(master)
    assert summary["has_source_split"] is False


def test_run_audit_writes_master_buckets_and_cleaned_manifest(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_image(images_dir / "clean.jpg", color=(120, 120, 120))
    _write_image(images_dir / "dark.jpg", color=(10, 10, 10))
    train_csv = _write_csv(
        tmp_path / "train.csv",
        [
            {"image_id": "clean.jpg", "target": 0},
            {"image_id": "dark.jpg", "target": 1},
        ],
    )
    config = _base_config(tmp_path, train_csv, images_dir)

    summary = run_audit(config)
    assert (tmp_path / "out" / "data_quality_master.csv").exists()
    assert (tmp_path / "out" / "clean_train_rows.csv").exists()
    assert (tmp_path / "out" / "manual_review_required_rows.csv").exists()
    assert (tmp_path / "out" / "cleaned_training_manifest.csv").exists()
    assert (tmp_path / "out" / "reports" / "input_file_inventory.csv").exists()
    assert (tmp_path / "out" / "reports" / "validation_alignment_summary.json").exists()
    assert summary["total_rows"] == 2
    assert summary["no_training_started"] is True


def test_summary_confirms_no_training_submission_or_leaderboard_usage(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    master = pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0, "data_quality_bucket": "clean_train", "exclude_from_training": False, "exclusion_reason": "", "label_issue_score": 0.1, "review_priority": 99, "roi_quality_flag": "ok", "duplicate_conflict_flag": False, "oof_probability": pd.NA, "anomaly_score": 0.1},
        ]
    )
    write_reports(master, output_root=tmp_path, warnings=[], duplicate_conflicts=pd.DataFrame())
    summary = json.loads((reports_dir / "data_quality_summary.json").read_text(encoding="utf-8"))
    assert summary["no_training_started"] is True
    assert summary["no_submission_created"] is True
    assert summary["no_test_labels_used"] is True
    assert summary["no_leaderboard_tuning"] is True
