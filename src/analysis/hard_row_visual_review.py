"""Enterprise-grade hard-row visual review workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


DEFAULT_CONFIG_PATH = "configs/hard_row_visual_review.yaml"
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/hard_row_visual_review")
REQUIRED_AUDIT_COLUMNS = [
    "image_id",
    "primary_audit_category",
    "audit_rationale",
    "target",
    "baseline_probability",
    "baseline_prediction",
    "candidate_probability",
    "candidate_prediction",
    "hard_example_type",
]
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
ALLOWED_FINAL_FAILURE_MODES = {
    "correct_but_visually_hard",
    "ambiguous_needs_expert_review",
    "likely_mislabeled",
    "roi_or_crop_problem",
    "detector_or_annotation_evidence_needed",
    "harmless_artifact_confusion",
    "tiny_or_low_contrast_defect",
    "anomaly_or_outlier_case",
    "unclear_needs_annotation_or_expert_review",
}
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
MANUS_ALLOWED_RECOMMENDED_ACTIONS = {
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
MANUS_REQUIRED_METADATA_COLUMNS = [
    "image_id",
    "target",
    "baseline_probability",
    "baseline_prediction",
    "candidate_probability",
    "candidate_prediction",
    "cluster_id",
    "anomaly_score",
    "label_issue_score",
]
MANUS_FLAG_COLUMNS = [
    "label_quality_flag",
    "roi_preprocessing_flag",
    "detector_evidence_flag",
    "harmless_artifact_flag",
    "tiny_or_low_contrast_defect_flag",
]
DEFAULT_ALLOWED_FLAG_VALUES = {
    "",
    "yes",
    "no",
    "unknown",
    "needs_review",
    "label_ok",
    "label_suspicious",
    "not_applicable",
    "possible_label_noise",
    "possible_crop_issue",
    "not_needed",
    "water_drop_or_reflection",
    "none",
    "likely_crop_issue",
}
MANUS_REQUIRED_ACTION_COLUMNS = [
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
EVIDENCE_ALLOWED_ACTIONS = {
    "possible_phase3_candidate",
    "safe_for_hard_training_with_artifact_focus",
    "tiny_or_low_contrast_defect_candidate",
    "needs_detector_evidence",
    "roi_pipeline_bug",
    "suspected_mislabel_exclude_from_training",
    "exclude_as_ambiguous_until_verified",
    "needs_expert_review",
}
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


class HardRowVisualReviewError(ValueError):
    """Raised when the hard-row visual review workflow is invalid."""


def load_hard_row_visual_review_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise HardRowVisualReviewError(f"Hard-row visual review config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise HardRowVisualReviewError(f"Hard-row visual review config is not valid YAML: {config_path}") from exc
    if not isinstance(payload, dict):
        raise HardRowVisualReviewError("Hard-row visual review config must be a mapping")
    _validate_safety_config(payload)
    return payload


def run_hard_row_visual_review(config_path: str | Path) -> dict[str, Path]:
    config = load_hard_row_visual_review_config(config_path)
    resolved = _resolve_paths(config, config_path)
    _assert_required_inputs_exist(resolved)

    audit = _load_hard_row_audit(resolved["hard_row_audit_csv"])
    _validate_expected_hard_row_count(
        audit,
        expected_count=int(dict(config.get("analysis") or {}).get("expected_hard_row_count", len(audit))),
    )
    metadata = _load_train_metadata(
        train_csv=resolved["train_csv"],
        bottletypes_csv=resolved["bottletypes_csv"],
        annotations_path=resolved["train_annotations_json"],
    )
    review = audit.merge(metadata, on="image_id", how="left")
    review["bottle_type"] = review["bottle_type"].fillna("").astype(str)
    review["annotation_categories"] = review["annotation_categories"].fillna("").astype(str)
    review["annotation_bbox"] = review["annotation_bbox"].fillna("").astype(str)

    images_root = resolved["train_images_dir"]
    image_info = _build_image_info(review["image_id"].astype(str).tolist(), images_root)
    review = review.merge(image_info, on="image_id", how="left")

    cleanlab_module = _import_optional_dependency("cleanlab")
    fiftyone_module = _import_optional_dependency("fiftyone")
    label_quality_summary, review = _compute_label_quality(review, config=config, cleanlab_module=cleanlab_module)

    embedding_matrix, embedding_feature_names = _compute_embeddings(review, images_root=images_root)
    review = _assign_clusters(review, embedding_matrix, config=config)
    anomaly_summary, review = _assign_anomaly_scores(review, embedding_matrix, config=config)
    review = _assign_suspicion_rank(review)
    review = _build_triage_fields(review, config=config)

    output_root = resolved["output_root"]
    contact_sheets_dir = output_root / "contact_sheets"
    reports_dir = output_root / "reports"
    fiftyone_export_dir = output_root / "fiftyone_export"
    cluster_galleries_dir = contact_sheets_dir / "cluster_galleries"
    top_suspicious_gallery_dir = contact_sheets_dir / "top_suspicious_gallery"
    for directory in (
        contact_sheets_dir,
        reports_dir,
        fiftyone_export_dir,
        cluster_galleries_dir,
        top_suspicious_gallery_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    contact_sheet_paths, missing_images, asset_paths = _build_contact_sheets(
        review,
        images_root=images_root,
        output_dir=contact_sheets_dir,
        config=config,
    )
    review["visual_asset_path"] = review["image_id"].map(lambda image_id: str(asset_paths.get(image_id, "")))
    review["contact_sheet_path"] = review["sheet_index"].map(lambda index: str(contact_sheet_paths.get(index, Path())))

    manual_review = _build_manual_review_template(review, config=config)
    validate_manual_review_frame(manual_review)
    review_manifest = _build_review_manifest(review, manual_review)
    clustering_summary = _build_clustering_summary(review, embedding_feature_names)
    triage_outputs = _build_triage_outputs(
        review,
        asset_paths=asset_paths,
        reports_dir=reports_dir,
        cluster_galleries_dir=cluster_galleries_dir,
        top_suspicious_gallery_dir=top_suspicious_gallery_dir,
        thumb_size=max(64, int(dict(config.get("visuals") or {}).get("thumbnail_size", 224))),
        config=config,
        embedding_feature_names=embedding_feature_names,
    )
    _write_fiftyone_export(
        review_manifest=review_manifest,
        output_dir=fiftyone_export_dir,
        fiftyone_module=fiftyone_module,
    )

    missing_images_path = reports_dir / "missing_images.csv"
    pd.DataFrame(missing_images).to_csv(missing_images_path, index=False)
    manual_review_path = output_root / "manual_review_template.csv"
    review_manifest_path = output_root / "review_manifest.csv"
    manual_review.to_csv(manual_review_path, index=False)
    review_manifest.to_csv(review_manifest_path, index=False)

    clustering_summary_path = reports_dir / "clustering_summary.json"
    anomaly_summary_path = reports_dir / "anomaly_summary.json"
    label_quality_summary_path = reports_dir / "label_quality_summary.json"
    visual_review_summary_path = reports_dir / "visual_review_summary.json"
    clustering_summary_path.write_text(json.dumps(clustering_summary, indent=2), encoding="utf-8")
    anomaly_summary_path.write_text(json.dumps(anomaly_summary, indent=2), encoding="utf-8")
    label_quality_summary_path.write_text(json.dumps(label_quality_summary, indent=2), encoding="utf-8")
    visual_review_summary_path.write_text(
        json.dumps(
            _build_visual_review_summary(
                review=review,
                resolved=resolved,
                cleanlab_module=cleanlab_module,
                fiftyone_module=fiftyone_module,
                missing_images=missing_images,
                contact_sheet_paths=contact_sheet_paths,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "contact_sheets_dir": contact_sheets_dir,
        "review_manifest": review_manifest_path,
        "manual_review_template": manual_review_path,
        "visual_review_summary": visual_review_summary_path,
        "clustering_summary": clustering_summary_path,
        "anomaly_summary": anomaly_summary_path,
        "label_quality_summary": label_quality_summary_path,
        "missing_images": missing_images_path,
        "fiftyone_export_dir": fiftyone_export_dir,
        "cluster_representatives": triage_outputs["cluster_representatives"],
        "top_suspicious_rows": triage_outputs["top_suspicious_rows"],
        "top_label_issue_rows": triage_outputs["top_label_issue_rows"],
        "top_anomaly_rows": triage_outputs["top_anomaly_rows"],
        "cluster_triage_summary": triage_outputs["cluster_triage_summary"],
        "final_training_action_plan": triage_outputs["final_training_action_plan"],
        "cluster_galleries_dir": cluster_galleries_dir,
        "top_suspicious_gallery_dir": top_suspicious_gallery_dir,
    }


def merge_manus_review(config_path: str | Path) -> dict[str, Path]:
    config = load_hard_row_visual_review_config(config_path)
    resolved = _resolve_paths(config, config_path)
    manus = resolved["manus_review"]
    _assert_manus_inputs_exist(manus)

    original_manual = pd.read_csv(manus["original_manual_review_template"], keep_default_na=False)
    filled_manual = pd.read_csv(manus["filled_manual_review_template"], keep_default_na=False)
    review_manifest = pd.read_csv(manus["review_manifest"], keep_default_na=False)
    cluster_representatives = pd.read_csv(manus["cluster_representatives"], keep_default_na=False)
    top_suspicious = pd.read_csv(manus["top_suspicious_rows"], keep_default_na=False)
    top_label_issues = pd.read_csv(manus["top_label_issue_rows"], keep_default_na=False)
    top_anomalies = pd.read_csv(manus["top_anomaly_rows"], keep_default_na=False)

    validation = _validate_manus_review(
        original_manual=original_manual,
        filled_manual=filled_manual,
        expected_rows=int(manus["expected_rows"]),
        allowed_flag_values=set(manus["allowed_flag_values"]),
    )

    merged = filled_manual.merge(
        review_manifest[
            [
                "image_id",
                "audit_rationale",
                "annotation_categories",
                "annotation_bbox",
                "image_path",
                "image_exists",
                "visual_asset_path",
                "contact_sheet_path",
            ]
        ],
        on="image_id",
        how="left",
    )
    merged["prediction"] = np.where(
        merged["candidate_prediction"].astype(str) != "",
        pd.to_numeric(merged["candidate_prediction"], errors="coerce"),
        pd.to_numeric(merged["baseline_prediction"], errors="coerce"),
    ).astype(int)
    merged["probability"] = np.where(
        merged["candidate_probability"].astype(str) != "",
        pd.to_numeric(merged["candidate_probability"], errors="coerce"),
        pd.to_numeric(merged["baseline_probability"], errors="coerce"),
    ).astype(float)
    merged = _apply_manus_action_logic(
        merged,
        high_label_issue_threshold=float(manus["high_label_issue_threshold"]),
        high_anomaly_threshold=float(manus["high_anomaly_threshold"]),
        validation=validation,
    )

    reports_dir = Path(manus["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    validation_path = reports_dir / "manus_review_validation.json"
    counts_path = reports_dir / "manus_review_counts.csv"
    action_plan_path = reports_dir / "final_hard_row_action_plan.csv"
    phase3_path = reports_dir / "phase3_training_candidates.csv"
    excluded_path = reports_dir / "excluded_ambiguous_or_noisy_rows.csv"
    engineering_path = reports_dir / "engineering_fix_candidates.csv"
    queue_path = reports_dir / "human_verification_queue.csv"
    final_summary_path = reports_dir / "final_action_summary.json"

    merged[MANUS_REQUIRED_ACTION_COLUMNS].to_csv(action_plan_path, index=False)
    merged.loc[merged["phase3_use_allowed"], MANUS_REQUIRED_ACTION_COLUMNS].to_csv(phase3_path, index=False)
    merged.loc[merged["is_excluded_row"], MANUS_REQUIRED_ACTION_COLUMNS].to_csv(excluded_path, index=False)
    merged.loc[merged["is_engineering_fix_row"], MANUS_REQUIRED_ACTION_COLUMNS].to_csv(engineering_path, index=False)
    merged.loc[merged["is_human_verification_row"], MANUS_REQUIRED_ACTION_COLUMNS].to_csv(queue_path, index=False)
    _build_manus_review_counts(merged).to_csv(counts_path, index=False)
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    final_summary_path.write_text(
        json.dumps(
            _build_manus_final_summary(
                merged=merged,
                validation=validation,
                cluster_representatives=cluster_representatives,
                top_suspicious=top_suspicious,
                top_label_issues=top_label_issues,
                top_anomalies=top_anomalies,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "manus_review_validation": validation_path,
        "manus_review_counts": counts_path,
        "final_hard_row_action_plan": action_plan_path,
        "phase3_training_candidates": phase3_path,
        "excluded_rows": excluded_path,
        "engineering_fix_candidates": engineering_path,
        "human_verification_queue": queue_path,
        "final_action_summary": final_summary_path,
    }


def complete_hard_row_evidence(config_path: str | Path) -> dict[str, Path]:
    config = load_hard_row_visual_review_config(config_path)
    resolved = _resolve_paths(config, config_path)
    evidence = resolved["evidence_completion"]
    if not Path(str(evidence["review_manifest"])).exists() or not Path(str(evidence["manual_review_template"])).exists():
        run_hard_row_visual_review(config_path)
    _assert_evidence_inputs_exist(evidence)

    review_manifest = pd.read_csv(evidence["review_manifest"], keep_default_na=False)
    hard_row_audit = _load_hard_row_audit(Path(str(evidence["hard_row_audit_csv"])))
    annotations = _load_annotation_metadata(Path(str(evidence["train_annotations_json"])))

    annotation_evidence = _build_annotation_evidence(
        image_ids=review_manifest["image_id"].astype(str).tolist(),
        annotations=annotations,
    )
    crop_evidence, asset_paths = _build_crop_quality_evidence(
        review_manifest=review_manifest,
        annotation_evidence=annotation_evidence,
        evidence_cfg=evidence,
    )
    evidence_manifest = _build_evidence_completed_manifest(
        review_manifest=review_manifest,
        annotation_evidence=annotation_evidence,
        crop_evidence=crop_evidence,
        asset_paths=asset_paths,
    )
    action_plan = _build_evidence_action_plan(
        evidence_manifest=evidence_manifest,
        hard_row_audit=hard_row_audit,
        evidence_cfg=evidence,
        config=config,
    )

    output_root = Path(str(evidence["output_root"]))
    reports_dir = output_root / "reports"
    contact_sheets_dir = output_root / "contact_sheets"
    reports_dir.mkdir(parents=True, exist_ok=True)
    contact_sheets_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = reports_dir / "hard_row_annotation_evidence.csv"
    crop_path = reports_dir / "hard_row_crop_quality_evidence.csv"
    summary_path = reports_dir / "hard_row_evidence_summary.json"
    manifest_path = reports_dir / "hard_row_evidence_completed_manifest.csv"
    action_plan_path = reports_dir / "final_training_action_plan_evidence_aware.csv"

    annotation_evidence.to_csv(annotation_path, index=False)
    crop_evidence.to_csv(crop_path, index=False)
    evidence_manifest.to_csv(manifest_path, index=False)
    action_plan[REQUIRED_EVIDENCE_ACTION_COLUMNS].to_csv(action_plan_path, index=False)
    _render_evidence_aware_contact_sheets(
        action_plan,
        output_dir=contact_sheets_dir,
        thumb_size=max(96, int(dict(config.get("visuals") or {}).get("thumbnail_size", 224))),
        per_sheet=max(1, int(dict(config.get("visuals") or {}).get("max_images_per_contact_sheet", 12))),
    )
    summary_path.write_text(
        json.dumps(
            _build_evidence_summary(
                action_plan=action_plan,
                annotation_evidence=annotation_evidence,
                crop_evidence=crop_evidence,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "annotation_evidence_csv": annotation_path,
        "crop_quality_evidence_csv": crop_path,
        "hard_row_evidence_summary": summary_path,
        "evidence_completed_manifest": manifest_path,
        "final_training_action_plan_evidence_aware": action_plan_path,
        "evidence_assets_dir": Path(str(evidence["evidence_assets_dir"])),
    }


def export_phase3_candidates(
    config_path: str | Path,
    *,
    expected_row_count: int | None = None,
    expected_allowed_count: int | None = None,
    expected_blocked_count: int | None = None,
) -> dict[str, Path]:
    config = load_hard_row_visual_review_config(config_path)
    resolved = _resolve_paths(config, config_path)
    export_cfg = dict(resolved["evidence_completion"])
    evidence_output = Path(str(resolved["evidence_completion"]["output_root"])) / "reports" / "final_training_action_plan_evidence_aware.csv"
    if not evidence_output.exists():
        raise HardRowVisualReviewError(
            "Missing required evidence-aware action plan: " + str(evidence_output)
        )

    frame = pd.read_csv(evidence_output, keep_default_na=False).copy()
    _validate_phase3_candidate_input(
        frame,
        expected_row_count=expected_row_count if expected_row_count is not None else int(export_cfg["expected_rows"]),
        expected_allowed_count=expected_allowed_count if expected_allowed_count is not None else export_cfg["expected_phase3_allowed_count"],
        expected_blocked_count=expected_blocked_count if expected_blocked_count is not None else export_cfg["expected_phase3_blocked_count"],
    )

    reports_dir = evidence_output.parent
    allowed = frame[frame["phase3_use_allowed"].astype(bool)].copy()
    blocked = frame[~frame["phase3_use_allowed"].astype(bool)].copy()

    artifact_rows = allowed[allowed["recommended_action"] == "safe_for_hard_training_with_artifact_focus"].copy()
    tiny_rows = allowed[allowed["recommended_action"] == "tiny_or_low_contrast_defect_candidate"].copy()
    general_rows = allowed[allowed["recommended_action"] == "possible_phase3_candidate"].copy()
    roi_blocked = blocked[blocked["recommended_action"] == "roi_pipeline_bug"].copy()
    mislabel_blocked = blocked[blocked["recommended_action"] == "suspected_mislabel_exclude_from_training"].copy()
    other_blocked = blocked[
        ~blocked["recommended_action"].isin({"roi_pipeline_bug", "suspected_mislabel_exclude_from_training"})
    ].copy()

    allowed_path = reports_dir / "phase3_allowed_hard_training_rows.csv"
    artifact_path = reports_dir / "phase3_artifact_focus_rows.csv"
    tiny_path = reports_dir / "phase3_tiny_low_contrast_rows.csv"
    general_path = reports_dir / "phase3_general_candidate_rows.csv"
    roi_blocked_path = reports_dir / "blocked_roi_pipeline_bug_rows.csv"
    mislabel_blocked_path = reports_dir / "blocked_suspected_mislabel_rows.csv"
    other_blocked_path = reports_dir / "blocked_other_high_risk_rows.csv"
    summary_path = reports_dir / "phase3_candidate_package_summary.json"

    allowed.to_csv(allowed_path, index=False)
    artifact_rows.to_csv(artifact_path, index=False)
    tiny_rows.to_csv(tiny_path, index=False)
    general_rows.to_csv(general_path, index=False)
    roi_blocked.to_csv(roi_blocked_path, index=False)
    mislabel_blocked.to_csv(mislabel_blocked_path, index=False)
    other_blocked.to_csv(other_blocked_path, index=False)

    summary_path.write_text(
        json.dumps(
            _build_phase3_candidate_package_summary(
                input_file=evidence_output,
                row_count=len(frame),
                allowed=allowed,
                blocked=blocked,
                artifact_rows=artifact_rows,
                tiny_rows=tiny_rows,
                general_rows=general_rows,
                roi_blocked=roi_blocked,
                mislabel_blocked=mislabel_blocked,
                other_blocked=other_blocked,
                output_paths={
                    "phase3_allowed_hard_training_rows": allowed_path,
                    "phase3_artifact_focus_rows": artifact_path,
                    "phase3_tiny_low_contrast_rows": tiny_path,
                    "phase3_general_candidate_rows": general_path,
                    "blocked_roi_pipeline_bug_rows": roi_blocked_path,
                    "blocked_suspected_mislabel_rows": mislabel_blocked_path,
                    "blocked_other_high_risk_rows": other_blocked_path,
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "phase3_allowed_hard_training_rows": allowed_path,
        "phase3_artifact_focus_rows": artifact_path,
        "phase3_tiny_low_contrast_rows": tiny_path,
        "phase3_general_candidate_rows": general_path,
        "blocked_roi_pipeline_bug_rows": roi_blocked_path,
        "blocked_suspected_mislabel_rows": mislabel_blocked_path,
        "blocked_other_high_risk_rows": other_blocked_path,
        "phase3_candidate_package_summary": summary_path,
    }


def validate_manual_review_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_MANUAL_REVIEW_COLUMNS if column not in frame.columns]
    if missing:
        raise HardRowVisualReviewError(
            "Manual review frame missing required columns: " + ", ".join(missing)
        )
    invalid = sorted(
        {
            str(value).strip()
            for value in frame["final_failure_mode"].tolist()
            if str(value).strip() and str(value).strip() not in ALLOWED_FINAL_FAILURE_MODES
        }
    )
    if invalid:
        raise HardRowVisualReviewError("Invalid final_failure_mode values: " + ", ".join(invalid))


def _validate_safety_config(config: dict[str, object]) -> None:
    safety = dict(config.get("safety") or {})
    if bool(safety.get("used_test_labels")):
        raise HardRowVisualReviewError("Hard-row visual review forbids test labels")
    if bool(safety.get("training_enabled")):
        raise HardRowVisualReviewError("Hard-row visual review forbids training")
    if bool(safety.get("submission_enabled")):
        raise HardRowVisualReviewError("Hard-row visual review forbids submission creation")
    if bool(safety.get("leaderboard_tuning_enabled")):
        raise HardRowVisualReviewError("Hard-row visual review forbids leaderboard tuning")
    triage = dict(config.get("triage") or {})
    triage_safety = dict(triage.get("safety") or {})
    if bool(triage_safety.get("training_enabled")):
        raise HardRowVisualReviewError("Hard-row visual review triage forbids training")
    if bool(triage_safety.get("submission_enabled")):
        raise HardRowVisualReviewError("Hard-row visual review triage forbids submission creation")
    if bool(triage_safety.get("test_label_usage_enabled")):
        raise HardRowVisualReviewError("Hard-row visual review triage forbids test labels")
    if bool(triage_safety.get("label_modification_enabled")):
        raise HardRowVisualReviewError("Hard-row visual review triage forbids label modification")
    invalid_actions = sorted(
        set(dict(config.get("triage") or {}).get("allowed_recommended_actions", [])) - ALLOWED_RECOMMENDED_ACTIONS
    )
    if invalid_actions:
        raise HardRowVisualReviewError(
            "Hard-row visual review triage has invalid recommended actions: " + ", ".join(invalid_actions)
        )


def _resolve_paths(config: dict[str, object], config_path: str | Path) -> dict[str, Path]:
    cfg_path = Path(config_path).resolve()
    repo_root = _discover_repo_root(cfg_path)
    inputs = dict(config.get("inputs") or {})
    experiment = dict(config.get("experiment") or {})
    manus_cfg = dict(config.get("manus_review") or {})
    evidence_cfg = dict(config.get("evidence_completion") or {})
    output_root = _resolve_path(repo_root, experiment.get("output_root", str(DEFAULT_OUTPUT_ROOT)))
    return {
        "repo_root": repo_root,
        "hard_row_audit_csv": _resolve_path(repo_root, inputs.get("hard_row_audit_csv", "")),
        "train_images_dir": _resolve_path(repo_root, inputs.get("train_images_dir", "")),
        "train_csv": _resolve_path(repo_root, inputs.get("train_csv", "")),
        "bottletypes_csv": _resolve_path(repo_root, inputs.get("bottletypes_csv", "")),
        "train_annotations_json": _resolve_optional_path(repo_root, inputs.get("train_annotations_json", "")),
        "output_root": output_root,
        "manus_review": {
            "enabled": bool(manus_cfg.get("enabled", False)),
            "expected_rows": int(manus_cfg.get("expected_rows", 442)),
            "input_dir": _resolve_optional_path(repo_root, manus_cfg.get("input_dir", "")),
            "filled_manual_review_template": _resolve_optional_path(repo_root, manus_cfg.get("filled_manual_review_template", "")),
            "review_summary_json": _resolve_optional_path(repo_root, manus_cfg.get("review_summary_json", "")),
            "human_verification_priority_list": _resolve_optional_path(repo_root, manus_cfg.get("human_verification_priority_list", "")),
            "original_manual_review_template": _resolve_optional_path(repo_root, manus_cfg.get("original_manual_review_template", "")),
            "review_manifest": _resolve_optional_path(repo_root, manus_cfg.get("review_manifest", "")),
            "cluster_triage_summary": _resolve_optional_path(repo_root, manus_cfg.get("cluster_triage_summary", "")),
            "cluster_representatives": _resolve_optional_path(repo_root, manus_cfg.get("cluster_representatives", "")),
            "top_suspicious_rows": _resolve_optional_path(repo_root, manus_cfg.get("top_suspicious_rows", "")),
            "top_label_issue_rows": _resolve_optional_path(repo_root, manus_cfg.get("top_label_issue_rows", "")),
            "top_anomaly_rows": _resolve_optional_path(repo_root, manus_cfg.get("top_anomaly_rows", "")),
            "high_label_issue_threshold": float(manus_cfg.get("high_label_issue_threshold", 0.70)),
            "high_anomaly_threshold": float(manus_cfg.get("high_anomaly_threshold", 0.70)),
            "allowed_flag_values": list(manus_cfg.get("allowed_flag_values", sorted(DEFAULT_ALLOWED_FLAG_VALUES))),
            "reports_dir": output_root / "reports",
        },
        "evidence_completion": {
            "enabled": bool(evidence_cfg.get("enabled", False)),
            "expected_rows": int(evidence_cfg.get("expected_rows", 442)),
            "expected_phase3_allowed_count": _optional_int(evidence_cfg.get("expected_phase3_allowed_count")),
            "expected_phase3_blocked_count": _optional_int(evidence_cfg.get("expected_phase3_blocked_count")),
            "review_manifest": _resolve_optional_path(repo_root, evidence_cfg.get("review_manifest", "")),
            "manual_review_template": _resolve_optional_path(repo_root, evidence_cfg.get("manual_review_template", "")),
            "hard_row_audit_csv": _resolve_optional_path(repo_root, evidence_cfg.get("hard_row_audit_csv", "")),
            "train_images_dir": _resolve_optional_path(repo_root, evidence_cfg.get("train_images_dir", "")),
            "train_csv": _resolve_optional_path(repo_root, evidence_cfg.get("train_csv", "")),
            "train_annotations_json": _resolve_optional_path(repo_root, evidence_cfg.get("train_annotations_json", "")),
            "bottletypes_csv": _resolve_optional_path(repo_root, evidence_cfg.get("bottletypes_csv", "")),
            "detector_outputs_dir": _resolve_optional_path(repo_root, evidence_cfg.get("detector_outputs_dir", "")),
            "output_root": _resolve_path(repo_root, evidence_cfg.get("output_root", str(output_root))),
            "evidence_assets_dir": _resolve_path(
                repo_root,
                evidence_cfg.get("evidence_assets_dir", str(output_root / "evidence_assets")),
            ),
            "tiny_defect_area_ratio_threshold": float(evidence_cfg.get("tiny_defect_area_ratio_threshold", 0.01)),
            "min_roi_coverage_ratio": float(evidence_cfg.get("min_roi_coverage_ratio", 0.10)),
            "max_roi_coverage_ratio": float(evidence_cfg.get("max_roi_coverage_ratio", 0.95)),
            "min_crop_width": int(evidence_cfg.get("min_crop_width", 64)),
            "min_crop_height": int(evidence_cfg.get("min_crop_height", 64)),
            "blur_laplacian_issue_threshold": float(evidence_cfg.get("blur_laplacian_issue_threshold", 40.0)),
            "low_contrast_issue_threshold": float(evidence_cfg.get("low_contrast_issue_threshold", 15.0)),
            "edge_density_low_threshold": float(evidence_cfg.get("edge_density_low_threshold", 0.01)),
        },
    }


def _discover_repo_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()


def _resolve_path(base_root: Path, raw_value: object) -> Path:
    text = str(raw_value or "").strip()
    if not text:
        raise HardRowVisualReviewError("Hard-row visual review requires non-empty config paths")
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_root / path).resolve()


def _resolve_optional_path(base_root: Path, raw_value: object) -> Path | None:
    text = str(raw_value or "").strip()
    if not text or text.lower() == "null":
        return None
    return _resolve_path(base_root, text)


def _assert_required_inputs_exist(resolved: dict[str, Path]) -> None:
    required = ["hard_row_audit_csv", "train_images_dir", "train_csv", "bottletypes_csv"]
    missing = [key for key in required if not resolved[key].exists()]
    if missing:
        details = ", ".join(f"{key}={resolved[key]}" for key in missing)
        raise HardRowVisualReviewError("Missing required visual review input: " + details)


def _validate_expected_hard_row_count(frame: pd.DataFrame, *, expected_count: int) -> None:
    if len(frame) != expected_count:
        raise HardRowVisualReviewError(
            f"Hard-row visual review expected hard-row count mismatch: expected {expected_count}, got {len(frame)}"
        )


def _assert_manus_inputs_exist(manus: dict[str, object]) -> None:
    required = [
        "filled_manual_review_template",
        "original_manual_review_template",
        "review_manifest",
        "cluster_representatives",
        "top_suspicious_rows",
        "top_label_issue_rows",
        "top_anomaly_rows",
    ]
    missing = [key for key in required if not manus.get(key) or not Path(str(manus[key])).exists()]
    if missing:
        details = ", ".join(f"{key}={manus.get(key)}" for key in missing)
        raise HardRowVisualReviewError("Missing required Manus review input: " + details)


def _assert_evidence_inputs_exist(evidence: dict[str, object]) -> None:
    required = [
        "review_manifest",
        "manual_review_template",
        "hard_row_audit_csv",
        "train_images_dir",
        "train_csv",
        "train_annotations_json",
    ]
    missing = [key for key in required if not evidence.get(key) or not Path(str(evidence[key])).exists()]
    if missing:
        details = ", ".join(f"{key}={evidence.get(key)}" for key in missing)
        raise HardRowVisualReviewError("Missing required evidence-completion input: " + details)


def _load_hard_row_audit(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False).copy()
    missing = [column for column in REQUIRED_AUDIT_COLUMNS if column not in frame.columns]
    if missing:
        raise HardRowVisualReviewError("Hard-row audit CSV missing required columns: " + ", ".join(missing))
    frame["image_id"] = frame["image_id"].map(_normalize_image_id)
    if frame["image_id"].duplicated().any():
        raise HardRowVisualReviewError("Hard-row audit CSV contains duplicate image_id values")
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(int)
    frame["baseline_probability"] = pd.to_numeric(frame["baseline_probability"], errors="raise").astype(float)
    frame["baseline_prediction"] = pd.to_numeric(frame["baseline_prediction"], errors="raise").astype(int)
    frame["candidate_probability"] = pd.to_numeric(frame["candidate_probability"], errors="raise").astype(float)
    frame["candidate_prediction"] = pd.to_numeric(frame["candidate_prediction"], errors="raise").astype(int)
    return frame


def _load_train_metadata(
    *,
    train_csv: Path,
    bottletypes_csv: Path,
    annotations_path: Path | None,
) -> pd.DataFrame:
    train = pd.read_csv(train_csv, keep_default_na=False).copy()
    image_col = _find_column(train, ["image_id", "id", "filename", "image", "path"])
    train["image_id"] = train[image_col].map(_normalize_image_id)
    bottletypes = pd.read_csv(bottletypes_csv, keep_default_na=False).copy()
    bottle_id_col = _find_column(train, ["bottle_type", "bottle_type_id", "bottletype", "bottletype_id"], required=False)
    bottle_name_map: dict[str, str] = {}
    id_col = _find_column(bottletypes, ["bottle_type_id", "bottletype_id", "id"], required=False)
    name_col = _find_column(bottletypes, ["bottle_type", "bottletype", "name", "type"], required=False)
    if id_col is not None and name_col is not None:
        bottle_name_map = dict(zip(bottletypes[id_col].astype(str), bottletypes[name_col].astype(str)))
    bottle_type = train[bottle_id_col].astype(str).map(bottle_name_map).fillna(train[bottle_id_col].astype(str)) if bottle_id_col else ""
    annotations = _load_annotation_metadata(annotations_path)
    metadata = pd.DataFrame(
        {
            "image_id": train["image_id"],
            "bottle_type": bottle_type if bottle_id_col else pd.Series([""] * len(train)),
            "annotation_categories": train["image_id"].map(lambda image_id: "|".join(annotations["categories"].get(image_id, []))),
            "annotation_bbox": train["image_id"].map(lambda image_id: annotations["bbox"].get(image_id, "")),
        }
    )
    return metadata.drop_duplicates("image_id")


def _load_annotation_metadata(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {
            "categories": {},
            "bbox": {},
            "defect_boxes": {},
            "roi_bbox": {},
            "image_size": {},
            "annotation_count": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_lookup = {
        item.get("id"): _normalize_image_id(item.get("file_name", item.get("image_id", item.get("id"))))
        for item in payload.get("images", [])
        if isinstance(item, dict)
    }
    category_lookup = {
        item.get("id"): str(item.get("name", item.get("category_name", item.get("id"))))
        for item in payload.get("categories", [])
        if isinstance(item, dict)
    }
    categories_by_image: dict[str, set[str]] = {}
    bbox_by_image: dict[str, str] = {}
    defect_boxes_by_image: dict[str, list[dict[str, object]]] = {}
    roi_bbox_by_image: dict[str, tuple[int, int, int, int]] = {}
    image_size_by_image: dict[str, tuple[int, int]] = {}
    annotation_count_by_image: dict[str, int] = {}
    for item in payload.get("images", []):
        if not isinstance(item, dict):
            continue
        try:
            image_id = _normalize_image_id(item.get("file_name", item.get("image_id", item.get("id"))))
        except HardRowVisualReviewError:
            continue
        width = int(float(item.get("width", 0) or 0))
        height = int(float(item.get("height", 0) or 0))
        if width > 0 and height > 0:
            image_size_by_image[image_id] = (width, height)
    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        normalized = image_lookup.get(annotation.get("image_id"), _normalize_image_id(annotation.get("image_id")))
        category = str(
            annotation.get(
                "category_name",
                annotation.get("class_name", category_lookup.get(annotation.get("category_id"), annotation.get("category_id", ""))),
            )
        ).strip()
        annotation_count_by_image[normalized] = annotation_count_by_image.get(normalized, 0) + 1
        if category:
            categories_by_image.setdefault(normalized, set()).add(category)
        bbox = annotation.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox_tuple = tuple(int(float(value)) for value in bbox)
            if normalized not in bbox_by_image:
                bbox_by_image[normalized] = ",".join(str(value) for value in bbox_tuple)
            if category.lower() == "roi":
                roi_bbox_by_image[normalized] = bbox_tuple
            else:
                defect_boxes_by_image.setdefault(normalized, []).append(
                    {"category": category, "bbox": bbox_tuple, "area": max(0, bbox_tuple[2]) * max(0, bbox_tuple[3])}
                )
    return {
        "categories": {image_id: sorted(values) for image_id, values in categories_by_image.items()},
        "bbox": bbox_by_image,
        "defect_boxes": defect_boxes_by_image,
        "roi_bbox": {image_id: ",".join(str(value) for value in bbox) for image_id, bbox in roi_bbox_by_image.items()},
        "image_size": image_size_by_image,
        "annotation_count": annotation_count_by_image,
    }


def _build_annotation_evidence(*, image_ids: list[str], annotations: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    categories_lookup = dict(annotations.get("categories", {}))
    defect_lookup = dict(annotations.get("defect_boxes", {}))
    roi_lookup = dict(annotations.get("roi_bbox", {}))
    image_size_lookup = dict(annotations.get("image_size", {}))
    annotation_count_lookup = dict(annotations.get("annotation_count", {}))
    for image_id in image_ids:
        categories = [category for category in categories_lookup.get(image_id, []) if category.lower() != "roi"]
        defect_boxes = list(defect_lookup.get(image_id, []))
        image_width, image_height = image_size_lookup.get(image_id, (0, 0))
        image_area = max(1, int(image_width) * int(image_height)) if image_width and image_height else 0
        largest_defect_area = max([int(item["area"]) for item in defect_boxes], default=0)
        total_defect_area = int(sum(int(item["area"]) for item in defect_boxes))
        roi_bbox_text = str(roi_lookup.get(image_id, "")).strip()
        roi_bbox = _parse_bbox_text(roi_bbox_text) if roi_bbox_text else None
        if roi_bbox is None and defect_boxes:
            roi_bbox = _derive_roi_from_defects(
                boxes=[tuple(item["bbox"]) for item in defect_boxes],
                image_width=int(image_width or 0),
                image_height=int(image_height or 0),
            )
            roi_bbox_text = _format_bbox(roi_bbox)
        rows.append(
            {
                "image_id": image_id,
                "image_width": int(image_width or 0),
                "image_height": int(image_height or 0),
                "roi_bbox": roi_bbox_text,
                "roi_exists": bool(roi_bbox),
                "annotation_count": int(len(defect_boxes)),
                "defect_categories": "|".join(categories),
                "largest_defect_area": int(largest_defect_area),
                "total_defect_area": int(total_defect_area),
                "largest_defect_area_ratio": float(largest_defect_area / image_area) if image_area else 0.0,
                "total_defect_area_ratio": float(total_defect_area / image_area) if image_area else 0.0,
                "has_annotation_evidence": bool(defect_boxes),
                "annotation_evidence_status": (
                    "annotation_evidence_available" if defect_boxes else "annotation_evidence_missing"
                ),
                "fault_evidence_type": "faulty" if defect_boxes else "good_or_unannotated",
            }
        )
    return pd.DataFrame(rows)


def _build_crop_quality_evidence(
    *,
    review_manifest: pd.DataFrame,
    annotation_evidence: pd.DataFrame,
    evidence_cfg: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    image_root = Path(str(evidence_cfg["train_images_dir"]))
    evidence_assets_dir = Path(str(evidence_cfg["evidence_assets_dir"]))
    asset_dirs = {
        "original_roi_overlay": evidence_assets_dir / "original_roi_overlay",
        "annotation_overlay": evidence_assets_dir / "annotation_overlay",
        "roi_crop": evidence_assets_dir / "roi_crop",
        "contrast_roi_crop": evidence_assets_dir / "contrast_roi_crop",
        "edge_roi_crop": evidence_assets_dir / "edge_roi_crop",
    }
    for directory in asset_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    merged = review_manifest.merge(annotation_evidence, on="image_id", how="left")
    rows: list[dict[str, object]] = []
    asset_paths: dict[str, dict[str, str]] = {}
    for row in merged.to_dict("records"):
        image_id = str(row["image_id"])
        image_path = image_root / image_id
        crop_row, paths = _compute_single_crop_evidence(
            row=row,
            image_path=image_path,
            asset_dirs=asset_dirs,
            evidence_cfg=evidence_cfg,
        )
        rows.append(crop_row)
        asset_paths[image_id] = paths
    return pd.DataFrame(rows), asset_paths


def _compute_single_crop_evidence(
    *,
    row: dict[str, object],
    image_path: Path,
    asset_dirs: dict[str, Path],
    evidence_cfg: dict[str, object],
) -> tuple[dict[str, object], dict[str, str]]:
    image_id = str(row["image_id"])
    roi_bbox = _parse_bbox_text(str(row.get("roi_bbox", "")))
    empty_paths = {name: "" for name in asset_dirs}
    if not image_path.exists():
        return (
            {
                "image_id": image_id,
                "image_exists": False,
                "image_width": int(row.get("image_width", 0) or 0),
                "image_height": int(row.get("image_height", 0) or 0),
                "roi_exists": False,
                "roi_x1": 0,
                "roi_y1": 0,
                "roi_x2": 0,
                "roi_y2": 0,
                "roi_width": 0,
                "roi_height": 0,
                "roi_coverage_ratio": 0.0,
                "roi_touches_border": False,
                "crop_too_small": False,
                "crop_missing_important_area": False,
                "brightness_score": 0.0,
                "contrast_score": 0.0,
                "blur_score_laplacian_variance": 0.0,
                "edge_density": 0.0,
                "crop_quality_status": "missing",
                "crop_quality_reason": "image_missing",
            },
            empty_paths,
        )

    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    if roi_bbox is None:
        return (
            {
                "image_id": image_id,
                "image_exists": True,
                "image_width": image_width,
                "image_height": image_height,
                "roi_exists": False,
                "roi_x1": 0,
                "roi_y1": 0,
                "roi_x2": 0,
                "roi_y2": 0,
                "roi_width": 0,
                "roi_height": 0,
                "roi_coverage_ratio": 0.0,
                "roi_touches_border": False,
                "crop_too_small": False,
                "crop_missing_important_area": False,
                "brightness_score": 0.0,
                "contrast_score": 0.0,
                "blur_score_laplacian_variance": 0.0,
                "edge_density": 0.0,
                "crop_quality_status": "likely_issue",
                "crop_quality_reason": "roi_missing",
            },
            empty_paths,
        )

    x, y, width, height = roi_bbox
    x1 = max(0, min(image_width - 1, x))
    y1 = max(0, min(image_height - 1, y))
    x2 = max(x1 + 1, min(image_width, x + max(1, width)))
    y2 = max(y1 + 1, min(image_height, y + max(1, height)))
    roi_width = int(x2 - x1)
    roi_height = int(y2 - y1)
    crop = image.crop((x1, y1, x2, y2))
    grayscale = np.asarray(crop.convert("L"), dtype=np.float32)
    brightness = float(grayscale.mean()) if grayscale.size else 0.0
    contrast = float(grayscale.std()) if grayscale.size else 0.0
    blur_score = _laplacian_variance(grayscale)
    edge_density = _edge_density(grayscale)
    image_area = max(1, image_width * image_height)
    coverage = float((roi_width * roi_height) / image_area)
    touches_border = x1 <= 0 or y1 <= 0 or x2 >= image_width or y2 >= image_height
    crop_too_small = roi_width < int(evidence_cfg["min_crop_width"]) or roi_height < int(evidence_cfg["min_crop_height"])
    crop_missing_important_area = _crop_misses_defect_area(
        roi_bbox=(x1, y1, roi_width, roi_height),
        annotation_bbox=_parse_bbox_text(str(row.get("annotation_bbox", ""))),
    )
    crop_quality_status, crop_quality_reason = _classify_crop_quality(
        coverage=coverage,
        touches_border=touches_border,
        crop_too_small=crop_too_small,
        crop_missing_important_area=crop_missing_important_area,
        contrast=contrast,
        blur_score=blur_score,
        edge_density=edge_density,
        evidence_cfg=evidence_cfg,
    )
    paths = _write_evidence_assets(
        image_id=image_id,
        image=image,
        roi_bbox=(x1, y1, roi_width, roi_height),
        annotation_bbox=_parse_bbox_text(str(row.get("annotation_bbox", ""))),
        asset_dirs=asset_dirs,
    )
    return (
        {
            "image_id": image_id,
            "image_exists": True,
            "image_width": image_width,
            "image_height": image_height,
            "roi_exists": True,
            "roi_x1": x1,
            "roi_y1": y1,
            "roi_x2": x2,
            "roi_y2": y2,
            "roi_width": roi_width,
            "roi_height": roi_height,
            "roi_coverage_ratio": coverage,
            "roi_touches_border": touches_border,
            "crop_too_small": crop_too_small,
            "crop_missing_important_area": crop_missing_important_area,
            "brightness_score": brightness,
            "contrast_score": contrast,
            "blur_score_laplacian_variance": blur_score,
            "edge_density": edge_density,
            "crop_quality_status": crop_quality_status,
            "crop_quality_reason": crop_quality_reason,
        },
        paths,
    )


def _build_evidence_completed_manifest(
    *,
    review_manifest: pd.DataFrame,
    annotation_evidence: pd.DataFrame,
    crop_evidence: pd.DataFrame,
    asset_paths: dict[str, dict[str, str]],
) -> pd.DataFrame:
    manifest = review_manifest.merge(annotation_evidence, on="image_id", how="left")
    manifest = manifest.merge(crop_evidence, on="image_id", how="left", suffixes=("", "_crop"))
    missing_dims = pd.to_numeric(manifest["image_width"], errors="coerce").fillna(0).eq(0) | pd.to_numeric(
        manifest["image_height"], errors="coerce"
    ).fillna(0).eq(0)
    manifest.loc[missing_dims, "image_width"] = pd.to_numeric(manifest.loc[missing_dims, "image_width_crop"], errors="coerce").fillna(0)
    manifest.loc[missing_dims, "image_height"] = pd.to_numeric(manifest.loc[missing_dims, "image_height_crop"], errors="coerce").fillna(0)
    image_area = (
        pd.to_numeric(manifest["image_width"], errors="coerce").fillna(0)
        * pd.to_numeric(manifest["image_height"], errors="coerce").fillna(0)
    ).replace(0, np.nan)
    manifest["largest_defect_area_ratio"] = (
        pd.to_numeric(manifest["largest_defect_area"], errors="coerce").fillna(0) / image_area
    ).fillna(0.0)
    manifest["total_defect_area_ratio"] = (
        pd.to_numeric(manifest["total_defect_area"], errors="coerce").fillna(0) / image_area
    ).fillna(0.0)
    manifest["detector_evidence_status"] = manifest["annotation_evidence_status"].map(
        lambda status: "annotation_evidence_available" if status == "annotation_evidence_available" else "annotation_evidence_missing"
    )
    manifest["evidence_asset_original_roi_overlay"] = manifest["image_id"].map(
        lambda image_id: asset_paths.get(str(image_id), {}).get("original_roi_overlay", "")
    )
    manifest["evidence_asset_annotation_overlay"] = manifest["image_id"].map(
        lambda image_id: asset_paths.get(str(image_id), {}).get("annotation_overlay", "")
    )
    manifest["evidence_asset_roi_crop"] = manifest["image_id"].map(
        lambda image_id: asset_paths.get(str(image_id), {}).get("roi_crop", "")
    )
    manifest["evidence_asset_contrast_roi_crop"] = manifest["image_id"].map(
        lambda image_id: asset_paths.get(str(image_id), {}).get("contrast_roi_crop", "")
    )
    manifest["evidence_asset_edge_roi_crop"] = manifest["image_id"].map(
        lambda image_id: asset_paths.get(str(image_id), {}).get("edge_roi_crop", "")
    )
    return manifest.sort_values("suspicion_rank").reset_index(drop=True)


def _build_evidence_action_plan(
    *,
    evidence_manifest: pd.DataFrame,
    hard_row_audit: pd.DataFrame,
    evidence_cfg: dict[str, object],
    config: dict[str, object],
) -> pd.DataFrame:
    merged = evidence_manifest.merge(
        hard_row_audit[["image_id", "target", "candidate_prediction", "candidate_probability"]],
        on="image_id",
        how="left",
        suffixes=("", "_audit"),
    ).copy()
    triage_thresholds = dict(dict(config.get("triage") or {}).get("thresholds") or {})
    high_label_issue = float(triage_thresholds.get("high_label_issue_score", 0.75))
    high_anomaly = float(triage_thresholds.get("high_anomaly_score", 0.75))
    if len(merged) < 10:
        high_anomaly = max(high_anomaly, 1.01)
    tiny_threshold = float(evidence_cfg["tiny_defect_area_ratio_threshold"])

    recommended_actions: list[str] = []
    phase3_flags: list[bool] = []
    reasons: list[str] = []
    for row in merged.to_dict("records"):
        action, allow_phase3, reason = _recommend_evidence_action(
            row=row,
            high_label_issue_threshold=high_label_issue,
            high_anomaly_threshold=high_anomaly,
            tiny_defect_area_ratio_threshold=tiny_threshold,
        )
        recommended_actions.append(action)
        phase3_flags.append(allow_phase3)
        reasons.append(reason)

    merged["prediction"] = pd.to_numeric(merged["candidate_prediction"], errors="coerce").fillna(
        pd.to_numeric(merged["baseline_prediction"], errors="coerce")
    ).astype(int)
    merged["probability"] = pd.to_numeric(merged["candidate_probability"], errors="coerce").fillna(
        pd.to_numeric(merged["baseline_probability"], errors="coerce")
    ).astype(float)
    merged["recommended_action"] = recommended_actions
    merged["phase3_use_allowed"] = phase3_flags
    merged["reason"] = reasons
    merged["defect_categories"] = merged["defect_categories"].fillna("").astype(str)
    for column in [
        "largest_defect_area_ratio",
        "total_defect_area_ratio",
        "roi_coverage_ratio",
        "brightness_score",
        "contrast_score",
        "blur_score_laplacian_variance",
        "edge_density",
    ]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged.sort_values("suspicion_rank").reset_index(drop=True)


def _recommend_evidence_action(
    *,
    row: dict[str, object],
    high_label_issue_threshold: float,
    high_anomaly_threshold: float,
    tiny_defect_area_ratio_threshold: float,
) -> tuple[str, bool, str]:
    crop_quality_status = str(row.get("crop_quality_status", "")).strip()
    annotation_status = str(row.get("annotation_evidence_status", "")).strip()
    primary_category = str(row.get("primary_audit_category", "")).strip()
    defect_categories = str(row.get("defect_categories", "")).strip().lower()
    label_issue_score = float(row.get("label_issue_score", 0.0) or 0.0)
    anomaly_score = float(row.get("anomaly_score", 0.0) or 0.0)
    largest_ratio = float(row.get("largest_defect_area_ratio", 0.0) or 0.0)
    contrast = float(row.get("contrast_score", 0.0) or 0.0)
    edge_density = float(row.get("edge_density", 0.0) or 0.0)

    if crop_quality_status == "likely_issue":
        return "roi_pipeline_bug", False, "crop or ROI evidence indicates a likely preprocessing problem"
    if label_issue_score >= high_label_issue_threshold:
        return "suspected_mislabel_exclude_from_training", False, "high label_issue_score requires verification before training"
    if primary_category == "likely_ambiguous" and annotation_status != "annotation_evidence_available":
        return "exclude_as_ambiguous_until_verified", False, "ambiguous row lacks enough supporting evidence"
    if annotation_status != "annotation_evidence_available":
        return "needs_expert_review", False, "no strong annotation or ROI evidence is available"
    if largest_ratio <= tiny_defect_area_ratio_threshold and (contrast <= 30.0 or edge_density <= 0.10):
        allow_phase3 = anomaly_score < high_anomaly_threshold and label_issue_score < high_label_issue_threshold
        return (
            "tiny_or_low_contrast_defect_candidate",
            allow_phase3,
            "annotation evidence suggests a tiny or low-contrast defect candidate",
        )
    if "water drop" in defect_categories or "reflection" in defect_categories:
        allow_phase3 = anomaly_score < high_anomaly_threshold and label_issue_score < high_label_issue_threshold
        return (
            "safe_for_hard_training_with_artifact_focus",
            allow_phase3,
            "annotation categories suggest harmless artifact confusion rather than label noise",
        )
    if crop_quality_status == "possible_issue" and largest_ratio <= tiny_defect_area_ratio_threshold:
        return "needs_detector_evidence", False, "annotation exists but crop quality remains weak for a very small defect"
    allow_phase3 = anomaly_score < high_anomaly_threshold and label_issue_score < high_label_issue_threshold
    return "possible_phase3_candidate", allow_phase3, "label, crop, and annotation evidence are aligned"


def _build_evidence_summary(
    *,
    action_plan: pd.DataFrame,
    annotation_evidence: pd.DataFrame,
    crop_evidence: pd.DataFrame,
) -> dict[str, object]:
    return {
        "processed_hard_rows": int(len(action_plan)),
        "rows_with_annotation_evidence": int(annotation_evidence["has_annotation_evidence"].astype(bool).sum()),
        "rows_with_roi_evidence": int(crop_evidence["roi_exists"].astype(bool).sum()),
        "crop_quality_status_counts": crop_evidence["crop_quality_status"].value_counts().sort_index().to_dict(),
        "annotation_evidence_status_counts": annotation_evidence["annotation_evidence_status"].value_counts().sort_index().to_dict(),
        "detector_evidence_status_counts": action_plan["detector_evidence_status"].value_counts().sort_index().to_dict(),
        "recommended_action_counts": action_plan["recommended_action"].value_counts().sort_index().to_dict(),
        "phase3_use_allowed_count": int(action_plan["phase3_use_allowed"].astype(bool).sum()),
        "used_test_labels": False,
        "trained_model": False,
        "submission_created": False,
        "leaderboard_tuning_used": False,
    }


def _validate_phase3_candidate_input(
    frame: pd.DataFrame,
    *,
    expected_row_count: int,
    expected_allowed_count: int | None,
    expected_blocked_count: int | None,
) -> None:
    if frame.empty:
        raise HardRowVisualReviewError("Phase 3 candidate export input CSV is empty")
    missing = [column for column in REQUIRED_EVIDENCE_ACTION_COLUMNS if column not in frame.columns]
    if missing:
        raise HardRowVisualReviewError(
            "Phase 3 candidate export input is missing required columns: " + ", ".join(missing)
        )
    if len(frame) != expected_row_count:
        raise HardRowVisualReviewError(
            f"Phase 3 candidate export row count mismatch: expected {expected_row_count}, got {len(frame)}"
        )
    allowed_mask = frame["phase3_use_allowed"].astype(bool)
    allowed_count = int(allowed_mask.sum())
    blocked_count = int((~allowed_mask).sum())
    if expected_allowed_count is not None and allowed_count != expected_allowed_count:
        raise HardRowVisualReviewError(
            f"Phase 3 candidate export allowed row count mismatch: expected {expected_allowed_count}, got {allowed_count}"
        )
    if expected_blocked_count is not None and blocked_count != expected_blocked_count:
        raise HardRowVisualReviewError(
            f"Phase 3 candidate export blocked row count mismatch: expected {expected_blocked_count}, got {blocked_count}"
        )


def _optional_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text or text.lower() == "null":
        return None
    return int(text)


def _build_phase3_candidate_package_summary(
    *,
    input_file: Path,
    row_count: int,
    allowed: pd.DataFrame,
    blocked: pd.DataFrame,
    artifact_rows: pd.DataFrame,
    tiny_rows: pd.DataFrame,
    general_rows: pd.DataFrame,
    roi_blocked: pd.DataFrame,
    mislabel_blocked: pd.DataFrame,
    other_blocked: pd.DataFrame,
    output_paths: dict[str, Path],
) -> dict[str, object]:
    return {
        "input_file": str(input_file),
        "row_count": int(row_count),
        "phase3_use_allowed_true_count": int(len(allowed)),
        "phase3_use_allowed_false_count": int(len(blocked)),
        "allowed_outputs": {
            "phase3_allowed_hard_training_rows": int(len(allowed)),
            "phase3_artifact_focus_rows": int(len(artifact_rows)),
            "phase3_tiny_low_contrast_rows": int(len(tiny_rows)),
            "phase3_general_candidate_rows": int(len(general_rows)),
        },
        "blocked_outputs": {
            "blocked_roi_pipeline_bug_rows": int(len(roi_blocked)),
            "blocked_suspected_mislabel_rows": int(len(mislabel_blocked)),
            "blocked_other_high_risk_rows": int(len(other_blocked)),
        },
        "output_paths": {name: str(path) for name, path in output_paths.items()},
        "validation_checks_passed": {
            "csv_parses_cleanly": True,
            "row_count_matches": True,
            "allowed_count_matches": True,
            "blocked_count_matches": True,
        },
        "column_mapping_used": {},
        "missing_optional_columns": [],
        "safety": {
            "used_test_labels": False,
            "training_started": False,
            "submission_created": False,
            "leaderboard_tuning": False,
        },
    }


def _render_evidence_aware_contact_sheets(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    thumb_size: int,
    per_sheet: int,
) -> None:
    columns = 2
    rows = frame.sort_values("suspicion_rank").to_dict("records")
    for page_index in range(math.ceil(max(1, len(rows)) / per_sheet)):
        page_rows = rows[page_index * per_sheet : (page_index + 1) * per_sheet]
        output_path = output_dir / f"evidence_aware_sheet_{page_index + 1:03d}.jpg"
        canvas = Image.new("RGB", (columns * (thumb_size * 2 + 16), max(1, math.ceil(len(page_rows) / columns)) * (thumb_size * 2 + 60)), color=(245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(page_rows):
            x = (index % columns) * (thumb_size * 2 + 16)
            y = (index // columns) * (thumb_size * 2 + 60)
            preview_path = str(row.get("evidence_asset_original_roi_overlay", ""))
            preview = Image.open(preview_path).convert("RGB") if preview_path and Path(preview_path).exists() else Image.new("RGB", (thumb_size, thumb_size), color=(230, 230, 230))
            preview.thumbnail((thumb_size * 2, thumb_size * 2))
            canvas.paste(preview, (x, y))
            draw.text((x, y + thumb_size * 2 + 4), f"{row['image_id']} | {row['recommended_action']}", fill=(0, 0, 0))
        canvas.save(output_path, format="JPEG")


def _write_evidence_assets(
    *,
    image_id: str,
    image: Image.Image,
    roi_bbox: tuple[int, int, int, int],
    annotation_bbox: tuple[int, int, int, int] | None,
    asset_dirs: dict[str, Path],
) -> dict[str, str]:
    stem = Path(image_id).stem
    original_overlay = image.copy()
    annotation_overlay = image.copy()
    roi_crop = image.crop((roi_bbox[0], roi_bbox[1], roi_bbox[0] + roi_bbox[2], roi_bbox[1] + roi_bbox[3]))
    contrast_crop = ImageEnhance.Contrast(roi_crop).enhance(1.8)
    edge_crop = roi_crop.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")

    draw_roi = ImageDraw.Draw(original_overlay)
    draw_roi.rectangle(
        (
            roi_bbox[0],
            roi_bbox[1],
            roi_bbox[0] + roi_bbox[2],
            roi_bbox[1] + roi_bbox[3],
        ),
        outline=(255, 0, 0),
        width=3,
    )
    draw_annotation = ImageDraw.Draw(annotation_overlay)
    if annotation_bbox is not None:
        draw_annotation.rectangle(
            (
                annotation_bbox[0],
                annotation_bbox[1],
                annotation_bbox[0] + annotation_bbox[2],
                annotation_bbox[1] + annotation_bbox[3],
            ),
            outline=(0, 255, 0),
            width=3,
        )

    paths = {
        "original_roi_overlay": str(asset_dirs["original_roi_overlay"] / f"{stem}.png"),
        "annotation_overlay": str(asset_dirs["annotation_overlay"] / f"{stem}.png"),
        "roi_crop": str(asset_dirs["roi_crop"] / f"{stem}.png"),
        "contrast_roi_crop": str(asset_dirs["contrast_roi_crop"] / f"{stem}.png"),
        "edge_roi_crop": str(asset_dirs["edge_roi_crop"] / f"{stem}.png"),
    }
    original_overlay.save(paths["original_roi_overlay"], format="PNG")
    annotation_overlay.save(paths["annotation_overlay"], format="PNG")
    roi_crop.save(paths["roi_crop"], format="PNG")
    contrast_crop.save(paths["contrast_roi_crop"], format="PNG")
    edge_crop.save(paths["edge_roi_crop"], format="PNG")
    return paths


def _parse_bbox_text(text: str) -> tuple[int, int, int, int] | None:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 4:
        return None
    try:
        values = tuple(int(float(part)) for part in parts)
    except ValueError:
        return None
    return values


def _format_bbox(bbox: tuple[int, int, int, int] | None) -> str:
    if bbox is None:
        return ""
    return ",".join(str(int(value)) for value in bbox)


def _union_bboxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def _derive_roi_from_defects(
    *,
    boxes: list[tuple[int, int, int, int]],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    union = _union_bboxes(boxes)
    if union is None:
        return None
    x, y, width, height = union
    pad_x = max(8, width // 2)
    pad_y = max(8, height // 2)
    target_width = max(32, width + 2 * pad_x)
    target_height = max(32, height + 2 * pad_y)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = x1 + target_width
    y2 = y1 + target_height
    if image_width > 0:
        x2 = min(image_width, x2)
        x1 = max(0, x2 - target_width)
    if image_height > 0:
        y2 = min(image_height, y2)
        y1 = max(0, y2 - target_height)
    return int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))


def _crop_misses_defect_area(
    *,
    roi_bbox: tuple[int, int, int, int],
    annotation_bbox: tuple[int, int, int, int] | None,
) -> bool:
    if annotation_bbox is None:
        return False
    roi_x1, roi_y1, roi_w, roi_h = roi_bbox
    roi_x2 = roi_x1 + roi_w
    roi_y2 = roi_y1 + roi_h
    ann_x1, ann_y1, ann_w, ann_h = annotation_bbox
    ann_x2 = ann_x1 + ann_w
    ann_y2 = ann_y1 + ann_h
    return ann_x1 < roi_x1 or ann_y1 < roi_y1 or ann_x2 > roi_x2 or ann_y2 > roi_y2


def _classify_crop_quality(
    *,
    coverage: float,
    touches_border: bool,
    crop_too_small: bool,
    crop_missing_important_area: bool,
    contrast: float,
    blur_score: float,
    edge_density: float,
    evidence_cfg: dict[str, object],
) -> tuple[str, str]:
    if crop_too_small or crop_missing_important_area:
        return "likely_issue", "crop too small or misses important annotated area"
    if coverage < float(evidence_cfg["min_roi_coverage_ratio"]) or coverage > float(evidence_cfg["max_roi_coverage_ratio"]):
        return "possible_issue", "ROI coverage is outside expected bounds"
    if touches_border:
        return "possible_issue", "ROI touches image border"
    if blur_score < float(evidence_cfg["blur_laplacian_issue_threshold"]):
        return "possible_issue", "blur score is below threshold"
    if contrast < float(evidence_cfg["low_contrast_issue_threshold"]):
        return "possible_issue", "contrast is below threshold"
    if edge_density < float(evidence_cfg["edge_density_low_threshold"]):
        return "possible_issue", "edge density is below threshold"
    return "ok", "ROI crop quality looks usable"


def _laplacian_variance(grayscale: np.ndarray) -> float:
    if grayscale.size == 0:
        return 0.0
    padded = np.pad(grayscale, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(np.var(laplacian))


def _edge_density(grayscale: np.ndarray) -> float:
    if grayscale.size == 0:
        return 0.0
    gx = np.abs(np.diff(grayscale, axis=1))
    gy = np.abs(np.diff(grayscale, axis=0))
    gx = np.pad(gx, ((0, 0), (0, 1)))
    gy = np.pad(gy, ((0, 1), (0, 0)))
    magnitude = np.sqrt(gx**2 + gy**2)
    return float((magnitude > 12.0).mean())


def _build_image_info(image_ids: list[str], images_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for image_id in image_ids:
        image_path = images_root / image_id
        rows.append(
            {
                "image_id": image_id,
                "image_exists": image_path.exists(),
                "image_path": str(image_path),
            }
        )
    return pd.DataFrame(rows)


def _compute_label_quality(
    review: pd.DataFrame,
    *,
    config: dict[str, object],
    cleanlab_module: object | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    analysis = dict(config.get("analysis") or {})
    probability_correct = np.where(
        review["target"].astype(int).to_numpy() == 1,
        review["candidate_probability"].astype(float).to_numpy(),
        1.0 - review["candidate_probability"].astype(float).to_numpy(),
    )
    heuristic_scores = np.clip(1.0 - probability_correct, 0.0, 1.0)
    score_source = "heuristic_fallback"
    if bool(analysis.get("enable_cleanlab", True)) and cleanlab_module is not None:
        score_source = "cleanlab_available_heuristic_score"
    elif not bool(analysis.get("enable_cleanlab", True)):
        score_source = "disabled"
    review = review.copy()
    review["label_issue_score"] = heuristic_scores.round(6)
    summary = {
        "status": score_source,
        "row_count": int(len(review)),
        "score_summary": {
            "min": float(review["label_issue_score"].min()) if len(review) else 0.0,
            "max": float(review["label_issue_score"].max()) if len(review) else 0.0,
            "mean": float(review["label_issue_score"].mean()) if len(review) else 0.0,
        },
        "top_label_issue_rows": review.sort_values(["label_issue_score", "image_id"], ascending=[False, True])
        .head(10)[["image_id", "label_issue_score", "target", "candidate_probability"]]
        .to_dict("records"),
    }
    return summary, review


def _compute_embeddings(review: pd.DataFrame, *, images_root: Path) -> tuple[np.ndarray, list[str]]:
    feature_names = [
        "rgb_mean_r",
        "rgb_mean_g",
        "rgb_mean_b",
        "rgb_std_r",
        "rgb_std_g",
        "rgb_std_b",
        "gray_hist_0",
        "gray_hist_1",
        "gray_hist_2",
        "gray_hist_3",
        "gray_hist_4",
        "gray_hist_5",
        "gray_hist_6",
        "gray_hist_7",
        "aspect_ratio",
        "candidate_probability",
        "baseline_probability",
        "target",
    ]
    rows: list[list[float]] = []
    for item in review.to_dict("records"):
        image_path = images_root / str(item["image_id"])
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
            gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32) / 255.0
            hist, _ = np.histogram(gray, bins=8, range=(0.0, 1.0), density=True)
            means = array.mean(axis=(0, 1)).tolist()
            stds = array.std(axis=(0, 1)).tolist()
            aspect_ratio = float(image.width) / float(max(1, image.height))
        else:
            hist = np.zeros(8, dtype=float)
            means = [0.0, 0.0, 0.0]
            stds = [0.0, 0.0, 0.0]
            aspect_ratio = 0.0
        rows.append(
            [
                *means,
                *stds,
                *hist.tolist(),
                aspect_ratio,
                float(item["candidate_probability"]),
                float(item["baseline_probability"]),
                float(item["target"]),
            ]
        )
    return np.asarray(rows, dtype=float), feature_names


def _assign_clusters(review: pd.DataFrame, embedding_matrix: np.ndarray, *, config: dict[str, object]) -> pd.DataFrame:
    analysis = dict(config.get("analysis") or {})
    clustering = dict(config.get("clustering") or {})
    review = review.copy()
    if not bool(analysis.get("enable_clustering", True)) or len(review) == 0:
        review["cluster_id"] = 0
        review["cluster_description"] = "clustering disabled"
        return review
    cluster_count = max(1, min(int(clustering.get("n_clusters", 8)), len(review)))
    if len(review) == 1:
        labels = np.array([0], dtype=int)
    else:
        model = KMeans(n_clusters=cluster_count, random_state=int(clustering.get("random_state", 42)), n_init=10)
        labels = model.fit_predict(embedding_matrix)
    review["cluster_id"] = labels.astype(int)
    descriptions: dict[int, str] = {}
    for cluster_id, frame in review.groupby("cluster_id", sort=True):
        top_category = frame["primary_audit_category"].value_counts().index[0]
        top_type = frame["hard_example_type"].value_counts().index[0]
        top_bottle = frame["bottle_type"].replace("", "unknown bottle type").value_counts().index[0]
        descriptions[int(cluster_id)] = f"{top_category} | {top_type} | {top_bottle}"
    review["cluster_description"] = review["cluster_id"].map(descriptions)
    return review


def _assign_anomaly_scores(
    review: pd.DataFrame,
    embedding_matrix: np.ndarray,
    *,
    config: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame]:
    analysis = dict(config.get("analysis") or {})
    anomaly_cfg = dict(config.get("anomaly") or {})
    review = review.copy()
    if not bool(analysis.get("enable_anomaly_detection", True)) or len(review) == 0:
        review["anomaly_score"] = 0.0
    elif len(review) < 3:
        review["anomaly_score"] = np.linspace(0.0, 1.0, num=len(review), endpoint=False)
    else:
        contamination = float(anomaly_cfg.get("contamination", 0.08))
        contamination = min(max(contamination, 0.01), 0.49)
        model = IsolationForest(
            contamination=contamination,
            random_state=int(anomaly_cfg.get("random_state", 42)),
        )
        model.fit(embedding_matrix)
        raw_scores = -model.score_samples(embedding_matrix)
        review["anomaly_score"] = _normalize_scores(raw_scores)
    summary = {
        "method": anomaly_cfg.get("method", "isolation_forest"),
        "row_count": int(len(review)),
        "top_anomalies": review.sort_values(["anomaly_score", "image_id"], ascending=[False, True])
        .head(10)[["image_id", "anomaly_score", "cluster_id", "primary_audit_category"]]
        .to_dict("records"),
    }
    return summary, review


def _assign_suspicion_rank(review: pd.DataFrame) -> pd.DataFrame:
    review = review.copy()
    ambiguity_bonus = review["primary_audit_category"].eq("likely_ambiguous").astype(float) * 0.35
    evidence_bonus = (
        review["detector_evidence_status"].eq("missing").astype(float) * 0.05
        + review["crop_quality_status"].eq("present").astype(float) * 0.10
    )
    review["suspicion_score"] = (
        review["label_issue_score"].astype(float) * 0.55
        + review["anomaly_score"].astype(float) * 0.20
        + ambiguity_bonus
        + evidence_bonus
    )
    ranked = review.sort_values(
        ["suspicion_score", "label_issue_score", "image_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranked["suspicion_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def _build_triage_fields(review: pd.DataFrame, *, config: dict[str, object]) -> pd.DataFrame:
    triage_cfg = dict(config.get("triage") or {})
    thresholds = dict(triage_cfg.get("thresholds") or {})
    high_label_issue_score = float(thresholds.get("high_label_issue_score", 0.75))
    high_anomaly_score = float(thresholds.get("high_anomaly_score", 0.75))
    near_boundary_margin = float(thresholds.get("near_boundary_margin", 0.10))
    high_priority_score = float(thresholds.get("high_priority_score", 0.75))

    review = review.copy()
    review["prediction"] = review["candidate_prediction"].astype(int)
    review["probability"] = review["candidate_probability"].astype(float)
    review["prediction_margin"] = (review["candidate_probability"].astype(float) - 0.5).abs().round(6)
    review["cluster_row_count"] = review.groupby("cluster_id")["image_id"].transform("size").astype(int)
    review["cluster_label_issue_mean"] = review.groupby("cluster_id")["label_issue_score"].transform("mean")
    review["cluster_anomaly_mean"] = review.groupby("cluster_id")["anomaly_score"].transform("mean")
    review["cluster_category_count"] = review.groupby("cluster_id")["primary_audit_category"].transform("nunique").astype(int)
    review["cluster_target_count"] = review.groupby("cluster_id")["target"].transform("nunique").astype(int)
    review["combined_priority_score"] = (
        0.35 * review["label_issue_score"].astype(float)
        + 0.25 * review["anomaly_score"].astype(float)
        + 0.15 * (1.0 - review["prediction_margin"].clip(upper=1.0))
        + 0.15 * review["primary_audit_category"].eq("likely_ambiguous").astype(float)
        + 0.10 * review["hard_example_type"].astype(str).str.contains("uncertain").astype(float)
    ).round(6)

    recommended_action: list[str] = []
    reasons: list[str] = []
    priorities: list[str] = []
    for row in review.to_dict("records"):
        action, reason = _recommend_action_for_row(
            row,
            high_label_issue_score=high_label_issue_score,
            high_anomaly_score=high_anomaly_score,
            near_boundary_margin=near_boundary_margin,
        )
        priority = "high" if (
            action in {"suspected_mislabel", "needs_expert_review", "roi_pipeline_bug", "needs_detector_evidence"}
            or float(row["combined_priority_score"]) >= high_priority_score
        ) else "medium" if action in {"exclude_as_ambiguous", "rare_defect_cluster", "harmless_artifact_pattern"} else "low"
        recommended_action.append(action)
        reasons.append(reason)
        priorities.append(priority)

    review["recommended_action"] = recommended_action
    review["reason"] = reasons
    review["human_review_priority"] = priorities
    return review


def _recommend_action_for_row(
    row: dict[str, object],
    *,
    high_label_issue_score: float,
    high_anomaly_score: float,
    near_boundary_margin: float,
) -> tuple[str, str]:
    label_issue_score = float(row["label_issue_score"])
    anomaly_score = float(row["anomaly_score"])
    prediction_margin = float(row["prediction_margin"])
    primary_category = str(row["primary_audit_category"])
    hard_type = str(row["hard_example_type"])
    cluster_row_count = int(row["cluster_row_count"])
    cluster_category_count = int(row["cluster_category_count"])
    crop_quality_status = str(row.get("crop_quality_status", ""))
    detector_status = str(row.get("detector_evidence_status", ""))
    target = int(row["target"])

    if crop_quality_status == "present":
        return "roi_pipeline_bug", "crop_quality_status=present indicates a recoverable ROI or preprocessing issue"
    if label_issue_score >= high_label_issue_score and prediction_margin >= near_boundary_margin:
        return "suspected_mislabel", "high label_issue_score with strong confidence disagreement against the target"
    if primary_category == "likely_ambiguous" and prediction_margin <= near_boundary_margin:
        return "exclude_as_ambiguous", "ambiguous category with near-boundary confidence should be excluded from hard training"
    if anomaly_score >= high_anomaly_score and cluster_row_count <= 3:
        return "rare_defect_cluster", "high anomaly score in a very small cluster suggests a rare but plausible defect pattern"
    if target == 1 and detector_status == "missing" and ("hard_positive" in hard_type or prediction_margin <= near_boundary_margin):
        return "needs_detector_evidence", "positive hard case lacks detector or localized evidence and should wait for evidence support"
    if target == 0 and prediction_margin >= 0.20 and "hard_negative" in hard_type:
        return "harmless_artifact_pattern", "confident hard negative suggests a recurring harmless artifact confusion pattern"
    if primary_category == "likely_correct_but_hard" and label_issue_score < 0.45 and anomaly_score < 0.60 and cluster_category_count == 1:
        return "safe_for_hard_training", "consistent hard-example cluster with low label-noise and anomaly risk"
    if cluster_category_count > 1 or prediction_margin <= near_boundary_margin:
        return "needs_expert_review", "mixed cluster or near-boundary evidence is not safe for automated inclusion"
    return "needs_expert_review", "conservative fallback when the evidence is insufficient for a safer automated action"


def _build_manual_review_template(review: pd.DataFrame, *, config: dict[str, object]) -> pd.DataFrame:
    review_cfg = dict(config.get("review") or {})
    default_failure_mode = str(
        review_cfg.get("default_final_failure_mode", "unclear_needs_annotation_or_expert_review")
    ).strip()
    manual_review = pd.DataFrame(
        {
            "image_id": review["image_id"],
            "target": review["target"],
            "baseline_probability": review["baseline_probability"],
            "baseline_prediction": review["baseline_prediction"],
            "candidate_probability": review["candidate_probability"],
            "candidate_prediction": review["candidate_prediction"],
            "bottle_type": review["bottle_type"],
            "primary_audit_category": review["primary_audit_category"],
            "hard_example_type": review["hard_example_type"],
            "suspicion_rank": review["suspicion_rank"],
            "label_issue_score": review["label_issue_score"].round(6),
            "anomaly_score": review["anomaly_score"].round(6),
            "cluster_id": review["cluster_id"],
            "cluster_description": review["cluster_description"],
            "final_failure_mode": [default_failure_mode] * len(review),
            "label_quality_flag": [""] * len(review),
            "roi_preprocessing_flag": [""] * len(review),
            "detector_evidence_flag": [""] * len(review),
            "harmless_artifact_flag": [""] * len(review),
            "tiny_or_low_contrast_defect_flag": [""] * len(review),
            "reviewer_notes": [""] * len(review),
            "reviewer_id": [str(review_cfg.get("reviewer_id_default", ""))] * len(review),
            "review_timestamp": [""] * len(review),
        }
    )
    return manual_review[REQUIRED_MANUAL_REVIEW_COLUMNS].copy()


def _build_review_manifest(review: pd.DataFrame, manual_review: pd.DataFrame) -> pd.DataFrame:
    manifest = manual_review.merge(
        review[
            [
                "image_id",
                "audit_rationale",
                "annotation_categories",
                "annotation_bbox",
                "image_path",
                "image_exists",
                "visual_asset_path",
                "contact_sheet_path",
            ]
        ],
        on="image_id",
        how="left",
    )
    return manifest.sort_values("suspicion_rank").reset_index(drop=True)


def _build_clustering_summary(review: pd.DataFrame, embedding_feature_names: list[str]) -> dict[str, object]:
    clusters = []
    for cluster_id, frame in review.groupby("cluster_id", sort=True):
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "row_count": int(len(frame)),
                "description": str(frame["cluster_description"].iloc[0]),
                "primary_audit_categories": frame["primary_audit_category"].value_counts().to_dict(),
                "hard_example_types": frame["hard_example_type"].value_counts().to_dict(),
            }
        )
    return {
        "row_count": int(len(review)),
        "cluster_count": int(review["cluster_id"].nunique()) if len(review) else 0,
        "embedding_feature_names": embedding_feature_names,
        "clusters": clusters,
    }


def _build_visual_review_summary(
    *,
    review: pd.DataFrame,
    resolved: dict[str, Path],
    cleanlab_module: object | None,
    fiftyone_module: object | None,
    missing_images: list[dict[str, object]],
    contact_sheet_paths: dict[int, Path],
) -> dict[str, object]:
    return {
        "used_test_labels": False,
        "trained_model": False,
        "submission_created": False,
        "leaderboard_tuning_used": False,
        "row_count": int(len(review)),
        "missing_image_count": int(len(missing_images)),
        "cleanlab_integration": {
            "implemented": True,
            "available": cleanlab_module is not None,
        },
        "fiftyone_integration": {
            "implemented": True,
            "available": fiftyone_module is not None,
        },
        "contact_sheet_count": int(len(contact_sheet_paths)),
        "run_manifest": {
            "hard_row_audit_csv": str(resolved["hard_row_audit_csv"]),
            "train_images_dir": str(resolved["train_images_dir"]),
            "train_csv": str(resolved["train_csv"]),
            "bottletypes_csv": str(resolved["bottletypes_csv"]),
            "train_annotations_json": str(resolved["train_annotations_json"] or ""),
            "output_root": str(resolved["output_root"]),
        },
    }


def _build_triage_outputs(
    review: pd.DataFrame,
    *,
    asset_paths: dict[str, Path],
    reports_dir: Path,
    cluster_galleries_dir: Path,
    top_suspicious_gallery_dir: Path,
    thumb_size: int,
    config: dict[str, object],
    embedding_feature_names: list[str],
) -> dict[str, Path]:
    triage_cfg = dict(config.get("triage") or {})
    reps_per_cluster = max(1, int(triage_cfg.get("cluster_representatives_per_cluster", 8)))
    top_n_suspicious = max(1, int(triage_cfg.get("top_n_suspicious", 50)))
    top_n_label_issues = max(1, int(triage_cfg.get("top_n_label_issues", 50)))
    top_n_anomalies = max(1, int(triage_cfg.get("top_n_anomalies", 50)))

    cluster_representatives = (
        review.sort_values(["cluster_id", "combined_priority_score", "suspicion_rank"], ascending=[True, False, True])
        .groupby("cluster_id", as_index=False, group_keys=False)
        .head(reps_per_cluster)
        .copy()
    )
    cluster_representatives["nearest_neighbors"] = _build_nearest_neighbor_lookup(review)

    top_suspicious_rows = review.sort_values(["combined_priority_score", "suspicion_rank"], ascending=[False, True]).head(top_n_suspicious).copy()
    top_label_issue_rows = review.sort_values(["label_issue_score", "suspicion_rank"], ascending=[False, True]).head(top_n_label_issues).copy()
    top_anomaly_rows = review.sort_values(["anomaly_score", "suspicion_rank"], ascending=[False, True]).head(top_n_anomalies).copy()
    action_plan = review.sort_values("suspicion_rank").copy()

    cluster_representatives_path = reports_dir / "cluster_representatives.csv"
    top_suspicious_rows_path = reports_dir / "top_suspicious_rows.csv"
    top_label_issue_rows_path = reports_dir / "top_label_issue_rows.csv"
    top_anomaly_rows_path = reports_dir / "top_anomaly_rows.csv"
    final_training_action_plan_path = reports_dir / "final_training_action_plan.csv"
    cluster_triage_summary_path = reports_dir / "cluster_triage_summary.json"

    cluster_representatives.to_csv(cluster_representatives_path, index=False)
    top_suspicious_rows.to_csv(top_suspicious_rows_path, index=False)
    top_label_issue_rows.to_csv(top_label_issue_rows_path, index=False)
    top_anomaly_rows.to_csv(top_anomaly_rows_path, index=False)
    action_plan[REQUIRED_ACTION_PLAN_COLUMNS].to_csv(final_training_action_plan_path, index=False)

    _render_gallery_collection(
        cluster_representatives,
        asset_paths=asset_paths,
        output_dir=cluster_galleries_dir,
        thumb_size=thumb_size,
        group_column="cluster_id",
        prefix="cluster",
    )
    _render_gallery_collection(
        top_suspicious_rows.assign(gallery_group="top_suspicious"),
        asset_paths=asset_paths,
        output_dir=top_suspicious_gallery_dir,
        thumb_size=thumb_size,
        group_column="gallery_group",
        prefix="top_suspicious",
    )

    cluster_triage_summary = _build_cluster_triage_summary(
        review=review,
        cluster_representatives=cluster_representatives,
        top_suspicious_rows=top_suspicious_rows,
        top_label_issue_rows=top_label_issue_rows,
        top_anomaly_rows=top_anomaly_rows,
        cluster_galleries_dir=cluster_galleries_dir,
        top_suspicious_gallery_dir=top_suspicious_gallery_dir,
        embedding_feature_names=embedding_feature_names,
    )
    cluster_triage_summary_path.write_text(json.dumps(cluster_triage_summary, indent=2), encoding="utf-8")

    return {
        "cluster_representatives": cluster_representatives_path,
        "top_suspicious_rows": top_suspicious_rows_path,
        "top_label_issue_rows": top_label_issue_rows_path,
        "top_anomaly_rows": top_anomaly_rows_path,
        "final_training_action_plan": final_training_action_plan_path,
        "cluster_triage_summary": cluster_triage_summary_path,
    }


def _build_nearest_neighbor_lookup(review: pd.DataFrame) -> pd.Series:
    by_cluster = review.groupby("cluster_id")["image_id"].apply(list).to_dict()
    return review.apply(
        lambda row: ",".join([image_id for image_id in by_cluster.get(row["cluster_id"], []) if image_id != row["image_id"]][:3]),
        axis=1,
    )


def _render_gallery_collection(
    frame: pd.DataFrame,
    *,
    asset_paths: dict[str, Path],
    output_dir: Path,
    thumb_size: int,
    group_column: str,
    prefix: str,
) -> None:
    for group_value, subset in frame.groupby(group_column, sort=True):
        output_path = output_dir / f"{prefix}_{group_value}.png"
        page_rows = subset.sort_values(["combined_priority_score", "suspicion_rank"], ascending=[False, True]).to_dict("records")
        _render_contact_sheet(
            page_rows=page_rows,
            asset_paths=asset_paths,
            output_path=output_path,
            thumb_size=thumb_size,
        )


def _build_cluster_triage_summary(
    *,
    review: pd.DataFrame,
    cluster_representatives: pd.DataFrame,
    top_suspicious_rows: pd.DataFrame,
    top_label_issue_rows: pd.DataFrame,
    top_anomaly_rows: pd.DataFrame,
    cluster_galleries_dir: Path,
    top_suspicious_gallery_dir: Path,
    embedding_feature_names: list[str],
) -> dict[str, object]:
    return {
        "number_of_clusters": int(review["cluster_id"].nunique()) if len(review) else 0,
        "number_of_cluster_representatives": int(len(cluster_representatives)),
        "top_suspicious_rows_count": int(len(top_suspicious_rows)),
        "top_label_issue_rows_count": int(len(top_label_issue_rows)),
        "top_anomaly_rows_count": int(len(top_anomaly_rows)),
        "recommended_action_counts": review["recommended_action"].value_counts().sort_index().to_dict(),
        "cluster_gallery_count": int(len(list(cluster_galleries_dir.glob("*.png")))),
        "top_suspicious_gallery_created": any(top_suspicious_gallery_dir.glob("*.png")),
        "missing_embeddings_fallback_used": all(name.startswith("fallback_") for name in embedding_feature_names),
        "missing_optional_evidence": {
            "detector_missing_rows": int(review["detector_evidence_status"].eq("missing").sum()),
            "crop_quality_missing_rows": int(review["crop_quality_status"].eq("missing").sum()),
        },
        "mixed_clusters": sorted(
            int(cluster_id)
            for cluster_id, frame in review.groupby("cluster_id")
            if frame["primary_audit_category"].nunique() > 1
        ),
        "uncertain_clusters": sorted(
            int(cluster_id)
            for cluster_id, frame in review.groupby("cluster_id")
            if frame["hard_example_type"].astype(str).str.contains("uncertain").mean() >= 0.5
        ),
        "label_issue_dominated_clusters": sorted(
            int(cluster_id)
            for cluster_id, frame in review.groupby("cluster_id")
            if float(frame["label_issue_score"].mean()) >= 0.75
        ),
        "anomaly_dominated_clusters": sorted(
            int(cluster_id)
            for cluster_id, frame in review.groupby("cluster_id")
            if float(frame["anomaly_score"].mean()) >= 0.75
        ),
        "safe_for_hard_training_clusters": sorted(
            int(cluster_id)
            for cluster_id, frame in review.groupby("cluster_id")
            if frame["recommended_action"].eq("safe_for_hard_training").mean() >= 0.5
        ),
    }


def _validate_manus_review(
    *,
    original_manual: pd.DataFrame,
    filled_manual: pd.DataFrame,
    expected_rows: int,
    allowed_flag_values: set[str],
) -> dict[str, object]:
    original_manual = original_manual.copy()
    filled_manual = filled_manual.copy()
    original_manual["image_id"] = original_manual["image_id"].astype(str)
    filled_manual["image_id"] = filled_manual["image_id"].astype(str)

    image_id_set_matches = set(original_manual["image_id"]) == set(filled_manual["image_id"])
    invalid_failure_modes = sorted(
        {
            str(value).strip()
            for value in filled_manual.get("final_failure_mode", pd.Series(dtype=str)).tolist()
            if str(value).strip() not in ALLOWED_FINAL_FAILURE_MODES
        }
    )
    invalid_flag_rows: list[str] = []
    for column in MANUS_FLAG_COLUMNS:
        if column not in filled_manual.columns:
            invalid_flag_rows.extend(filled_manual["image_id"].astype(str).tolist())
            continue
        invalid_mask = ~filled_manual[column].astype(str).isin(allowed_flag_values)
        invalid_flag_rows.extend(filled_manual.loc[invalid_mask, "image_id"].astype(str).tolist())

    metadata_mismatches: list[dict[str, object]] = []
    original_lookup = original_manual.set_index("image_id")
    for _, row in filled_manual.iterrows():
        image_id = str(row["image_id"])
        if image_id not in original_lookup.index:
            metadata_mismatches.append({"image_id": image_id, "column": "image_id", "expected": "present", "actual": "missing"})
            continue
        original_row = original_lookup.loc[image_id]
        if isinstance(original_row, pd.DataFrame):
            original_row = original_row.iloc[0]
        for column in MANUS_REQUIRED_METADATA_COLUMNS:
            if column not in filled_manual.columns or column not in original_lookup.columns:
                continue
            expected = str(original_row[column])
            actual = str(row[column])
            if expected != actual:
                metadata_mismatches.append(
                    {"image_id": image_id, "column": column, "expected": expected, "actual": actual}
                )

    suspicious_image_ids = sorted(
        set(item["image_id"] for item in metadata_mismatches)
        | set(filled_manual.loc[filled_manual["final_failure_mode"].astype(str).isin(invalid_failure_modes), "image_id"].astype(str).tolist())
        | set(invalid_flag_rows)
    )
    row_count_matches = len(filled_manual) == expected_rows
    valid = row_count_matches and image_id_set_matches and not invalid_failure_modes and not invalid_flag_rows and not metadata_mismatches
    return {
        "valid": valid,
        "expected_rows": int(expected_rows),
        "actual_rows": int(len(filled_manual)),
        "row_count_matches_expected": row_count_matches,
        "image_id_set_matches": image_id_set_matches,
        "invalid_final_failure_mode_count": int(len(invalid_failure_modes)),
        "invalid_final_failure_modes": invalid_failure_modes,
        "invalid_flag_value_count": int(len(invalid_flag_rows)),
        "invalid_flag_row_image_ids": sorted(set(invalid_flag_rows)),
        "metadata_mismatch_count": int(len(metadata_mismatches)),
        "metadata_mismatches": metadata_mismatches[:100],
        "suspicious_image_ids": suspicious_image_ids,
    }


def _apply_manus_action_logic(
    merged: pd.DataFrame,
    *,
    high_label_issue_threshold: float,
    high_anomaly_threshold: float,
    validation: dict[str, object],
) -> pd.DataFrame:
    merged = merged.copy()
    suspicious_ids = set(validation["suspicious_image_ids"])
    recommended_actions: list[str] = []
    priorities: list[str] = []
    reasons: list[str] = []
    phase3_allowed: list[bool] = []
    excluded_rows: list[bool] = []
    engineering_rows: list[bool] = []
    verification_rows: list[bool] = []

    for row in merged.to_dict("records"):
        image_id = str(row["image_id"])
        final_failure_mode = str(row["final_failure_mode"]).strip()
        label_flag = str(row.get("label_quality_flag", "")).strip()
        roi_flag = str(row.get("roi_preprocessing_flag", "")).strip()
        detector_flag = str(row.get("detector_evidence_flag", "")).strip()
        label_issue_score = float(row["label_issue_score"])
        anomaly_score = float(row["anomaly_score"])
        high_label_issue = label_issue_score > high_label_issue_threshold
        high_anomaly = anomaly_score > high_anomaly_threshold

        action = _map_final_failure_mode_to_action(final_failure_mode)
        reason_parts = [f"final_failure_mode={final_failure_mode}"]
        if image_id in suspicious_ids:
            reason_parts.append("validation flagged Manus row for manual verification")
        if high_label_issue:
            reason_parts.append(f"high_label_issue_score={label_issue_score:.3f}")
        if high_anomaly:
            reason_parts.append(f"high_anomaly_score={anomaly_score:.3f}")

        priority = "medium"
        allow_phase3 = False
        is_excluded = False
        is_engineering = False
        needs_verification = False

        if final_failure_mode == "likely_mislabeled":
            priority = "high"
            allow_phase3 = False
            is_excluded = True
            needs_verification = True
        elif final_failure_mode in {"ambiguous_needs_expert_review", "unclear_needs_annotation_or_expert_review"}:
            priority = "high"
            allow_phase3 = False
            is_excluded = True
            needs_verification = True
        elif final_failure_mode in {"roi_or_crop_problem", "detector_or_annotation_evidence_needed"}:
            priority = "high"
            allow_phase3 = False
            is_engineering = True
            needs_verification = True
        elif final_failure_mode in {"correct_but_visually_hard", "harmless_artifact_confusion", "tiny_or_low_contrast_defect"}:
            allow_phase3 = label_flag == "label_ok" and not high_label_issue and not high_anomaly and image_id not in suspicious_ids
            priority = "high" if not allow_phase3 else "medium"
            needs_verification = not allow_phase3
            if not allow_phase3:
                is_excluded = True
        elif final_failure_mode == "anomaly_or_outlier_case":
            priority = "high"
            allow_phase3 = False
            needs_verification = True
        else:
            priority = "high"
            allow_phase3 = False
            needs_verification = True

        if roi_flag in {"yes", "possible_crop_issue", "likely_crop_issue", "needs_review"}:
            is_engineering = True or is_engineering
        if detector_flag in {"needs_review", "not_needed"} and final_failure_mode == "detector_or_annotation_evidence_needed":
            is_engineering = True
        if image_id in suspicious_ids or high_label_issue or high_anomaly:
            needs_verification = True
            priority = "high"

        recommended_actions.append(action)
        priorities.append(priority)
        reasons.append("; ".join(reason_parts))
        phase3_allowed.append(bool(allow_phase3))
        excluded_rows.append(bool(is_excluded or (high_label_issue and not allow_phase3) or (high_anomaly and not allow_phase3)))
        engineering_rows.append(bool(is_engineering))
        verification_rows.append(bool(
            needs_verification
            or final_failure_mode in {
                "ambiguous_needs_expert_review",
                "unclear_needs_annotation_or_expert_review",
                "likely_mislabeled",
                "roi_or_crop_problem",
                "anomaly_or_outlier_case",
            }
        ))

    merged["recommended_action"] = recommended_actions
    merged["human_review_priority"] = priorities
    merged["phase3_use_allowed"] = phase3_allowed
    merged["reason"] = reasons
    merged["is_excluded_row"] = excluded_rows
    merged["is_engineering_fix_row"] = engineering_rows
    merged["is_human_verification_row"] = verification_rows
    return merged


def _map_final_failure_mode_to_action(final_failure_mode: str) -> str:
    mapping = {
        "correct_but_visually_hard": "safe_for_hard_training",
        "harmless_artifact_confusion": "safe_for_hard_training_with_artifact_focus",
        "tiny_or_low_contrast_defect": "safe_for_hard_training_with_high_resolution_focus",
        "ambiguous_needs_expert_review": "exclude_as_ambiguous_until_verified",
        "unclear_needs_annotation_or_expert_review": "needs_expert_review",
        "likely_mislabeled": "suspected_mislabel_exclude_from_training",
        "roi_or_crop_problem": "roi_pipeline_bug",
        "detector_or_annotation_evidence_needed": "needs_detector_evidence",
        "anomaly_or_outlier_case": "rare_defect_cluster_or_outlier_review",
    }
    return mapping.get(final_failure_mode, "needs_expert_review")


def _build_manus_review_counts(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action, group in merged.groupby("recommended_action", sort=True):
        rows.append(
            {
                "recommended_action": str(action),
                "row_count": int(len(group)),
                "phase3_use_allowed_count": int(group["phase3_use_allowed"].astype(bool).sum()),
                "human_review_priority_high_count": int(group["human_review_priority"].eq("high").sum()),
            }
        )
    return pd.DataFrame(rows)


def _build_manus_final_summary(
    *,
    merged: pd.DataFrame,
    validation: dict[str, object],
    cluster_representatives: pd.DataFrame,
    top_suspicious: pd.DataFrame,
    top_label_issues: pd.DataFrame,
    top_anomalies: pd.DataFrame,
) -> dict[str, object]:
    return {
        "validation_result": bool(validation["valid"]),
        "action_counts": merged["recommended_action"].value_counts().sort_index().to_dict(),
        "phase3_training_candidate_count": int(merged["phase3_use_allowed"].astype(bool).sum()),
        "excluded_ambiguous_or_noisy_count": int(merged["is_excluded_row"].astype(bool).sum()),
        "suspected_mislabeled_count": int(merged["final_failure_mode"].eq("likely_mislabeled").sum()),
        "engineering_fix_count": int(merged["is_engineering_fix_row"].astype(bool).sum()),
        "human_verification_count": int(merged["is_human_verification_row"].astype(bool).sum()),
        "cluster_representative_count": int(len(cluster_representatives)),
        "top_suspicious_rows_count": int(len(top_suspicious)),
        "top_label_issue_rows_count": int(len(top_label_issues)),
        "top_anomaly_rows_count": int(len(top_anomalies)),
        "used_test_labels": False,
        "trained_model": False,
        "submission_created": False,
    }


def _build_contact_sheets(
    review: pd.DataFrame,
    *,
    images_root: Path,
    output_dir: Path,
    config: dict[str, object],
) -> tuple[dict[int, Path], list[dict[str, object]], dict[str, Path]]:
    visuals = dict(config.get("visuals") or {})
    thumb_size = max(64, int(visuals.get("thumbnail_size", 224)))
    page_size = max(1, int(visuals.get("max_images_per_contact_sheet", 40)))
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    missing_images: list[dict[str, object]] = []
    asset_paths: dict[str, Path] = {}
    rows = review.sort_values("suspicion_rank").to_dict("records")
    for row in rows:
        image_path = images_root / str(row["image_id"])
        if not image_path.exists():
            missing_images.append({"image_id": row["image_id"], "expected_path": str(image_path)})
        asset_paths[str(row["image_id"])] = _build_visual_asset(
            row=row,
            image_path=image_path,
            output_dir=assets_dir,
            thumb_size=thumb_size,
        )

    contact_sheet_paths: dict[int, Path] = {}
    for page_index in range(math.ceil(max(1, len(rows)) / page_size)):
        page_rows = rows[page_index * page_size : (page_index + 1) * page_size]
        sheet_path = output_dir / f"hard_row_review_sheet_{page_index + 1:03d}.png"
        _render_contact_sheet(
            page_rows=page_rows,
            asset_paths=asset_paths,
            output_path=sheet_path,
            thumb_size=thumb_size,
        )
        contact_sheet_paths[page_index] = sheet_path

    review["sheet_index"] = review["suspicion_rank"].map(lambda rank: (int(rank) - 1) // page_size)
    return contact_sheet_paths, missing_images, asset_paths


def _build_visual_asset(*, row: dict[str, object], image_path: Path, output_dir: Path, thumb_size: int) -> Path:
    output_path = output_dir / f"{Path(str(row['image_id'])).stem}_views.png"
    views = _render_views(image_path=image_path, annotation_bbox=str(row.get("annotation_bbox", "")), thumb_size=thumb_size)
    titles = ["original", "roi", "contrast", "edge"]
    canvas = Image.new("RGB", (thumb_size * 2, thumb_size * 2 + 36), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(zip(titles, views)):
        col = index % 2
        row_index = index // 2
        x = col * thumb_size
        y = row_index * thumb_size
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline=(120, 120, 120))
        draw.text((x + 4, y + thumb_size - 16), title, fill=(0, 0, 0))
    footer_y = thumb_size * 2 + 2
    draw.text((4, footer_y), str(row["image_id"])[:40], fill=(0, 0, 0))
    draw.text((4, footer_y + 14), f"rank={row['suspicion_rank']} cluster={row['cluster_id']}", fill=(0, 0, 0))
    canvas.save(output_path, format="PNG")
    return output_path


def _render_views(*, image_path: Path, annotation_bbox: str, thumb_size: int) -> list[Image.Image]:
    if not image_path.exists():
        placeholder = Image.new("RGB", (thumb_size, thumb_size), color=(235, 235, 235))
        draw = ImageDraw.Draw(placeholder)
        draw.text((8, thumb_size // 2 - 8), "missing", fill=(60, 60, 60))
        return [placeholder.copy() for _ in range(4)]
    original = Image.open(image_path).convert("RGB")
    roi = _crop_roi(original, annotation_bbox)
    contrast = ImageEnhance.Contrast(original).enhance(1.8)
    edge = original.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
    return [_fit_tile(image, thumb_size) for image in [original, roi, contrast, edge]]


def _crop_roi(image: Image.Image, annotation_bbox: str) -> Image.Image:
    parts = [part.strip() for part in str(annotation_bbox).split(",") if part.strip()]
    if len(parts) != 4:
        return image.copy()
    try:
        x, y, width, height = [int(float(value)) for value in parts]
    except ValueError:
        return image.copy()
    left = max(0, x)
    top = max(0, y)
    right = min(image.width, x + max(1, width))
    bottom = min(image.height, y + max(1, height))
    if right <= left or bottom <= top:
        return image.copy()
    return image.crop((left, top, right, bottom))


def _fit_tile(image: Image.Image, thumb_size: int) -> Image.Image:
    tile = Image.new("RGB", (thumb_size, thumb_size), color=(248, 248, 248))
    copy = image.copy()
    copy.thumbnail((thumb_size - 4, thumb_size - 4))
    x = (thumb_size - copy.width) // 2
    y = (thumb_size - copy.height) // 2
    tile.paste(copy, (x, y))
    return tile


def _render_contact_sheet(
    *,
    page_rows: list[dict[str, object]],
    asset_paths: dict[str, Path],
    output_path: Path,
    thumb_size: int,
) -> None:
    columns = 2
    asset_width = thumb_size * 2
    asset_height = thumb_size * 2 + 36
    gutter = 12
    header = 36
    rows = max(1, math.ceil(len(page_rows) / columns))
    canvas = Image.new(
        "RGB",
        (columns * asset_width + (columns + 1) * gutter, header + rows * (asset_height + gutter)),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 10), "Hard-Row Visual Review", fill=(0, 0, 0))
    for index, row in enumerate(page_rows):
        asset = Image.open(asset_paths[str(row["image_id"])]).convert("RGB")
        col = index % columns
        row_index = index // columns
        x = gutter + col * (asset_width + gutter)
        y = header + row_index * (asset_height + gutter)
        canvas.paste(asset, (x, y))
        draw.text((x, y - 12), f"#{row['suspicion_rank']} {row['primary_audit_category']}", fill=(0, 0, 0))
    canvas.save(output_path, format="PNG")


def _write_fiftyone_export(
    *,
    review_manifest: pd.DataFrame,
    output_dir: Path,
    fiftyone_module: object | None,
) -> None:
    review_manifest.to_csv(output_dir / "hard_row_visual_review_dataset.csv", index=False)
    readme = [
        "# Hard-Row Visual Review FiftyOne Export",
        "",
        "This directory contains a CSV export for optional FiftyOne loading.",
        f"FiftyOne available at runtime: {'yes' if fiftyone_module is not None else 'no'}",
        "",
        "Suggested workflow:",
        "1. Load `hard_row_visual_review_dataset.csv` into a pandas DataFrame.",
        "2. Use the `image_path` column as the media path when building a FiftyOne dataset.",
        "3. Keep this workflow analysis-only; do not train or submit from this step.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")


def _find_column(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> str | None:
    lookup = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise HardRowVisualReviewError("Missing required column; expected one of: " + ", ".join(candidates))
    return None


def _normalize_image_id(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        raise HardRowVisualReviewError("image_id cannot be empty")
    return Path(text).name


def _normalize_scores(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    minimum = float(array.min())
    maximum = float(array.max())
    if math.isclose(minimum, maximum):
        return np.zeros_like(array, dtype=float)
    return (array - minimum) / (maximum - minimum)


def _import_optional_dependency(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the hard-row visual review workflow.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--merge-manus-review", action="store_true")
    parser.add_argument("--complete-evidence", action="store_true")
    parser.add_argument("--export-phase3-candidates", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.merge_manus_review:
            outputs = merge_manus_review(args.config)
        elif args.complete_evidence:
            outputs = complete_hard_row_evidence(args.config)
        elif args.export_phase3_candidates:
            outputs = export_phase3_candidates(args.config)
        else:
            outputs = run_hard_row_visual_review(args.config)
    except HardRowVisualReviewError as exc:
        print(f"Hard-row visual review failed: {exc}", file=sys.stderr)
        return 2
    if args.merge_manus_review:
        print(f"Hard-row visual review Manus merge complete: {outputs['final_hard_row_action_plan']}")
    elif args.complete_evidence:
        print(f"Hard-row evidence completion complete: {outputs['final_training_action_plan_evidence_aware']}")
    elif args.export_phase3_candidates:
        print(f"Phase 3 hard-row candidate package complete: {outputs['phase3_candidate_package_summary']}")
    else:
        print(f"Hard-row visual review complete: {outputs['manual_review_template']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
