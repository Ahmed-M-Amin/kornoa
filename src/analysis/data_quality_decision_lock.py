"""Spec 017 automated decision-lock workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, UnidentifiedImageError


DEFAULT_CONFIG_PATH = Path("configs/data_quality_decision_lock.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/data_quality_decision_lock")
REQUIRED_MASTER_COLUMNS = ("image_id", "target")
REQUIRED_CONFIG_PATHS = (
    "spec016_master_path",
    "spec016_cleaned_manifest_path",
    "spec016_manual_review_required_path",
    "spec016_review_template_path",
    "spec016_summary_path",
    "spec016_exclusion_reasons_path",
    "spec014_allowed_hard_rows_path",
    "spec014_blocked_roi_pipeline_bug_rows_path",
    "spec014_blocked_suspected_mislabel_rows_path",
    "spec014_blocked_other_high_risk_rows_path",
)
FORBIDDEN_SAFETY_FLAGS = (
    "allow_test_labels",
    "public_leaderboard_input",
    "generate_submission",
    "train_model",
    "apply_relabels",
)
DECISION_COLUMNS = [
    "image_id",
    "target",
    "source_bucket",
    "decision",
    "decision_reason",
    "decision_confidence",
    "evidence_sources",
    "blocked_flag",
    "blocked_reason",
    "duplicate_group_id",
    "duplicate_conflict_flag",
    "roi_quality_group",
    "roi_severity",
    "prediction_probability",
    "prediction_risk_score",
    "prediction_risk_level",
    "embedding_status",
    "cluster_id",
    "cluster_outlier_score",
    "cleanlab_label_quality_score",
    "label_issue_score",
    "manual_review_required",
    "review_priority",
    "image_exists",
    "label_exists",
    "risk_group_id",
]
SUMMARY_PATH = Path("reports/decision_lock_summary.json")


class DecisionLockError(ValueError):
    """Raised when Spec 017 inputs or safety gates are invalid."""


@dataclass(frozen=True)
class DecisionLockConfig:
    spec016_master_path: Path
    spec016_cleaned_manifest_path: Path
    spec016_manual_review_required_path: Path
    spec016_review_template_path: Path
    spec016_summary_path: Path
    spec016_exclusion_reasons_path: Path
    spec014_allowed_hard_rows_path: Path
    spec014_blocked_roi_pipeline_bug_rows_path: Path
    spec014_blocked_suspected_mislabel_rows_path: Path
    spec014_blocked_other_high_risk_rows_path: Path
    oof_predictions_path: Path | None
    v2b_predictions_path: Path | None
    image_embeddings_path: Path | None
    cleanlab_scores_path: Path | None
    cluster_assignments_path: Path | None
    train_csv_path: Path
    train_images_dir: Path
    output_root: Path
    label_issue_threshold: float
    prediction_high_risk_threshold: float
    prediction_critical_risk_threshold: float
    contact_sheet_limit: int
    adjudication_queue_limit: int


def _optional_path(value: Any, base_dir: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _required_path(value: Any, base_dir: Path, field_name: str) -> Path:
    path = _optional_path(value, base_dir)
    if path is None:
        raise DecisionLockError(f"Required config path is missing: {field_name}")
    return path


def _discover_repo_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / ".git").exists():
            return candidate.resolve()
    return config_path.parent.resolve()


def _validate_safety(payload: dict[str, Any]) -> None:
    safety = payload.get("safety", {})
    if not isinstance(safety, dict):
        raise DecisionLockError("safety config must be a mapping")
    for field in FORBIDDEN_SAFETY_FLAGS:
        if bool(safety.get(field, False)):
            raise DecisionLockError(f"Safety flag must remain false for Spec 017: {field}")


def load_config(config_path: str | Path) -> DecisionLockConfig:
    path = Path(config_path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise DecisionLockError(f"Decision-lock config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise DecisionLockError(f"Decision-lock config is not valid YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise DecisionLockError("Decision-lock config must be a mapping")

    _validate_safety(payload)
    base_dir = _discover_repo_root(path.resolve())
    inputs = payload.get("inputs", {})
    optional = payload.get("optional_evidence", {})
    dataset = payload.get("dataset", {})
    output = payload.get("output", {})
    quality = payload.get("quality", {})
    if not isinstance(inputs, dict) or not isinstance(dataset, dict):
        raise DecisionLockError("inputs and dataset sections must be mappings")

    for field_name in REQUIRED_CONFIG_PATHS:
        if field_name not in inputs:
            raise DecisionLockError(f"Missing required config field inputs.{field_name}")

    return DecisionLockConfig(
        spec016_master_path=_required_path(inputs.get("spec016_master_path"), base_dir, "inputs.spec016_master_path"),
        spec016_cleaned_manifest_path=_required_path(
            inputs.get("spec016_cleaned_manifest_path"),
            base_dir,
            "inputs.spec016_cleaned_manifest_path",
        ),
        spec016_manual_review_required_path=_required_path(
            inputs.get("spec016_manual_review_required_path"),
            base_dir,
            "inputs.spec016_manual_review_required_path",
        ),
        spec016_review_template_path=_required_path(
            inputs.get("spec016_review_template_path"),
            base_dir,
            "inputs.spec016_review_template_path",
        ),
        spec016_summary_path=_required_path(inputs.get("spec016_summary_path"), base_dir, "inputs.spec016_summary_path"),
        spec016_exclusion_reasons_path=_required_path(
            inputs.get("spec016_exclusion_reasons_path"),
            base_dir,
            "inputs.spec016_exclusion_reasons_path",
        ),
        spec014_allowed_hard_rows_path=_required_path(
            inputs.get("spec014_allowed_hard_rows_path"),
            base_dir,
            "inputs.spec014_allowed_hard_rows_path",
        ),
        spec014_blocked_roi_pipeline_bug_rows_path=_required_path(
            inputs.get("spec014_blocked_roi_pipeline_bug_rows_path"),
            base_dir,
            "inputs.spec014_blocked_roi_pipeline_bug_rows_path",
        ),
        spec014_blocked_suspected_mislabel_rows_path=_required_path(
            inputs.get("spec014_blocked_suspected_mislabel_rows_path"),
            base_dir,
            "inputs.spec014_blocked_suspected_mislabel_rows_path",
        ),
        spec014_blocked_other_high_risk_rows_path=_required_path(
            inputs.get("spec014_blocked_other_high_risk_rows_path"),
            base_dir,
            "inputs.spec014_blocked_other_high_risk_rows_path",
        ),
        oof_predictions_path=_optional_path(optional.get("oof_predictions_path"), base_dir),
        v2b_predictions_path=_optional_path(optional.get("v2b_predictions_path"), base_dir),
        image_embeddings_path=_optional_path(optional.get("image_embeddings_path"), base_dir),
        cleanlab_scores_path=_optional_path(optional.get("cleanlab_scores_path"), base_dir),
        cluster_assignments_path=_optional_path(optional.get("cluster_assignments_path"), base_dir),
        train_csv_path=_required_path(dataset.get("train_csv_path"), base_dir, "dataset.train_csv_path"),
        train_images_dir=_required_path(dataset.get("train_images_dir"), base_dir, "dataset.train_images_dir"),
        output_root=_optional_path(output.get("analysis_root"), base_dir) or DEFAULT_OUTPUT_ROOT.resolve(),
        label_issue_threshold=float(quality.get("label_issue_threshold", 0.80)),
        prediction_high_risk_threshold=float(quality.get("prediction_high_risk_threshold", 0.75)),
        prediction_critical_risk_threshold=float(quality.get("prediction_critical_risk_threshold", 0.90)),
        contact_sheet_limit=int(quality.get("contact_sheet_limit", 64)),
        adjudication_queue_limit=int(quality.get("adjudication_queue_limit", 800)),
    )


def _as_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _normalize_text(value: Any, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value).strip()


def _normalize_probability_frame(frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    data = frame.copy()
    probability_col = next(
        (
            column
            for column in ("probability", "target_probability", "oof_probability", "v2b_probability")
            if column in data.columns
        ),
        None,
    )
    if "image_id" not in data.columns or probability_col is None:
        raise DecisionLockError(f"Optional prediction evidence for {column_name} requires image_id and probability")
    result = data[["image_id", probability_col]].copy()
    result["image_id"] = result["image_id"].astype(str)
    result = result.rename(columns={probability_col: column_name})
    result[column_name] = pd.to_numeric(result[column_name], errors="coerce")
    return result.drop_duplicates("image_id")


def _load_required_csv(path: Path, label: str, required_columns: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise DecisionLockError(f"Required input missing: {label}: {path}")
    frame = pd.read_csv(path)
    if required_columns:
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise DecisionLockError(f"{label} missing required columns: {missing}")
    if "image_id" in frame.columns:
        frame = frame.copy()
        frame["image_id"] = frame["image_id"].astype(str)
    return frame


def _load_optional_csv(path: Path | None, warnings: list[str], label: str) -> pd.DataFrame | None:
    if path is None:
        warnings.append(f"{label} not configured")
        return None
    if not path.exists():
        warnings.append(f"{label} missing: {path}")
        return None
    frame = pd.read_csv(path)
    if "image_id" in frame.columns:
        frame = frame.copy()
        frame["image_id"] = frame["image_id"].astype(str)
    return frame


def _ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    reports = output_root / "reports"
    contacts = output_root / "contact_sheets"
    for path in (output_root, reports, contacts):
        path.mkdir(parents=True, exist_ok=True)
    return {"root": output_root, "reports": reports, "contact_sheets": contacts}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _source_config_hash(config: DecisionLockConfig) -> str:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    return json.dumps(payload, sort_keys=True)


def _roi_severity(flag: str) -> str:
    mapping = {
        "ok": "none",
        "": "none",
        "missing_image": "critical",
        "missing_or_unreadable_image": "critical",
        "too_dark": "high",
        "too_bright": "high",
        "small_image": "high",
        "low_contrast_or_weak_texture": "medium",
    }
    return mapping.get(flag, "medium")


def _risk_level(score: float, high_threshold: float, critical_threshold: float) -> str:
    if score >= critical_threshold:
        return "critical"
    if score >= high_threshold:
        return "high"
    if score >= 0.55:
        return "medium"
    if score >= 0.35:
        return "low"
    return "none"


def _choose_probability(row: pd.Series) -> float | None:
    for column in ("oof_probability", "v2b_probability", "oof_probability_optional", "v2b_probability_optional"):
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value):
            return float(value)
    return None


def _required_input_inventory(config: DecisionLockConfig) -> dict[str, Path]:
    return {
        "spec016_master": config.spec016_master_path,
        "spec016_cleaned_manifest": config.spec016_cleaned_manifest_path,
        "spec016_manual_review_required": config.spec016_manual_review_required_path,
        "spec016_review_template": config.spec016_review_template_path,
        "spec016_summary": config.spec016_summary_path,
        "spec016_exclusion_reasons": config.spec016_exclusion_reasons_path,
        "spec014_allowed_hard_rows": config.spec014_allowed_hard_rows_path,
        "spec014_blocked_roi_pipeline_bug_rows": config.spec014_blocked_roi_pipeline_bug_rows_path,
        "spec014_blocked_suspected_mislabel_rows": config.spec014_blocked_suspected_mislabel_rows_path,
        "spec014_blocked_other_high_risk_rows": config.spec014_blocked_other_high_risk_rows_path,
        "train_csv": config.train_csv_path,
        "train_images_dir": config.train_images_dir,
    }


def _optional_input_inventory(config: DecisionLockConfig) -> dict[str, Path | None]:
    return {
        "oof_predictions": config.oof_predictions_path,
        "v2b_predictions": config.v2b_predictions_path,
        "image_embeddings": config.image_embeddings_path,
        "cleanlab_scores": config.cleanlab_scores_path,
        "cluster_assignments": config.cluster_assignments_path,
    }


def _build_blocked_map(config: DecisionLockConfig) -> tuple[dict[str, str], set[str], list[str]]:
    warnings: list[str] = []
    blocked_map: dict[str, str] = {}
    blocked_union: set[str] = set()
    sources = [
        (config.spec014_blocked_roi_pipeline_bug_rows_path, "spec014_roi_pipeline_bug"),
        (config.spec014_blocked_suspected_mislabel_rows_path, "spec014_suspected_mislabel"),
        (config.spec014_blocked_other_high_risk_rows_path, "spec014_other_high_risk"),
    ]
    for path, label in sources:
        frame = _load_required_csv(path, label, ("image_id",))
        ids = set(frame["image_id"].astype(str))
        overlap = blocked_union & ids
        if overlap:
            warnings.append(f"Blocked sets overlap for {label}: {len(overlap)} image_ids")
        blocked_union |= ids
        for image_id in ids:
            blocked_map[image_id] = label
    return blocked_map, blocked_union, warnings


def _apply_optional_evidence(master: pd.DataFrame, config: DecisionLockConfig, warnings: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    result = master.copy()
    statuses: dict[str, str] = {}

    oof = _load_optional_csv(config.oof_predictions_path, warnings, "OOF predictions")
    if oof is not None:
        normalized = _normalize_probability_frame(oof, "oof_probability_optional")
        result = result.merge(normalized, on="image_id", how="left")
        statuses["prediction_status"] = "partial" if len(normalized) < len(result) else "available"
    else:
        statuses["prediction_status"] = "missing"
        result["oof_probability_optional"] = pd.NA

    v2b = _load_optional_csv(config.v2b_predictions_path, warnings, "V2B predictions")
    if v2b is not None:
        normalized = _normalize_probability_frame(v2b, "v2b_probability_optional")
        result = result.merge(normalized, on="image_id", how="left")
        if statuses["prediction_status"] == "available" and len(normalized) < len(result):
            statuses["prediction_status"] = "partial"
        elif statuses["prediction_status"] == "missing":
            statuses["prediction_status"] = "partial" if len(normalized) < len(result) else "available"
    else:
        result["v2b_probability_optional"] = pd.NA

    embeddings = _load_optional_csv(config.image_embeddings_path, warnings, "Image embeddings")
    if embeddings is not None:
        statuses["embedding_status"] = "partial" if len(embeddings) < len(result) else "available"
        allowed_columns = [
            column
            for column in ("image_id", "duplicate_group_id", "near_duplicate_score", "cluster_id", "cluster_outlier_score")
            if column in embeddings.columns
        ]
        if len(allowed_columns) > 1:
            result = result.merge(embeddings[allowed_columns].drop_duplicates("image_id"), on="image_id", how="left", suffixes=("", "_embedding"))
    else:
        statuses["embedding_status"] = "missing"

    cleanlab = _load_optional_csv(config.cleanlab_scores_path, warnings, "Cleanlab scores")
    if cleanlab is not None and "image_id" in cleanlab.columns:
        statuses["cleanlab_status"] = "partial" if len(cleanlab) < len(result) else "available"
        score_column = next((column for column in cleanlab.columns if column != "image_id"), None)
        if score_column is not None:
            merged = cleanlab[["image_id", score_column]].rename(columns={score_column: "cleanlab_label_quality_score"})
            result = result.merge(merged.drop_duplicates("image_id"), on="image_id", how="left")
    else:
        statuses["cleanlab_status"] = "missing"
        result["cleanlab_label_quality_score"] = pd.NA

    clusters = _load_optional_csv(config.cluster_assignments_path, warnings, "Cluster assignments")
    if clusters is not None and "image_id" in clusters.columns:
        status = "partial" if len(clusters) < len(result) else "available"
        for column in ("cluster_id", "cluster_outlier_score"):
            if column in clusters.columns and column not in result.columns:
                result = result.merge(clusters[["image_id", column]].drop_duplicates("image_id"), on="image_id", how="left")
        if statuses["embedding_status"] == "missing":
            statuses["embedding_status"] = status
    return result, statuses


def _decision_evidence(row: pd.Series) -> str:
    parts: list[str] = []
    if _normalize_text(row.get("blocked_reason")):
        parts.append("spec014_blocked")
    if _as_bool(row.get("duplicate_conflict_flag")):
        parts.append("duplicate_conflict")
    if _normalize_text(row.get("roi_quality_group")) not in {"", "ok"}:
        parts.append("roi_audit")
    if pd.notna(row.get("prediction_probability")):
        parts.append("prediction_risk")
    if pd.notna(row.get("cleanlab_label_quality_score")) or pd.notna(row.get("label_issue_score")):
        parts.append("label_issue")
    if _normalize_text(row.get("cluster_id")):
        parts.append("cluster")
    return "|".join(dict.fromkeys(parts)) or "baseline_audit"


def _build_decision_frame(
    master: pd.DataFrame,
    cleaned_manifest: pd.DataFrame,
    manual_review: pd.DataFrame,
    train: pd.DataFrame,
    config: DecisionLockConfig,
    blocked_map: dict[str, str],
    blocked_union: set[str],
) -> pd.DataFrame:
    frame = master.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    train["image_id"] = train["image_id"].astype(str)

    train_ids = set(train["image_id"].astype(str))
    cleaned_ids = set(cleaned_manifest["image_id"].astype(str)) if "image_id" in cleaned_manifest.columns else set()
    manual_ids = set(manual_review["image_id"].astype(str)) if "image_id" in manual_review.columns else set()
    allowed_frame = _load_required_csv(config.spec014_allowed_hard_rows_path, "Spec 014 allowed hard rows", ("image_id",))
    allowed_ids = set(allowed_frame["image_id"].astype(str))

    frame["label_exists"] = frame["image_id"].isin(train_ids)
    frame["image_exists"] = frame["image_id"].map(lambda image_id: (config.train_images_dir / image_id).exists())
    frame["blocked_reason"] = frame["image_id"].map(blocked_map).fillna(frame.get("blocked_reason", pd.Series(index=frame.index, dtype=object)).fillna(""))
    frame["blocked_flag"] = frame["image_id"].isin(blocked_union)
    frame["manual_review_required"] = frame["image_id"].isin(manual_ids) | frame.get("data_quality_bucket", "").astype(str).eq("manual_review_required")
    frame["source_bucket"] = np.where(
        frame["blocked_flag"],
        "blocked",
        np.where(
            frame["image_id"].isin(allowed_ids) | frame.get("data_quality_bucket", "").astype(str).eq("hard_valid_train"),
            "hard_valid_train",
            np.where(
                frame["image_id"].isin(cleaned_ids) | frame.get("data_quality_bucket", "").astype(str).eq("clean_train"),
                "clean_train",
                np.where(frame["manual_review_required"], "manual_review_required", frame.get("data_quality_bucket", "").fillna("unclassified")),
            ),
        ),
    )
    frame["roi_quality_group"] = frame.get("roi_quality_flag", pd.Series(index=frame.index, dtype=object)).fillna("ok").astype(str)
    frame["roi_severity"] = frame["roi_quality_group"].map(_roi_severity)
    frame["duplicate_group_id"] = frame.get("duplicate_group_id", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str)
    frame["duplicate_conflict_flag"] = frame.get("duplicate_conflict_flag", pd.Series(index=frame.index, dtype=object)).map(_as_bool)
    frame["cluster_id"] = frame.get("cluster_id", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str)
    frame["cluster_outlier_score"] = pd.to_numeric(frame.get("cluster_outlier_score"), errors="coerce")
    frame["label_issue_score"] = pd.to_numeric(frame.get("label_issue_score"), errors="coerce").fillna(0.0)
    frame["cleanlab_label_quality_score"] = pd.to_numeric(frame.get("cleanlab_label_quality_score"), errors="coerce")

    frame["prediction_probability"] = frame.apply(_choose_probability, axis=1)
    frame["prediction_probability"] = pd.to_numeric(frame["prediction_probability"], errors="coerce")
    target_prob = np.where(frame["target"].fillna(0).astype(int).eq(1), frame["prediction_probability"], 1.0 - frame["prediction_probability"])
    frame["prediction_risk_score"] = pd.Series(1.0 - target_prob, index=frame.index, dtype="float64")
    frame.loc[frame["prediction_probability"].isna(), "prediction_risk_score"] = 0.0
    frame["prediction_risk_level"] = frame["prediction_risk_score"].map(
        lambda value: _risk_level(float(value), config.prediction_high_risk_threshold, config.prediction_critical_risk_threshold)
    )

    def classify(row: pd.Series) -> pd.Series:
        blocked_reason = _normalize_text(row["blocked_reason"])
        prediction_risk = _normalize_text(row["prediction_risk_level"])
        roi_severity = _normalize_text(row["roi_severity"])
        label_issue = float(row["label_issue_score"])
        cleanlab_score = float(row["cleanlab_label_quality_score"]) if pd.notna(row["cleanlab_label_quality_score"]) else label_issue
        corroborating_roi_signals = int(label_issue >= config.label_issue_threshold) + int(
            prediction_risk in {"high", "critical"}
        )
        decision = "defer"
        reason = "unclassified_defer"
        confidence = "low"
        if _as_bool(row["blocked_flag"]):
            decision = "auto_exclude"
            reason = blocked_reason or "spec014_blocked"
            confidence = "high"
        elif _as_bool(row["duplicate_conflict_flag"]):
            decision = "auto_exclude"
            reason = "duplicate_conflict"
            confidence = "high"
        elif not _as_bool(row["label_exists"]) or not _as_bool(row["image_exists"]):
            decision = "defer"
            reason = "missing_training_artifact"
            confidence = "high"
        elif row["source_bucket"] in {"clean_train", "hard_valid_train"}:
            if roi_severity in {"high", "critical"} and corroborating_roi_signals >= 1:
                decision = "auto_exclude"
                reason = "severe_roi_with_corroboration"
                confidence = "high"
            elif prediction_risk == "critical" or cleanlab_score >= config.label_issue_threshold:
                decision = "needs_adjudication"
                reason = "unexpected_high_risk_clean_candidate"
                confidence = "medium"
            else:
                decision = "auto_keep"
                reason = "clean_or_allowed_hard_candidate"
                confidence = "high" if prediction_risk in {"none", "low"} else "medium"
        elif row["source_bucket"] == "manual_review_required":
            exclusion_reason = _normalize_text(row.get("exclusion_reason"))
            if (
                exclusion_reason == "roi_or_crop_problem_requires_review"
                and roi_severity in {"low", "medium"}
                and label_issue < config.label_issue_threshold
                and prediction_risk not in {"high", "critical"}
            ):
                decision = "auto_keep"
                reason = "roi_review_downgraded_after_recalibration"
                confidence = "medium"
            elif (
                exclusion_reason in {
                    "high_label_issue_score",
                    "spec014_roi_pipeline_bug_requires_review",
                    "spec014_other_high_risk_requires_review",
                    "duplicate_label_conflict",
                }
                or label_issue >= config.label_issue_threshold
                or cleanlab_score >= config.label_issue_threshold
                or prediction_risk in {"high", "critical"}
            ):
                decision = "needs_adjudication"
                reason = exclusion_reason or "high_risk_manual_review_candidate"
                confidence = "low"
            else:
                decision = "defer"
                reason = exclusion_reason or "manual_review_without_keep_signal"
                confidence = "low"
        elif _normalize_text(row.get("exclusion_reason")) == "missing_or_unreadable_image":
            decision = "defer"
            reason = "missing_or_unreadable_image"
            confidence = "high"
        else:
            decision = "auto_exclude"
            reason = _normalize_text(row.get("exclusion_reason")) or "exclude_from_training_source"
            confidence = "medium"
        return pd.Series({"decision": decision, "decision_reason": reason, "decision_confidence": confidence})

    decisions = frame.apply(classify, axis=1)
    frame = pd.concat([frame, decisions], axis=1)
    frame["evidence_sources"] = frame.apply(_decision_evidence, axis=1)
    frame["embedding_status"] = np.where(
        frame["cluster_id"].ne("") | frame["duplicate_group_id"].ne(""),
        "available",
        np.where(
            frame.get("image_embeddings_path", pd.Series(index=frame.index, dtype=object)).notna(),
            "partial",
            "missing",
        ),
    )
    frame["risk_group_id"] = np.where(
        frame["duplicate_group_id"].ne(""),
        frame["duplicate_group_id"],
        np.where(frame["roi_quality_group"].ne("ok"), "roi:" + frame["roi_quality_group"], "risk:" + frame["decision_reason"]),
    )
    return frame


def _roi_group_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    roi_rows = decisions.loc[decisions["roi_quality_group"].fillna("ok").astype(str).ne("ok")].copy()
    if roi_rows.empty:
        return pd.DataFrame(
            columns=[
                "roi_quality_group",
                "row_count",
                "target_0_count",
                "target_1_count",
                "representative_image_ids",
                "group_risk_score",
                "proposed_group_action",
            ]
        )
    grouped = []
    for group_name, group in roi_rows.groupby("roi_quality_group", dropna=False):
        action_counts = group["decision"].value_counts()
        if action_counts.get("auto_keep", 0) >= max(action_counts.get("auto_exclude", 0), action_counts.get("needs_adjudication", 0)):
            action = "keep_candidate"
        elif action_counts.get("auto_exclude", 0) >= action_counts.get("needs_adjudication", 0):
            action = "exclude_candidate"
        else:
            action = "adjudicate_candidate"
        representatives = "|".join(group.sort_values(["review_priority", "label_issue_score"], ascending=[True, False])["image_id"].head(6))
        grouped.append(
            {
                "roi_quality_group": group_name,
                "row_count": int(len(group)),
                "target_0_count": int(group["target"].fillna(-1).eq(0).sum()),
                "target_1_count": int(group["target"].fillna(-1).eq(1).sum()),
                "representative_image_ids": representatives,
                "group_risk_score": float(group["prediction_risk_score"].fillna(0.0).mean() + group["label_issue_score"].fillna(0.0).mean()),
                "proposed_group_action": action,
            }
        )
    return pd.DataFrame(grouped).sort_values(["group_risk_score", "row_count"], ascending=[False, False])


def _adjudication_queue(decisions: pd.DataFrame, limit: int) -> pd.DataFrame:
    queue = decisions.loc[decisions["decision"].eq("needs_adjudication")].copy()
    if queue.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "target",
                "risk_rank",
                "risk_bucket",
                "risk_group_id",
                "primary_issue",
                "secondary_issue",
                "suggested_decision",
                "suggested_reason",
                "evidence_sources",
                "reviewer_decision",
                "reviewer_confidence",
                "reviewer_notes",
            ]
        )
    severity_weight = queue["roi_severity"].map({"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}).fillna(0)
    queue["risk_score"] = queue["label_issue_score"].fillna(0.0) * 100.0 + queue["prediction_risk_score"].fillna(0.0) * 100.0 + severity_weight * 10.0
    queue["risk_bucket"] = np.where(
        queue["prediction_risk_level"].isin(["high", "critical"]),
        "prediction_or_label_risk",
        np.where(queue["roi_quality_group"].ne("ok"), "roi_recalibration", "other"),
    )
    queue["primary_issue"] = queue["decision_reason"]
    queue["secondary_issue"] = np.where(queue["roi_quality_group"].ne("ok"), queue["roi_quality_group"], queue["prediction_risk_level"])
    queue["suggested_decision"] = "defer"
    queue["suggested_reason"] = queue["decision_reason"]
    queue = queue.sort_values(["risk_score", "risk_group_id", "image_id"], ascending=[False, True, True]).head(limit).copy()
    queue["risk_rank"] = np.arange(1, len(queue) + 1)
    queue["reviewer_decision"] = ""
    queue["reviewer_confidence"] = ""
    queue["reviewer_notes"] = ""
    columns = [
        "image_id",
        "target",
        "risk_rank",
        "risk_bucket",
        "risk_group_id",
        "primary_issue",
        "secondary_issue",
        "suggested_decision",
        "suggested_reason",
        "evidence_sources",
        "reviewer_decision",
        "reviewer_confidence",
        "reviewer_notes",
    ]
    return queue[columns]


def _ensure_contact_sheet(
    rows: pd.DataFrame,
    images_dir: Path,
    output_path: Path,
    max_rows: int = 64,
    thumb_size: tuple[int, int] = (160, 160),
    columns: int = 4,
) -> None:
    selected = rows.head(max_rows).copy()
    if selected.empty:
        sheet = Image.new("RGB", (thumb_size[0], thumb_size[1]), color=(245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), "No rows", fill=(0, 0, 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path)
        return
    caption_height = 56
    cell_width = thumb_size[0]
    cell_height = thumb_size[1] + caption_height
    grid_rows = int(np.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (columns * cell_width, grid_rows * cell_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for index, (_, row) in enumerate(selected.iterrows()):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        image_path = images_dir / str(row["image_id"])
        try:
            with Image.open(image_path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail(thumb_size)
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            thumb = Image.new("RGB", thumb_size, color=(230, 230, 230))
        sheet.paste(thumb, (x, y))
        caption = f"{row.get('image_id', '')}\n{row.get('decision_reason', '')}"
        draw.text((x + 4, y + thumb_size[1] + 4), caption[:120], fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _contact_sheet_status(contact_paths: Sequence[Path]) -> str:
    return "available" if all(path.exists() for path in contact_paths) else "missing"


def _validate_outputs(
    approved_manifest: pd.DataFrame,
    decisions: pd.DataFrame,
    blocked_union: set[str],
) -> tuple[int, int, int]:
    blocked_overlap_count = int(approved_manifest["image_id"].isin(blocked_union).sum())
    duplicate_conflict_approved_count = int(
        decisions.loc[
            decisions["decision"].eq("auto_keep") & decisions["duplicate_conflict_flag"].astype(bool),
            "image_id",
        ].nunique()
    )
    unresolved_review_approved_count = int(
        decisions.loc[
            decisions["decision"].eq("auto_keep") & decisions["manual_review_required"].astype(bool) & decisions["decision_reason"].ne("roi_review_downgraded_after_recalibration"),
            "image_id",
        ].nunique()
    )
    if blocked_overlap_count or duplicate_conflict_approved_count or unresolved_review_approved_count:
        raise DecisionLockError(
            "Safety gate failed for approved manifest: "
            f"blocked_overlap={blocked_overlap_count}, "
            f"duplicate_conflict_approved={duplicate_conflict_approved_count}, "
            f"unresolved_review_approved={unresolved_review_approved_count}"
        )
    return blocked_overlap_count, duplicate_conflict_approved_count, unresolved_review_approved_count


def run_decision_lock(config: DecisionLockConfig) -> dict[str, Any]:
    for label, path in _required_input_inventory(config).items():
        if not path.exists():
            raise DecisionLockError(f"Required input missing: {label}: {path}")

    warnings: list[str] = []
    output_dirs = _ensure_output_dirs(config.output_root)

    master = _load_required_csv(config.spec016_master_path, "Spec 016 master", REQUIRED_MASTER_COLUMNS)
    cleaned_manifest = _load_required_csv(config.spec016_cleaned_manifest_path, "Spec 016 cleaned manifest", ("image_id",))
    manual_review = _load_required_csv(config.spec016_manual_review_required_path, "Spec 016 manual review", ("image_id",))
    _load_required_csv(config.spec016_review_template_path, "Spec 016 review template", ("image_id",))
    json.loads(config.spec016_summary_path.read_text(encoding="utf-8"))
    _load_required_csv(config.spec016_exclusion_reasons_path, "Spec 016 exclusion reasons", ("exclusion_reason", "count"))
    train = _load_required_csv(config.train_csv_path, "Train labels", ("image_id", "target"))

    blocked_map, blocked_union, blocked_warnings = _build_blocked_map(config)
    warnings.extend(blocked_warnings)
    allowed = _load_required_csv(config.spec014_allowed_hard_rows_path, "Spec 014 allowed hard rows", ("image_id",))
    allowed_overlap_count = int(allowed["image_id"].isin(blocked_union).sum())
    if allowed_overlap_count:
        raise DecisionLockError(f"Allowed Spec 014 rows overlap blocked rows: {allowed_overlap_count}")

    master, evidence_statuses = _apply_optional_evidence(master, config, warnings)
    decisions = _build_decision_frame(master, cleaned_manifest, manual_review, train, config, blocked_map, blocked_union)

    for column in DECISION_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = pd.NA
    decisions = decisions[DECISION_COLUMNS].copy()

    approved_manifest = decisions.loc[decisions["decision"].eq("auto_keep"), [
        "image_id",
        "target",
        "decision",
        "decision_reason",
        "decision_confidence",
    ]].copy()
    auto_keep = decisions.loc[decisions["decision"].eq("auto_keep")].copy()
    auto_exclude = decisions.loc[decisions["decision"].eq("auto_exclude")].copy()
    needs_adjudication = decisions.loc[decisions["decision"].eq("needs_adjudication")].copy()
    deferred = decisions.loc[decisions["decision"].eq("defer")].copy()
    roi_summary = _roi_group_summary(decisions)
    adjudication_queue = _adjudication_queue(decisions, config.adjudication_queue_limit)
    rule_audit = decisions[[
        "image_id",
        "decision",
        "decision_reason",
        "decision_confidence",
        "evidence_sources",
        "source_bucket",
        "risk_group_id",
    ]].copy()

    _write_csv(config.output_root / "decision_master.csv", decisions)
    _write_csv(config.output_root / "approved_cleaned_training_manifest.csv", approved_manifest)
    _write_csv(config.output_root / "auto_keep_rows.csv", auto_keep)
    _write_csv(config.output_root / "auto_exclude_rows.csv", auto_exclude)
    _write_csv(config.output_root / "needs_adjudication_rows.csv", needs_adjudication)
    _write_csv(config.output_root / "deferred_uncertain_rows.csv", deferred)
    _write_csv(config.output_root / "adjudication_queue.csv", adjudication_queue)
    _write_csv(output_dirs["reports"] / "roi_recalibration_summary.csv", roi_summary)
    _write_csv(output_dirs["reports"] / "decision_rule_audit.csv", rule_audit)

    top_adjudication = needs_adjudication.sort_values(["label_issue_score", "prediction_risk_score"], ascending=[False, False])
    top_auto_exclude = auto_exclude.sort_values(["label_issue_score", "prediction_risk_score"], ascending=[False, False])
    roi_representatives = decisions.loc[decisions["roi_quality_group"].ne("ok")].sort_values(
        ["prediction_risk_score", "label_issue_score"],
        ascending=[False, False],
    )
    adjudication_sheet = output_dirs["contact_sheets"] / "needs_adjudication_top_risk.jpg"
    auto_exclude_sheet = output_dirs["contact_sheets"] / "auto_exclude_representatives.jpg"
    roi_sheet = output_dirs["contact_sheets"] / "roi_group_representatives.jpg"
    _ensure_contact_sheet(top_adjudication, config.train_images_dir, adjudication_sheet, max_rows=config.contact_sheet_limit)
    _ensure_contact_sheet(top_auto_exclude, config.train_images_dir, auto_exclude_sheet, max_rows=config.contact_sheet_limit)
    _ensure_contact_sheet(roi_representatives, config.train_images_dir, roi_sheet, max_rows=config.contact_sheet_limit)

    evidence_inventory = {
        "spec016_inputs_status": "available",
        "spec014_inputs_status": "available",
        "embedding_status": evidence_statuses.get("embedding_status", "missing"),
        "prediction_status": evidence_statuses.get("prediction_status", "missing"),
        "cleanlab_status": evidence_statuses.get("cleanlab_status", "missing"),
        "contact_sheet_status": _contact_sheet_status([adjudication_sheet, auto_exclude_sheet, roi_sheet]),
        "warnings": warnings,
    }
    _write_json(output_dirs["reports"] / "evidence_inventory.json", evidence_inventory)

    blocked_overlap_count, duplicate_conflict_approved_count, unresolved_review_approved_count = _validate_outputs(
        approved_manifest,
        decisions,
        blocked_union,
    )
    missing_approved_image_count = int(auto_keep["image_exists"].eq(False).sum())
    if missing_approved_image_count:
        raise DecisionLockError(f"Approved rows without images: {missing_approved_image_count}")

    summary = {
        "total_rows": int(len(decisions)),
        "auto_keep_count": int(len(auto_keep)),
        "auto_exclude_count": int(len(auto_exclude)),
        "needs_adjudication_count": int(len(needs_adjudication)),
        "defer_count": int(len(deferred)),
        "approved_manifest_count": int(len(approved_manifest)),
        "blocked_overlap_count": blocked_overlap_count,
        "duplicate_conflict_approved_count": duplicate_conflict_approved_count,
        "unresolved_review_approved_count": unresolved_review_approved_count,
        "missing_approved_image_count": missing_approved_image_count,
        "allowed_blocked_overlap_count": allowed_overlap_count,
        "original_labels_unchanged": True,
        "no_training_started": True,
        "no_submission_created": True,
        "no_test_labels_used": True,
        "no_leaderboard_tuning": True,
        "warnings": warnings,
        "source_config_hash": _source_config_hash(config),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    _write_json(output_dirs["reports"] / "decision_lock_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spec 017 decision-lock workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Build locked training-manifest decisions")
    run_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "run":
        summary = run_decision_lock(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
