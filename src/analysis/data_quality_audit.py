"""Controlled data quality audit workflow for cleaned training manifests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError

try:
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional fast path
    cv2 = None


DEFAULT_CONFIG_PATH = Path("configs/data_quality_audit.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/data_quality_audit")
REQUIRED_TRAIN_COLUMNS = ("image_id", "target")
FORBIDDEN_SAFETY_FLAGS = (
    "allow_test_labels",
    "public_leaderboard_input",
    "generate_submission",
    "train_model",
    "apply_relabels",
)
MASTER_COLUMNS = [
    "image_id",
    "target",
    "bottle_type",
    "source_split",
    "v2b_probability",
    "v2b_prediction",
    "v2b_correct",
    "v2b_confidence",
    "v2b_loss_or_risk_score",
    "decision_boundary_distance",
    "prediction_source",
    "oof_probability",
    "oof_prediction",
    "oof_correct",
    "oof_confidence",
    "oof_loss_or_risk_score",
    "label_quality_reliability",
    "hard_row_status",
    "hard_example_type",
    "spec014_final_failure_mode",
    "spec014_recommended_action",
    "spec014_phase3_use_allowed",
    "blocked_reason",
    "roi_quality_flag",
    "crop_quality_score",
    "annotation_evidence_flag",
    "image_width",
    "image_height",
    "image_mean_brightness",
    "image_contrast",
    "edge_strength",
    "background_heavy_score",
    "cluster_id",
    "anomaly_score",
    "label_issue_score",
    "duplicate_group_id",
    "near_duplicate_score",
    "duplicate_conflict_flag",
    "manual_review_status",
    "data_quality_bucket",
    "exclude_from_training",
    "exclusion_reason",
    "review_priority",
    "evidence_summary",
    "audit_run_id",
    "created_at",
    "source_config_hash",
]


class DataQualityAuditError(ValueError):
    """Raised when the data quality audit inputs are invalid."""


@dataclass(frozen=True)
class DataQualityConfig:
    train_csv_path: Path
    train_images_dir: Path
    output_root: Path
    v2b_validation_predictions_path: Path | None
    oof_train_predictions_path: Path | None
    image_embeddings_path: Path | None
    bottletypes_path: Path | None
    train_annotations_path: Path | None
    hard_row_quality_audit_path: Path | None
    hard_row_review_manifest_path: Path | None
    final_hard_row_action_plan_path: Path | None
    phase3_training_candidates_path: Path | None
    excluded_ambiguous_or_noisy_rows_path: Path | None
    engineering_fix_candidates_path: Path | None
    human_verification_queue_path: Path | None
    allowed_hard_rows_path: Path | None
    blocked_suspected_mislabel_rows_path: Path | None
    blocked_roi_pipeline_bug_rows_path: Path | None
    blocked_other_high_risk_rows_path: Path | None
    confident_wrong_probability_threshold: float
    label_issue_threshold: float
    dark_brightness_threshold: float
    bright_brightness_threshold: float
    low_contrast_threshold: float
    weak_edge_threshold: float
    min_image_width: int
    min_image_height: int
    review_template_limit: int
    contact_sheet_limit: int
    use_cleanlab_if_available: bool
    use_fiftyone_if_available: bool
    use_imagehash_if_available: bool


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


def _discover_repo_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / ".git").exists():
            return candidate.resolve()
    return config_path.parent.resolve()


def _required_path(value: Any, base_dir: Path) -> Path:
    path = _optional_path(value, base_dir)
    if path is None:
        raise DataQualityAuditError("Required config path is missing")
    return path


def _validate_safety_config(payload: dict[str, Any]) -> None:
    safety = payload.get("safety", {})
    if not isinstance(safety, dict):
        raise DataQualityAuditError("safety config must be a mapping")
    for field in FORBIDDEN_SAFETY_FLAGS:
        if bool(safety.get(field, False)):
            raise DataQualityAuditError(f"Safety flag must remain false for Spec 016: {field}")


def load_config(config_path: str | Path) -> DataQualityConfig:
    path = Path(config_path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise DataQualityAuditError(f"Data quality audit config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise DataQualityAuditError(f"Data quality audit config is not valid YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise DataQualityAuditError("Data quality audit config must be a mapping")
    _validate_safety_config(payload)

    base_dir = _discover_repo_root(path.resolve())
    dataset = payload.get("dataset", {})
    predictions = payload.get("predictions", {})
    optional_inputs = payload.get("optional_inputs", {})
    spec014 = payload.get("spec014", {})
    output = payload.get("output", {})
    quality = payload.get("quality", {})
    optional_tools = payload.get("optional_tools", {})

    if not isinstance(dataset, dict) or not isinstance(predictions, dict):
        raise DataQualityAuditError("dataset and predictions sections must be mappings")

    return DataQualityConfig(
        train_csv_path=_required_path(dataset.get("train_csv_path"), base_dir),
        train_images_dir=_required_path(dataset.get("train_images_dir"), base_dir),
        output_root=_optional_path(output.get("analysis_root"), base_dir) or DEFAULT_OUTPUT_ROOT.resolve(),
        v2b_validation_predictions_path=_optional_path(
            predictions.get("v2b_validation_predictions_path"), base_dir
        ),
        oof_train_predictions_path=_optional_path(predictions.get("oof_train_predictions_path"), base_dir),
        image_embeddings_path=_optional_path(predictions.get("image_embeddings_path"), base_dir),
        bottletypes_path=_optional_path(optional_inputs.get("bottletypes_path"), base_dir),
        train_annotations_path=_optional_path(optional_inputs.get("train_annotations_path"), base_dir),
        hard_row_quality_audit_path=_optional_path(
            optional_inputs.get("hard_row_quality_audit_path"), base_dir
        ),
        hard_row_review_manifest_path=_optional_path(
            optional_inputs.get("hard_row_review_manifest_path"), base_dir
        ),
        final_hard_row_action_plan_path=_optional_path(
            optional_inputs.get("final_hard_row_action_plan_path"), base_dir
        ),
        phase3_training_candidates_path=_optional_path(
            optional_inputs.get("phase3_training_candidates_path"), base_dir
        ),
        excluded_ambiguous_or_noisy_rows_path=_optional_path(
            optional_inputs.get("excluded_ambiguous_or_noisy_rows_path"), base_dir
        ),
        engineering_fix_candidates_path=_optional_path(
            optional_inputs.get("engineering_fix_candidates_path"), base_dir
        ),
        human_verification_queue_path=_optional_path(
            optional_inputs.get("human_verification_queue_path"), base_dir
        ),
        allowed_hard_rows_path=_optional_path(spec014.get("allowed_hard_rows_path"), base_dir),
        blocked_suspected_mislabel_rows_path=_optional_path(
            spec014.get("blocked_suspected_mislabel_rows_path"), base_dir
        ),
        blocked_roi_pipeline_bug_rows_path=_optional_path(
            spec014.get("blocked_roi_pipeline_bug_rows_path"), base_dir
        ),
        blocked_other_high_risk_rows_path=_optional_path(
            spec014.get("blocked_other_high_risk_rows_path"), base_dir
        ),
        confident_wrong_probability_threshold=float(
            quality.get("confident_wrong_probability_threshold", 0.90)
        ),
        label_issue_threshold=float(quality.get("label_issue_threshold", 0.80)),
        dark_brightness_threshold=float(quality.get("dark_brightness_threshold", 20.0)),
        bright_brightness_threshold=float(quality.get("bright_brightness_threshold", 235.0)),
        low_contrast_threshold=float(quality.get("low_contrast_threshold", 8.0)),
        weak_edge_threshold=float(quality.get("weak_edge_threshold", 2.0)),
        min_image_width=int(quality.get("min_image_width", 32)),
        min_image_height=int(quality.get("min_image_height", 32)),
        review_template_limit=int(quality.get("review_template_limit", 800)),
        contact_sheet_limit=int(quality.get("contact_sheet_limit", 64)),
        use_cleanlab_if_available=bool(optional_tools.get("use_cleanlab_if_available", True)),
        use_fiftyone_if_available=bool(optional_tools.get("use_fiftyone_if_available", False)),
        use_imagehash_if_available=bool(optional_tools.get("use_imagehash_if_available", True)),
    )


def _load_optional_csv(path: Path | None, warnings: list[str], label: str) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        warnings.append(f"{label} missing: {path}")
        return None
    return pd.read_csv(path)


def _ensure_master_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in MASTER_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[MASTER_COLUMNS]


def _normalize_prediction_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = frame.copy()
    data.columns = [str(column) for column in data.columns]
    probability_col = next(
        (
            column
            for column in ("probability", "target_probability", "v2b_probability", "oof_probability")
            if column in data.columns
        ),
        None,
    )
    prediction_col = next(
        (
            column
            for column in ("prediction", "predicted_label", "v2b_prediction", "oof_prediction")
            if column in data.columns
        ),
        None,
    )
    if "image_id" not in data.columns or probability_col is None:
        raise DataQualityAuditError(f"{prefix} predictions require image_id and probability columns")

    data = data.rename(
        columns={
            probability_col: f"{prefix}_probability",
            prediction_col: f"{prefix}_prediction" if prediction_col else "prediction",
        }
    )
    if f"{prefix}_prediction" not in data.columns:
        prob = pd.to_numeric(data[f"{prefix}_probability"], errors="coerce")
        data[f"{prefix}_prediction"] = (prob >= 0.5).astype("Int64")

    keep = ["image_id", f"{prefix}_probability", f"{prefix}_prediction"]
    return data[keep].copy()


def _prediction_risk(prob: pd.Series, target: pd.Series) -> pd.Series:
    target_prob = np.where(target == 1, prob, 1.0 - prob)
    return pd.Series(1.0 - target_prob, index=prob.index, dtype="float64")


def _source_config_hash(config: DataQualityConfig) -> str:
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    payload = json.dumps(serializable, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_master_table(config: DataQualityConfig) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if not config.train_csv_path.exists():
        raise DataQualityAuditError(f"Training CSV not found: {config.train_csv_path}")
    if not config.train_images_dir.exists():
        raise DataQualityAuditError(f"Training images directory not found: {config.train_images_dir}")

    train = pd.read_csv(config.train_csv_path)
    missing_columns = [column for column in REQUIRED_TRAIN_COLUMNS if column not in train.columns]
    if missing_columns:
        raise DataQualityAuditError(f"Training CSV missing required columns: {missing_columns}")

    master = train.copy()
    master["image_id"] = master["image_id"].astype(str)
    master["target"] = pd.to_numeric(master["target"], errors="raise").astype(int)
    if "bottle_type" not in master.columns:
        master["bottle_type"] = pd.NA
    if "source_split" not in master.columns:
        master["source_split"] = "train"

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    master["created_at"] = now
    master["audit_run_id"] = now.replace(":", "").replace("+00:00", "Z")
    master["source_config_hash"] = _source_config_hash(config)

    if config.v2b_validation_predictions_path and config.v2b_validation_predictions_path.exists():
        v2b = _normalize_prediction_columns(
            pd.read_csv(config.v2b_validation_predictions_path),
            "v2b",
        )
        master = master.merge(v2b, on="image_id", how="left")
        master["prediction_source"] = np.where(master["v2b_probability"].notna(), "v2b_validation", "")
    else:
        warnings.append("V2B validation predictions missing")
        master["v2b_probability"] = pd.NA
        master["v2b_prediction"] = pd.NA
        master["prediction_source"] = ""

    if config.oof_train_predictions_path and config.oof_train_predictions_path.exists():
        oof = _normalize_prediction_columns(
            pd.read_csv(config.oof_train_predictions_path),
            "oof",
        )
        master = master.merge(oof, on="image_id", how="left")
        master["label_quality_reliability"] = np.where(master["oof_probability"].notna(), "oof", "limited")
    else:
        warnings.append("OOF train predictions missing; label-quality scoring is limited")
        master["oof_probability"] = pd.NA
        master["oof_prediction"] = pd.NA
        master["label_quality_reliability"] = "limited"

    for prefix in ("v2b", "oof"):
        prob_col = f"{prefix}_probability"
        pred_col = f"{prefix}_prediction"
        if prob_col not in master.columns:
            master[prob_col] = pd.NA
        if pred_col not in master.columns:
            master[pred_col] = pd.NA
        probability = pd.to_numeric(master[prob_col], errors="coerce")
        prediction = pd.to_numeric(master[pred_col], errors="coerce")
        target = pd.to_numeric(master["target"], errors="coerce")
        master[f"{prefix}_correct"] = prediction.eq(target).astype("object")
        master.loc[probability.isna(), f"{prefix}_correct"] = pd.NA
        master[f"{prefix}_confidence"] = pd.Series(
            np.maximum(probability, 1.0 - probability),
            index=master.index,
            dtype="float64",
        )
        master[f"{prefix}_loss_or_risk_score"] = _prediction_risk(probability, target)

    decision_source = pd.to_numeric(master["oof_probability"], errors="coerce")
    decision_source = decision_source.fillna(pd.to_numeric(master["v2b_probability"], errors="coerce"))
    master["decision_boundary_distance"] = decision_source.sub(0.5).abs()
    master["anomaly_score"] = pd.Series(
        1.0 - master["decision_boundary_distance"].fillna(0.5) * 2.0,
        index=master.index,
        dtype="float64",
    ).clip(lower=0.0, upper=1.0)

    return _ensure_master_columns(master), warnings


def _read_image_id_set(path: Path | None, label: str, warnings: list[str]) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        warnings.append(f"{label} missing: {path}")
        return set()
    frame = pd.read_csv(path)
    if "image_id" not in frame.columns:
        warnings.append(f"{label} has no image_id column: {path}")
        return set()
    return set(frame["image_id"].astype(str))


def apply_spec014_evidence(
    master: pd.DataFrame, config: DataQualityConfig
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    result = master.copy()

    allowed = _load_optional_csv(config.allowed_hard_rows_path, warnings, "Spec 014 allowed hard rows")
    if allowed is not None:
        if "image_id" not in allowed.columns:
            warnings.append(f"Spec 014 allowed hard rows has no image_id column: {config.allowed_hard_rows_path}")
        else:
            allowed = allowed.copy()
            allowed["image_id"] = allowed["image_id"].astype(str)
            rename_map = {
                "final_failure_mode": "spec014_final_failure_mode",
                "recommended_action": "spec014_recommended_action",
                "phase3_use_allowed": "spec014_phase3_use_allowed",
            }
            allowed = allowed.rename(columns=rename_map)
            keep = [
                column
                for column in (
                    "image_id",
                    "hard_example_type",
                    "spec014_final_failure_mode",
                    "spec014_recommended_action",
                    "spec014_phase3_use_allowed",
                )
                if column in allowed.columns
            ]
            result = result.merge(allowed[keep].drop_duplicates("image_id"), on="image_id", how="left")
            allowed_ids = set(allowed["image_id"])
            result.loc[result["image_id"].isin(allowed_ids), "hard_row_status"] = "spec014_allowed_hard"

    blocked_specs = [
        (config.blocked_suspected_mislabel_rows_path, "spec014_suspected_mislabel"),
        (config.blocked_roi_pipeline_bug_rows_path, "spec014_roi_pipeline_bug"),
        (config.blocked_other_high_risk_rows_path, "spec014_other_high_risk"),
    ]
    for path, reason in blocked_specs:
        ids = _read_image_id_set(path, reason, warnings)
        if not ids:
            continue
        mask = result["image_id"].isin(ids)
        result.loc[mask, "hard_row_status"] = "spec014_blocked"
        result.loc[mask, "blocked_reason"] = reason

    return _ensure_master_columns(result), warnings


def _image_quality_metrics(path: Path) -> dict[str, Any]:
    try:
        if cv2 is not None:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(path)
            height, width = image.shape[:2]
            scale = min(128 / max(height, width), 1.0)
            if scale < 1.0:
                resized = cv2.resize(
                    image,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                resized = image
            edges = cv2.Canny(resized, 50, 150)
            return {
                "roi_quality_flag": "ok",
                "crop_quality_score": float(min(float(resized.std()) / 32.0, 1.0)),
                "annotation_evidence_flag": "",
                "image_width": int(width),
                "image_height": int(height),
                "image_mean_brightness": float(resized.mean()),
                "image_contrast": float(resized.std()),
                "edge_strength": float(edges.mean()),
                "background_heavy_score": float((resized > 245).mean()),
            }

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            analysis = rgb.copy()
            analysis.thumbnail((128, 128))
            gray = analysis.convert("L")
            arr = np.asarray(gray, dtype=np.float32)
            edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
            return {
                "roi_quality_flag": "ok",
                "crop_quality_score": float(min(arr.std() / 32.0, 1.0)),
                "annotation_evidence_flag": "",
                "image_width": int(rgb.width),
                "image_height": int(rgb.height),
                "image_mean_brightness": float(arr.mean()),
                "image_contrast": float(arr.std()),
                "edge_strength": float(edges.mean()),
                "background_heavy_score": float((arr > 245).mean()),
            }
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return {
            "roi_quality_flag": "missing_image",
            "crop_quality_score": 0.0,
            "annotation_evidence_flag": "",
            "image_width": pd.NA,
            "image_height": pd.NA,
            "image_mean_brightness": pd.NA,
            "image_contrast": pd.NA,
            "edge_strength": pd.NA,
            "background_heavy_score": pd.NA,
        }


def apply_roi_quality_audit(
    master: pd.DataFrame,
    images_dir: Path,
    dark_threshold: float,
    bright_threshold: float,
    low_contrast_threshold: float,
    weak_edge_threshold: float,
    min_image_width: int = 32,
    min_image_height: int = 32,
) -> pd.DataFrame:
    image_ids = master["image_id"].astype(str).tolist()

    def compute_metrics(image_id: str) -> dict[str, Any]:
        metrics = _image_quality_metrics(images_dir / image_id)
        if metrics["roi_quality_flag"] == "ok":
            if (
                int(metrics["image_width"]) < min_image_width
                or int(metrics["image_height"]) < min_image_height
            ):
                metrics["roi_quality_flag"] = "very_small_image"
            elif metrics["image_mean_brightness"] < dark_threshold:
                metrics["roi_quality_flag"] = "mostly_dark"
            elif metrics["image_mean_brightness"] > bright_threshold:
                metrics["roi_quality_flag"] = "mostly_bright"
            elif (
                metrics["image_contrast"] < low_contrast_threshold
                or metrics["edge_strength"] < weak_edge_threshold
            ):
                metrics["roi_quality_flag"] = "low_contrast_or_weak_texture"
            elif metrics["background_heavy_score"] > 0.80:
                metrics["roi_quality_flag"] = "background_heavy"
        metrics["image_id"] = image_id
        return metrics

    max_workers = min(16, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(compute_metrics, image_ids))

    quality = pd.DataFrame(rows)
    result = master.drop(
        columns=[
            "roi_quality_flag",
            "crop_quality_score",
            "annotation_evidence_flag",
            "image_width",
            "image_height",
            "image_mean_brightness",
            "image_contrast",
            "edge_strength",
            "background_heavy_score",
        ],
        errors="ignore",
    )
    result = result.merge(quality, on="image_id", how="left")
    return _ensure_master_columns(result)


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_duplicate_audit(master: pd.DataFrame, images_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = master.copy()
    file_paths = [images_dir / image_id for image_id in result["image_id"].astype(str)]
    file_sizes = [path.stat().st_size if path.exists() and path.is_file() else -1 for path in file_paths]
    result["_file_size"] = file_sizes
    duplicated_sizes = (
        result.loc[result["_file_size"] >= 0, "_file_size"].value_counts().loc[lambda counts: counts > 1].index
    )
    duplicated_size_set = set(duplicated_sizes.tolist())
    result["_file_hash"] = [
        _file_hash(path) if size in duplicated_size_set else ""
        for path, size in zip(file_paths, file_sizes)
    ]
    result["duplicate_group_id"] = ""
    result["near_duplicate_score"] = 0.0
    result["duplicate_conflict_flag"] = False

    valid_hashes = result["_file_hash"].astype(str) != ""
    counts = result.loc[valid_hashes, "_file_hash"].value_counts()
    duplicate_hashes = [value for value, count in counts.items() if count > 1]

    for index, file_hash in enumerate(sorted(duplicate_hashes), start=1):
        mask = result["_file_hash"] == file_hash
        group_id = f"exact_dup_{index:05d}"
        result.loc[mask, "duplicate_group_id"] = group_id
        result.loc[mask, "near_duplicate_score"] = 1.0
        if result.loc[mask, "target"].nunique(dropna=True) > 1:
            result.loc[mask, "duplicate_conflict_flag"] = True

    conflicts = result.loc[result["duplicate_conflict_flag"]].drop(columns=["_file_hash", "_file_size"]).copy()
    return _ensure_master_columns(result.drop(columns=["_file_hash", "_file_size"])), conflicts


def apply_label_issue_scoring(master: pd.DataFrame) -> pd.DataFrame:
    result = master.copy()
    probability = pd.to_numeric(result["oof_probability"], errors="coerce")
    probability = probability.fillna(pd.to_numeric(result["v2b_probability"], errors="coerce"))
    target = pd.to_numeric(result["target"], errors="coerce")
    predicted = (probability >= 0.5).astype("Int64")
    confidence = pd.Series(np.maximum(probability, 1.0 - probability), index=result.index, dtype="float64")
    correct = predicted.eq(target)

    base_score = _prediction_risk(probability, target).fillna(0.0)
    blocked_bonus = result["blocked_reason"].notna().astype(float) * 0.15
    duplicate_bonus = result["duplicate_conflict_flag"].fillna(False).astype(bool).astype(float) * 0.20
    roi_bonus = result["roi_quality_flag"].fillna("ok").ne("ok").astype(float) * 0.10
    result["label_issue_score"] = (base_score + blocked_bonus + duplicate_bonus + roi_bonus).clip(0.0, 1.0)
    result["review_priority"] = 99

    confident_wrong = (~correct.fillna(False)) & (confidence >= 0.90)
    high_label_issue = result["label_issue_score"] >= 0.80
    blocked = result["blocked_reason"].notna()
    duplicate_conflict = result["duplicate_conflict_flag"].fillna(False).astype(bool)
    roi_problem = result["roi_quality_flag"].fillna("ok").ne("ok")
    boundary = probability.sub(0.5).abs() <= 0.05

    result.loc[boundary, "review_priority"] = 7
    result.loc[roi_problem, "review_priority"] = 6
    result.loc[duplicate_conflict, "review_priority"] = 4
    result.loc[blocked, "review_priority"] = 3
    result.loc[high_label_issue, "review_priority"] = 2
    result.loc[confident_wrong, "review_priority"] = 1

    result["decision_boundary_distance"] = probability.sub(0.5).abs()
    if "v2b_confidence" in result.columns:
        result["v2b_confidence"] = result["v2b_confidence"].fillna(confidence)
    else:
        result["v2b_confidence"] = confidence
    return _ensure_master_columns(result)


def apply_optional_cleanlab_scoring(
    master: pd.DataFrame, enabled: bool
) -> tuple[pd.DataFrame, list[str]]:
    result = master.copy()
    warnings: list[str] = []
    if not enabled:
        warnings.append("Cleanlab disabled by config")
        return result, warnings
    try:
        importlib.import_module("cleanlab")
    except ImportError:
        warnings.append("Cleanlab not installed; using deterministic fallback label_issue_score")
        return result, warnings

    warnings.append(
        "Cleanlab installed but deterministic fallback retained unless full prediction matrices are provided"
    )
    return result, warnings


def load_optional_embeddings(path: Path | None) -> tuple[pd.DataFrame | None, str]:
    if path is None:
        return None, "not_configured"
    if not path.exists():
        return None, "missing"
    frame = pd.read_csv(path)
    if "image_id" not in frame.columns:
        return None, "invalid_missing_image_id"
    return frame, "available"


def _build_evidence_summary(row: pd.Series) -> str:
    parts = []
    for column in (
        "data_quality_bucket",
        "exclusion_reason",
        "blocked_reason",
        "roi_quality_flag",
        "hard_row_status",
    ):
        value = row.get(column)
        if pd.notna(value) and str(value):
            parts.append(f"{column}={value}")
    return "; ".join(parts)


def assign_data_quality_buckets(master: pd.DataFrame, label_issue_threshold: float) -> pd.DataFrame:
    result = master.copy()
    result["data_quality_bucket"] = "clean_train"
    result["exclude_from_training"] = False
    result["exclusion_reason"] = ""

    def set_bucket(mask: pd.Series, bucket: str, reason: str, exclude: bool = True) -> None:
        result.loc[mask, "data_quality_bucket"] = bucket
        result.loc[mask, "exclude_from_training"] = exclude
        result.loc[mask, "exclusion_reason"] = reason

    roi = result["roi_quality_flag"].fillna("ok").astype(str)
    blocked_reason = result["blocked_reason"].astype("string")
    duplicate_conflict = result["duplicate_conflict_flag"].fillna(False).astype(bool)
    label_issue = pd.to_numeric(result["label_issue_score"], errors="coerce").fillna(0.0)
    hard_status = result["hard_row_status"].astype("string")
    correct = result["hard_row_status"].astype("string")

    set_bucket(roi.eq("missing_image"), "exclude_from_training", "missing_or_unreadable_image")
    set_bucket(blocked_reason.eq("spec014_suspected_mislabel"), "exclude_from_training", "spec014_suspected_mislabel")
    set_bucket(
        blocked_reason.eq("spec014_roi_pipeline_bug"),
        "manual_review_required",
        "spec014_roi_pipeline_bug_requires_review",
    )
    set_bucket(
        blocked_reason.eq("spec014_other_high_risk"),
        "manual_review_required",
        "spec014_other_high_risk_requires_review",
    )
    set_bucket(duplicate_conflict, "manual_review_required", "duplicate_label_conflict")
    set_bucket(label_issue.ge(label_issue_threshold), "manual_review_required", "high_label_issue_score")
    set_bucket(
        roi.ne("ok") & roi.ne("missing_image"),
        "manual_review_required",
        "roi_or_crop_problem_requires_review",
    )

    eligible_hard = (
        hard_status.eq("spec014_allowed_hard")
        & result["exclusion_reason"].eq("")
        & roi.eq("ok")
        & ~duplicate_conflict
        & label_issue.lt(label_issue_threshold)
    )
    result.loc[eligible_hard, "data_quality_bucket"] = "hard_valid_train"
    result.loc[eligible_hard, "exclude_from_training"] = False
    result.loc[eligible_hard, "exclusion_reason"] = ""

    manual = result["data_quality_bucket"].eq("manual_review_required")
    result["manual_review_status"] = np.where(manual, "pending", "")
    result["evidence_summary"] = result.apply(_build_evidence_summary, axis=1)
    return _ensure_master_columns(result)


def build_cleaned_training_manifest(master: pd.DataFrame) -> pd.DataFrame:
    allowed = master["data_quality_bucket"].isin(["clean_train", "hard_valid_train"])
    not_excluded = ~master["exclude_from_training"].fillna(True).astype(bool)
    manifest = master.loc[allowed & not_excluded].copy()
    manifest["hard_training_allowed"] = manifest["data_quality_bucket"].eq("hard_valid_train")
    columns = [
        "image_id",
        "target",
        "data_quality_bucket",
        "hard_training_allowed",
        "exclude_from_training",
        "exclusion_reason",
        "review_priority",
        "evidence_summary",
    ]
    for column in columns:
        if column not in manifest.columns:
            manifest[column] = pd.NA
    return manifest[columns]


def build_input_file_inventory(paths: dict[str, Path | None]) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        exists = bool(path and path.exists())
        size_bytes = 0
        if exists and path:
            if path.is_file():
                size_bytes = int(path.stat().st_size)
            else:
                size_bytes = 0
        rows.append(
            {
                "input_name": name,
                "path": "" if path is None else str(path),
                "exists": exists,
                "size_bytes": size_bytes,
            }
        )
    return pd.DataFrame(rows)


def build_validation_alignment_summary(master: pd.DataFrame) -> dict[str, Any]:
    has_source_split = "source_split" in master.columns and master["source_split"].notna().any()
    summary: dict[str, Any] = {
        "has_source_split": bool(has_source_split),
        "note": "Validation alignment requires source_split or fold metadata. Missing metadata limits split-drift analysis.",
    }
    if not has_source_split:
        return summary
    grouped = (
        master.groupby(["source_split", "data_quality_bucket"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    summary["bucket_counts_by_split"] = grouped.to_dict(orient="records")
    return summary


def write_reports(
    master: pd.DataFrame,
    output_root: Path,
    warnings: list[str],
    duplicate_conflicts: pd.DataFrame,
    review_template_limit: int = 800,
) -> None:
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    bucket_counts = (
        master["data_quality_bucket"]
        .value_counts(dropna=False)
        .rename_axis("data_quality_bucket")
        .reset_index(name="count")
    )
    bucket_counts.to_csv(reports_dir / "bucket_counts.csv", index=False)

    exclusion_counts = (
        master.loc[master["exclusion_reason"].fillna("").astype(str) != "", "exclusion_reason"]
        .value_counts()
        .rename_axis("exclusion_reason")
        .reset_index(name="count")
    )
    exclusion_counts.to_csv(reports_dir / "exclusion_reason_counts.csv", index=False)

    ranked = master.sort_values(["review_priority", "label_issue_score"], ascending=[True, False])
    ranked.to_csv(reports_dir / "top_suspicious_rows.csv", index=False)
    ranked.head(review_template_limit).to_csv(output_root / "review_decision_template.csv", index=False)
    master.sort_values("label_issue_score", ascending=False).head(review_template_limit).to_csv(
        reports_dir / "top_label_issue_rows.csv",
        index=False,
    )
    master.loc[master["review_priority"].eq(1)].sort_values("label_issue_score", ascending=False).to_csv(
        reports_dir / "top_confident_wrong_rows.csv",
        index=False,
    )
    anomaly_source = "anomaly_score" if master["anomaly_score"].notna().any() else "label_issue_score"
    master.sort_values(anomaly_source, ascending=False).head(review_template_limit).to_csv(
        reports_dir / "top_anomaly_rows.csv",
        index=False,
    )
    duplicate_conflicts.to_csv(reports_dir / "duplicate_conflicts.csv", index=False)
    master.loc[master["roi_quality_flag"].fillna("ok").ne("ok")].to_csv(
        reports_dir / "roi_quality_issues.csv",
        index=False,
    )

    summary = {
        "total_rows": int(len(master)),
        "clean_train_count": int(master["data_quality_bucket"].eq("clean_train").sum()),
        "hard_valid_train_count": int(master["data_quality_bucket"].eq("hard_valid_train").sum()),
        "exclude_from_training_count": int(master["data_quality_bucket"].eq("exclude_from_training").sum()),
        "manual_review_required_count": int(master["data_quality_bucket"].eq("manual_review_required").sum()),
        "warnings": warnings,
        "oof_prediction_status": "available" if master["oof_probability"].notna().any() else "missing",
        "label_quality_reliability": "oof" if master["oof_probability"].notna().any() else "limited",
        "no_training_started": True,
        "no_submission_created": True,
        "no_test_labels_used": True,
        "no_leaderboard_tuning": True,
        "original_labels_unchanged": True,
    }
    (reports_dir / "data_quality_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manual_review_plan = {
        "priority_1": "confident wrong predictions",
        "priority_2": "top label issue rows",
        "priority_3": "Spec 014 blocked or ambiguous rows",
        "priority_4": "duplicate label conflicts",
        "priority_5": "cluster outliers or anomalies",
        "priority_6": "ROI/crop issues",
        "priority_7": "decision-boundary unclear rows",
        "default_review_limit": int(review_template_limit),
    }
    (reports_dir / "manual_review_plan.json").write_text(
        json.dumps(manual_review_plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_contact_sheet(
    rows: pd.DataFrame,
    images_dir: Path,
    output_path: Path,
    max_rows: int = 64,
    thumb_size: tuple[int, int] = (160, 160),
    columns: int = 4,
) -> None:
    selected = rows.head(max_rows).copy()
    if selected.empty:
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
        caption = f'{row.get("image_id", "")}\nt={row.get("target", "")} {row.get("data_quality_bucket", "")}'
        draw.text((x + 4, y + thumb_size[1] + 4), caption[:120], fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def build_contact_sheets_from_outputs(config: DataQualityConfig) -> list[Path]:
    master_path = config.output_root / "data_quality_master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing data quality master table: {master_path}")
    master = pd.read_csv(master_path)
    contact_root = config.output_root / "contact_sheets"
    contact_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for bucket in ["manual_review_required", "exclude_from_training", "hard_valid_train", "clean_train"]:
        rows = master.loc[master["data_quality_bucket"].eq(bucket)].sort_values(
            ["review_priority", "label_issue_score"],
            ascending=[True, False],
        )
        path = contact_root / f"{bucket}.jpg"
        build_contact_sheet(rows, config.train_images_dir, path, max_rows=config.contact_sheet_limit)
        if path.exists():
            outputs.append(path)
    return outputs


def run_audit(config: DataQualityConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    master, master_warnings = build_master_table(config)
    warnings.extend(master_warnings)

    bottletypes = _load_optional_csv(config.bottletypes_path, warnings, "Bottle types")
    if bottletypes is not None and "image_id" in bottletypes.columns:
        master = master.drop(columns=["bottle_type"], errors="ignore").merge(
            bottletypes[["image_id", "bottle_type"]].drop_duplicates("image_id"),
            on="image_id",
            how="left",
        )
    elif bottletypes is not None:
        warnings.append(f"Bottle types has no image_id column: {config.bottletypes_path}")

    master, spec_warnings = apply_spec014_evidence(master, config)
    warnings.extend(spec_warnings)

    master = apply_roi_quality_audit(
        master,
        images_dir=config.train_images_dir,
        dark_threshold=config.dark_brightness_threshold,
        bright_threshold=config.bright_brightness_threshold,
        low_contrast_threshold=config.low_contrast_threshold,
        weak_edge_threshold=config.weak_edge_threshold,
        min_image_width=config.min_image_width,
        min_image_height=config.min_image_height,
    )
    master, duplicate_conflicts = apply_duplicate_audit(master, config.train_images_dir)
    master = apply_label_issue_scoring(master)
    master, cleanlab_warnings = apply_optional_cleanlab_scoring(
        master,
        enabled=config.use_cleanlab_if_available,
    )
    warnings.extend(cleanlab_warnings)
    _, embedding_status = load_optional_embeddings(config.image_embeddings_path)
    if embedding_status != "available":
        warnings.append(f"Embeddings status: {embedding_status}")

    master = assign_data_quality_buckets(master, label_issue_threshold=config.label_issue_threshold)
    manifest = build_cleaned_training_manifest(master)

    master.to_csv(config.output_root / "data_quality_master.csv", index=False)
    master.loc[master["data_quality_bucket"].eq("clean_train")].to_csv(
        config.output_root / "clean_train_rows.csv",
        index=False,
    )
    master.loc[master["data_quality_bucket"].eq("hard_valid_train")].to_csv(
        config.output_root / "hard_valid_train_rows.csv",
        index=False,
    )
    master.loc[master["data_quality_bucket"].eq("exclude_from_training")].to_csv(
        config.output_root / "exclude_from_training_rows.csv",
        index=False,
    )
    master.loc[master["data_quality_bucket"].eq("manual_review_required")].to_csv(
        config.output_root / "manual_review_required_rows.csv",
        index=False,
    )
    manifest.to_csv(config.output_root / "cleaned_training_manifest.csv", index=False)

    write_reports(
        master,
        output_root=config.output_root,
        warnings=warnings,
        duplicate_conflicts=duplicate_conflicts,
        review_template_limit=config.review_template_limit,
    )

    inventory = build_input_file_inventory(
        {
            "train_csv": config.train_csv_path,
            "train_images_dir": config.train_images_dir,
            "v2b_validation_predictions": config.v2b_validation_predictions_path,
            "oof_train_predictions": config.oof_train_predictions_path,
            "image_embeddings": config.image_embeddings_path,
            "spec014_allowed_hard_rows": config.allowed_hard_rows_path,
            "spec014_blocked_suspected_mislabel": config.blocked_suspected_mislabel_rows_path,
            "spec014_blocked_roi_pipeline_bug": config.blocked_roi_pipeline_bug_rows_path,
            "spec014_blocked_other_high_risk": config.blocked_other_high_risk_rows_path,
        }
    )
    reports_dir = config.output_root / "reports"
    inventory.to_csv(reports_dir / "input_file_inventory.csv", index=False)
    alignment = build_validation_alignment_summary(master)
    (reports_dir / "validation_alignment_summary.json").write_text(
        json.dumps(alignment, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_path = reports_dir / "data_quality_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run data quality audit workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Build data quality audit outputs")
    run_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    contact_parser = subparsers.add_parser(
        "build-contact-sheets",
        help="Build review contact sheets from existing data quality outputs",
    )
    contact_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "run":
        summary = run_audit(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "build-contact-sheets":
        paths = build_contact_sheets_from_outputs(config)
        print(json.dumps({"contact_sheets": [str(path) for path in paths]}, indent=2))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
