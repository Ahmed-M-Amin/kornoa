"""Hybrid classifier-detector fusion for SPEC-009."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


class HybridFusionError(ValueError):
    """Raised when hybrid fusion inputs or parameters are invalid."""


@dataclass(frozen=True)
class HybridFusionConfig:
    classifier_threshold: float
    uncertainty_margin: float
    detector_conf_threshold: float
    conditional_area_thresholds: dict[str, float] = field(default_factory=dict)
    default_conditional_area_threshold: float | None = None
    always_faulty_categories: set[str] = field(default_factory=set)
    conditional_categories: set[str] = field(default_factory=set)
    detector_usage_preference: float = 0.30
    optimization_metric: str = "f1"

    def __post_init__(self) -> None:
        _validate_unit_interval("classifier_threshold", self.classifier_threshold)
        _validate_unit_interval("uncertainty_margin", self.uncertainty_margin)
        _validate_unit_interval("detector_conf_threshold", self.detector_conf_threshold)
        _validate_unit_interval("detector_usage_preference", self.detector_usage_preference)
        if self.default_conditional_area_threshold is not None:
            _validate_nonnegative("default_conditional_area_threshold", self.default_conditional_area_threshold)
        for category, threshold in self.conditional_area_thresholds.items():
            if not str(category).strip():
                raise HybridFusionError("conditional_area_thresholds contains an empty category")
            _validate_nonnegative(f"conditional_area_thresholds[{category!r}]", threshold)
        object.__setattr__(self, "always_faulty_categories", {str(item) for item in self.always_faulty_categories})
        object.__setattr__(self, "conditional_categories", {str(item) for item in self.conditional_categories})
        object.__setattr__(
            self,
            "conditional_area_thresholds",
            {str(key): float(value) for key, value in self.conditional_area_thresholds.items()},
        )

    def to_report_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["always_faulty_categories"] = sorted(self.always_faulty_categories)
        data["conditional_categories"] = sorted(self.conditional_categories)
        return data


def load_best_threshold(path: str | Path) -> float:
    report_path = Path(path)
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HybridFusionError(f"Threshold file not found: {report_path}") from exc
    except json.JSONDecodeError as exc:
        raise HybridFusionError(f"Threshold file is not valid JSON: {report_path}") from exc
    for key in ("threshold", "best_threshold", "classifier_threshold"):
        if key in raw:
            try:
                threshold = float(raw[key])
            except (TypeError, ValueError) as exc:
                raise HybridFusionError(f"Threshold value is not numeric in {report_path}") from exc
            _validate_unit_interval("threshold", threshold)
            return threshold
    raise HybridFusionError(f"Threshold file must contain threshold field: {report_path}")


def normalize_image_id(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text:
        raise HybridFusionError("image_id cannot be empty")
    return Path(text).name


def normalize_classifier_predictions(
    df: pd.DataFrame,
    classifier_threshold: Optional[float] = None,
    require_label: bool = False,
) -> pd.DataFrame:
    image_col = _find_column(df, ["image_id", "id", "filename", "image", "path"])
    score_col = _find_column(
        df,
        [
            "classifier_score",
            "prob_bad",
            "prob",
            "probability",
            "score",
            "confidence",
            "positive_prob",
            "pred_prob",
            "target_prob",
        ],
    )
    target_col = _find_column(
        df,
        ["classifier_target", "classifier_prediction", "pred", "prediction", "target_pred", "predicted_target"],
        required=False,
    )
    label_col = _find_column(df, ["y_true", "label", "ground_truth", "true_target", "true_label"], required=False)
    if label_col is None and require_label:
        target_as_label = _find_column(df, ["target"], required=False)
        label_col = target_as_label
    if require_label and label_col is None:
        raise HybridFusionError("Classifier validation predictions must contain labels")

    out = pd.DataFrame()
    out["image_id"] = df[image_col].map(normalize_image_id)
    out["classifier_score"] = pd.to_numeric(df[score_col], errors="raise").astype(float)
    if ((out["classifier_score"] < 0) | (out["classifier_score"] > 1)).any():
        raise HybridFusionError("classifier_score values must be between 0 and 1")
    if target_col is not None:
        out["classifier_target"] = pd.to_numeric(df[target_col], errors="raise").astype(int)
    elif classifier_threshold is not None:
        _validate_unit_interval("classifier_threshold", classifier_threshold)
        out["classifier_target"] = (out["classifier_score"] >= classifier_threshold).astype(int)
    else:
        raise HybridFusionError("classifier_target is missing and no classifier_threshold was provided")
    if label_col is not None:
        out["y_true"] = pd.to_numeric(df[label_col], errors="raise").astype(int)
    return out.sort_values("image_id").reset_index(drop=True)


def normalize_detector_predictions(df: pd.DataFrame) -> pd.DataFrame:
    image_col = _find_column(df, ["image_id", "id", "filename", "image", "path"])
    conf_col = _find_column(
        df,
        ["detector_confidence", "detector_score", "confidence", "conf", "max_conf", "max_detector_conf", "box_conf", "score"],
    )
    category_col = _find_column(df, ["defect_category", "category", "category_name", "class_name", "label_name"], required=False)
    area_col = _find_column(df, ["defect_area", "area", "bbox_area", "mask_area"], required=False)
    binary_col = _find_column(df, ["detector_target", "prediction", "predicted_target", "target"], required=False)

    out = pd.DataFrame()
    out["image_id"] = df[image_col].map(normalize_image_id)
    out["detector_confidence"] = pd.to_numeric(df[conf_col], errors="raise").astype(float)
    if category_col is not None:
        out["defect_category"] = df[category_col].fillna("").astype(str)
    else:
        out["defect_category"] = ""
    if area_col is not None:
        out["defect_area"] = pd.to_numeric(df[area_col], errors="coerce")
    else:
        out["defect_area"] = pd.NA
    if binary_col is not None:
        out["detector_binary_target"] = pd.to_numeric(df[binary_col], errors="coerce").fillna(0).astype(int)
        out.loc[out["detector_binary_target"] == 1, "defect_category"] = out.loc[
            out["detector_binary_target"] == 1, "defect_category"
        ].replace("", "binary_detector_positive")
    else:
        out["detector_binary_target"] = pd.NA

    return out.sort_values(["image_id", "detector_confidence"], ascending=[True, False]).reset_index(drop=True)


def apply_hybrid_fusion(
    classifier_df: pd.DataFrame,
    detector_df: pd.DataFrame,
    config: HybridFusionConfig,
    require_detector_for_uncertain: bool = False,
) -> pd.DataFrame:
    classifier = _ensure_classifier_frame(classifier_df, config.classifier_threshold)
    detector = _ensure_detector_frame(detector_df)
    merged = classifier.merge(detector, on="image_id", how="left")
    merged = _select_rule_aware_detector_evidence(merged, config)
    if require_detector_for_uncertain and merged["detector_confidence"].isna().any():
        missing = merged.loc[merged["detector_confidence"].isna(), "image_id"].tolist()
        raise HybridFusionError("Missing detector evidence for uncertain images: " + ", ".join(missing))

    merged["classifier_uncertain"] = (
        (merged["classifier_score"] - config.classifier_threshold).abs() <= config.uncertainty_margin
    )
    merged["evidence_available"] = merged["detector_confidence"].notna()
    merged["detector_used"] = merged["classifier_uncertain"] & merged["detector_rejects"]
    merged["hybrid_target"] = merged["classifier_target"].astype(int)
    merged.loc[merged["detector_used"], "hybrid_target"] = 1
    merged["decision_source"] = "classifier"
    merged.loc[merged["detector_used"], "decision_source"] = "detector"

    for column in ("classifier_uncertain", "evidence_available", "detector_rejects", "detector_used", "used_default_area_threshold"):
        merged[column] = merged[column].map(bool).astype(object)
    if "defect_category" not in merged:
        merged["defect_category"] = ""
    if "defect_area" not in merged:
        merged["defect_area"] = pd.NA
    if "detector_confidence" not in merged:
        merged["detector_confidence"] = pd.NA
    return merged.sort_values("image_id").reset_index(drop=True)


def compute_binary_metrics(y_true: Iterable[object], y_pred: Iterable[object]) -> dict[str, float | int]:
    truth = [int(value) for value in y_true]
    pred = [int(value) for value in y_pred]
    if len(truth) != len(pred):
        raise HybridFusionError("y_true and y_pred lengths differ")
    tp = sum(1 for y, p in zip(truth, pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(truth, pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(truth, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(truth, pred) if y == 1 and p == 0)
    total = len(truth)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support_positive": sum(1 for value in truth if value == 1),
        "support_negative": sum(1 for value in truth if value == 0),
    }


def build_overlap_report(classifier_df: pd.DataFrame, detector_df: pd.DataFrame) -> dict[str, object]:
    classifier_ids = set(classifier_df["image_id"].map(normalize_image_id))
    detector_ids = set(detector_df["image_id"].map(normalize_image_id))
    labeled_ids = set(classifier_df.loc[classifier_df.get("y_true", pd.Series(index=classifier_df.index)).notna(), "image_id"].map(normalize_image_id))
    overlap = sorted(classifier_ids & detector_ids & labeled_ids)
    return {
        "classifier_val_count": len(classifier_ids),
        "detector_val_count": len(detector_ids),
        "overlap_count": len(overlap),
        "classifier_only_count": len((classifier_ids - detector_ids) & labeled_ids),
        "detector_only_count": len(detector_ids - classifier_ids),
        "overlap_ratio_classifier": len(overlap) / len(classifier_ids) if classifier_ids else 0.0,
        "overlap_ratio_detector": len(overlap) / len(detector_ids) if detector_ids else 0.0,
        "overlap_image_ids": overlap,
        "used_for_parameter_search": "validation_overlap_only",
    }


def summarize_detector_usage(fused_df: pd.DataFrame) -> dict[str, int | float]:
    count = int(fused_df["detector_used"].map(bool).sum()) if "detector_used" in fused_df else 0
    total = len(fused_df)
    changed = int((fused_df["hybrid_target"].astype(int) != fused_df["classifier_target"].astype(int)).sum()) if total else 0
    result: dict[str, int | float] = {
        "count": count,
        "ratio": count / total if total else 0.0,
        "changed_predictions": changed,
        "fallback_area_threshold_uses": int(fused_df.get("used_default_area_threshold", pd.Series(dtype=bool)).map(bool).sum()),
    }
    if "y_true" in fused_df:
        classifier_correct = fused_df["classifier_target"].astype(int) == fused_df["y_true"].astype(int)
        hybrid_correct = fused_df["hybrid_target"].astype(int) == fused_df["y_true"].astype(int)
        result.update(
            {
                "changed_incorrect_to_correct": int((~classifier_correct & hybrid_correct).sum()),
                "changed_correct_to_incorrect": int((classifier_correct & ~hybrid_correct).sum()),
                "changed_fn_to_tp": int(
                    (
                        (fused_df["y_true"].astype(int) == 1)
                        & (fused_df["classifier_target"].astype(int) == 0)
                        & (fused_df["hybrid_target"].astype(int) == 1)
                    ).sum()
                ),
                "changed_fp_to_tn": int(
                    (
                        (fused_df["y_true"].astype(int) == 0)
                        & (fused_df["classifier_target"].astype(int) == 1)
                        & (fused_df["hybrid_target"].astype(int) == 0)
                    ).sum()
                ),
            }
        )
    return result


def search_hybrid_parameters(
    classifier_val_df: pd.DataFrame,
    detector_val_df: pd.DataFrame,
    base_classifier_threshold: float,
    classifier_threshold_candidates: Optional[Iterable[float]] = None,
    uncertainty_margin_candidates: Optional[Iterable[float]] = None,
    detector_conf_threshold_candidates: Optional[Iterable[float]] = None,
    *,
    always_faulty_categories: Optional[set[str]] = None,
    conditional_categories: Optional[set[str]] = None,
    conditional_area_threshold_candidates: Optional[dict[str, Iterable[float]]] = None,
    default_conditional_area_threshold_candidates: Optional[Iterable[float]] = None,
    detector_usage_preference: float = 0.30,
) -> tuple[HybridFusionConfig, dict[str, object], pd.DataFrame]:
    classifier = _ensure_classifier_frame(classifier_val_df, base_classifier_threshold)
    detector = _ensure_detector_frame(detector_val_df)
    if "y_true" not in classifier:
        raise HybridFusionError("Classifier validation predictions must contain y_true for search")
    report = build_overlap_report(classifier, detector)
    if report["overlap_count"] == 0:
        raise HybridFusionError("Validation overlap is empty")
    overlap_ids = set(report["overlap_image_ids"])
    classifier = classifier[classifier["image_id"].isin(overlap_ids)].sort_values("image_id").reset_index(drop=True)
    detector = detector[detector["image_id"].isin(overlap_ids)].sort_values("image_id").reset_index(drop=True)

    threshold_candidates = list(classifier_threshold_candidates or _default_threshold_candidates(base_classifier_threshold))
    margin_candidates = list(uncertainty_margin_candidates or [i / 100 for i in range(0, 31)])
    detector_candidates = list(detector_conf_threshold_candidates or [i / 100 for i in range(5, 100, 5)])
    area_candidate_sets = _area_candidate_sets(conditional_area_threshold_candidates)
    default_area_candidates = list(default_conditional_area_threshold_candidates or [None])

    best: tuple[HybridFusionConfig, dict[str, object], pd.DataFrame] | None = None
    for threshold, margin, detector_threshold, area_thresholds, default_area in itertools.product(
        threshold_candidates,
        margin_candidates,
        detector_candidates,
        area_candidate_sets,
        default_area_candidates,
    ):
        config = HybridFusionConfig(
            classifier_threshold=float(threshold),
            uncertainty_margin=float(margin),
            detector_conf_threshold=float(detector_threshold),
            conditional_area_thresholds=area_thresholds,
            default_conditional_area_threshold=default_area,
            always_faulty_categories=always_faulty_categories or set(),
            conditional_categories=conditional_categories or set(area_thresholds),
            detector_usage_preference=detector_usage_preference,
        )
        candidate_classifier = classifier.copy()
        candidate_classifier["classifier_target"] = (
            candidate_classifier["classifier_score"] >= config.classifier_threshold
        ).astype(int)
        fused = apply_hybrid_fusion(candidate_classifier, detector, config)
        hybrid_metrics = compute_binary_metrics(fused["y_true"], fused["hybrid_target"])
        classifier_metrics = compute_binary_metrics(fused["y_true"], fused["classifier_target"])
        detector_target = fused["detector_rejects"].astype(int)
        detector_metrics = compute_binary_metrics(fused["y_true"], detector_target)
        usage = summarize_detector_usage(fused)
        over_cap = float(usage["ratio"]) > detector_usage_preference
        metrics = {
            "overlap_count": len(fused),
            "classifier_baseline": classifier_metrics,
            "detector_baseline": detector_metrics,
            "hybrid": hybrid_metrics,
            "detector_usage": usage,
            "fallback_area_threshold_uses": usage.get("fallback_area_threshold_uses", 0),
            "usage_tradeoff_reported": over_cap,
            "overlap_report": report,
        }
        if best is None or _is_better_candidate(
            candidate_metrics=hybrid_metrics,
            candidate_usage=usage,
            candidate_config=config,
            best_metrics=best[1]["hybrid"],
            best_usage=best[1]["detector_usage"],
            best_config=best[0],
            detector_usage_preference=detector_usage_preference,
        ):
            best = (config, metrics, fused)
    assert best is not None
    return best[0], best[1], best[2]


def _evaluate_detector_rejection(row: pd.Series, config: HybridFusionConfig) -> tuple[bool, str, bool]:
    if pd.isna(row.get("detector_confidence")):
        return False, "", False
    confidence = float(row["detector_confidence"])
    if confidence < config.detector_conf_threshold:
        return False, "", False
    category = str(row.get("defect_category", "") or "")
    binary_target = row.get("detector_binary_target")
    if not pd.isna(binary_target) and int(binary_target) == 1:
        return True, "binary_detector_positive", False
    if category in config.always_faulty_categories:
        return True, f"always_faulty:{category}", False
    if category in config.conditional_categories:
        threshold = config.conditional_area_thresholds.get(category)
        used_default = False
        if threshold is None:
            threshold = config.default_conditional_area_threshold
            used_default = True
        if threshold is None:
            return False, "", False
        area = row.get("defect_area")
        if pd.isna(area):
            return False, "", used_default
        if float(area) >= float(threshold):
            return True, f"conditional_area:{category}", used_default
        return False, "", used_default
    return False, "", False


def _select_rule_aware_detector_evidence(merged: pd.DataFrame, config: HybridFusionConfig) -> pd.DataFrame:
    if merged.empty:
        return merged
    marked = _mark_detector_rejections(merged, config)
    marked["_selection_confidence"] = pd.to_numeric(marked["detector_confidence"], errors="coerce").fillna(-1.0)
    marked = marked.sort_values(
        ["image_id", "detector_rejects", "_selection_confidence"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    selected = marked.groupby("image_id", sort=True, dropna=False).head(1).drop(columns=["_selection_confidence"])
    return selected.reset_index(drop=True)


def _mark_detector_rejections(df: pd.DataFrame, config: HybridFusionConfig) -> pd.DataFrame:
    marked = df.copy()
    confidence = pd.to_numeric(marked["detector_confidence"], errors="coerce")
    confidence_ok = confidence >= config.detector_conf_threshold
    category = marked.get("defect_category", pd.Series("", index=marked.index)).fillna("").astype(str)

    marked["detector_rejects"] = False
    marked["detector_rejection_reason"] = ""
    marked["used_default_area_threshold"] = False

    binary_target = pd.to_numeric(marked.get("detector_binary_target", pd.Series(pd.NA, index=marked.index)), errors="coerce")
    binary_mask = confidence_ok & (binary_target == 1)
    marked.loc[binary_mask, "detector_rejects"] = True
    marked.loc[binary_mask, "detector_rejection_reason"] = "binary_detector_positive"

    always_mask = confidence_ok & ~marked["detector_rejects"] & category.isin(config.always_faulty_categories)
    marked.loc[always_mask, "detector_rejects"] = True
    marked.loc[always_mask, "detector_rejection_reason"] = "always_faulty:" + category[always_mask]

    area = pd.to_numeric(marked.get("defect_area", pd.Series(pd.NA, index=marked.index)), errors="coerce")
    for conditional_category in config.conditional_categories:
        threshold = config.conditional_area_thresholds.get(conditional_category)
        used_default = False
        if threshold is None:
            threshold = config.default_conditional_area_threshold
            used_default = True
        if threshold is None:
            continue
        mask = (
            confidence_ok
            & ~marked["detector_rejects"]
            & (category == conditional_category)
            & area.notna()
            & (area >= float(threshold))
        )
        marked.loc[mask, "detector_rejects"] = True
        marked.loc[mask, "detector_rejection_reason"] = "conditional_area:" + conditional_category
        marked.loc[mask, "used_default_area_threshold"] = used_default
    return marked


def _is_better_candidate(
    *,
    candidate_metrics: dict[str, float | int],
    candidate_usage: dict[str, float | int],
    candidate_config: HybridFusionConfig,
    best_metrics: dict[str, float | int],
    best_usage: dict[str, float | int],
    best_config: HybridFusionConfig,
    detector_usage_preference: float,
) -> bool:
    f1_tolerance = 0.002
    candidate_f1 = float(candidate_metrics["f1"])
    best_f1 = float(best_metrics["f1"])
    if candidate_f1 > best_f1 + f1_tolerance:
        return True
    if candidate_f1 < best_f1 - f1_tolerance:
        return False

    candidate_ratio = float(candidate_usage["ratio"])
    best_ratio = float(best_usage["ratio"])
    candidate_under_cap = candidate_ratio <= detector_usage_preference
    best_under_cap = best_ratio <= detector_usage_preference
    if candidate_under_cap != best_under_cap:
        return candidate_under_cap
    if candidate_ratio != best_ratio:
        return candidate_ratio < best_ratio
    if candidate_config.uncertainty_margin != best_config.uncertainty_margin:
        return candidate_config.uncertainty_margin < best_config.uncertainty_margin
    if candidate_config.detector_conf_threshold != best_config.detector_conf_threshold:
        return candidate_config.detector_conf_threshold > best_config.detector_conf_threshold
    return candidate_f1 > best_f1


def _ensure_classifier_frame(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if {"image_id", "classifier_score"}.issubset(df.columns):
        out = df.copy()
        out["image_id"] = out["image_id"].map(normalize_image_id)
        if "classifier_target" not in out:
            out["classifier_target"] = (pd.to_numeric(out["classifier_score"]) >= threshold).astype(int)
        return out.sort_values("image_id").reset_index(drop=True)
    return normalize_classifier_predictions(df, threshold, require_label="y_true" in df.columns)


def _ensure_detector_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["image_id", "detector_confidence", "defect_category", "defect_area"])
    if {"image_id", "detector_confidence"}.issubset(df.columns):
        out = df.copy()
        out["image_id"] = out["image_id"].map(normalize_image_id)
        if "defect_category" not in out:
            out["defect_category"] = ""
        if "defect_area" not in out:
            out["defect_area"] = pd.NA
        if "detector_binary_target" not in out:
            out["detector_binary_target"] = pd.NA
        return out.sort_values("image_id").reset_index(drop=True)
    return normalize_detector_predictions(df)


def _find_column(df: pd.DataFrame, aliases: list[str], *, required: bool = True) -> str | None:
    lookup = {column.lower(): column for column in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise HybridFusionError(f"Missing required columns; expected one of: {', '.join(aliases)}")
    return None


def _validate_unit_interval(name: str, value: float) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HybridFusionError(f"{name} must be numeric") from exc
    if not 0.0 <= numeric <= 1.0:
        raise HybridFusionError(f"{name} must be between 0.0 and 1.0")


def _validate_nonnegative(name: str, value: float) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HybridFusionError(f"{name} must be numeric") from exc
    if numeric < 0.0:
        raise HybridFusionError(f"{name} must be non-negative")


def _default_threshold_candidates(base: float) -> list[float]:
    return sorted({min(0.99, max(0.01, round(base + offset / 100, 4))) for offset in range(-10, 11)})


def _area_candidate_sets(candidates: Optional[dict[str, Iterable[float]]]) -> list[dict[str, float]]:
    if not candidates:
        return [{}]
    keys = sorted(candidates)
    values = [list(candidates[key]) for key in keys]
    return [{key: float(value) for key, value in zip(keys, combo)} for combo in itertools.product(*values)]
