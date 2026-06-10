"""V2.1 validation error intelligence for the locked V2B baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score


REQUIRED_BASELINE_KEYS = {
    "validation_predictions",
    "validation_labels",
    "threshold",
    "metrics",
}
OPTIONAL_BASELINE_KEYS = {"checkpoint", "submission"}
OPTIONAL_EVIDENCE_KEYS = {"detector", "image_quality"}
FORBIDDEN_ANALYSIS_FLAGS = {"allow_test_labels", "generate_submission", "train_model"}
AUDIT_COLUMNS = [
    "image_id",
    "true_label",
    "v2b_probability",
    "v2b_prediction",
    "error_type",
    "threshold",
    "threshold_distance",
    "is_near_threshold",
    "is_high_confidence",
    "is_over_rejected_reusable",
    "detector_category",
    "detector_confidence",
    "detector_bbox",
    "detector_area",
    "detector_evidence_status",
    "brightness",
    "blur",
    "crop_size",
    "crop_confidence",
    "image_quality_status",
]
REQUIRED_OUTPUT_FILENAMES = {
    "audit": ("audit", "v2b_validation_error_audit.csv"),
    "summary": ("reports", "v2b_error_intelligence_summary.json"),
    "group_counts": ("reports", "v2b_error_group_counts.csv"),
    "target_distribution": ("reports", "v2b_target_distribution_report.json"),
    "optional_evidence_report": ("reports", "v2b_optional_evidence_report.json"),
    "provenance": ("reports", "v2b_audit_provenance.json"),
    "high_confidence_false_positives": ("review_groups", "high_confidence_false_positives.csv"),
    "high_confidence_false_negatives": ("review_groups", "high_confidence_false_negatives.csv"),
    "near_threshold_false_positives": ("review_groups", "near_threshold_false_positives.csv"),
    "near_threshold_false_negatives": ("review_groups", "near_threshold_false_negatives.csv"),
    "over_rejected_reusable": ("review_groups", "over_rejected_reusable.csv"),
}


class V21ErrorIntelligenceError(ValueError):
    """Raised when V2.1 error intelligence inputs are invalid."""


def load_error_intelligence_config(path: str | Path) -> dict:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise V21ErrorIntelligenceError(f"V2.1 config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise V21ErrorIntelligenceError(f"V2.1 config is not valid YAML: {config_path}") from exc
    if not isinstance(raw, dict):
        raise V21ErrorIntelligenceError("V2.1 config must be a mapping")
    baseline = raw.get("baseline")
    if not isinstance(baseline, dict):
        raise V21ErrorIntelligenceError("V2.1 config must contain a baseline mapping")
    missing = sorted(REQUIRED_BASELINE_KEYS - set(baseline))
    if missing:
        raise V21ErrorIntelligenceError("V2.1 config missing required baseline keys: " + ", ".join(missing))

    analysis = dict(raw.get("analysis") or {})
    if int(analysis.get("positive_class", 1)) != 1:
        raise V21ErrorIntelligenceError("V2.1 requires analysis.positive_class = 1")
    analysis.setdefault("near_threshold_distance", 0.05)
    analysis.setdefault("high_confidence_distance", 0.30)
    analysis.setdefault("metric_tolerance", 0.000001)
    raw["analysis"] = analysis
    validate_config_safety(raw)
    return raw


def validate_config_safety(config: dict) -> None:
    analysis = config.get("analysis") or {}
    for flag in FORBIDDEN_ANALYSIS_FLAGS:
        if bool(analysis.get(flag)):
            raise V21ErrorIntelligenceError(f"V2.1 config forbids analysis.{flag}=true")

    baseline = config.get("baseline") or {}
    label_path = str(baseline.get("validation_labels", "")).lower()
    if "test.csv" in label_path or "sample_submission" in label_path:
        raise V21ErrorIntelligenceError("baseline.validation_labels must point to validation labels, not test labels")

    flattened = json.dumps(config).lower()
    forbidden_markers = [
        '"public_score"',
        '"public_leaderboard"',
        '"kaggle_score"',
        '"sample_solution"',
    ]
    for marker in forbidden_markers:
        if marker in flattened:
            cleaned_marker = marker.replace('"', "")
            raise V21ErrorIntelligenceError(
                f"V2.1 config includes forbidden leakage marker: {cleaned_marker}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V2.1 V2B validation error audit.")
    parser.add_argument("--config", default="configs/v2_1_error_intelligence.yaml")
    args = parser.parse_args(argv)
    try:
        outputs = run_error_intelligence(args.config)
    except V21ErrorIntelligenceError as exc:
        print(f"V2.1 error intelligence failed: {exc}", file=sys.stderr)
        return 2
    print(f"V2.1 error intelligence complete: {outputs['summary']}")
    return 0


def run_error_intelligence(config_path: str | Path) -> dict[str, Path]:
    config = load_error_intelligence_config(config_path)
    baseline = config["baseline"]
    evidence_cfg = dict(config.get("evidence") or {})
    analysis = config["analysis"]

    _assert_required_inputs_exist(baseline)

    output_root = Path(config.get("output", {}).get("root", "outputs/analysis/v2_1_error_intelligence"))
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {key: output_root.joinpath(*parts) for key, parts in REQUIRED_OUTPUT_FILENAMES.items()}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    labels = _load_validation_labels(baseline["validation_labels"])
    predictions = _load_v2b_predictions(baseline["validation_predictions"])
    threshold = _load_threshold(baseline["threshold"])
    baseline_metrics = _load_baseline_metrics(baseline["metrics"])
    audit = _build_audit(labels, predictions, threshold, analysis)

    detector_path = evidence_cfg.get("detector")
    detector_summary = _load_detector_evidence(detector_path)
    quality_path = evidence_cfg.get("image_quality")
    quality_summary = _load_image_quality_evidence(quality_path)
    audit = _attach_optional_evidence(audit, detector_summary, quality_summary)

    computed_metrics = _compute_metrics(audit["true_label"], audit["v2b_prediction"])
    baseline_match = _compare_with_locked_baseline(computed_metrics, baseline_metrics, float(analysis["metric_tolerance"]))
    if not baseline_match:
        raise V21ErrorIntelligenceError("Computed validation metrics do not match the locked baseline")

    review_groups = _build_review_groups(audit)
    for key, frame in review_groups.items():
        frame.to_csv(outputs[key], index=False)

    audit = audit[AUDIT_COLUMNS].copy()
    audit.to_csv(outputs["audit"], index=False)

    group_counts = _build_group_counts(review_groups)
    group_counts.to_csv(outputs["group_counts"], index=False)

    target_distribution = _build_target_distribution_report(audit)
    optional_evidence_report = _build_optional_evidence_report(audit, detector_path, quality_path)
    provenance = _build_provenance_report(config_path, baseline, evidence_cfg, outputs)
    summary = _build_summary(
        baseline_match=baseline_match,
        computed_metrics=computed_metrics,
        baseline_metrics=baseline_metrics,
        target_distribution=target_distribution,
        review_group_counts=group_counts,
        optional_evidence_report=optional_evidence_report,
        provenance=provenance,
    )

    _write_json(outputs["summary"], summary)
    _write_json(outputs["target_distribution"], target_distribution)
    _write_json(outputs["optional_evidence_report"], optional_evidence_report)
    _write_json(outputs["provenance"], provenance)
    return outputs


def _assert_required_inputs_exist(baseline: dict) -> None:
    missing = []
    for key in sorted(REQUIRED_BASELINE_KEYS):
        path = Path(str(baseline[key]))
        if not path.exists():
            missing.append(f"{key}={path}")
    if missing:
        raise V21ErrorIntelligenceError("Missing required V2.1 baseline input: " + ", ".join(missing))


def _load_validation_labels(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    image_col = _find_column(df, ["image_id", "id", "filename", "image", "path"])
    target_col = _find_column(df, ["target", "label", "true_label", "y_true"])
    out = pd.DataFrame(
        {
            "image_id": df[image_col].map(_normalize_image_id),
            "true_label": pd.to_numeric(df[target_col], errors="raise").astype(int),
        }
    )
    if out["image_id"].duplicated().any():
        raise V21ErrorIntelligenceError("Validation labels contain duplicate image_id values")
    if not set(out["true_label"]).issubset({0, 1}):
        raise V21ErrorIntelligenceError("Validation labels must be binary")
    return out


def _load_v2b_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    image_col = _find_column(df, ["image_id", "id", "filename", "image", "path"])
    score_col = _find_column(df, ["prob_bad", "probability", "classifier_score", "score", "confidence", "positive_prob"])
    out = pd.DataFrame(
        {
            "image_id": df[image_col].map(_normalize_image_id),
            "v2b_probability": pd.to_numeric(df[score_col], errors="raise").astype(float),
        }
    )
    if out["image_id"].duplicated().any():
        raise V21ErrorIntelligenceError("V2B validation predictions contain duplicate image_id values")
    if ((out["v2b_probability"] < 0) | (out["v2b_probability"] > 1)).any():
        raise V21ErrorIntelligenceError("V2B probabilities must be between 0 and 1")
    return out


def _load_threshold(path: str | Path) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("threshold", "best_threshold", "classifier_threshold"):
        if key in payload:
            value = float(payload[key])
            if 0.0 <= value <= 1.0:
                return value
    raise V21ErrorIntelligenceError(f"Threshold JSON missing valid threshold field: {path}")


def _load_baseline_metrics(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    confusion = payload.get("confusion_counts")
    if not isinstance(confusion, dict):
        raise V21ErrorIntelligenceError(f"Baseline metrics missing confusion_counts: {path}")
    required = {"tp", "fp", "fn", "tn"}
    if not required <= set(confusion):
        raise V21ErrorIntelligenceError(f"Baseline metrics confusion_counts missing keys: {path}")
    return {
        "f1_score": float(payload.get("f1_score", payload.get("f1", 0.0))),
        "precision": float(payload.get("precision", 0.0)),
        "recall": float(payload.get("recall", 0.0)),
        "confusion_counts": {key: int(confusion[key]) for key in sorted(required)},
    }


def _build_audit(labels: pd.DataFrame, predictions: pd.DataFrame, threshold: float, analysis: dict) -> pd.DataFrame:
    audit = predictions.merge(labels, on="image_id", how="left")
    if audit["true_label"].isna().any():
        missing_ids = sorted(audit.loc[audit["true_label"].isna(), "image_id"].astype(str).tolist())
        raise V21ErrorIntelligenceError("V2B predictions are missing validation labels for: " + ", ".join(missing_ids))
    audit["true_label"] = audit["true_label"].astype(int)
    audit["threshold"] = float(threshold)
    audit["v2b_prediction"] = (audit["v2b_probability"].astype(float) >= float(threshold)).astype(int)
    audit["threshold_distance"] = (audit["v2b_probability"].astype(float) - float(threshold)).abs().round(6)
    audit["is_near_threshold"] = audit["threshold_distance"] <= float(analysis["near_threshold_distance"])
    audit["is_high_confidence"] = audit["threshold_distance"] >= float(analysis["high_confidence_distance"])
    audit["is_over_rejected_reusable"] = (audit["true_label"] == 0) & (audit["v2b_prediction"] == 1)
    audit["error_type"] = audit.apply(_error_type_from_row, axis=1)
    return audit


def _error_type_from_row(row: pd.Series) -> str:
    if int(row["true_label"]) == 1 and int(row["v2b_prediction"]) == 1:
        return "TP"
    if int(row["true_label"]) == 0 and int(row["v2b_prediction"]) == 0:
        return "TN"
    if int(row["true_label"]) == 0 and int(row["v2b_prediction"]) == 1:
        return "FP"
    return "FN"


def _load_detector_evidence(path_value: object) -> pd.DataFrame:
    columns = ["image_id", "detector_category", "detector_confidence", "detector_bbox", "detector_area"]
    if not path_value or not Path(str(path_value)).exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path_value)
    image_col = _find_column(df, ["image_id", "id", "filename", "image", "path"])
    category_col = _find_column(df, ["category", "category_name", "class_name", "label"], required=False)
    conf_col = _find_column(df, ["confidence", "conf", "score", "max_detector_conf"], required=False)
    bbox_col = _find_column(df, ["bbox", "box", "bounding_box"], required=False)
    area_col = _find_column(df, ["area", "bbox_area"], required=False)
    detector = pd.DataFrame({"image_id": df[image_col].map(_normalize_image_id)})
    detector["detector_category"] = df[category_col].astype(str) if category_col else ""
    detector["detector_confidence"] = pd.to_numeric(df[conf_col], errors="coerce") if conf_col else 0.0
    detector["detector_bbox"] = df[bbox_col].astype(str) if bbox_col else ""
    detector["detector_area"] = pd.to_numeric(df[area_col], errors="coerce") if area_col else pd.NA
    detector = detector.sort_values(["detector_confidence", "image_id"], ascending=[False, True])
    return detector.drop_duplicates("image_id", keep="first")


def _load_image_quality_evidence(path_value: object) -> pd.DataFrame:
    columns = ["image_id", "brightness", "blur", "crop_size", "crop_confidence"]
    if not path_value or not Path(str(path_value)).exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path_value)
    image_col = _find_column(df, ["image_id", "id", "filename", "image", "path"])
    out = pd.DataFrame({"image_id": df[image_col].map(_normalize_image_id)})
    out["brightness"] = _extract_numeric(df, ["brightness"], default=pd.NA)
    out["blur"] = _extract_numeric(df, ["blur"], default=pd.NA)
    out["crop_size"] = _extract_numeric(df, ["crop_size", "crop_area"], default=pd.NA)
    out["crop_confidence"] = _extract_numeric(df, ["crop_confidence", "confidence"], default=pd.NA)
    return out.drop_duplicates("image_id")


def _attach_optional_evidence(audit: pd.DataFrame, detector: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    merged = audit.merge(detector, on="image_id", how="left")
    merged["detector_evidence_status"] = merged["detector_category"].notna().map(lambda present: "present" if present else "missing")
    merged = merged.merge(quality, on="image_id", how="left")
    merged["image_quality_status"] = merged["brightness"].notna().map(lambda present: "present" if present else "missing")
    return merged


def _compute_metrics(y_true: Sequence[object], y_pred: Sequence[object]) -> dict[str, object]:
    y_true_int = [int(value) for value in y_true]
    y_pred_int = [int(value) for value in y_pred]
    tp = sum(1 for truth, pred in zip(y_true_int, y_pred_int) if truth == 1 and pred == 1)
    fp = sum(1 for truth, pred in zip(y_true_int, y_pred_int) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true_int, y_pred_int) if truth == 1 and pred == 0)
    tn = sum(1 for truth, pred in zip(y_true_int, y_pred_int) if truth == 0 and pred == 0)
    return {
        "f1_score": float(f1_score(y_true_int, y_pred_int, average="binary", pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true_int, y_pred_int, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true_int, y_pred_int, pos_label=1, zero_division=0)),
        "confusion_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _compare_with_locked_baseline(computed: dict[str, object], baseline: dict[str, object], tolerance: float) -> bool:
    scalar_keys = ["f1_score", "precision", "recall"]
    if any(abs(float(computed[key]) - float(baseline[key])) > tolerance for key in scalar_keys):
        return False
    return computed["confusion_counts"] == baseline["confusion_counts"]


def _build_review_groups(audit: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "high_confidence_false_positives": audit.loc[(audit["error_type"] == "FP") & audit["is_high_confidence"], AUDIT_COLUMNS],
        "high_confidence_false_negatives": audit.loc[(audit["error_type"] == "FN") & audit["is_high_confidence"], AUDIT_COLUMNS],
        "near_threshold_false_positives": audit.loc[(audit["error_type"] == "FP") & audit["is_near_threshold"], AUDIT_COLUMNS],
        "near_threshold_false_negatives": audit.loc[(audit["error_type"] == "FN") & audit["is_near_threshold"], AUDIT_COLUMNS],
        "over_rejected_reusable": audit.loc[audit["is_over_rejected_reusable"], AUDIT_COLUMNS],
    }


def _build_group_counts(review_groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    group_names = {
        "high_confidence_false_positives": "high_confidence_false_positive",
        "high_confidence_false_negatives": "high_confidence_false_negative",
        "near_threshold_false_positives": "near_threshold_false_positive",
        "near_threshold_false_negatives": "near_threshold_false_negative",
        "over_rejected_reusable": "over_rejected_reusable",
    }
    selection_rules = {
        "high_confidence_false_positives": "error_type=FP and abs(probability-threshold)>=0.30",
        "high_confidence_false_negatives": "error_type=FN and abs(probability-threshold)>=0.30",
        "near_threshold_false_positives": "error_type=FP and abs(probability-threshold)<=0.05",
        "near_threshold_false_negatives": "error_type=FN and abs(probability-threshold)<=0.05",
        "over_rejected_reusable": "true_label=0 and v2b_prediction=1",
    }
    return pd.DataFrame(
        [
            {"group_name": group_names[key], "row_count": int(len(frame)), "selection_rule": selection_rules[key]}
            for key, frame in review_groups.items()
        ]
    )


def _build_target_distribution_report(audit: pd.DataFrame) -> dict[str, object]:
    true_counts = audit["true_label"].value_counts().to_dict()
    pred_counts = audit["v2b_prediction"].value_counts().to_dict()
    return {
        "target_distribution": {"target_0": int(true_counts.get(0, 0)), "target_1": int(true_counts.get(1, 0))},
        "prediction_distribution": {"target_0": int(pred_counts.get(0, 0)), "target_1": int(pred_counts.get(1, 0))},
    }


def _build_optional_evidence_report(audit: pd.DataFrame, detector_path: object, quality_path: object) -> dict[str, object]:
    return {
        "optional_evidence": {
            "detector": str(detector_path or ""),
            "image_quality": str(quality_path or ""),
        },
        "missing_optional_evidence_counts": {
            "detector": int((audit["detector_evidence_status"] == "missing").sum()),
            "image_quality": int((audit["image_quality_status"] == "missing").sum()),
        },
    }


def _build_provenance_report(config_path: str | Path, baseline: dict, evidence: dict, outputs: dict[str, Path]) -> dict[str, object]:
    return {
        "config_path": str(config_path),
        "baseline": {key: str(value) for key, value in baseline.items()},
        "optional_evidence": {key: str(value) for key, value in evidence.items()},
        "outputs": {key: str(value) for key, value in outputs.items()},
    }


def _build_summary(
    *,
    baseline_match: bool,
    computed_metrics: dict[str, object],
    baseline_metrics: dict[str, object],
    target_distribution: dict[str, object],
    review_group_counts: pd.DataFrame,
    optional_evidence_report: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, object]:
    return {
        "baseline_match": baseline_match,
        "computed_metrics": computed_metrics,
        "baseline_metrics": baseline_metrics,
        "target_distribution": target_distribution["target_distribution"],
        "prediction_distribution": target_distribution["prediction_distribution"],
        "review_group_counts": {
            row["group_name"]: int(row["row_count"]) for row in review_group_counts.to_dict("records")
        },
        "missing_optional_evidence_counts": optional_evidence_report["missing_optional_evidence_counts"],
        "provenance": provenance,
        "safety_flags": {
            "used_test_labels": False,
            "used_sample_submission_labels": False,
            "used_public_leaderboard": False,
            "trained_model": False,
            "generated_submission": False,
        },
    }


def _find_column(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> str | None:
    lookup = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise V21ErrorIntelligenceError("Missing required column; expected one of: " + ", ".join(candidates))
    return None


def _extract_numeric(df: pd.DataFrame, candidates: Sequence[str], *, default: object) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series([default] * len(df), index=df.index)


def _normalize_image_id(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        raise V21ErrorIntelligenceError("image_id cannot be empty")
    return Path(text).name


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
