"""V1/V2B artifact audit for controlled classifier improvement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml
from sklearn.metrics import f1_score


REQUIRED_PATH_KEYS = {
    "train_csv",
    "train_annotations",
    "bottletypes",
    "v1_predictions",
    "v1_threshold",
    "v1_metrics",
    "v2b_predictions",
    "v2b_threshold",
    "v2b_metrics",
}
OPTIONAL_PATH_KEYS = {
    "dataset_root",
    "v3_detector_val_predictions",
    "v4_hybrid_metrics",
    "v4_hybrid_diff",
}
ERROR_COLUMNS = [
    "image_id",
    "y_true",
    "v1_score",
    "v1_target",
    "v2b_score",
    "v2b_target",
    "v1_correct",
    "v2b_correct",
    "bottle_type",
    "annotation_categories",
]
ERROR_SPLIT_FILENAMES = {
    "v1_wrong_v2b_correct": "v1_wrong_v2b_correct.csv",
    "v1_correct_v2b_wrong": "v1_correct_v2b_wrong.csv",
    "both_wrong": "both_wrong.csv",
    "both_correct": "both_correct.csv",
    "high_confidence_wrong": "high_confidence_wrong.csv",
    "uncertain": "uncertain.csv",
}


class ArtifactAuditError(ValueError):
    """Raised when artifact audit inputs are invalid."""


def load_audit_config(path: str | Path) -> dict:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ArtifactAuditError(f"Artifact audit config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ArtifactAuditError(f"Artifact audit config is not valid YAML: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ArtifactAuditError("Artifact audit config must be a mapping")
    validate_required_path_keys(raw)
    return raw


def validate_required_path_keys(config: dict) -> None:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ArtifactAuditError("Artifact audit config must contain a paths mapping")
    missing = sorted(REQUIRED_PATH_KEYS - set(paths))
    if missing:
        raise ArtifactAuditError("Artifact audit config missing required paths: " + ", ".join(missing))


def compute_binary_f1(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
    return float(
        f1_score(
            [int(value) for value in y_true],
            [int(value) for value in y_pred],
            average="binary",
            pos_label=1,
            zero_division=0,
        )
    )


def run_audit(config_path: str | Path) -> dict[str, Path]:
    config = load_audit_config(config_path)
    paths = config["paths"]
    analysis = config.get("analysis", {})
    positive_class = int(analysis.get("positive_class", 1))
    if positive_class != 1:
        raise ArtifactAuditError("Artifact audit requires positive_class: 1")
    _assert_required_inputs_exist(paths)

    output_root = Path(config.get("output", {}).get("root", "outputs/analysis/v1_v2b_audit"))
    reports_dir = output_root / "reports"
    errors_dir = output_root / "errors"
    reports_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_train_labels(paths["train_csv"])
    bottle_types = _load_bottle_types(paths["train_csv"], paths["bottletypes"])
    annotation_categories = _load_annotation_categories(paths["train_annotations"])
    v1_threshold = _load_threshold(paths["v1_threshold"])
    v2b_threshold = _load_threshold(paths["v2b_threshold"])
    v1 = _load_classifier_predictions(paths["v1_predictions"], labels, v1_threshold, prefix="v1")
    v2b = _load_classifier_predictions(paths["v2b_predictions"], labels, v2b_threshold, prefix="v2b")

    audit = v1.merge(v2b, on=["image_id", "y_true"], how="inner")
    if audit.empty:
        raise ArtifactAuditError("V1 and V2B validation predictions have no overlapping image_id rows")
    audit["v1_correct"] = audit["v1_target"].astype(int) == audit["y_true"].astype(int)
    audit["v2b_correct"] = audit["v2b_target"].astype(int) == audit["y_true"].astype(int)
    audit = audit.merge(bottle_types, on="image_id", how="left")
    audit["bottle_type"] = audit["bottle_type"].fillna("")
    audit["annotation_categories"] = audit["image_id"].map(lambda image_id: "|".join(annotation_categories.get(image_id, [])))

    threshold_grid = build_threshold_sensitivity(audit, analysis)
    threshold_path = reports_dir / "threshold_sensitivity.csv"
    threshold_grid.to_csv(threshold_path, index=False)

    error_paths = _write_error_splits(audit, errors_dir, v2b_threshold, analysis)
    bottle_report_path = reports_dir / "bottle_type_error_report.csv"
    _build_bottle_type_report(audit).to_csv(bottle_report_path, index=False)
    category_report_path = reports_dir / "annotation_category_error_report.csv"
    _build_annotation_category_report(audit).to_csv(category_report_path, index=False)
    detector_overlap_path = reports_dir / "v1_v2b_detector_overlap_report.json"
    _write_json(detector_overlap_path, _build_detector_overlap_report(audit, paths.get("v3_detector_val_predictions")))
    summary_path = reports_dir / "v1_v2b_error_summary.json"
    _write_json(
        summary_path,
        _build_summary(config, paths, audit, threshold_grid, v1_threshold, v2b_threshold),
    )

    return {
        "summary": summary_path,
        "threshold_sensitivity": threshold_path,
        "bottle_type_error_report": bottle_report_path,
        "annotation_category_error_report": category_report_path,
        "detector_overlap_report": detector_overlap_path,
        "errors_dir": errors_dir,
        **error_paths,
    }


def build_threshold_sensitivity(audit: pd.DataFrame, analysis: dict) -> pd.DataFrame:
    start = float(analysis.get("threshold_grid_start", 0.01))
    end = float(analysis.get("threshold_grid_end", 0.99))
    step = float(analysis.get("threshold_grid_step", 0.01))
    thresholds: list[float] = []
    value = start
    while value <= end + (step / 10):
        thresholds.append(round(value, 6))
        value += step
    rows = []
    y_true = audit["y_true"].astype(int)
    for threshold in thresholds:
        v1_pred = (audit["v1_score"].astype(float) >= threshold).astype(int)
        v2b_pred = (audit["v2b_score"].astype(float) >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "v1_f1": compute_binary_f1(y_true, v1_pred),
                "v2b_f1": compute_binary_f1(y_true, v2b_pred),
                "v1_prediction_positive_count": int(v1_pred.sum()),
                "v2b_prediction_positive_count": int(v2b_pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def _assert_required_inputs_exist(paths: dict) -> None:
    missing = [key for key in sorted(REQUIRED_PATH_KEYS) if not Path(paths[key]).exists()]
    if missing:
        details = ", ".join(f"{key}={paths[key]}" for key in missing)
        raise ArtifactAuditError("Missing required audit input: " + details)


def _load_train_labels(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    image_col = _find_column(df, ["image_id", "id", "filename", "image", "path"])
    target_col = _find_column(df, ["target", "label", "true_label", "y_true"])
    labels = pd.DataFrame(
        {
            "image_id": df[image_col].map(_normalize_image_id),
            "y_true": pd.to_numeric(df[target_col], errors="raise").astype(int),
        }
    )
    if not set(labels["y_true"]).issubset({0, 1}):
        raise ArtifactAuditError("train.csv labels must be binary")
    return labels.drop_duplicates("image_id")


def _load_classifier_predictions(path: str | Path, labels: pd.DataFrame, threshold: float, *, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    image_col = _find_column(df, ["image_id", "id", "filename", "image", "path"])
    score_col = _find_column(
        df,
        ["prob_bad", "probability", "classifier_score", "score", "confidence", "positive_prob", "pred_prob"],
    )
    pred_col = _find_column(
        df,
        ["classifier_prediction", "predicted_label", "prediction", "predicted_target", "target_pred"],
        required=False,
    )
    out = pd.DataFrame({"image_id": df[image_col].map(_normalize_image_id)})
    out[f"{prefix}_score"] = pd.to_numeric(df[score_col], errors="raise").astype(float)
    if ((out[f"{prefix}_score"] < 0) | (out[f"{prefix}_score"] > 1)).any():
        raise ArtifactAuditError(f"{prefix} scores must be between 0 and 1")
    if pred_col is None:
        out[f"{prefix}_target"] = (out[f"{prefix}_score"] >= threshold).astype(int)
    else:
        out[f"{prefix}_target"] = pd.to_numeric(df[pred_col], errors="raise").astype(int)
    out = out.merge(labels, on="image_id", how="inner")
    if out.empty:
        raise ArtifactAuditError(f"{prefix} predictions do not overlap train.csv validation labels")
    return out.drop_duplicates("image_id")


def _load_threshold(path: str | Path) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("threshold", "best_threshold", "classifier_threshold"):
        if key in payload:
            threshold = float(payload[key])
            if not 0.0 <= threshold <= 1.0:
                raise ArtifactAuditError(f"Threshold must be in [0, 1]: {path}")
            return threshold
    raise ArtifactAuditError(f"Threshold JSON missing threshold field: {path}")


def _load_bottle_types(train_csv: str | Path, bottletypes_csv: str | Path) -> pd.DataFrame:
    train = pd.read_csv(train_csv)
    image_col = _find_column(train, ["image_id", "id", "filename", "image", "path"])
    train["_audit_image_id"] = train[image_col].map(_normalize_image_id)
    bottle_col = _find_column(train, ["bottle_type", "bottle_type_id", "bottletype", "bottletype_id"], required=False)
    if bottle_col is None:
        return pd.DataFrame({"image_id": train["_audit_image_id"], "bottle_type": ""})

    bottletypes = pd.read_csv(bottletypes_csv)
    id_col = _find_column(bottletypes, ["bottle_type_id", "bottletype_id", "id"], required=False)
    name_col = _find_column(bottletypes, ["bottle_type", "bottletype", "name", "type"], required=False)
    if id_col is not None and name_col is not None:
        mapping = dict(zip(bottletypes[id_col].astype(str), bottletypes[name_col].astype(str)))
        bottle_type = train[bottle_col].astype(str).map(mapping).fillna(train[bottle_col].astype(str))
    else:
        bottle_type = train[bottle_col].astype(str)
    return pd.DataFrame({"image_id": train["_audit_image_id"], "bottle_type": bottle_type})


def _load_annotation_categories(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_image: dict[str, set[str]] = {}
    if isinstance(payload, dict) and "annotations" in payload:
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
        for annotation in payload.get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            image_id = annotation.get("image_id")
            normalized = image_lookup.get(image_id, _normalize_image_id(image_id))
            category = str(
                annotation.get(
                    "category_name",
                    annotation.get("class_name", category_lookup.get(annotation.get("category_id"), annotation.get("category_id", ""))),
                )
            )
            if category:
                by_image.setdefault(normalized, set()).add(category)
    return {image_id: sorted(categories) for image_id, categories in by_image.items()}


def _write_error_splits(audit: pd.DataFrame, errors_dir: Path, v2b_threshold: float, analysis: dict) -> dict[str, Path]:
    high_margin = float(analysis.get("high_confidence_wrong_margin", 0.20))
    uncertainty_margin = float(analysis.get("uncertainty_margin", 0.05))
    masks = {
        "v1_wrong_v2b_correct": (~audit["v1_correct"]) & audit["v2b_correct"],
        "v1_correct_v2b_wrong": audit["v1_correct"] & (~audit["v2b_correct"]),
        "both_wrong": (~audit["v1_correct"]) & (~audit["v2b_correct"]),
        "both_correct": audit["v1_correct"] & audit["v2b_correct"],
        "high_confidence_wrong": (~audit["v2b_correct"])
        & ((audit["v2b_score"].astype(float) - float(v2b_threshold)).abs() >= high_margin),
        "uncertain": (audit["v2b_score"].astype(float) - float(v2b_threshold)).abs() <= uncertainty_margin,
    }
    outputs: dict[str, Path] = {}
    for split_name, filename in ERROR_SPLIT_FILENAMES.items():
        path = errors_dir / filename
        audit.loc[masks[split_name], ERROR_COLUMNS].to_csv(path, index=False)
        outputs[split_name] = path
    return outputs


def _build_bottle_type_report(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bottle_type, group in audit.groupby("bottle_type", dropna=False):
        rows.append(_group_error_row("bottle_type", str(bottle_type), group))
    return pd.DataFrame(rows).sort_values(["v2b_error_count", "count"], ascending=[False, False])


def _build_annotation_category_report(audit: pd.DataFrame) -> pd.DataFrame:
    exploded = audit.copy()
    exploded["annotation_category"] = exploded["annotation_categories"].map(lambda value: str(value).split("|") if value else [""])
    exploded = exploded.explode("annotation_category")
    rows = []
    for category, group in exploded.groupby("annotation_category", dropna=False):
        rows.append(_group_error_row("annotation_category", str(category), group))
    return pd.DataFrame(rows).sort_values(["v2b_error_count", "count"], ascending=[False, False])


def _group_error_row(key: str, value: str, group: pd.DataFrame) -> dict[str, object]:
    return {
        key: value,
        "count": int(len(group)),
        "v1_error_count": int((~group["v1_correct"]).sum()),
        "v2b_error_count": int((~group["v2b_correct"]).sum()),
        "v2b_fixed_count": int(((~group["v1_correct"]) & group["v2b_correct"]).sum()),
        "v2b_regressed_count": int((group["v1_correct"] & (~group["v2b_correct"])).sum()),
    }


def _build_detector_overlap_report(audit: pd.DataFrame, detector_path: object) -> dict[str, object]:
    if not detector_path or not Path(str(detector_path)).exists():
        return {"status": "missing", "path": str(detector_path or ""), "v2b_wrong_with_detector_evidence": 0}
    detector = pd.read_csv(detector_path)
    image_col = _find_column(detector, ["image_id", "id", "filename", "image", "path"])
    conf_col = _find_column(detector, ["max_detector_conf", "detector_confidence", "confidence", "conf", "score"], required=False)
    pred_col = _find_column(detector, ["prediction", "detector_target", "predicted_target", "target"], required=False)
    evidence = pd.DataFrame({"image_id": detector[image_col].map(_normalize_image_id)})
    if pred_col is not None:
        evidence["detector_positive"] = pd.to_numeric(detector[pred_col], errors="coerce").fillna(0).astype(int) == 1
    elif conf_col is not None:
        evidence["detector_positive"] = pd.to_numeric(detector[conf_col], errors="coerce").fillna(0.0).astype(float) > 0
    else:
        evidence["detector_positive"] = False
    positive_ids = set(evidence.loc[evidence["detector_positive"], "image_id"])
    v2b_wrong = audit.loc[~audit["v2b_correct"], "image_id"]
    return {
        "status": "found",
        "path": str(detector_path),
        "v2b_wrong_count": int(len(v2b_wrong)),
        "v2b_wrong_with_detector_evidence": int(sum(image_id in positive_ids for image_id in v2b_wrong)),
        "v2b_wrong_with_detector_evidence_image_ids": sorted(image_id for image_id in v2b_wrong if image_id in positive_ids),
    }


def _build_summary(
    config: dict,
    paths: dict,
    audit: pd.DataFrame,
    threshold_grid: pd.DataFrame,
    v1_threshold: float,
    v2b_threshold: float,
) -> dict[str, object]:
    counts = {name: int(mask.sum()) for name, mask in _summary_masks(audit).items()}
    best_v2b = threshold_grid.sort_values(["v2b_f1", "threshold"], ascending=[False, True]).iloc[0].to_dict()
    return {
        "selection_source": "validation_only",
        "positive_class": int(config.get("analysis", {}).get("positive_class", 1)),
        "zero_division": 0,
        "row_count": int(len(audit)),
        "thresholds": {"v1": v1_threshold, "v2b": v2b_threshold},
        "f1": {
            "v1": compute_binary_f1(audit["y_true"], audit["v1_target"]),
            "v2b": compute_binary_f1(audit["y_true"], audit["v2b_target"]),
        },
        "v1_vs_v2b_counts": counts,
        "best_v2b_validation_threshold": {
            "threshold": float(best_v2b["threshold"]),
            "f1": float(best_v2b["v2b_f1"]),
        },
        "stable_v2b_threshold_range": _stable_threshold_range(threshold_grid),
        "required_inputs": {key: _path_status(paths.get(key)) for key in sorted(REQUIRED_PATH_KEYS)},
        "optional_inputs": {key: _path_status(paths.get(key)) for key in sorted(OPTIONAL_PATH_KEYS) if key in paths},
    }


def _summary_masks(audit: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "v1_wrong_v2b_correct": (~audit["v1_correct"]) & audit["v2b_correct"],
        "v1_correct_v2b_wrong": audit["v1_correct"] & (~audit["v2b_correct"]),
        "both_wrong": (~audit["v1_correct"]) & (~audit["v2b_correct"]),
        "both_correct": audit["v1_correct"] & audit["v2b_correct"],
    }


def _stable_threshold_range(threshold_grid: pd.DataFrame) -> dict[str, float]:
    best_f1 = float(threshold_grid["v2b_f1"].max())
    stable = threshold_grid[threshold_grid["v2b_f1"] >= best_f1 - 0.01]
    return {
        "min_threshold": float(stable["threshold"].min()),
        "max_threshold": float(stable["threshold"].max()),
        "best_f1": best_f1,
    }


def _path_status(path: object) -> dict[str, str]:
    text = str(path or "")
    return {"path": text, "status": "found" if text and Path(text).exists() else "missing"}


def _find_column(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> str | None:
    lookup = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ArtifactAuditError("Missing required column; expected one of: " + ", ".join(candidates))
    return None


def _normalize_image_id(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        raise ArtifactAuditError("image_id cannot be empty")
    return Path(text).name


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V1/V2B artifact audit.")
    parser.add_argument("--config", default="configs/artifact_audit.yaml")
    args = parser.parse_args(argv)
    try:
        outputs = run_audit(args.config)
    except ArtifactAuditError as exc:
        print(f"Artifact audit failed: {exc}", file=sys.stderr)
        return 2
    print(f"Artifact audit complete: {outputs['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
