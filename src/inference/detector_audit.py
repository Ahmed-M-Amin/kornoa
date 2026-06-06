"""V3B detector audit and calibration reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import yaml
from PIL import Image

from src.inference.fusion import compute_binary_metrics


DETAIL_COLUMNS = [
    "image_id",
    "split",
    "class_id",
    "class_name",
    "box_conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_area",
    "image_area",
    "box_area_ratio",
]


class DetectorAuditError(ValueError):
    """Raised when detector audit inputs are invalid."""


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DetectorAuditError(f"Detector audit config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise DetectorAuditError(f"Detector audit config is not valid YAML: {config_path}") from exc
    if not isinstance(raw, dict):
        raise DetectorAuditError("Detector audit config must be a mapping")
    return raw.get("detector_audit", raw)


def run_audit(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    output_dir = Path(config.get("output_dir", "outputs/detector/v3b_audit"))
    reports_dir = output_dir / "reports"
    predictions_dir = output_dir / "predictions"
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    val_df = _load_predictions(config["val_predictions"], require_target=True)
    test_df = _load_predictions(config["test_predictions"], require_target=False)
    threshold_report = _load_threshold_report(config.get("threshold_report"))
    current_threshold = float(threshold_report.get("best_threshold", 0.01))

    grid = build_threshold_grid(val_df)
    grid_path = reports_dir / "v3_threshold_grid.csv"
    grid.to_csv(grid_path, index=False)

    calibration = build_calibration_report(val_df, test_df, current_threshold, threshold_report)
    calibration_path = reports_dir / "v3_detector_calibration_report.json"
    _write_json(calibration_path, calibration)

    category_names = _load_category_names(config.get("category_mapping"))
    val_detail_path = predictions_dir / "val_detector_detailed_boxes.csv"
    test_detail_path = predictions_dir / "test_detector_detailed_boxes.csv"
    detail_status = export_detailed_boxes(config, val_df, test_df, category_names, val_detail_path, test_detail_path)

    val_details = pd.read_csv(val_detail_path)
    category_audit = build_category_area_rule_audit(val_df, grid, val_details)
    category_audit.update(detail_status)
    category_audit_path = reports_dir / "category_area_rule_audit.json"
    _write_json(category_audit_path, category_audit)

    return {
        "calibration_report": calibration_path,
        "threshold_grid": grid_path,
        "val_detailed_boxes": val_detail_path,
        "test_detailed_boxes": test_detail_path,
        "category_area_rule_audit": category_audit_path,
    }


def build_threshold_grid(val_df: pd.DataFrame) -> pd.DataFrame:
    thresholds = [round(value / 100, 2) for value in range(1, 100)]
    rows = []
    y_true = val_df["target"].astype(int)
    confidence = val_df["max_detector_conf"].astype(float)
    for threshold in thresholds:
        prediction = (confidence >= threshold).astype(int)
        metrics = compute_binary_metrics(y_true, prediction)
        positive_count = int(prediction.sum())
        rows.append(
            {
                "threshold": threshold,
                "prediction_positive_count": positive_count,
                "prediction_positive_ratio": positive_count / len(prediction) if len(prediction) else 0.0,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
            }
        )
    return pd.DataFrame(rows)


def build_calibration_report(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    current_threshold: float,
    threshold_report: dict,
) -> dict[str, object]:
    y_true = val_df["target"].astype(int)
    all_positive = pd.Series([1] * len(val_df))
    current_prediction = (val_df["max_detector_conf"].astype(float) >= current_threshold).astype(int)
    all_positive_metrics = compute_binary_metrics(y_true, all_positive)
    current_metrics = compute_binary_metrics(y_true, current_prediction)
    return {
        "all_positive_f1_baseline": all_positive_metrics["f1"],
        "current_detector_f1": current_metrics["f1"],
        "current_threshold": current_threshold,
        "threshold_reported_f1": threshold_report.get("best_f1"),
        "val_target_distribution": _distribution(val_df["target"]),
        "val_prediction_distribution": _distribution(current_prediction),
        "test_prediction_distribution": _distribution(_test_prediction_series(test_df, current_threshold)),
        "confidence_stats_by_target": _stats_by_target(val_df, "max_detector_conf"),
        "n_boxes_stats_by_target": _stats_by_target(val_df, "n_boxes"),
        "reason_all_positive_selected": _all_positive_reason(all_positive_metrics["f1"], current_metrics["f1"], current_prediction),
    }


def export_detailed_boxes(
    config: dict,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    category_names: dict[int, str],
    val_output: Path,
    test_output: Path,
) -> dict[str, object]:
    if not bool(config.get("run_yolo_detailed", False)):
        _write_empty_detail_csv(val_output)
        _write_empty_detail_csv(test_output)
        return {
            "detailed_box_export_status": "skipped",
            "detailed_box_export_reason": "run_yolo_detailed is false; existing binary predictions do not contain per-box coordinates",
        }
    try:
        from ultralytics import YOLO
    except Exception as exc:
        _write_empty_detail_csv(val_output)
        _write_empty_detail_csv(test_output)
        return {"detailed_box_export_status": "unavailable", "detailed_box_export_reason": f"ultralytics unavailable: {exc}"}

    model_path = Path(config["model_path"])
    dataset_root = Path(config["dataset_root"])
    model = YOLO(str(model_path))
    conf = float(config.get("yolo_confidence", 0.001))
    batch_size = int(config.get("yolo_batch_size", 8))
    _predict_detail_csv(
        model,
        _resolve_image_paths(val_df["image_id"], dataset_root / "train_images"),
        "val",
        category_names,
        val_output,
        conf,
        batch_size,
    )
    _predict_detail_csv(
        model,
        _resolve_image_paths(test_df["image_id"], dataset_root / "test_images"),
        "test",
        category_names,
        test_output,
        conf,
        batch_size,
    )
    return {"detailed_box_export_status": "generated", "detailed_box_export_reason": ""}


def build_category_area_rule_audit(val_df: pd.DataFrame, grid: pd.DataFrame, val_details: pd.DataFrame) -> dict[str, object]:
    best_conf = grid.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0].to_dict()
    report: dict[str, object] = {
        "best_conf_only_rule": _json_ready(best_conf),
        "best_category_area_rule": None,
        "harmless_category_false_positive_rate": {},
        "faulty_category_true_positive_rate": {},
        "conditional_area_best_thresholds": {},
    }
    if val_details.empty:
        report["best_category_area_rule"] = {
            "status": "unavailable",
            "reason": "Detailed box predictions are empty; category/area rules cannot be evaluated.",
        }
        return report

    labels = val_df[["image_id", "target"]].copy()
    details = val_details.merge(labels, on="image_id", how="inner")
    if details.empty:
        report["best_category_area_rule"] = {
            "status": "unavailable",
            "reason": "Detailed box predictions do not overlap validation labels.",
        }
        return report

    grouped = details.groupby("class_name", dropna=False)
    harmless: dict[str, float] = {}
    faulty: dict[str, float] = {}
    for class_name, group in grouped:
        name = str(class_name)
        image_level = group.groupby("image_id").agg(target=("target", "max")).reset_index()
        fp_images = image_level[image_level["target"].astype(int) == 0]
        tp_images = image_level[image_level["target"].astype(int) == 1]
        harmless[name] = float(len(fp_images) / max(1, int((labels["target"].astype(int) == 0).sum())))
        faulty[name] = float(len(tp_images) / max(1, int((labels["target"].astype(int) == 1).sum())))

    best_rule = _search_category_area_rule(labels, details)
    report["best_category_area_rule"] = best_rule
    report["harmless_category_false_positive_rate"] = harmless
    report["faulty_category_true_positive_rate"] = faulty
    report["conditional_area_best_thresholds"] = best_rule.get("area_thresholds", {}) if isinstance(best_rule, dict) else {}
    return report


def _search_category_area_rule(labels: pd.DataFrame, details: pd.DataFrame) -> dict[str, object]:
    candidates = []
    for confidence_threshold in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]:
        for area_threshold in [0.0, 0.001, 0.005, 0.01, 0.05]:
            selected = details[
                (details["box_conf"].astype(float) >= confidence_threshold)
                & (details["box_area_ratio"].astype(float) >= area_threshold)
            ]
            positive_ids = set(selected["image_id"])
            pred = labels["image_id"].map(lambda image_id: 1 if image_id in positive_ids else 0)
            metrics = compute_binary_metrics(labels["target"].astype(int), pred)
            candidates.append(
                {
                    "confidence_threshold": confidence_threshold,
                    "area_threshold": area_threshold,
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "accuracy": metrics["accuracy"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tn": metrics["tn"],
                    "area_thresholds": {"all_categories": area_threshold},
                }
            )
    return max(candidates, key=lambda row: (float(row["f1"]), float(row["precision"]), -float(row["area_threshold"])))


def _predict_detail_csv(
    model: object,
    image_paths: list[Path],
    split: str,
    category_names: dict[int, str],
    output: Path,
    conf: float,
    batch_size: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS)
        writer.writeheader()
        for result in model.predict([str(path) for path in image_paths], conf=conf, batch=batch_size, verbose=False, stream=True):
            image_path = Path(result.path)
            width, height = _image_size(image_path)
            image_area = float(width * height)
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confs = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            for coords, box_conf, class_id in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = [float(value) for value in coords]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                writer.writerow(
                    {
                        "image_id": image_path.name,
                        "split": split,
                        "class_id": int(class_id),
                        "class_name": category_names.get(int(class_id), ""),
                        "box_conf": float(box_conf),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "box_area": area,
                        "image_area": image_area,
                        "box_area_ratio": area / image_area if image_area else 0.0,
                    }
                )


def _load_predictions(path: str | Path, *, require_target: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_id", "max_detector_conf", "n_boxes"}
    if require_target:
        required.add("target")
    missing = required - set(df.columns)
    if missing:
        raise DetectorAuditError(f"Detector predictions missing columns: {', '.join(sorted(missing))}")
    return df


def _load_threshold_report(path: object) -> dict:
    if not path:
        return {}
    threshold_path = Path(str(path))
    if not threshold_path.exists():
        return {}
    return json.loads(threshold_path.read_text(encoding="utf-8"))


def _load_category_names(path: object) -> dict[int, str]:
    if not path or not Path(str(path)).exists():
        return {}
    raw = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    items = raw.get("items", raw if isinstance(raw, list) else [])
    names = {}
    for item in items:
        if isinstance(item, dict) and "class_index" in item:
            names[int(item["class_index"])] = str(item.get("category_name", ""))
    return names


def _resolve_image_paths(image_ids: pd.Series, image_dir: Path) -> list[Path]:
    paths = [image_dir / str(image_id) for image_id in image_ids]
    return [path for path in paths if path.exists()]


def _write_empty_detail_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=DETAIL_COLUMNS).to_csv(path, index=False)


def _distribution(values: pd.Series) -> dict[str, int]:
    counts = values.astype(int).value_counts().to_dict()
    return {str(key): int(counts.get(key, 0)) for key in sorted({0, 1} | set(counts))}


def _stats_by_target(df: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for target, group in df.groupby("target"):
        series = pd.to_numeric(group[column], errors="coerce")
        stats[str(int(target))] = {
            "count": int(series.count()),
            "min": float(series.min()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "max": float(series.max()),
        }
    return stats


def _test_prediction_series(test_df: pd.DataFrame, threshold: float) -> pd.Series:
    if "prediction" in test_df:
        return test_df["prediction"].astype(int)
    if "target" in test_df:
        return test_df["target"].astype(int)
    return (test_df["max_detector_conf"].astype(float) >= threshold).astype(int)


def _all_positive_reason(all_positive_f1: float, current_f1: float, current_prediction: pd.Series) -> str:
    if int(current_prediction.sum()) == len(current_prediction):
        return "Current threshold predicts every validation image as positive, matching the all-positive baseline."
    if abs(float(all_positive_f1) - float(current_f1)) < 1e-12:
        return "Current detector F1 ties the all-positive baseline."
    return "Current detector threshold does not exactly match the all-positive baseline."


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), indent=2), encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit V3B detector calibration and category/area rules.")
    parser.add_argument("--config", type=Path, default=Path("configs/detector_v3b_audit.yaml"))
    args = parser.parse_args(argv)
    run_audit(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
