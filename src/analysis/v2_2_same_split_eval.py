"""V2.2 same-split validation evaluation before submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.hard_examples import HardExampleError, load_hard_examples
from src.inference.predict import predict_images


DEFAULT_CONFIG_PATH = "configs/v2_2_hard_examples.yaml"
DEFAULT_V2B_VALIDATION_PREDICTIONS = (
    "artifacts/kaggle_v2b_artifacts/kaggle_v2/v2b_effnet_b1/kaggle_v2/predictions/val_classifier_predictions.csv"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/v2_2_same_split_eval")
METRIC_COLUMNS = [
    "section",
    "model",
    "row_count",
    "f1",
    "precision",
    "recall",
    "tp",
    "fp",
    "tn",
    "fn",
    "prediction_0_count",
    "prediction_1_count",
    "target_0_count",
    "target_1_count",
    "threshold",
]
ERROR_COLUMNS = [
    "image_id",
    "target",
    "v2b_probability",
    "v2b_prediction",
    "v22_probability",
    "v22_prediction",
    "v22_threshold",
    "is_hard_example",
    "hard_example_type",
]
OUTPUT_FILES = {
    "summary": ("reports", "v2_2_same_split_summary.json"),
    "metrics": ("reports", "v2_2_same_split_metrics.csv"),
    "threshold_sweep": ("reports", "v2_2_threshold_sweep_on_v2b_val.csv"),
    "v2b_wrong_v22_correct": ("errors", "v2b_wrong_v22_correct.csv"),
    "v2b_correct_v22_wrong": ("errors", "v2b_correct_v22_wrong.csv"),
    "both_wrong": ("errors", "both_wrong.csv"),
    "both_correct": ("errors", "both_correct.csv"),
}
SECTION_NAMES = [
    "all_original_v2b_validation_rows",
    "hard_example_rows_only",
    "original_v2b_validation_excluding_hard_examples",
]


class V22SameSplitEvalError(ValueError):
    """Raised when V2.2 same-split evaluation inputs are invalid."""


def load_same_split_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise V22SameSplitEvalError(f"V2.2 config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise V22SameSplitEvalError(f"V2.2 config is not valid YAML: {config_path}") from exc
    if not isinstance(payload, dict):
        raise V22SameSplitEvalError("V2.2 config must be a mapping")
    _validate_safety_config(payload)
    return payload


def run_same_split_evaluation(config_path: str | Path) -> dict[str, Path]:
    config = load_same_split_config(config_path)
    resolved = _resolve_paths(config, config_path)
    _assert_required_inputs_exist(resolved)

    output_root = resolved["analysis_output_root"]
    outputs = {key: output_root.joinpath(*parts) for key, parts in OUTPUT_FILES.items()}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    threshold = _load_threshold(resolved["v22_threshold"])
    v2b_rows = _load_v2b_validation_rows(resolved["v2b_predictions"], dataset_root=resolved["dataset_root"])
    image_paths = _resolve_image_paths(v2b_rows["image_id"].tolist(), resolved["train_images_root"])
    v22_rows = _infer_v22_predictions(
        image_paths=image_paths,
        checkpoint_path=resolved["v22_checkpoint"],
        threshold_path=resolved["v22_threshold"],
        model_name=str(config.get("model", {}).get("model_name", "efficientnet_b1")),
        image_size=int(config.get("model", {}).get("image_size", 384)),
        batch_size=int(config.get("training", {}).get("batch_size", 4)),
        device=str(config.get("training", {}).get("device", "auto")),
        threshold=threshold,
    )

    combined = v2b_rows.merge(v22_rows, on="image_id", how="inner")
    if len(combined) != len(v2b_rows):
        raise V22SameSplitEvalError("V2.2 inference did not return the same row set as V2B validation predictions")

    hard_tags = _load_hard_example_tags(config.get("hard_examples", {}))
    combined = combined.merge(hard_tags, on="image_id", how="left")
    combined["hard_example_type"] = combined["hard_example_type"].fillna("")
    combined["is_hard_example"] = combined["hard_example_type"].astype(str) != ""

    metrics_rows = _build_metrics_rows(combined, threshold)
    pd.DataFrame(metrics_rows, columns=METRIC_COLUMNS).to_csv(outputs["metrics"], index=False)

    threshold_sweep = _build_threshold_sweep(combined)
    threshold_sweep.to_csv(outputs["threshold_sweep"], index=False)

    error_splits = _build_error_splits(combined)
    for key, frame in error_splits.items():
        frame.to_csv(outputs[key], index=False)

    summary = _build_summary(
        config=config,
        resolved=resolved,
        threshold=threshold,
        metrics_rows=metrics_rows,
        combined=combined,
    )
    outputs["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return outputs


def _validate_safety_config(config: dict[str, object]) -> None:
    safety = dict(config.get("safety") or {})
    if bool(safety.get("allow_test_labels")):
        raise V22SameSplitEvalError("V2.2 same-split evaluation forbids allow_test_labels=true because test labels are not allowed")
    if bool(safety.get("generate_submission")):
        raise V22SameSplitEvalError("V2.2 same-split evaluation forbids generate_submission=true")
    if bool(safety.get("train_model")) or bool(safety.get("train_classifier")):
        raise V22SameSplitEvalError("V2.2 same-split evaluation forbids training")
    prediction_path = str((config.get("data") or {}).get("original_v2b_validation_predictions", "")).lower()
    if "sample_submission" in prediction_path or "test" in Path(prediction_path).name:
        raise V22SameSplitEvalError("V2.2 same-split evaluation cannot use test labels or submission labels")
    metrics = dict(config.get("metrics") or {})
    if int(metrics.get("positive_class", 1)) != 1:
        raise V22SameSplitEvalError("V2.2 same-split evaluation requires positive_class=1")
    if int(metrics.get("zero_division", 0)) != 0:
        raise V22SameSplitEvalError("V2.2 same-split evaluation requires zero_division=0")


def _resolve_paths(config: dict[str, object], config_path: str | Path) -> dict[str, Path]:
    cfg_path = Path(config_path)
    data_cfg = dict(config.get("data") or {})
    output_cfg = dict(config.get("output") or {})
    model_cfg = dict(config.get("model") or {})
    dataset_root = _resolve_path(cfg_path, data_cfg.get("dataset_root", ""))
    v22_output_root = _resolve_path(cfg_path, output_cfg.get("root", ""))
    analysis_output_root = _resolve_path(cfg_path, output_cfg.get("analysis_root", str(DEFAULT_OUTPUT_ROOT)))
    return {
        "dataset_root": dataset_root,
        "train_images_root": dataset_root / "train_images",
        "v2b_predictions": _resolve_path(
            cfg_path, data_cfg.get("original_v2b_validation_predictions", DEFAULT_V2B_VALIDATION_PREDICTIONS)
        ),
        "v22_output_root": v22_output_root,
        "v22_checkpoint": _resolve_path(
            cfg_path, model_cfg.get("checkpoint", str(v22_output_root / "models" / "classifier_best.pth"))
        ),
        "v22_threshold": _resolve_path(
            cfg_path, model_cfg.get("threshold_report", str(v22_output_root / "reports" / "best_threshold.json"))
        ),
        "analysis_output_root": analysis_output_root,
    }


def _resolve_path(config_path: Path, raw_value: object) -> Path:
    path = Path(str(raw_value))
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _assert_required_inputs_exist(resolved: dict[str, Path]) -> None:
    if not resolved["v2b_predictions"].exists():
        raise V22SameSplitEvalError(f"V2B validation predictions not found: {resolved['v2b_predictions']}")
    if not resolved["v22_checkpoint"].exists():
        raise V22SameSplitEvalError(f"V2.2 checkpoint not found: {resolved['v22_checkpoint']}")
    if not resolved["v22_threshold"].exists():
        raise V22SameSplitEvalError(f"V2.2 threshold report not found: {resolved['v22_threshold']}")
    if not resolved["train_images_root"].exists():
        raise V22SameSplitEvalError(f"Train images root not found: {resolved['train_images_root']}")


def _load_threshold(path: str | Path) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("threshold", "best_threshold", "classifier_threshold"):
        if key in payload:
            value = float(payload[key])
            if 0.0 <= value <= 1.0:
                return value
    raise V22SameSplitEvalError(f"V2.2 threshold report missing valid threshold: {path}")


def _load_v2b_validation_rows(path: str | Path, *, dataset_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    image_col = _find_column(frame, ["image_id", "id", "filename", "image", "path"])
    probability_col = _find_column(frame, ["probability", "prob_bad", "classifier_score", "score", "confidence"])
    target_col = _find_column(frame, ["true_label", "target", "label", "y_true"], required=False)
    prediction_col = _find_column(
        frame,
        ["predicted_label", "classifier_prediction", "prediction", "predicted_target", "target_pred"],
        required=False,
    )
    if target_col is None:
        frame = frame.merge(_load_dataset_targets(dataset_root), on=image_col, how="left")
        target_col = "target"
    normalized = pd.DataFrame({"image_id": frame[image_col].map(_normalize_image_id)})
    normalized["target"] = pd.to_numeric(frame[target_col], errors="raise").astype(int)
    normalized["v2b_probability"] = pd.to_numeric(frame[probability_col], errors="raise").astype(float)
    if prediction_col is None:
        threshold = float(_find_inline_threshold(frame))
        normalized["v2b_prediction"] = (normalized["v2b_probability"] >= threshold).astype(int)
    else:
        normalized["v2b_prediction"] = pd.to_numeric(frame[prediction_col], errors="raise").astype(int)
    if normalized["image_id"].duplicated().any():
        raise V22SameSplitEvalError("V2B validation predictions contain duplicate image_id values")
    if not set(normalized["target"]).issubset({0, 1}):
        raise V22SameSplitEvalError("V2B validation targets must be binary")
    return normalized


def _load_dataset_targets(dataset_root: Path) -> pd.DataFrame:
    train_csv = dataset_root / "train.csv"
    if not train_csv.exists():
        raise V22SameSplitEvalError("V2B validation predictions are missing labels and dataset train.csv is unavailable")
    frame = pd.read_csv(train_csv)
    image_col = _find_column(frame, ["image_id", "id", "filename", "image", "path"])
    target_col = _find_column(frame, ["target", "label", "true_label", "y_true"])
    return pd.DataFrame({image_col: frame[image_col], "target": pd.to_numeric(frame[target_col], errors="raise").astype(int)})


def _find_inline_threshold(frame: pd.DataFrame) -> float:
    threshold_col = _find_column(frame, ["threshold", "best_threshold", "classifier_threshold"], required=False)
    if threshold_col is None:
        raise V22SameSplitEvalError("V2B validation predictions are missing predicted_label and threshold columns")
    values = pd.to_numeric(frame[threshold_col], errors="raise").astype(float).unique().tolist()
    if not values:
        raise V22SameSplitEvalError("V2B validation predictions contain no threshold values")
    return float(values[0])


def _resolve_image_paths(image_ids: Sequence[str], train_images_root: Path) -> list[Path]:
    paths = [train_images_root / image_id for image_id in image_ids]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise V22SameSplitEvalError("Missing train_images files for same-split evaluation: " + ", ".join(missing[:20]))
    return paths


def _infer_v22_predictions(
    *,
    image_paths: list[Path],
    checkpoint_path: Path,
    threshold_path: Path,
    model_name: str,
    image_size: int,
    batch_size: int,
    device: str,
    threshold: float,
) -> pd.DataFrame:
    predictions = predict_images(
        image_paths,
        model_path=checkpoint_path,
        threshold_path=threshold_path,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )
    return pd.DataFrame(
        {
            "image_id": [prediction.image_id for prediction in predictions],
            "v22_probability": [float(prediction.probability) for prediction in predictions],
            "v22_prediction": [int(prediction.target) for prediction in predictions],
            "v22_threshold": [float(threshold) for _ in predictions],
        }
    )


def _load_hard_example_tags(raw_cfg: object) -> pd.DataFrame:
    cfg = dict(raw_cfg or {})
    try:
        report = load_hard_examples(
            hard_negatives=cfg.get("hard_negatives", ""),
            hard_positives=cfg.get("hard_positives", ""),
            uncertain_examples=cfg.get("uncertain_examples", ""),
        )
    except HardExampleError as exc:
        raise V22SameSplitEvalError(str(exc)) from exc
    tags: dict[str, set[str]] = {}
    for image_id in report.hard_negative_image_ids:
        tags.setdefault(image_id, set()).add("hard_negative")
    for image_id in report.hard_positive_image_ids:
        tags.setdefault(image_id, set()).add("hard_positive")
    for image_id in report.uncertain_image_ids:
        tags.setdefault(image_id, set()).add("uncertain")
    return pd.DataFrame(
        {
            "image_id": list(tags),
            "hard_example_type": [",".join(sorted(tags[image_id])) for image_id in tags],
        }
    )


def _build_metrics_rows(combined: pd.DataFrame, threshold: float) -> list[dict[str, object]]:
    sections = {
        "all_original_v2b_validation_rows": combined,
        "hard_example_rows_only": combined.loc[combined["is_hard_example"]],
        "original_v2b_validation_excluding_hard_examples": combined.loc[~combined["is_hard_example"]],
    }
    rows: list[dict[str, object]] = []
    for section_name in SECTION_NAMES:
        section = sections[section_name].copy()
        for model_name, prediction_col in (("v2b", "v2b_prediction"), ("v2_2", "v22_prediction")):
            metrics = _compute_metrics(section["target"], section[prediction_col])
            rows.append(
                {
                    "section": section_name,
                    "model": model_name,
                    "row_count": int(len(section)),
                    "f1": float(metrics["f1"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "tp": int(metrics["tp"]),
                    "fp": int(metrics["fp"]),
                    "tn": int(metrics["tn"]),
                    "fn": int(metrics["fn"]),
                    "prediction_0_count": int(metrics["prediction_0_count"]),
                    "prediction_1_count": int(metrics["prediction_1_count"]),
                    "target_0_count": int(metrics["target_0_count"]),
                    "target_1_count": int(metrics["target_1_count"]),
                    "threshold": float(threshold),
                }
            )
    return rows


def _compute_metrics(y_true: Sequence[object], y_pred: Sequence[object]) -> dict[str, object]:
    truths = [int(value) for value in y_true]
    predictions = [int(value) for value in y_pred]
    tp = sum(1 for truth, pred in zip(truths, predictions) if truth == 1 and pred == 1)
    fp = sum(1 for truth, pred in zip(truths, predictions) if truth == 0 and pred == 1)
    tn = sum(1 for truth, pred in zip(truths, predictions) if truth == 0 and pred == 0)
    fn = sum(1 for truth, pred in zip(truths, predictions) if truth == 1 and pred == 0)
    return {
        "f1": float(f1_score(truths, predictions, average="binary", pos_label=1, zero_division=0)),
        "precision": float(precision_score(truths, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(truths, predictions, pos_label=1, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "prediction_0_count": sum(1 for value in predictions if value == 0),
        "prediction_1_count": sum(1 for value in predictions if value == 1),
        "target_0_count": sum(1 for value in truths if value == 0),
        "target_1_count": sum(1 for value in truths if value == 1),
    }


def _build_threshold_sweep(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_true = combined["target"].astype(int)
    probabilities = combined["v22_probability"].astype(float)
    for idx in range(1, 100):
        threshold = round(idx / 100, 2)
        predicted = (probabilities >= threshold).astype(int)
        metrics = _compute_metrics(y_true, predicted)
        rows.append(
            {
                "threshold": threshold,
                "f1": float(metrics["f1"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "tn": int(metrics["tn"]),
                "fn": int(metrics["fn"]),
                "prediction_0_count": int(metrics["prediction_0_count"]),
                "prediction_1_count": int(metrics["prediction_1_count"]),
            }
        )
    return pd.DataFrame(rows)


def _build_error_splits(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    v2b_correct = combined["v2b_prediction"].astype(int) == combined["target"].astype(int)
    v22_correct = combined["v22_prediction"].astype(int) == combined["target"].astype(int)
    return {
        "v2b_wrong_v22_correct": combined.loc[(~v2b_correct) & v22_correct, ERROR_COLUMNS].reset_index(drop=True),
        "v2b_correct_v22_wrong": combined.loc[v2b_correct & (~v22_correct), ERROR_COLUMNS].reset_index(drop=True),
        "both_wrong": combined.loc[(~v2b_correct) & (~v22_correct), ERROR_COLUMNS].reset_index(drop=True),
        "both_correct": combined.loc[v2b_correct & v22_correct, ERROR_COLUMNS].reset_index(drop=True),
    }


def _build_summary(
    *,
    config: dict[str, object],
    resolved: dict[str, Path],
    threshold: float,
    metrics_rows: list[dict[str, object]],
    combined: pd.DataFrame,
) -> dict[str, object]:
    metrics_by_section: dict[str, dict[str, dict[str, object]]] = {}
    for row in metrics_rows:
        metrics_by_section.setdefault(str(row["section"]), {})[str(row["model"])] = row
    all_v2b = metrics_by_section["all_original_v2b_validation_rows"]["v2b"]
    all_v22 = metrics_by_section["all_original_v2b_validation_rows"]["v2_2"]
    recommended_decision = (
        "same_split_pass_pending_manual_review"
        if float(all_v22["f1"]) >= float(all_v2b["f1"])
        else "hold_submission_v22_underperforms_v2b_on_locked_split"
    )
    return {
        "evaluation_mode": "validation_only_same_split",
        "used_test_labels": False,
        "trained_model": False,
        "submission_created": False,
        "v2b_prediction_file": str(resolved["v2b_predictions"]),
        "v22_checkpoint": str(resolved["v22_checkpoint"]),
        "v22_threshold": float(threshold),
        "hard_example_files": {
            "hard_negatives": str((config.get("hard_examples") or {}).get("hard_negatives", "")),
            "hard_positives": str((config.get("hard_examples") or {}).get("hard_positives", "")),
            "uncertain_examples": str((config.get("hard_examples") or {}).get("uncertain_examples", "")),
        },
        "sections": metrics_by_section,
        "recommended_decision": recommended_decision,
        "row_count": int(len(combined)),
    }


def _find_column(frame: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> str | None:
    lookup = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise V22SameSplitEvalError("Missing required column; expected one of: " + ", ".join(candidates))
    return None


def _normalize_image_id(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        raise V22SameSplitEvalError("image_id cannot be empty")
    return Path(text).name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V2.2 same-split validation evaluation.")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 1
    try:
        outputs = run_same_split_evaluation(args.config)
    except V22SameSplitEvalError as exc:
        print(f"V2.2 same-split evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(f"V2.2 same-split evaluation complete: {outputs['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
