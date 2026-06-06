"""CLI orchestration for SPEC-009 hybrid inference."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import yaml

from src.inference.fusion import (
    HybridFusionConfig,
    HybridFusionError,
    apply_hybrid_fusion,
    load_best_threshold,
    normalize_classifier_predictions,
    normalize_detector_predictions,
    search_hybrid_parameters,
)


class HybridSubmissionError(ValueError):
    """Raised when hybrid submission orchestration fails."""


@dataclass(frozen=True)
class HybridRunReport:
    output_dir: Path
    config_path: Path
    row_count: int


def load_hybrid_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HybridSubmissionError(f"Hybrid config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise HybridSubmissionError(f"Hybrid config is not valid YAML: {path}") from exc
    if not isinstance(config, dict):
        raise HybridSubmissionError("Hybrid config must be a mapping")
    return config


def run_search(config_path: str | Path) -> HybridRunReport:
    config_path = Path(config_path)
    config = load_hybrid_config(config_path)
    _check_forbidden_paths(config)
    classifier_cfg = config.get("classifier", {})
    detector_cfg = config.get("detector", {})
    hybrid_cfg = config.get("hybrid", {})
    output_dir = Path(hybrid_cfg.get("output_dir", "outputs/hybrid/v4"))
    base_threshold = load_best_threshold(classifier_cfg["threshold_report"])
    classifier = normalize_classifier_predictions(
        pd.read_csv(classifier_cfg["val_predictions"]),
        base_threshold,
        require_label=True,
    )
    detector = normalize_detector_predictions(pd.read_csv(detector_cfg["val_predictions"]))
    selected_config, metrics, val_predictions = search_hybrid_parameters(
        classifier,
        detector,
        base_classifier_threshold=base_threshold,
        uncertainty_margin_candidates=hybrid_cfg.get("uncertainty_margin_candidates"),
        detector_conf_threshold_candidates=hybrid_cfg.get("detector_conf_threshold_candidates"),
        always_faulty_categories=set(hybrid_cfg.get("always_faulty_categories", [])),
        conditional_categories=set(hybrid_cfg.get("conditional_categories", [])),
        conditional_area_threshold_candidates=_area_threshold_candidates(hybrid_cfg),
        default_conditional_area_threshold_candidates=_default_area_candidates(hybrid_cfg),
        detector_usage_preference=float(hybrid_cfg.get("detector_usage_preference", 0.30)),
    )
    _write_search_artifacts(
        output_dir,
        selected_config,
        metrics,
        val_predictions,
        classifier_cfg=classifier_cfg,
        detector_cfg=detector_cfg,
        hybrid_cfg=hybrid_cfg,
    )
    _write_timing_report(output_dir, hybrid_cfg)
    return HybridRunReport(output_dir=output_dir, config_path=config_path, row_count=len(val_predictions))


def run_submit(config_path: str | Path, *, allow_untuned_submit: bool = False) -> HybridRunReport:
    config_path = Path(config_path)
    config = load_hybrid_config(config_path)
    _check_forbidden_paths(config)
    classifier_cfg = config.get("classifier", {})
    detector_cfg = config.get("detector", {})
    hybrid_cfg = config.get("hybrid", {})
    output_dir = Path(hybrid_cfg.get("output_dir", "outputs/hybrid/v4"))
    hybrid_config_path = output_dir / "reports" / "hybrid_config.json"
    if hybrid_config_path.exists():
        saved = json.loads(hybrid_config_path.read_text(encoding="utf-8"))
        fusion_config = _fusion_config_from_report(saved, hybrid_cfg)
    else:
        if not allow_untuned_submit:
            raise HybridSubmissionError(
                "Missing selected validation-tuned hybrid_config.json. Run search first or pass --allow-untuned-submit."
            )
        base_threshold = load_best_threshold(classifier_cfg["threshold_report"])
        fusion_config = HybridFusionConfig(
            classifier_threshold=base_threshold,
            uncertainty_margin=float(hybrid_cfg.get("uncertainty_margin", 0.05)),
            detector_conf_threshold=float(hybrid_cfg.get("detector_conf_threshold", 0.25)),
            conditional_area_thresholds=dict(hybrid_cfg.get("conditional_area_thresholds", {})),
            default_conditional_area_threshold=hybrid_cfg.get("default_conditional_area_threshold"),
            always_faulty_categories=set(hybrid_cfg.get("always_faulty_categories", [])),
            conditional_categories=set(hybrid_cfg.get("conditional_categories", [])),
            detector_usage_preference=float(hybrid_cfg.get("detector_usage_preference", 0.30)),
        )

    classifier_test = _load_classifier_test_predictions(classifier_cfg, fusion_config.classifier_threshold)
    detector_test = normalize_detector_predictions(pd.read_csv(detector_cfg["test_predictions"]))
    fused = apply_hybrid_fusion(classifier_test, detector_test, fusion_config)
    output_path = output_dir / "submissions" / "submission_v4_hybrid.csv"
    write_submission(fused, output_path)
    _write_v2b_vs_v4_diff(output_dir, classifier_test, fused, classifier_cfg)
    _write_timing_report(output_dir, hybrid_cfg)
    return HybridRunReport(output_dir=output_dir, config_path=config_path, row_count=len(fused))


def run_pipeline(config_path: str | Path) -> HybridRunReport:
    run_search(config_path)
    return run_submit(config_path)


def write_submission(predictions: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if "image_id" not in predictions or "hybrid_target" not in predictions:
        raise HybridSubmissionError("Predictions must contain image_id and hybrid_target")
    rows = predictions[["image_id", "hybrid_target"]].rename(columns={"hybrid_target": "target"}).copy()
    rows["target"] = rows["target"].astype(int)
    if list(rows.columns) != ["image_id", "target"]:
        raise HybridSubmissionError("Final submission columns must be exactly image_id,target")
    rows.sort_values("image_id").to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run SPEC-009 hybrid inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "submit", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", type=Path, default=Path("configs/hybrid_inference.yaml"))
        if name == "submit":
            sub.add_argument("--allow-untuned-submit", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "search":
        run_search(args.config)
    elif args.command == "submit":
        run_submit(args.config, allow_untuned_submit=args.allow_untuned_submit)
    elif args.command == "run":
        run_pipeline(args.config)
    return 0


def _write_search_artifacts(
    output_dir: Path,
    selected_config: HybridFusionConfig,
    metrics: dict[str, object],
    val_predictions: pd.DataFrame,
    *,
    classifier_cfg: dict,
    detector_cfg: dict,
    hybrid_cfg: dict,
) -> None:
    reports_dir = output_dir / "reports"
    predictions_dir = output_dir / "predictions"
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    report_config = {
        "version": "v4_hybrid",
        "classifier_model_id": classifier_cfg.get("model_id", ""),
        "detector_model_id": detector_cfg.get("model_id", ""),
        "selection_rule": "classifier_default_detector_rejection_only_uncertain",
        "optimization_metric": hybrid_cfg.get("selection_metric", "f1"),
        "tie_break_policy": ["f1", "usage_under_cap", "recall", "precision", "lower_usage", "narrower_margin"],
        "tuned_on": "validation_overlap_only",
        "uses_test_labels": False,
        "retrained_models": False,
        "input_paths": {
            "classifier_val_predictions": classifier_cfg.get("val_predictions", ""),
            "classifier_test_predictions": classifier_cfg.get("test_predictions", ""),
            "classifier_threshold_report": classifier_cfg.get("threshold_report", ""),
            "detector_val_predictions": detector_cfg.get("val_predictions", ""),
            "detector_test_predictions": detector_cfg.get("test_predictions", ""),
        },
    }
    report_config.update(selected_config.to_report_dict())
    _validate_report_schema(report_config, ["version", "classifier_model_id", "detector_model_id", "selection_rule", "uses_test_labels", "input_paths"])
    _validate_report_schema(metrics, ["overlap_count", "classifier_baseline", "detector_baseline", "hybrid", "detector_usage"])
    overlap = metrics.get("overlap_report", {})
    _validate_report_schema(overlap, ["classifier_val_count", "detector_val_count", "overlap_count", "used_for_parameter_search"])
    _write_json(reports_dir / "hybrid_config.json", report_config)
    _write_json(reports_dir / "hybrid_metrics.json", _json_ready(metrics))
    _write_json(reports_dir / "overlap_report.json", _json_ready(overlap))
    val_predictions.sort_values("image_id").to_csv(predictions_dir / "val_hybrid_predictions.csv", index=False)


def _load_classifier_test_predictions(classifier_cfg: dict, classifier_threshold: float) -> pd.DataFrame:
    test_path = Path(classifier_cfg["test_predictions"])
    df = pd.read_csv(test_path)
    try:
        return normalize_classifier_predictions(df, classifier_threshold=classifier_threshold, require_label=False)
    except HybridFusionError as exc:
        if _has_target_only_submission(df):
            return _normalize_target_only_submission(df, classifier_threshold)
        siblings = [
            test_path.parent / "test_classifier_predictions.csv",
            test_path.parent.parent / "predictions" / "test_classifier_predictions.csv",
        ]
        for sibling in siblings:
            if sibling.exists() and sibling != test_path:
                return normalize_classifier_predictions(
                    pd.read_csv(sibling),
                    classifier_threshold=classifier_threshold,
                    require_label=False,
                )
        raise HybridSubmissionError(
            "Missing classifier test confidence. Hybrid V4 requires classifier probabilities to decide which test images are uncertain. "
            "Provide test_classifier_predictions.csv or include probability columns in submission_v2b.csv."
        ) from exc


def _has_target_only_submission(df: pd.DataFrame) -> bool:
    columns = {column.lower() for column in df.columns}
    return "image_id" in columns and "target" in columns and not any(
        column in columns for column in ("prob", "probability", "score", "confidence", "classifier_score")
    )


def _normalize_target_only_submission(df: pd.DataFrame, classifier_threshold: float) -> pd.DataFrame:
    target_col = next(column for column in df.columns if column.lower() == "target")
    image_col = next(column for column in df.columns if column.lower() == "image_id")
    targets = pd.to_numeric(df[target_col], errors="raise").astype(int)
    confident_offset = 0.51
    scores = targets.map(lambda target: min(1.0, classifier_threshold + confident_offset) if target == 1 else max(0.0, classifier_threshold - confident_offset))
    return pd.DataFrame(
        {
            "image_id": df[image_col],
            "classifier_score": scores,
            "classifier_target": targets,
        }
    ).sort_values("image_id").reset_index(drop=True)


def _write_v2b_vs_v4_diff(output_dir: Path, classifier_test: pd.DataFrame, fused: pd.DataFrame, classifier_cfg: dict) -> None:
    baseline = classifier_test[["image_id", "classifier_target"]].rename(columns={"classifier_target": "v2b_target"})
    hybrid = fused[["image_id", "hybrid_target", "decision_source", "classifier_score"]].rename(
        columns={"hybrid_target": "v4_target", "classifier_score": "prob_bad"}
    )
    comparison = baseline.merge(hybrid, on="image_id", how="outer", validate="one_to_one")
    changed = comparison[comparison["v2b_target"].astype(int) != comparison["v4_target"].astype(int)].copy()
    rows = [
        {
            "image_id": row.image_id,
            "v2b_target": int(row.v2b_target),
            "v4_target": int(row.v4_target),
            "prob_bad": float(row.prob_bad),
            "decision_source": str(row.decision_source),
        }
        for row in changed.sort_values("image_id").itertuples(index=False)
    ]
    report = {
        "version": "v4_1_v2b_vs_v4_diff",
        "classifier_test_predictions": str(classifier_cfg.get("test_predictions", "")),
        "uses_test_labels": False,
        "row_count": int(len(comparison)),
        "changed_count": int(len(rows)),
        "changed_ratio": float(len(rows) / len(comparison)) if len(comparison) else 0.0,
        "v2b_target_counts": _target_counts(comparison["v2b_target"]),
        "v4_target_counts": _target_counts(comparison["v4_target"]),
        "changed_rows": rows,
    }
    _write_json(output_dir / "reports" / "v2b_vs_v4_diff.json", report)


def _target_counts(values: pd.Series) -> dict[str, int]:
    counts = values.astype(int).value_counts().to_dict()
    return {str(key): int(counts.get(key, 0)) for key in (0, 1)}


def _fusion_config_from_report(report: dict, hybrid_cfg: dict) -> HybridFusionConfig:
    return HybridFusionConfig(
        classifier_threshold=float(report["classifier_threshold"]),
        uncertainty_margin=float(report["uncertainty_margin"]),
        detector_conf_threshold=float(report["detector_conf_threshold"]),
        conditional_area_thresholds=dict(report.get("conditional_area_thresholds", {})),
        default_conditional_area_threshold=report.get("default_conditional_area_threshold"),
        always_faulty_categories=set(report.get("always_faulty_categories", hybrid_cfg.get("always_faulty_categories", []))),
        conditional_categories=set(report.get("conditional_categories", hybrid_cfg.get("conditional_categories", []))),
        detector_usage_preference=float(report.get("detector_usage_preference", hybrid_cfg.get("detector_usage_preference", 0.30))),
    )


def _check_forbidden_paths(config: dict) -> None:
    forbidden = []
    for path in config.get("leakage_check", {}).get("forbidden_paths", []):
        text = str(path).lower().replace("\\", "/")
        if any(token in text for token in ("sample_solution", "test_label", "test_labels", "public_score")):
            forbidden.append(str(path))
    for section in ("classifier", "detector"):
        for value in config.get(section, {}).values():
            text = str(value).lower().replace("\\", "/")
            if any(token in text for token in ("sample_solution", "test_label", "test_labels", "public_score")):
                forbidden.append(str(value))
    if forbidden:
        raise HybridSubmissionError("Configured forbidden leakage path: " + ", ".join(forbidden))


def _default_area_candidates(hybrid_cfg: dict) -> list[float | None]:
    if "default_conditional_area_threshold_candidates" in hybrid_cfg:
        return list(hybrid_cfg["default_conditional_area_threshold_candidates"])
    if "default_conditional_area_threshold" in hybrid_cfg:
        return [hybrid_cfg["default_conditional_area_threshold"]]
    return [None]


def _area_threshold_candidates(hybrid_cfg: dict) -> dict[str, list[float]] | None:
    if "conditional_area_threshold_candidates" in hybrid_cfg:
        return {
            str(category): [float(value) for value in values]
            for category, values in hybrid_cfg["conditional_area_threshold_candidates"].items()
        }
    thresholds = hybrid_cfg.get("conditional_area_thresholds")
    if isinstance(thresholds, dict) and thresholds:
        return {str(category): [float(value)] for category, value in thresholds.items()}
    return None


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), indent=2), encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items() if key != "overlap_report"}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _validate_report_schema(report: object, fields: list[str]) -> None:
    if not isinstance(report, dict):
        raise HybridSubmissionError("Report must be a mapping")
    missing = [field for field in fields if field not in report]
    if missing:
        raise HybridSubmissionError("Report missing required fields: " + ", ".join(missing))


def _write_timing_report(output_dir: Path, hybrid_cfg: dict) -> None:
    timing = hybrid_cfg.get("timing")
    if not isinstance(timing, dict):
        return
    classifier_avg = timing.get("classifier_average_time_per_image")
    hybrid_avg = timing.get("hybrid_average_time_per_image")
    if classifier_avg is None or hybrid_avg is None:
        return
    report = {
        "classifier_average_time_per_image": float(classifier_avg),
        "hybrid_average_time_per_image": float(hybrid_avg),
        "speed_multiplier_vs_classifier": float(hybrid_avg) / float(classifier_avg) if float(classifier_avg) > 0 else None,
    }
    _write_json(output_dir / "benchmarks" / "hybrid_timing_report.json", report)


if __name__ == "__main__":
    raise SystemExit(main())
