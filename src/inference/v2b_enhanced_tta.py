"""V2B-enhanced validation TTA search and selected submission export."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


SUPPORTED_TTA_MODES = {"no_tta", "hflip_tta", "gated_hflip_tta"}
SUPPORTED_TTA_MARGINS = (0.04, 0.06, 0.08)
REQUIRED_CANDIDATE_FIELDS = (
    "mode",
    "margin",
    "threshold",
    "validation_f1",
    "validation_accuracy",
    "tp",
    "fp",
    "fn",
    "tn",
    "runtime_seconds",
    "images_per_second",
    "tta_usage_count",
    "tta_usage_ratio",
    "changed_count_vs_no_tta",
    "changed_ratio_vs_no_tta",
)
LABEL_COLUMNS_FORBIDDEN_IN_TEST = {"true_label", "label", "public_score", "sample_label"}


class V2BEnhancedTtaError(ValueError):
    """Raised when V2B-enhanced TTA inputs are invalid."""


@dataclass(frozen=True)
class V2BEnhancedTtaConfig:
    dataset_root: Path
    model_checkpoint: Path
    model_config: Path
    output_root: Path
    validation_predictions: Path
    test_predictions: Optional[Path]
    model_name: str
    image_size: int
    modes: tuple[str, ...]
    margins: tuple[float, ...]
    threshold_start: float
    threshold_end: float
    threshold_step: float
    positive_class: int
    zero_division: int
    save_validation_predictions: bool
    save_selected_submission: bool
    save_all_candidates_report: bool


@dataclass(frozen=True)
class TtaCandidate:
    mode: str
    margin: float
    threshold: float
    validation_f1: float
    validation_accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int
    runtime_seconds: float
    images_per_second: float
    tta_usage_count: int
    tta_usage_ratio: float
    changed_count_vs_no_tta: int
    changed_ratio_vs_no_tta: float


@dataclass(frozen=True)
class TtaSearchResult:
    output_root: Path
    selected_candidate: TtaCandidate
    search_grid_path: Path
    candidate_summary_path: Path
    selected_validation_predictions_path: Optional[Path]
    selected_submission_path: Optional[Path]


def load_tta_config(config_path: str | Path) -> V2BEnhancedTtaConfig:
    """Load and validate a V2B-enhanced TTA YAML config."""

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    paths = raw.get("paths", {})
    model = raw.get("model", {})
    tta = raw.get("tta", {})
    threshold = raw.get("threshold_search", {})
    output = raw.get("output", {})
    output_root = Path(paths.get("output_root", "outputs/kaggle_v2b_enhanced/tta"))
    config = V2BEnhancedTtaConfig(
        dataset_root=Path(paths.get("dataset_root", "1st-krones-vision-ai-challenge")),
        model_checkpoint=Path(paths.get("model_checkpoint", "outputs/kaggle_v2b_enhanced/448/models/classifier_best.pth")),
        model_config=Path(paths.get("model_config", "configs/v2b_enhanced_448.yaml")),
        output_root=output_root,
        validation_predictions=Path(paths.get("validation_predictions", output_root / "input" / "val_predictions.csv")),
        test_predictions=(
            None
            if paths.get("test_predictions") in (None, "")
            else Path(paths.get("test_predictions", output_root / "input" / "test_predictions.csv"))
        ),
        model_name=str(model.get("model_name", "efficientnet_b1")),
        image_size=int(model.get("image_size", 448)),
        modes=tuple(str(mode) for mode in tta.get("modes", ["no_tta", "hflip_tta", "gated_hflip_tta"])),
        margins=tuple(float(margin) for margin in tta.get("margins", SUPPORTED_TTA_MARGINS)),
        threshold_start=float(threshold.get("start", 0.15)),
        threshold_end=float(threshold.get("end", 0.60)),
        threshold_step=float(threshold.get("step", 0.01)),
        positive_class=int(threshold.get("positive_class", 1)),
        zero_division=int(threshold.get("zero_division", 0)),
        save_validation_predictions=bool(output.get("save_validation_predictions", True)),
        save_selected_submission=bool(output.get("save_selected_submission", True)),
        save_all_candidates_report=bool(output.get("save_all_candidates_report", True)),
    )
    _validate_config(config)
    return config


def threshold_candidates(start: float, end: float, step: float) -> list[float]:
    """Return inclusive rounded threshold candidates."""

    if step <= 0:
        raise V2BEnhancedTtaError("threshold_search.step must be positive")
    values: list[float] = []
    value = start
    while value <= end + (step / 10.0):
        values.append(round(value, 2))
        value += step
    return values


def run_tta_search(config_path: str | Path) -> TtaSearchResult:
    """Run validation search and write configured V2B-enhanced TTA artifacts."""

    config = load_tta_config(config_path)
    validation = _read_prediction_table(config.validation_predictions, require_labels=True)
    test = _read_prediction_table(config.test_predictions, require_labels=False) if config.test_predictions is not None else None
    output_root = config.output_root
    reports_dir = output_root / "reports"
    predictions_dir = output_root / "predictions"
    submissions_dir = output_root / "submissions"
    reports_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[TtaCandidate] = []
    selected_predictions: Optional[pd.DataFrame] = None
    selected_probabilities: Optional[pd.Series] = None
    for mode in config.modes:
        active_margins = config.margins if mode == "gated_hflip_tta" else (0.0,)
        for margin in active_margins:
            for threshold in threshold_candidates(config.threshold_start, config.threshold_end, config.threshold_step):
                started = time.perf_counter()
                probabilities, usage = apply_tta_probabilities(validation, mode=mode, threshold=threshold, margin=margin)
                targets = (probabilities >= threshold).astype(int)
                baseline_targets = (validation["prob_bad"].astype(float) >= threshold).astype(int)
                elapsed = max(time.perf_counter() - started, 0.0)
                candidate = _score_candidate(
                    mode=mode,
                    margin=margin,
                    threshold=threshold,
                    true_labels=validation["true_label"].astype(int),
                    targets=targets,
                    baseline_targets=baseline_targets,
                    runtime_seconds=elapsed,
                    tta_usage_count=int(usage.sum()),
                    positive_class=config.positive_class,
                    zero_division=config.zero_division,
                )
                candidates.append(candidate)
                if _is_better_candidate(candidate, candidates[0] if len(candidates) == 1 else _select_candidate(candidates[:-1])):
                    selected_predictions = _build_validation_predictions(validation, probabilities, targets, candidate)
                    selected_probabilities = probabilities

    selected_candidate = _select_candidate(candidates)
    if selected_predictions is None or selected_probabilities is None:
        probabilities, _usage = apply_tta_probabilities(
            validation,
            mode=selected_candidate.mode,
            threshold=selected_candidate.threshold,
            margin=selected_candidate.margin,
        )
        selected_predictions = _build_validation_predictions(
            validation,
            probabilities,
            (probabilities >= selected_candidate.threshold).astype(int),
            selected_candidate,
        )

    grid_path = reports_dir / "tta_search_grid.csv"
    pd.DataFrame([asdict(candidate) for candidate in candidates], columns=REQUIRED_CANDIDATE_FIELDS).to_csv(grid_path, index=False)
    summary_path = reports_dir / "tta_candidate_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "uses_test_labels": False,
                "selection_rule": "highest validation_f1, then validation_accuracy, lower tta_usage_ratio, lower threshold",
                "candidate_count": len(candidates),
                "selected_candidate": asdict(selected_candidate),
                "input_paths": {
                    "validation_predictions": str(config.validation_predictions),
                    "test_predictions": "" if config.test_predictions is None else str(config.test_predictions),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    val_path: Optional[Path] = None
    if config.save_validation_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        val_path = predictions_dir / "val_predictions_selected.csv"
        selected_predictions.to_csv(val_path, index=False)

    submission_path: Optional[Path] = None
    if config.save_selected_submission and test is not None:
        submissions_dir.mkdir(parents=True, exist_ok=True)
        submission_path = submissions_dir / "submission_v2b_enhanced_tta.csv"
        test_probabilities, _usage = apply_tta_probabilities(
            test,
            mode=selected_candidate.mode,
            threshold=selected_candidate.threshold,
            margin=selected_candidate.margin,
        )
        _write_submission(submission_path, test["image_id"], (test_probabilities >= selected_candidate.threshold).astype(int))

    return TtaSearchResult(
        output_root=output_root,
        selected_candidate=selected_candidate,
        search_grid_path=grid_path,
        candidate_summary_path=summary_path,
        selected_validation_predictions_path=val_path,
        selected_submission_path=submission_path,
    )


def apply_tta_probabilities(
    rows: pd.DataFrame,
    *,
    mode: str,
    threshold: float,
    margin: float,
) -> tuple[pd.Series, pd.Series]:
    """Apply a supported TTA mode to prediction rows."""

    if mode not in SUPPORTED_TTA_MODES:
        raise V2BEnhancedTtaError(f"Unsupported TTA mode: {mode}")
    base = rows["prob_bad"].astype(float)
    hflip = rows["prob_bad_hflip"].astype(float) if "prob_bad_hflip" in rows.columns else base
    if mode == "no_tta":
        usage = pd.Series(False, index=rows.index)
        return base, usage
    averaged = (base + hflip) / 2.0
    if mode == "hflip_tta":
        usage = pd.Series(True, index=rows.index)
        return averaged, usage
    usage = (base - threshold).abs() <= margin
    return base.where(~usage, averaged), usage


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run V2B-enhanced TTA validation search.")
    parser.add_argument("--config", default="configs/v2b_enhanced_tta.yaml")
    args = parser.parse_args(argv)
    result = run_tta_search(args.config)
    print(
        "V2B-enhanced TTA search complete: "
        f"mode={result.selected_candidate.mode}, "
        f"margin={result.selected_candidate.margin:.2f}, "
        f"threshold={result.selected_candidate.threshold:.2f}, "
        f"f1={result.selected_candidate.validation_f1:.4f}"
    )
    return 0


def _validate_config(config: V2BEnhancedTtaConfig) -> None:
    unsupported = sorted(set(config.modes) - SUPPORTED_TTA_MODES)
    if unsupported:
        raise V2BEnhancedTtaError("Unsupported TTA modes: " + ", ".join(unsupported))
    missing_margins = sorted(set(SUPPORTED_TTA_MARGINS) - set(config.margins))
    if missing_margins:
        raise V2BEnhancedTtaError("TTA margins must include: " + ", ".join(str(value) for value in missing_margins))
    if config.threshold_start != 0.15 or config.threshold_end != 0.60:
        raise V2BEnhancedTtaError("threshold_search range must be 0.15 to 0.60")
    if config.positive_class != 1:
        raise V2BEnhancedTtaError("threshold_search.positive_class must be 1")
    if config.zero_division != 0:
        raise V2BEnhancedTtaError("threshold_search.zero_division must be 0")


def _read_prediction_table(path: Path | None, *, require_labels: bool) -> pd.DataFrame:
    if path is None:
        raise V2BEnhancedTtaError("prediction path is required")
    if not path.exists():
        raise V2BEnhancedTtaError(f"prediction CSV not found: {path}")
    rows = pd.read_csv(path)
    if rows.empty:
        raise V2BEnhancedTtaError(f"prediction CSV contains no rows: {path}")
    if "probability" in rows.columns and "prob_bad" not in rows.columns:
        rows = rows.rename(columns={"probability": "prob_bad"})
    missing = {"image_id", "prob_bad"} - set(rows.columns)
    if missing:
        raise V2BEnhancedTtaError("prediction CSV missing columns: " + ", ".join(sorted(missing)))
    if require_labels and "true_label" not in rows.columns:
        if "label" in rows.columns:
            rows = rows.rename(columns={"label": "true_label"})
        else:
            raise V2BEnhancedTtaError("validation predictions require true_label")
    if not require_labels:
        forbidden = LABEL_COLUMNS_FORBIDDEN_IN_TEST & set(rows.columns)
        if forbidden:
            raise V2BEnhancedTtaError("test labels are forbidden in TTA submission input: " + ", ".join(sorted(forbidden)))
    if not rows["prob_bad"].astype(float).between(0, 1).all():
        raise V2BEnhancedTtaError("prob_bad must be between 0 and 1")
    if "prob_bad_hflip" in rows.columns and not rows["prob_bad_hflip"].astype(float).between(0, 1).all():
        raise V2BEnhancedTtaError("prob_bad_hflip must be between 0 and 1")
    return rows


def _score_candidate(
    *,
    mode: str,
    margin: float,
    threshold: float,
    true_labels: pd.Series,
    targets: pd.Series,
    baseline_targets: pd.Series,
    runtime_seconds: float,
    tta_usage_count: int,
    positive_class: int,
    zero_division: int,
) -> TtaCandidate:
    tn, fp, fn, tp = confusion_matrix(true_labels, targets, labels=[0, 1]).ravel()
    row_count = len(targets)
    changed_count = int((targets.to_numpy() != baseline_targets.to_numpy()).sum())
    return TtaCandidate(
        mode=mode,
        margin=float(margin),
        threshold=float(threshold),
        validation_f1=float(f1_score(true_labels, targets, pos_label=positive_class, zero_division=zero_division)),
        validation_accuracy=float(accuracy_score(true_labels, targets)),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        runtime_seconds=float(runtime_seconds),
        images_per_second=float(row_count / max(runtime_seconds, 1e-9)),
        tta_usage_count=int(tta_usage_count),
        tta_usage_ratio=float(tta_usage_count / row_count),
        changed_count_vs_no_tta=changed_count,
        changed_ratio_vs_no_tta=float(changed_count / row_count),
    )


def _select_candidate(candidates: Sequence[TtaCandidate]) -> TtaCandidate:
    if not candidates:
        raise V2BEnhancedTtaError("No TTA candidates were evaluated")
    return max(
        candidates,
        key=lambda item: (
            item.validation_f1,
            item.validation_accuracy,
            -item.tta_usage_ratio,
            -item.threshold,
            item.mode == "no_tta",
        ),
    )


def _is_better_candidate(candidate: TtaCandidate, current: TtaCandidate) -> bool:
    return _select_candidate([candidate, current]) == candidate


def _build_validation_predictions(
    rows: pd.DataFrame,
    probabilities: pd.Series,
    targets: pd.Series,
    candidate: TtaCandidate,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": rows["image_id"],
            "true_label": rows["true_label"].astype(int),
            "prob_bad": probabilities.astype(float),
            "target": targets.astype(int),
            "mode": candidate.mode,
            "margin": candidate.margin,
            "threshold": candidate.threshold,
        }
    )


def _write_submission(path: Path, image_ids: pd.Series, targets: pd.Series) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "target"])
        writer.writeheader()
        for image_id, target in zip(image_ids, targets):
            writer.writerow({"image_id": image_id, "target": int(target)})


if __name__ == "__main__":
    raise SystemExit(main())
