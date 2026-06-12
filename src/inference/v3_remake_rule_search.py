"""V3/V4 remake detector-rule search around original V2B predictions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score


ALWAYS_REJECT_CATEGORIES = {
    "binary_detector_positive",
    "Break/Crack",
    "Circlip",
    "Contamination dark",
    "Crown cap",
    "Foil/Semitransparent",
    "Foreign object",
    "Glass shard",
    "Insect",
    "Label",
    "Liquid",
    "Mold",
    "No base visible",
    "Paint residue",
    "Straw",
    "Yeast residue",
}
NEVER_REJECT_CATEGORIES = {"No fault", "Water drop", "Embossing", "Foam residue"}
CONDITIONAL_REJECT_CATEGORIES = {
    "Air bubble",
    "Chip",
    "Contamination light",
    "Glass imperfection",
    "Scuffing",
    "Scuffing heavy",
}


class V3RemakeRuleSearchError(ValueError):
    """Raised when V3-remake rule search inputs are invalid."""


@dataclass(frozen=True)
class RemakeRuleConfig:
    v2b_threshold: float
    detector_conf_threshold: float
    uncertainty_margin: float
    conditional_area_threshold: float = 0.05
    always_reject_categories: set[str] = field(default_factory=lambda: set(ALWAYS_REJECT_CATEGORIES))
    never_reject_categories: set[str] = field(default_factory=lambda: set(NEVER_REJECT_CATEGORIES))
    conditional_reject_categories: set[str] = field(default_factory=lambda: set(CONDITIONAL_REJECT_CATEGORIES))


@dataclass(frozen=True)
class SafetyConfig:
    max_changed_ratio_without_f1_gain: float = 0.15
    max_target1_distribution_shift: float = 0.08
    max_precision_drop: float = 0.05
    minimum_clear_f1_gain: float = 0.002


@dataclass(frozen=True)
class RuleSearchConfig:
    v2b_val_predictions: Path
    v2b_test_submission: Path
    v2b_threshold: Path
    detector_val_predictions: Path
    detector_test_predictions: Path
    category_mapping: Path
    train_csv: Path
    sample_submission: Path
    output_root: Path
    detector_conf_thresholds: tuple[float, ...]
    uncertainty_margins: tuple[float, ...]
    conditional_area_threshold: float
    f1_tie_tolerance: float
    safety: SafetyConfig


@dataclass(frozen=True)
class CandidateMetrics:
    detector_conf_threshold: float
    uncertainty_margin: float
    validation_f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    changed_count_vs_v2b: int
    zero_to_one_count: int
    one_to_zero_count: int
    detector_usage_percent: float
    target1_count: int
    target0_count: int
    target1_distribution_diff: float
    fixed_false_negatives: int
    added_false_positives: int
    rejected: bool
    rejected_reasons: list[str]


@dataclass(frozen=True)
class RuleSearchResult:
    output_root: Path
    best: CandidateMetrics
    grid_path: Path
    best_config_path: Path
    validation_metrics_path: Path
    diff_path: Path
    distribution_path: Path
    submission_path: Path
    uses_test_labels: bool = False


def compute_positive_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))


def load_rule_search_config(config_path: str | Path) -> RuleSearchConfig:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    paths = raw.get("paths", {})
    search = raw.get("search", {})
    safety = raw.get("safety", {})
    return RuleSearchConfig(
        v2b_val_predictions=Path(paths["v2b_val_predictions"]),
        v2b_test_submission=Path(paths["v2b_test_submission"]),
        v2b_threshold=Path(paths["v2b_threshold"]),
        detector_val_predictions=Path(paths["detector_val_predictions"]),
        detector_test_predictions=Path(paths["detector_test_predictions"]),
        category_mapping=Path(paths.get("category_mapping", "")),
        train_csv=Path(paths.get("train_csv", "")),
        sample_submission=Path(paths["sample_submission"]),
        output_root=Path(paths.get("output_root", "outputs/hybrid/v4_remake")),
        detector_conf_thresholds=tuple(float(value) for value in search.get("detector_conf_thresholds", [0.10, 0.15, 0.20, 0.25, 0.30, 0.35])),
        uncertainty_margins=tuple(float(value) for value in search.get("uncertainty_margins", [0.02, 0.04, 0.06, 0.08, 0.10, 0.12])),
        conditional_area_threshold=float(search.get("conditional_area_threshold", 0.05)),
        f1_tie_tolerance=float(search.get("f1_tie_tolerance", 0.002)),
        safety=SafetyConfig(
            max_changed_ratio_without_f1_gain=float(safety.get("max_changed_ratio_without_f1_gain", 0.15)),
            max_target1_distribution_shift=float(safety.get("max_target1_distribution_shift", 0.08)),
            max_precision_drop=float(safety.get("max_precision_drop", 0.05)),
            minimum_clear_f1_gain=float(safety.get("minimum_clear_f1_gain", 0.002)),
        ),
    )


def run_rule_search(config_path: str | Path) -> RuleSearchResult:
    config = load_rule_search_config(config_path)
    threshold = _load_threshold(config.v2b_threshold)
    classifier_val = _read_classifier_validation(config.v2b_val_predictions, threshold)
    detector_val = read_detector_predictions_for_rules(config.detector_val_predictions, config.category_mapping)
    baseline_metrics = _compute_candidate_metrics(
        fused=classifier_val.assign(v4_remake_target=classifier_val["v2b_target"], detector_used=False),
        detector_conf_threshold=0.0,
        uncertainty_margin=0.0,
        baseline=classifier_val,
        safety=config.safety,
        baseline_f1=None,
        baseline_precision=None,
    )
    candidates: list[CandidateMetrics] = [baseline_metrics]
    fused_by_key: dict[tuple[float, float], pd.DataFrame] = {
        (baseline_metrics.detector_conf_threshold, baseline_metrics.uncertainty_margin): classifier_val.assign(
            v4_remake_target=classifier_val["v2b_target"],
            detector_used=False,
            flip_direction="",
            category="",
        )
    }
    for detector_threshold in config.detector_conf_thresholds:
        for margin in config.uncertainty_margins:
            rule_config = RemakeRuleConfig(
                v2b_threshold=threshold,
                detector_conf_threshold=detector_threshold,
                uncertainty_margin=margin,
                conditional_area_threshold=config.conditional_area_threshold,
            )
            fused = apply_rules_to_validation(classifier_val, detector_val, rule_config)
            metrics = _compute_candidate_metrics(
                fused=fused,
                detector_conf_threshold=detector_threshold,
                uncertainty_margin=margin,
                baseline=classifier_val,
                safety=config.safety,
                baseline_f1=baseline_metrics.validation_f1,
                baseline_precision=baseline_metrics.precision,
            )
            candidates.append(metrics)
            fused_by_key[(detector_threshold, margin)] = fused
    best = select_best_candidate(candidates, f1_tie_tolerance=config.f1_tie_tolerance)
    best_fused = fused_by_key[(best.detector_conf_threshold, best.uncertainty_margin)]
    return _write_outputs(
        config=config,
        best=best,
        baseline=baseline_metrics,
        candidates=candidates,
        best_fused=best_fused,
        threshold=threshold,
    )


def apply_rules_to_validation(
    classifier: pd.DataFrame,
    detector: pd.DataFrame,
    config: RemakeRuleConfig,
) -> pd.DataFrame:
    classifier_df = _ensure_classifier_frame(classifier, config.v2b_threshold)
    detector_df = _ensure_detector_frame(detector)
    evidence = _select_detector_evidence(detector_df, config)
    fused = classifier_df.merge(evidence, on="image_id", how="left")
    fused["classifier_uncertain"] = (fused["v2b_probability"] - config.v2b_threshold).abs() <= config.uncertainty_margin
    fused["detector_rejects"] = fused["detector_rejects"].map(lambda value: bool(value) if not pd.isna(value) else False)
    fused["detector_used"] = (
        (fused["v2b_target"].astype(int) == 0)
        & fused["classifier_uncertain"]
        & fused["detector_rejects"]
    )
    fused["v4_remake_target"] = fused["v2b_target"].astype(int)
    fused.loc[fused["detector_used"], "v4_remake_target"] = 1
    fused["flip_direction"] = ""
    fused.loc[(fused["v2b_target"] == 0) & (fused["v4_remake_target"] == 1), "flip_direction"] = "0_to_1"
    return fused.sort_values("image_id").reset_index(drop=True)


def select_best_candidate(candidates: Sequence[CandidateMetrics], *, f1_tie_tolerance: float = 0.002) -> CandidateMetrics:
    valid = [candidate for candidate in candidates if not candidate.rejected]
    if not valid:
        raise V3RemakeRuleSearchError("All V3-remake candidates were rejected")
    best = valid[0]
    for candidate in valid[1:]:
        if candidate.validation_f1 > best.validation_f1 + f1_tie_tolerance:
            best = candidate
            continue
        if abs(candidate.validation_f1 - best.validation_f1) <= f1_tie_tolerance:
            if _tie_break_key(candidate) > _tie_break_key(best):
                best = candidate
    return best


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run V3/V4 remake detector rule search.")
    parser.add_argument("--config", type=Path, default=Path("configs/v3_remake_rule_search.yaml"))
    args = parser.parse_args(argv)
    result = run_rule_search(args.config)
    print(
        "V3-remake rule search complete: "
        f"f1={result.best.validation_f1:.6f}, "
        f"precision={result.best.precision:.6f}, "
        f"recall={result.best.recall:.6f}, "
        f"changed={result.best.changed_count_vs_v2b}, "
        f"submission={result.submission_path}"
    )
    return 0


def _read_classifier_validation(path: Path, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    image_col = _find_column(df, ["image_id", "id", "filename", "path"])
    score_col = _find_column(df, ["prob_bad", "probability", "classifier_score", "score", "confidence"])
    label_col = _find_column(df, ["true_label", "label", "y_true", "target"])
    out = pd.DataFrame(
        {
            "image_id": df[image_col].map(_normalize_image_id),
            "v2b_probability": pd.to_numeric(df[score_col], errors="raise").astype(float),
            "y_true": pd.to_numeric(df[label_col], errors="raise").astype(int),
        }
    )
    out["v2b_target"] = (out["v2b_probability"] >= threshold).astype(int)
    return out.sort_values("image_id").reset_index(drop=True)


def read_detector_predictions_for_rules(
    path: str | Path,
    category_mapping_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read detector predictions in category-level or binary detector artifact format."""

    category_path = Path(category_mapping_path) if category_mapping_path not in (None, "") else None
    return _read_detector_predictions(Path(path), category_path)


def _read_detector_predictions(path: Path, category_mapping_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    mapping = _load_category_mapping(category_mapping_path)
    image_col = _find_column(df, ["image_id", "id", "filename", "path"])
    conf_col = _find_column(df, ["detector_confidence", "confidence", "conf", "score", "max_conf", "max_detector_conf"])
    category_col = _find_column(df, ["category", "defect_category", "category_name", "class_name", "label_name"], required=False)
    area_col = _find_column(df, ["area", "defect_area", "bbox_area", "mask_area"], required=False)
    class_id_col = _find_column(df, ["category_id", "class_id", "class"], required=False)
    binary_col = _find_column(df, ["detector_binary_target", "prediction", "predicted_target", "target"], required=False)
    if category_col is not None:
        categories = df[category_col].fillna("").astype(str)
    elif class_id_col is not None:
        categories = df[class_id_col].map(lambda value: mapping.get(str(value), str(value))).fillna("").astype(str)
    elif binary_col is not None:
        binary = pd.to_numeric(df[binary_col], errors="coerce").fillna(0).astype(int)
        categories = binary.map(lambda value: "binary_detector_positive" if value == 1 else "No fault")
    else:
        categories = pd.Series("", index=df.index)
    return pd.DataFrame(
        {
            "image_id": df[image_col].map(_normalize_image_id),
            "detector_confidence": pd.to_numeric(df[conf_col], errors="raise").astype(float),
            "category": categories,
            "area": pd.to_numeric(df[area_col], errors="coerce") if area_col is not None else pd.Series(pd.NA, index=df.index),
        }
    ).sort_values(["image_id", "detector_confidence"], ascending=[True, False]).reset_index(drop=True)


def _ensure_classifier_frame(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["image_id"] = out["image_id"].map(_normalize_image_id)
    if "v2b_probability" not in out:
        out = out.rename(columns={"probability": "v2b_probability", "prob_bad": "v2b_probability"})
    if "v2b_target" not in out:
        out["v2b_target"] = (pd.to_numeric(out["v2b_probability"]) >= threshold).astype(int)
    return out


def _ensure_detector_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["image_id"] = out["image_id"].map(_normalize_image_id)
    if "detector_confidence" not in out and "confidence" in out:
        out = out.rename(columns={"confidence": "detector_confidence"})
    if "defect_category" in out and "category" not in out:
        out = out.rename(columns={"defect_category": "category"})
    if "defect_area" in out and "area" not in out:
        out = out.rename(columns={"defect_area": "area"})
    if "area" not in out:
        out["area"] = pd.NA
    return out


def _select_detector_evidence(detector: pd.DataFrame, config: RemakeRuleConfig) -> pd.DataFrame:
    rows = detector.copy()
    rows["detector_rejects"] = rows.apply(lambda row: _detector_row_rejects(row, config), axis=1)
    rows["_confidence"] = pd.to_numeric(rows["detector_confidence"], errors="coerce").fillna(-1.0)
    rows = rows.sort_values(["image_id", "detector_rejects", "_confidence"], ascending=[True, False, False])
    return rows.groupby("image_id", sort=True).head(1).drop(columns=["_confidence"]).reset_index(drop=True)


def _detector_row_rejects(row: pd.Series, config: RemakeRuleConfig) -> bool:
    if float(row.get("detector_confidence", 0.0)) < config.detector_conf_threshold:
        return False
    category = str(row.get("category", "") or "")
    if category in config.never_reject_categories:
        return False
    if category in config.always_reject_categories:
        return True
    if category in config.conditional_reject_categories:
        area = row.get("area")
        return not pd.isna(area) and float(area) >= config.conditional_area_threshold
    return False


def _compute_candidate_metrics(
    *,
    fused: pd.DataFrame,
    detector_conf_threshold: float,
    uncertainty_margin: float,
    baseline: pd.DataFrame,
    safety: SafetyConfig,
    baseline_f1: Optional[float],
    baseline_precision: Optional[float],
) -> CandidateMetrics:
    y_true = fused["y_true"].astype(int)
    pred = fused["v4_remake_target"].astype(int)
    v2b = fused["v2b_target"].astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    precision = float(precision_score(y_true, pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, pred, pos_label=1, zero_division=0))
    validation_f1 = compute_positive_f1(y_true.tolist(), pred.tolist())
    changed = pred != v2b
    zero_to_one = (v2b == 0) & (pred == 1)
    one_to_zero = (v2b == 1) & (pred == 0)
    target1_ratio = float((pred == 1).mean()) if len(pred) else 0.0
    baseline_target1_ratio = float((baseline["v2b_target"].astype(int) == 1).mean()) if len(baseline) else 0.0
    target1_diff = abs(target1_ratio - baseline_target1_ratio)
    fixed_fn = int(((y_true == 1) & (v2b == 0) & (pred == 1)).sum())
    added_fp = int(((y_true == 0) & (v2b == 0) & (pred == 1)).sum())
    rejected_reasons = _candidate_rejection_reasons(
        validation_f1=validation_f1,
        precision=precision,
        changed_ratio=float(changed.mean()) if len(changed) else 0.0,
        target1_distribution_diff=target1_diff,
        baseline_f1=baseline_f1,
        baseline_precision=baseline_precision,
        safety=safety,
    )
    return CandidateMetrics(
        detector_conf_threshold=float(detector_conf_threshold),
        uncertainty_margin=float(uncertainty_margin),
        validation_f1=validation_f1,
        precision=precision,
        recall=recall,
        tp=tp,
        fp=fp,
        fn=fn,
        changed_count_vs_v2b=int(changed.sum()),
        zero_to_one_count=int(zero_to_one.sum()),
        one_to_zero_count=int(one_to_zero.sum()),
        detector_usage_percent=float(fused.get("detector_used", pd.Series(False, index=fused.index)).astype(bool).mean() * 100.0),
        target1_count=int((pred == 1).sum()),
        target0_count=int((pred == 0).sum()),
        target1_distribution_diff=target1_diff,
        fixed_false_negatives=fixed_fn,
        added_false_positives=added_fp,
        rejected=bool(rejected_reasons),
        rejected_reasons=rejected_reasons,
    )


def _candidate_rejection_reasons(
    *,
    validation_f1: float,
    precision: float,
    changed_ratio: float,
    target1_distribution_diff: float,
    baseline_f1: Optional[float],
    baseline_precision: Optional[float],
    safety: SafetyConfig,
) -> list[str]:
    reasons: list[str] = []
    f1_gain = 0.0 if baseline_f1 is None else validation_f1 - baseline_f1
    if changed_ratio > safety.max_changed_ratio_without_f1_gain and f1_gain < safety.minimum_clear_f1_gain:
        reasons.append("changed_ratio_without_clear_f1_gain")
    if target1_distribution_diff > safety.max_target1_distribution_shift:
        reasons.append("target1_distribution_shift")
    if baseline_precision is not None and precision < baseline_precision - safety.max_precision_drop:
        reasons.append("precision_drop")
    return reasons


def _write_outputs(
    *,
    config: RuleSearchConfig,
    best: CandidateMetrics,
    baseline: CandidateMetrics,
    candidates: Sequence[CandidateMetrics],
    best_fused: pd.DataFrame,
    threshold: float,
) -> RuleSearchResult:
    reports = config.output_root / "reports"
    submissions = config.output_root / "submissions"
    reports.mkdir(parents=True, exist_ok=True)
    submissions.mkdir(parents=True, exist_ok=True)
    grid_path = reports / "v3_remake_grid.csv"
    pd.DataFrame([_candidate_to_row(candidate) for candidate in candidates]).to_csv(grid_path, index=False)
    best_config_path = reports / "v3_remake_best_config.json"
    _write_json(
        best_config_path,
        {
            "uses_test_labels": False,
            "uses_v2b_enhanced": False,
            "v2b_threshold": threshold,
            "selected_candidate": _candidate_to_row(best),
            "selection_rule": [
                "highest_validation_f1",
                "within_0.002_lower_detector_usage",
                "fewer_changed_rows",
                "closest_target_distribution",
                "narrower_margin",
                "higher_detector_confidence",
            ],
        },
    )
    validation_metrics_path = reports / "v4_remake_validation_metrics.json"
    _write_json(validation_metrics_path, {"baseline_v2b": _candidate_to_row(baseline), "v4_remake": _candidate_to_row(best)})
    diff_path = reports / "v2b_vs_v4_remake_diff.csv"
    best_fused[["image_id", "y_true", "v2b_target", "v4_remake_target", "flip_direction", "detector_used", "category"]].to_csv(diff_path, index=False)
    distribution_path = reports / "target_distribution_report.json"
    _write_json(distribution_path, _target_distribution_report(best, baseline))
    submission_path = submissions / "submission_v4_remake.csv"
    _write_conservative_submission(config.v2b_test_submission, config.sample_submission, submission_path)
    return RuleSearchResult(
        output_root=config.output_root,
        best=best,
        grid_path=grid_path,
        best_config_path=best_config_path,
        validation_metrics_path=validation_metrics_path,
        diff_path=diff_path,
        distribution_path=distribution_path,
        submission_path=submission_path,
        uses_test_labels=False,
    )


def _write_conservative_submission(v2b_submission_path: Path, sample_submission_path: Path, output_path: Path) -> None:
    v2b = pd.read_csv(v2b_submission_path)
    forbidden = {"true_label", "label", "y_true"} & set(v2b.columns)
    if forbidden:
        raise V3RemakeRuleSearchError("test labels are forbidden in V2B test submission: " + ", ".join(sorted(forbidden)))
    if "image_id" not in v2b or "target" not in v2b:
        raise V3RemakeRuleSearchError("V2B test submission must contain image_id,target")
    sample = pd.read_csv(sample_submission_path)
    sample_ids = sample["image_id"].map(_normalize_image_id).tolist()
    v2b["image_id"] = v2b["image_id"].map(_normalize_image_id)
    by_id = dict(zip(v2b["image_id"], v2b["target"].astype(int)))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "target"])
        writer.writeheader()
        for image_id in sample_ids:
            writer.writerow({"image_id": image_id, "target": by_id.get(image_id, 0)})


def _target_distribution_report(best: CandidateMetrics, baseline: CandidateMetrics) -> dict[str, object]:
    return {
        "original_v2b": {"target_0": baseline.target0_count, "target_1": baseline.target1_count},
        "v4_remake": {"target_0": best.target0_count, "target_1": best.target1_count},
        "target1_distribution_diff": best.target1_distribution_diff,
        "changed_count_vs_v2b": best.changed_count_vs_v2b,
    }


def _candidate_to_row(candidate: CandidateMetrics) -> dict[str, object]:
    row = asdict(candidate)
    row["rejected_reasons"] = ";".join(candidate.rejected_reasons)
    return row


def _tie_break_key(candidate: CandidateMetrics) -> tuple[float, int, float, float, float]:
    return (
        -candidate.detector_usage_percent,
        -candidate.changed_count_vs_v2b,
        -candidate.target1_distribution_diff,
        -candidate.uncertainty_margin,
        candidate.detector_conf_threshold,
    )


def _load_threshold(path: Path) -> float:
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("threshold", "best_threshold", "classifier_threshold"):
        if key in raw:
            return float(raw[key])
    raise V3RemakeRuleSearchError(f"Threshold report missing threshold: {path}")


def _load_category_mapping(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists() or not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "id_to_name" in raw and isinstance(raw["id_to_name"], dict):
            return {str(key): str(value) for key, value in raw["id_to_name"].items()}
        return {str(key): str(value) for key, value in raw.items()}
    return {}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _find_column(df: pd.DataFrame, aliases: Sequence[str], *, required: bool = True) -> Optional[str]:
    lookup = {column.lower(): column for column in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise V3RemakeRuleSearchError("Missing required column; expected one of: " + ", ".join(aliases))
    return None


def _normalize_image_id(value: object) -> str:
    return Path(str(value).replace("\\", "/")).name


if __name__ == "__main__":
    raise SystemExit(main())
