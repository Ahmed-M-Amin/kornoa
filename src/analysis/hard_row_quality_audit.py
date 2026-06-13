"""Hard-row quality audit workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml


DEFAULT_CONFIG_PATH = "configs/hard_row_quality_audit.yaml"
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/hard_row_quality_audit")
PRIMARY_AUDIT_CATEGORIES = {
    "likely_correct_but_hard",
    "likely_ambiguous",
    "likely_mislabeled",
    "likely_preprocessing_issue",
    "likely_detector_evidence_issue",
}
OUTPUT_FILES = {
    "audit": ("audit", "hard_row_audit.csv"),
    "category_counts": ("reports", "hard_row_audit_category_counts.csv"),
    "summary": ("reports", "hard_row_audit_summary.json"),
    "failure_modes": ("reports", "hard_row_failure_mode_summary.json"),
}
REQUIRED_CONTEXT_COLUMNS = [
    "image_id",
    "target",
    "baseline_probability",
    "baseline_prediction",
    "candidate_probability",
    "candidate_prediction",
]
AUDIT_COLUMNS = [
    "image_id",
    "primary_audit_category",
    "audit_rationale",
    "target",
    "baseline_probability",
    "baseline_prediction",
    "candidate_probability",
    "candidate_prediction",
    "hard_example_type",
    "detector_evidence_status",
    "detector_summary",
    "crop_quality_status",
    "crop_quality_summary",
    "secondary_notes",
]
FAILURE_MODE_METADATA = {
    "likely_correct_but_hard": (
        "classifier_hard_case_review",
        "Prioritize classifier hard-example mining or architecture refinement.",
    ),
    "likely_ambiguous": (
        "ambiguity_adjudication_review",
        "Review ambiguous rows and document a stable adjudication policy.",
    ),
    "likely_mislabeled": (
        "label_quality_review",
        "Review likely label issues before any future training branch.",
    ),
    "likely_preprocessing_issue": (
        "preprocessing_pipeline_review",
        "Inspect ROI cropping and preprocessing for recoverable context loss.",
    ),
    "likely_detector_evidence_issue": (
        "detector_evidence_review",
        "Inspect detector evidence generation and ROI coverage for missed cues.",
    ),
}
HARD_TYPE_ORDER = ("hard_negative", "hard_positive", "uncertain")
FALLBACK_REQUIRED_INPUTS = ("hard_negatives", "hard_positives", "uncertain_examples", "validation_predictions")


class HardRowQualityAuditError(ValueError):
    """Raised when the hard-row audit inputs are invalid."""


def load_hard_row_audit_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise HardRowQualityAuditError(f"Hard-row audit config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise HardRowQualityAuditError(f"Hard-row audit config is not valid YAML: {config_path}") from exc
    if not isinstance(payload, dict):
        raise HardRowQualityAuditError("Hard-row audit config must be a mapping")
    _validate_safety_config(payload)
    return payload


def run_hard_row_quality_audit(config_path: str | Path) -> dict[str, Path]:
    config = load_hard_row_audit_config(config_path)
    resolved = _resolve_paths(config, config_path)
    _assert_required_inputs_exist(resolved)

    governed_rows = _load_governed_rows(resolved)
    row_context = _load_row_context(resolved)
    row_context = _filter_row_context_to_governed(governed_rows, row_context)

    detector_evidence = _load_optional_evidence(
        resolved["detector_evidence"],
        summary_column="detector_summary",
        status_column="detector_evidence_status",
    )
    image_quality = _load_optional_evidence(
        resolved["image_quality_evidence"],
        summary_column="crop_quality_summary",
        status_column="crop_quality_status",
    )

    audit = _build_audit_rows(governed_rows, row_context, detector_evidence, image_quality)
    category_counts = _build_category_counts(audit)
    summary_payload = _build_summary_payload(audit, resolved=resolved, config_path=Path(config_path))
    failure_mode_summary = _build_failure_mode_summary(
        audit,
        config_path=Path(config_path),
        resolved=resolved,
    )

    output_root = resolved["analysis_output_root"]
    outputs = {key: output_root.joinpath(*parts) for key, parts in OUTPUT_FILES.items()}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    audit.to_csv(outputs["audit"], index=False)
    category_counts.to_csv(outputs["category_counts"], index=False)
    outputs["summary"].write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    outputs["failure_modes"].write_text(json.dumps(failure_mode_summary, indent=2), encoding="utf-8")
    return outputs


def _validate_safety_config(config: dict[str, object]) -> None:
    safety = dict(config.get("safety") or {})
    if bool(safety.get("allow_test_labels")):
        raise HardRowQualityAuditError("Hard-row audit forbids test labels")
    if bool(safety.get("public_leaderboard_input")):
        raise HardRowQualityAuditError("Hard-row audit forbids public leaderboard inputs")
    if bool(safety.get("generate_submission")):
        raise HardRowQualityAuditError("Hard-row audit forbids submission creation")
    if bool(safety.get("train_model")) or bool(safety.get("train_classifier")) or bool(safety.get("train_detector")):
        raise HardRowQualityAuditError("Hard-row audit forbids training")
    if bool(safety.get("apply_relabels")):
        raise HardRowQualityAuditError("Hard-row audit forbids relabel application")
    output_cfg = dict(config.get("output") or {})
    analysis_root = str(output_cfg.get("analysis_root", "")).lower().replace("\\", "/")
    if analysis_root and "/analysis" not in analysis_root and not analysis_root.endswith("analysis"):
        raise HardRowQualityAuditError("Hard-row audit requires an analysis output root under outputs/analysis")
    if any(part in {"submission", "submissions"} for part in Path(analysis_root).parts):
        raise HardRowQualityAuditError("Hard-row audit output cannot point to submission paths")


def _resolve_paths(config: dict[str, object], config_path: str | Path) -> dict[str, Path | None]:
    cfg_path = Path(config_path).resolve()
    repo_root = _discover_repo_root(cfg_path)
    data_cfg = dict(config.get("data") or {})
    evidence_cfg = dict(config.get("evidence") or {})
    output_cfg = dict(config.get("output") or {})
    return {
        "repo_root": repo_root,
        "governed_rows": _resolve_optional_path(repo_root, data_cfg.get("governed_rows", "")),
        "row_context": _resolve_optional_path(repo_root, data_cfg.get("row_context", "")),
        "hard_negatives": _resolve_optional_path(repo_root, data_cfg.get("hard_negatives", "")),
        "hard_positives": _resolve_optional_path(repo_root, data_cfg.get("hard_positives", "")),
        "uncertain_examples": _resolve_optional_path(repo_root, data_cfg.get("uncertain_examples", "")),
        "validation_predictions": _resolve_optional_path(repo_root, data_cfg.get("validation_predictions", "")),
        "detector_evidence": _resolve_optional_path(repo_root, evidence_cfg.get("detector", "")),
        "image_quality_evidence": _resolve_optional_path(repo_root, evidence_cfg.get("image_quality", "")),
        "analysis_output_root": _resolve_path(repo_root, output_cfg.get("analysis_root", str(DEFAULT_OUTPUT_ROOT))),
    }


def _discover_repo_root(config_path: Path) -> Path:
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()


def _resolve_path(base_root: Path, raw_value: object) -> Path:
    path = Path(str(raw_value))
    if path.is_absolute():
        return path
    return (base_root / path).resolve()


def _resolve_optional_path(base_root: Path, raw_value: object) -> Path | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    return _resolve_path(base_root, text)


def _assert_required_inputs_exist(resolved: dict[str, Path | None]) -> None:
    if resolved["governed_rows"] is not None and resolved["row_context"] is not None:
        if not resolved["governed_rows"].exists():
            raise HardRowQualityAuditError(f"Missing governed hard-row source: {resolved['governed_rows']}")
        if not resolved["row_context"].exists():
            raise HardRowQualityAuditError(f"Missing row-level context source: {resolved['row_context']}")
        return
    missing = [
        key
        for key in FALLBACK_REQUIRED_INPUTS
        if resolved[key] is None or not resolved[key].exists()
    ]
    if missing:
        if "validation_predictions" in missing:
            raise HardRowQualityAuditError(f"Missing validation predictions source: {resolved['validation_predictions']}")
        raise HardRowQualityAuditError("Missing fallback hard-row audit sources: " + ", ".join(missing))


def _read_required_csv(path: Path, *, required_columns: Sequence[str], missing_message: str) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise HardRowQualityAuditError(f"{missing_message} missing required columns: {', '.join(missing)}")
    return frame


def _load_governed_rows(resolved: dict[str, Path | None]) -> pd.DataFrame:
    governed_path = resolved["governed_rows"]
    if governed_path is not None and governed_path.exists():
        frame = _read_required_csv(
            governed_path,
            required_columns=["image_id", "hard_example_type", "primary_audit_category", "audit_rationale"],
            missing_message="Governed hard-row source",
        ).copy()
        frame["image_id"] = frame["image_id"].map(_normalize_image_id)
        frame["hard_example_type"] = frame["hard_example_type"].map(lambda value: str(value).strip())
        frame["primary_audit_category"] = frame["primary_audit_category"].map(lambda value: str(value).strip())
        frame["audit_rationale"] = frame["audit_rationale"].map(lambda value: str(value).strip())
        frame = _deduplicate_explicit_governed_rows(frame)
        _validate_audit_rows(frame)
        return frame[["image_id", "hard_example_type", "primary_audit_category", "audit_rationale"]].copy()
    return _build_governed_rows_from_fallback_sources(resolved)


def _deduplicate_explicit_governed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: list[dict[str, object]] = []
    for image_id, group in frame.groupby("image_id", sort=True):
        categories = [value for value in group["primary_audit_category"].tolist() if value]
        rationales = [value for value in group["audit_rationale"].tolist() if value]
        category = categories[0] if categories else ""
        rationale = rationales[0] if rationales else ""
        grouped_rows.append(
            {
                "image_id": image_id,
                "hard_example_type": _merge_hard_example_types(group["hard_example_type"].tolist()),
                "primary_audit_category": category,
                "audit_rationale": rationale,
            }
        )
    return pd.DataFrame(grouped_rows)


def _build_governed_rows_from_fallback_sources(resolved: dict[str, Path | None]) -> pd.DataFrame:
    rows = []
    rows.extend(_load_fallback_group(resolved["hard_negatives"], hard_type="hard_negative", default_category="likely_correct_but_hard"))
    rows.extend(_load_fallback_group(resolved["hard_positives"], hard_type="hard_positive", default_category="likely_correct_but_hard"))
    rows.extend(_load_fallback_group(resolved["uncertain_examples"], hard_type="uncertain", default_category="likely_ambiguous"))

    aggregated: dict[str, dict[str, object]] = {}
    for row in rows:
        image_id = str(row["image_id"])
        entry = aggregated.setdefault(
            image_id,
            {
                "image_id": image_id,
                "hard_types": [],
                "review_groups": [],
                "notes": [],
                "reasons": [],
            },
        )
        entry["hard_types"].append(str(row["hard_type"]))
        if row["review_group"]:
            entry["review_groups"].append(str(row["review_group"]))
        if row["reviewer_note"]:
            entry["notes"].append(str(row["reviewer_note"]))
        if row["auto_triage_reason"]:
            entry["reasons"].append(str(row["auto_triage_reason"]))

    governed_rows = []
    for image_id in sorted(aggregated):
        entry = aggregated[image_id]
        hard_example_type = _merge_hard_example_types(entry["hard_types"])
        category = "likely_ambiguous" if "uncertain" in entry["hard_types"] else "likely_correct_but_hard"
        rationale = _build_fallback_rationale(
            hard_example_type=hard_example_type,
            category=category,
            review_groups=entry["review_groups"],
            notes=entry["notes"],
            reasons=entry["reasons"],
        )
        governed_rows.append(
            {
                "image_id": image_id,
                "hard_example_type": hard_example_type,
                "primary_audit_category": category,
                "audit_rationale": rationale,
            }
        )
    frame = pd.DataFrame(governed_rows)
    _validate_audit_rows(frame)
    return frame


def _load_fallback_group(path: Path | None, *, hard_type: str, default_category: str) -> list[dict[str, object]]:
    assert path is not None
    frame = _read_required_csv(
        path,
        required_columns=["image_id"],
        missing_message=f"{hard_type} source",
    ).copy()
    review_group_column = "review_group" if "review_group" in frame.columns else None
    note_column = "reviewer_note" if "reviewer_note" in frame.columns else None
    reason_column = "auto_triage_reason" if "auto_triage_reason" in frame.columns else None
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "image_id": _normalize_image_id(row["image_id"]),
                "hard_type": hard_type,
                "default_category": default_category,
                "review_group": str(row[review_group_column]).strip() if review_group_column else "",
                "reviewer_note": str(row[note_column]).strip() if note_column else "",
                "auto_triage_reason": str(row[reason_column]).strip() if reason_column else "",
            }
        )
    return rows


def _merge_hard_example_types(values: Sequence[object]) -> str:
    normalized = []
    seen = set()
    for hard_type in HARD_TYPE_ORDER:
        for raw_value in values:
            for part in str(raw_value).split(","):
                text = part.strip()
                if text == hard_type and text not in seen:
                    normalized.append(text)
                    seen.add(text)
    return ",".join(normalized)


def _build_fallback_rationale(
    *,
    hard_example_type: str,
    category: str,
    review_groups: Sequence[str],
    notes: Sequence[str],
    reasons: Sequence[str],
) -> str:
    if category == "likely_ambiguous":
        prefix = "Auto-triage marked this row as ambiguous within the governed hard-example pool."
    else:
        prefix = "Auto-triage marked this row as likely correct but hard within the governed hard-example pool."
    context = []
    if hard_example_type:
        context.append(f"hard_example_type={hard_example_type}")
    if review_groups:
        context.append(f"review_groups={','.join(sorted(set(review_groups)))}")
    if reasons:
        context.append(f"reason={sorted(set(reasons))[0]}")
    if notes:
        context.append(f"note={sorted(set(notes))[0]}")
    return prefix + (" " + "; ".join(context) if context else "")


def _load_row_context(resolved: dict[str, Path | None]) -> pd.DataFrame:
    row_context_path = resolved["row_context"]
    if row_context_path is not None and row_context_path.exists():
        frame = _read_required_csv(
            row_context_path,
            required_columns=REQUIRED_CONTEXT_COLUMNS,
            missing_message="Row-level context source",
        ).copy()
        frame["image_id"] = frame["image_id"].map(_normalize_image_id)
        return _normalize_row_context_frame(frame)

    validation_path = resolved["validation_predictions"]
    assert validation_path is not None
    frame = _read_required_csv(
        validation_path,
        required_columns=["image_id", "true_label", "probability", "predicted_label"],
        missing_message="Validation predictions source",
    ).copy()
    frame = pd.DataFrame(
        {
            "image_id": frame["image_id"].map(_normalize_image_id),
            "target": frame["true_label"],
            "baseline_probability": frame["probability"],
            "baseline_prediction": frame["predicted_label"],
            "candidate_probability": frame["probability"],
            "candidate_prediction": frame["predicted_label"],
        }
    )
    return _normalize_row_context_frame(frame)


def _normalize_row_context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame["image_id"].duplicated().any():
        raise HardRowQualityAuditError("Row-level context source contains duplicate image_id values after normalization")
    frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype(int)
    frame["baseline_probability"] = pd.to_numeric(frame["baseline_probability"], errors="raise").astype(float)
    frame["baseline_prediction"] = pd.to_numeric(frame["baseline_prediction"], errors="raise").astype(int)
    frame["candidate_probability"] = pd.to_numeric(frame["candidate_probability"], errors="raise").astype(float)
    frame["candidate_prediction"] = pd.to_numeric(frame["candidate_prediction"], errors="raise").astype(int)
    if not set(frame["target"]).issubset({0, 1}):
        raise HardRowQualityAuditError("Row-level context targets must be binary")
    if not set(frame["baseline_prediction"]).issubset({0, 1}):
        raise HardRowQualityAuditError("Baseline predictions must be binary")
    if not set(frame["candidate_prediction"]).issubset({0, 1}):
        raise HardRowQualityAuditError("Candidate predictions must be binary")
    return frame[REQUIRED_CONTEXT_COLUMNS].copy()


def _load_optional_evidence(path: Path | None, *, summary_column: str, status_column: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["image_id", status_column, summary_column, "secondary_notes"])
    frame = pd.read_csv(path, keep_default_na=False).copy()
    if "image_id" not in frame.columns:
        raise HardRowQualityAuditError(f"Optional evidence source missing required columns: image_id ({path})")
    frame["image_id"] = frame["image_id"].map(_normalize_image_id)
    if frame["image_id"].duplicated().any():
        raise HardRowQualityAuditError(f"Optional evidence source contains duplicate image_id values after normalization: {path}")
    if summary_column not in frame.columns:
        frame[summary_column] = ""
    if status_column not in frame.columns:
        frame[status_column] = frame[summary_column].map(lambda value: "present" if str(value).strip() else "missing")
    frame[summary_column] = frame[summary_column].map(lambda value: str(value).strip())
    frame[status_column] = frame.apply(
        lambda row: _normalize_evidence_status(row[status_column], row[summary_column]),
        axis=1,
    )
    if "secondary_notes" not in frame.columns:
        frame["secondary_notes"] = ""
    frame["secondary_notes"] = frame["secondary_notes"].map(lambda value: str(value).strip())
    return frame[["image_id", status_column, summary_column, "secondary_notes"]].copy()


def _normalize_image_id(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        raise HardRowQualityAuditError("image_id cannot be empty")
    return Path(text).name


def _filter_row_context_to_governed(governed_rows: pd.DataFrame, row_context: pd.DataFrame) -> pd.DataFrame:
    governed_ids = set(governed_rows["image_id"])
    context_ids = set(row_context["image_id"])
    missing_context = sorted(governed_ids - context_ids)
    if missing_context:
        raise HardRowQualityAuditError(
            "Row-level context must preserve exact governed-row coverage after normalization: missing context for "
            + ", ".join(missing_context[:10])
        )
    filtered = row_context.loc[row_context["image_id"].isin(governed_ids)].copy()
    return filtered.sort_values("image_id").reset_index(drop=True)


def _validate_audit_rows(frame: pd.DataFrame) -> None:
    category_counts = frame["primary_audit_category"].map(lambda value: len([part for part in str(value).split(",") if part.strip()]))
    if not bool((category_counts == 1).all()):
        raise HardRowQualityAuditError("Each row must have exactly one primary audit category")
    if not set(frame["primary_audit_category"]).issubset(PRIMARY_AUDIT_CATEGORIES):
        invalid = sorted(set(frame["primary_audit_category"]) - PRIMARY_AUDIT_CATEGORIES)
        raise HardRowQualityAuditError("Invalid primary audit category values: " + ", ".join(invalid))
    if not bool(frame["audit_rationale"].map(bool).all()):
        raise HardRowQualityAuditError("Each row must include a non-empty audit rationale")


def _normalize_evidence_status(value: object, summary: object) -> str:
    text = str(value).strip().lower()
    if text in {"present", "missing"}:
        return text
    return "present" if str(summary).strip() else "missing"


def _build_audit_rows(
    governed_rows: pd.DataFrame,
    row_context: pd.DataFrame,
    detector_evidence: pd.DataFrame,
    image_quality: pd.DataFrame,
) -> pd.DataFrame:
    audit = governed_rows.merge(row_context, on="image_id", how="inner")
    if len(audit) != len(governed_rows):
        raise HardRowQualityAuditError("Audit row construction lost governed rows during context join")
    audit = audit.merge(detector_evidence, on="image_id", how="left", suffixes=("", "_detector"))
    audit = audit.merge(image_quality, on="image_id", how="left", suffixes=("", "_quality"))
    audit["detector_evidence_status"] = audit["detector_evidence_status"].fillna("missing")
    audit["detector_summary"] = audit["detector_summary"].fillna("").astype(str)
    audit["crop_quality_status"] = audit["crop_quality_status"].fillna("missing")
    audit["crop_quality_summary"] = audit["crop_quality_summary"].fillna("").astype(str)
    audit["secondary_notes"] = audit["secondary_notes"].fillna("").astype(str)
    for status_column, summary_column in (
        ("detector_evidence_status", "detector_summary"),
        ("crop_quality_status", "crop_quality_summary"),
    ):
        audit[status_column] = audit.apply(
            lambda row: _normalize_evidence_status(row[status_column], row[summary_column]),
            axis=1,
        )
    _validate_audit_rows(audit)
    audit = audit[AUDIT_COLUMNS].copy()
    if len(audit) != len(governed_rows):
        raise HardRowQualityAuditError("Audit row completeness check failed")
    return audit.sort_values("image_id").reset_index(drop=True)


def _build_category_counts(audit: pd.DataFrame) -> pd.DataFrame:
    counts = (
        audit["primary_audit_category"]
        .value_counts(sort=False)
        .rename_axis("primary_audit_category")
        .reset_index(name="row_count")
    )
    if int(counts["row_count"].sum()) != len(audit):
        raise HardRowQualityAuditError("Audit category counts must sum to the full audited row count")
    return counts


def _build_summary_payload(
    audit: pd.DataFrame,
    *,
    resolved: dict[str, Path | None],
    config_path: Path,
) -> dict[str, Any]:
    return {
        "used_test_labels": False,
        "trained_model": False,
        "submission_created": False,
        "relabels_applied": False,
        "leaderboard_tuning_used": False,
        "total_audited_rows": int(len(audit)),
        "run_manifest": {
            "config_path": str(config_path),
            "analysis_output_root": str(resolved["analysis_output_root"]),
            "governed_rows_path": str(resolved["governed_rows"] or ""),
            "row_context_path": str(resolved["row_context"] or ""),
            "hard_negatives_path": str(resolved["hard_negatives"] or ""),
            "hard_positives_path": str(resolved["hard_positives"] or ""),
            "uncertain_examples_path": str(resolved["uncertain_examples"] or ""),
            "validation_predictions_path": str(resolved["validation_predictions"] or ""),
        },
    }


def _build_failure_mode_summary(
    audit: pd.DataFrame,
    *,
    config_path: Path,
    resolved: dict[str, Path | None],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    ranked_counts = audit["primary_audit_category"].value_counts()
    for rank, (category, row_count) in enumerate(sorted(ranked_counts.items(), key=lambda item: (-item[1], item[0])), start=1):
        group_name, recommended_action = FAILURE_MODE_METADATA[str(category)]
        records.append(
            {
                "rank": rank,
                "failure_mode_group": group_name,
                "row_count": int(row_count),
                "recommended_next_action": recommended_action,
                "supporting_categories": [str(category)],
            }
        )
    return {
        "used_test_labels": False,
        "trained_model": False,
        "submission_created": False,
        "relabels_applied": False,
        "leaderboard_tuning_used": False,
        "total_audited_rows": int(len(audit)),
        "run_manifest": {
            "config_path": str(config_path),
            "governed_rows_path": str(resolved["governed_rows"] or ""),
            "row_context_path": str(resolved["row_context"] or ""),
            "detector_evidence_path": str(resolved["detector_evidence"] or ""),
            "image_quality_evidence_path": str(resolved["image_quality_evidence"] or ""),
            "analysis_output_root": str(resolved["analysis_output_root"]),
        },
        "failure_modes": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the hard-row quality audit.")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 1
    try:
        outputs = run_hard_row_quality_audit(args.config)
    except HardRowQualityAuditError as exc:
        print(f"Hard-row quality audit failed: {exc}", file=sys.stderr)
        return 2
    print(f"Hard-row quality audit complete: {outputs['audit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
