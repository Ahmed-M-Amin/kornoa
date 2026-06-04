"""Offline V1 hard-example mining for SPEC-006."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.inference.predict import DEFAULT_ARTIFACT_ROOT, DEFAULT_VALIDATION_PREDICTIONS_RELATIVE, load_threshold


DEFAULT_HARD_EXAMPLES_DIR = Path("outputs/hard_examples")
DEFAULT_HARD_EXAMPLE_SUMMARY = Path("outputs/reports/hard_example_summary.json")
DEFAULT_FALSE_POSITIVES_OUTPUT = DEFAULT_HARD_EXAMPLES_DIR / "false_positives.csv"
DEFAULT_FALSE_NEGATIVES_OUTPUT = DEFAULT_HARD_EXAMPLES_DIR / "false_negatives.csv"
DEFAULT_UNCERTAIN_OUTPUT = DEFAULT_HARD_EXAMPLES_DIR / "uncertain.csv"
DEFAULT_HIGH_LOSS_OUTPUT = DEFAULT_HARD_EXAMPLES_DIR / "high_loss_samples.csv"
DEFAULT_HARD_EXAMPLE_OUTPUTS = [
    DEFAULT_FALSE_POSITIVES_OUTPUT,
    DEFAULT_FALSE_NEGATIVES_OUTPUT,
    DEFAULT_UNCERTAIN_OUTPUT,
    DEFAULT_HIGH_LOSS_OUTPUT,
    DEFAULT_HARD_EXAMPLE_SUMMARY,
]
HARD_EXAMPLE_FILENAMES = {
    "false_positives": "false_positives.csv",
    "false_negatives": "false_negatives.csv",
    "uncertain": "uncertain.csv",
    "high_loss_samples": "high_loss_samples.csv",
}


class HardExampleMiningError(ValueError):
    """Raised when hard-example mining inputs are invalid."""


@dataclass(frozen=True)
class HardExampleRow:
    image_id: str
    true_label: int
    probability: float
    threshold: float
    predicted_label: int
    margin: float
    error_type: str


@dataclass(frozen=True)
class HardExampleMiningResult:
    false_positives_path: Path
    false_negatives_path: Path
    uncertain_path: Path
    high_loss_samples_path: Path
    summary_path: Path


@dataclass(frozen=True)
class ResolvedHardExampleSources:
    hard_examples_root: Path
    summary_path: Path
    found: bool
    hard_example_source_used: str
    generated: bool = False


@dataclass(frozen=True)
class HardExampleSourceReport:
    false_positives_path: Path
    false_negatives_path: Path
    uncertain_path: Path
    high_loss_samples_path: Path
    summary_path: Path
    loaded_counts: dict[str, int]
    summary_counts: dict[str, int]
    count_mismatches: dict[str, dict[str, int]]
    eligible_for_training_count: int
    excluded_validation_count: int
    validation_excluded_image_ids: list[str]
    used_for_oversampling_count: int
    train_validation_disjoint: bool
    excluded_rows: list[str]
    oversampled_image_ids: list[str]
    oversampled_image_ids_by_group: dict[str, list[str]]
    hard_example_source_used: str


def mine_hard_examples(
    *,
    artifact_root: str | Path | None = None,
    predictions_path: str | Path | None = None,
    threshold_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_HARD_EXAMPLES_DIR,
    summary_path: str | Path = DEFAULT_HARD_EXAMPLE_SUMMARY,
    uncertainty_margin: float = 0.05,
    high_loss_limit: int = 50,
) -> HardExampleMiningResult:
    """Mine validation-only hard-example CSVs."""

    active_artifact_root = _resolve_artifact_root(artifact_root) if artifact_root is not None else None
    active_predictions = (
        Path(predictions_path)
        if predictions_path is not None
        else active_artifact_root / DEFAULT_VALIDATION_PREDICTIONS_RELATIVE
        if active_artifact_root is not None
        else None
    )
    active_threshold = (
        Path(threshold_path)
        if threshold_path is not None
        else active_artifact_root / "reports" / "best_threshold.json"
        if active_artifact_root is not None
        else None
    )
    if active_predictions is None:
        raise HardExampleMiningError("predictions_path or artifact_root is required")
    if active_threshold is None:
        raise HardExampleMiningError("threshold_path or artifact_root is required")
    threshold = load_threshold(active_threshold)
    rows = _load_prediction_rows(active_predictions, threshold)
    false_positives = [row for row in rows if row.true_label == 0 and row.predicted_label == 1]
    false_negatives = [row for row in rows if row.true_label == 1 and row.predicted_label == 0]
    uncertain = [row for row in rows if row.margin <= uncertainty_margin]
    high_loss = _rank_high_loss(false_positives, false_negatives, uncertain)[:high_loss_limit]

    out_dir = Path(output_dir)
    fp_path = out_dir / "false_positives.csv"
    fn_path = out_dir / "false_negatives.csv"
    uncertain_path = out_dir / "uncertain.csv"
    high_loss_path = out_dir / "high_loss_samples.csv"
    summary = Path(summary_path)
    for path, output_rows in (
        (fp_path, false_positives),
        (fn_path, false_negatives),
        (uncertain_path, uncertain),
        (high_loss_path, high_loss),
    ):
        _write_rows(path, output_rows)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "source_predictions": str(active_predictions),
        "threshold_path": str(active_threshold),
        "threshold": threshold,
        "uncertainty_margin": uncertainty_margin,
        "high_loss_limit": high_loss_limit,
        "counts": {
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "uncertain": len(uncertain),
            "high_loss_samples": len(high_loss),
        },
        "artifact_paths": {
            "false_positives": str(fp_path),
            "false_negatives": str(fn_path),
            "uncertain": str(uncertain_path),
            "high_loss_samples": str(high_loss_path),
            "summary": str(summary),
        },
    }
    summary.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return HardExampleMiningResult(fp_path, fn_path, uncertain_path, high_loss_path, summary)


def prepare_hard_example_report(
    *,
    hard_examples_root: str | Path | None = None,
    hard_example_source: str | Path | None = None,
    summary_path: str | Path | None = None,
    train_image_ids: Sequence[str] = (),
    validation_image_ids: Sequence[str] = (),
    strategy: str = "analysis_only",
) -> HardExampleSourceReport:
    """Load V1 hard-example source sets and report V2 leakage exclusions."""

    resolved = resolve_hard_example_sources(
        hard_example_source=hard_example_source if hard_example_source is not None else hard_examples_root,
        summary_path=summary_path,
    )
    root = resolved.hard_examples_root
    paths = {
        name: root / filename for name, filename in HARD_EXAMPLE_FILENAMES.items()
    }
    loaded: dict[str, list[str]] = {name: _load_image_ids(path) for name, path in paths.items()}
    loaded_counts = {name: len(ids) for name, ids in loaded.items()}
    summary_counts = _load_summary_counts(resolved.summary_path)
    count_mismatches = {
        name: {"loaded": loaded_counts.get(name, 0), "summary": summary_counts.get(name, 0)}
        for name in sorted(set(loaded_counts) | set(summary_counts))
        if loaded_counts.get(name, 0) != summary_counts.get(name, 0)
    }

    train_set = set(train_image_ids)
    validation_set = set(validation_image_ids)
    all_hard_ids = _ordered_unique([image_id for ids in loaded.values() for image_id in ids])
    unmapped = sorted(image_id for image_id in all_hard_ids if image_id not in train_set and image_id not in validation_set)
    validation_excluded = sorted(image_id for image_id in all_hard_ids if image_id in validation_set)
    eligible_by_group = {
        group: [image_id for image_id in ids if image_id in train_set]
        for group, ids in loaded.items()
    }
    eligible = _ordered_unique(
        image_id
        for group in _oversampling_group_priority()
        for image_id in eligible_by_group.get(group, [])
    )
    oversampled = eligible if strategy == "oversample" else []
    return HardExampleSourceReport(
        false_positives_path=paths["false_positives"],
        false_negatives_path=paths["false_negatives"],
        uncertain_path=paths["uncertain"],
        high_loss_samples_path=paths["high_loss_samples"],
        summary_path=resolved.summary_path,
        loaded_counts=loaded_counts,
        summary_counts=summary_counts,
        count_mismatches=count_mismatches,
        eligible_for_training_count=len(eligible),
        excluded_validation_count=len(validation_excluded),
        validation_excluded_image_ids=validation_excluded,
        used_for_oversampling_count=len(oversampled),
        train_validation_disjoint=not bool(train_set & validation_set),
        excluded_rows=[*unmapped, *validation_excluded],
        oversampled_image_ids=oversampled,
        oversampled_image_ids_by_group=eligible_by_group if strategy == "oversample" else {},
        hard_example_source_used=resolved.hard_example_source_used,
    )


def resolve_hard_example_sources(
    *,
    hard_examples_root: str | Path | None = None,
    hard_example_source: str | Path | None = None,
    summary_path: str | Path | None = None,
    search_roots: Sequence[str | Path] | None = None,
) -> ResolvedHardExampleSources:
    """Resolve hard-example memory from explicit, V2, V1, local, or Kaggle-style sources."""

    requested_source = hard_example_source if hard_example_source is not None else hard_examples_root
    explicit_root = None
    if requested_source not in (None, "", "auto"):
        explicit_root = Path(str(requested_source))
    explicit_summary = Path(summary_path) if summary_path is not None else (
        _summary_for_hard_root(explicit_root) if explicit_root is not None else DEFAULT_HARD_EXAMPLE_SUMMARY
    )
    if explicit_root is not None:
        resolved = _resolve_source_root(explicit_root, explicit_summary)
        if resolved is not None:
            return resolved

    roots = [Path.cwd()] if search_roots is None else [Path(root) for root in search_roots]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "artifacts" / "kaggle_v2b_he_artifacts" / "hard_examples",
                root / "artifacts" / "kaggle_v2b_artifacts" / "hard_examples",
                root / "artifacts" / "kaggle_v2b_artifacts" / "outputs" / "hard_examples",
                root / "artifacts" / "kaggle_v1_artifacts" / "outputs" / "hard_examples",
                root / "outputs" / "hard_examples",
            ]
        )

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(sorted(kaggle_input.glob("*/outputs/hard_examples")))
        candidates.extend(sorted(kaggle_input.glob("*/*/outputs/hard_examples")))

    for candidate in candidates:
        resolved = _resolve_source_root(candidate, _summary_for_hard_root(candidate))
        if resolved is not None:
            return resolved

    fallback_root = explicit_root if explicit_root is not None else DEFAULT_HARD_EXAMPLES_DIR
    return ResolvedHardExampleSources(fallback_root, explicit_summary, False, str(fallback_root))


def _resolve_artifact_root(root: str | Path) -> Path:
    raw = Path(root)
    return raw / "kaggle_v1" if (raw / "kaggle_v1").is_dir() else raw


def _has_hard_example_files(root: Path) -> bool:
    return any((root / filename).exists() for filename in HARD_EXAMPLE_FILENAMES.values())


def _summary_for_hard_root(root: Path) -> Path:
    return root.parent / "reports" / "hard_example_summary.json"


def _resolve_source_root(source: Path, summary_path: Path | None = None) -> ResolvedHardExampleSources | None:
    hard_root = _find_hard_examples_root(source)
    if hard_root is not None:
        return ResolvedHardExampleSources(
            hard_examples_root=hard_root,
            summary_path=summary_path if summary_path is not None and summary_path.exists() else _summary_for_hard_root(hard_root),
            found=True,
            hard_example_source_used=str(hard_root),
            generated=False,
        )

    prediction_path, threshold_path = _find_prediction_and_threshold(source)
    if prediction_path is None or threshold_path is None:
        return None

    output_dir = source / "hard_examples"
    generated_summary = source / "reports" / "hard_example_summary.json"
    mine_hard_examples(
        predictions_path=prediction_path,
        threshold_path=threshold_path,
        output_dir=output_dir,
        summary_path=generated_summary,
    )
    return ResolvedHardExampleSources(
        hard_examples_root=output_dir,
        summary_path=generated_summary,
        found=True,
        hard_example_source_used=str(source),
        generated=True,
    )


def _find_hard_examples_root(source: Path) -> Path | None:
    if _has_hard_example_files(source):
        return source
    direct = source / "hard_examples"
    if _has_hard_example_files(direct):
        return direct
    if not source.exists() or not source.is_dir():
        return None
    for candidate in sorted(source.rglob("hard_examples")):
        if candidate.is_dir() and _has_hard_example_files(candidate):
            return candidate
    return None


def _find_prediction_and_threshold(source: Path) -> tuple[Path | None, Path | None]:
    if not source.exists():
        return None, None
    if source.is_file():
        return None, None
    predictions = sorted(source.rglob("val_classifier_predictions.csv"))
    thresholds = sorted(source.rglob("best_threshold.json"))
    return (predictions[0] if predictions else None, thresholds[0] if thresholds else None)


def _oversampling_group_priority() -> tuple[str, ...]:
    return ("false_negatives", "false_positives", "uncertain", "high_loss_samples")


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _load_image_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_id" not in reader.fieldnames:
            raise HardExampleMiningError(f"Hard-example CSV must contain image_id: {path}")
        return [row["image_id"] for row in reader if row.get("image_id")]


def _load_summary_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HardExampleMiningError(f"Hard-example summary is not valid JSON: {path}") from exc
    counts = raw.get("counts", raw)
    output: dict[str, int] = {}
    for key in ("false_positives", "false_negatives", "uncertain", "high_loss_samples"):
        if key in counts:
            output[key] = int(counts[key])
    return output


def _load_prediction_rows(path: Path, threshold: float) -> list[HardExampleRow]:
    if not path.exists():
        raise HardExampleMiningError(f"Validation predictions not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    output: list[HardExampleRow] = []
    for row in rows:
        try:
            probability = float(row["probability"])
            true_label = int(row["true_label"])
            predicted_label = int(row.get("predicted_label", int(probability >= threshold)))
            row_threshold = float(row.get("threshold", threshold))
        except (KeyError, TypeError, ValueError) as exc:
            raise HardExampleMiningError(f"Invalid validation prediction row in {path}: {row}") from exc
        output.append(
            HardExampleRow(
                image_id=row["image_id"],
                true_label=true_label,
                probability=probability,
                threshold=row_threshold,
                predicted_label=predicted_label,
                margin=abs(probability - threshold),
                error_type=_error_type(true_label, predicted_label),
            )
        )
    return output


def _error_type(true_label: int, predicted_label: int) -> str:
    if true_label == 0 and predicted_label == 1:
        return "false_positive"
    if true_label == 1 and predicted_label == 0:
        return "false_negative"
    return "uncertain"


def _rank_high_loss(
    false_positives: Sequence[HardExampleRow],
    false_negatives: Sequence[HardExampleRow],
    uncertain: Sequence[HardExampleRow],
) -> list[HardExampleRow]:
    ranked_fp = sorted(false_positives, key=lambda row: row.probability, reverse=True)
    ranked_fn = sorted(false_negatives, key=lambda row: row.probability)
    ranked_uncertain = sorted(uncertain, key=lambda row: row.margin)
    seen: set[str] = set()
    ranked: list[HardExampleRow] = []
    for row in [*ranked_fp, *ranked_fn, *ranked_uncertain]:
        if row.image_id not in seen:
            ranked.append(row)
            seen.add(row.image_id)
    return ranked


def _write_rows(path: Path, rows: Sequence[HardExampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "true_label", "probability", "threshold", "predicted_label", "margin", "error_type"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mine SPEC-006 V1 hard examples.")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument("--threshold-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HARD_EXAMPLES_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_HARD_EXAMPLE_SUMMARY)
    parser.add_argument("--uncertainty-margin", type=float, default=0.05)
    parser.add_argument("--high-loss-limit", type=int, default=50)
    args = parser.parse_args(argv)

    result = mine_hard_examples(
        artifact_root=args.artifact_root,
        predictions_path=args.predictions_path,
        threshold_path=args.threshold_path,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        uncertainty_margin=args.uncertainty_margin,
        high_loss_limit=args.high_loss_limit,
    )
    print(f"Generated hard examples at {result.false_positives_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
