"""Offline V1 hard-example mining for SPEC-006."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

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


def _resolve_artifact_root(root: str | Path) -> Path:
    raw = Path(root)
    return raw / "kaggle_v1" if (raw / "kaggle_v1").is_dir() else raw


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
