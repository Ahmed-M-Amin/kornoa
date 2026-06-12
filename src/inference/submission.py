"""V1 submission generation for SPEC-006."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from src.inference.predict import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_V2_ARTIFACT_ROOT,
    DEFAULT_V5_ARTIFACT_ROOT,
    V1InferenceError,
    predict_images,
    resolve_v1_artifact_paths,
    resolve_v2_artifact_paths,
    resolve_v5_artifact_paths,
    resolve_image_paths,
)
from src.training.train_classifier import (
    V5_METRICS_OUTPUT,
    V5_SUBMISSION_OUTPUT,
    V5_TEST_PREDICTIONS_OUTPUT,
    validate_v5_test_predictions,
)


DEFAULT_SUBMISSION_OUTPUT = Path("outputs/submissions/submission_v1.csv")
DEFAULT_V2_SUBMISSION_OUTPUT = Path("outputs/kaggle_v2/submissions/submission_v2.csv")
DEFAULT_V5_SUBMISSION_OUTPUT = V5_SUBMISSION_OUTPUT


class SubmissionError(V1InferenceError):
    """Raised when SPEC-006 submission inputs are invalid."""


@dataclass(frozen=True)
class SubmissionReport:
    output_path: Path
    row_count: int
    artifact_root: Path


def generate_submission(
    *,
    dataset_root: str | Path | None = None,
    sample_submission_path: str | Path | None = None,
    test_images_dir: str | Path | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    model_path: str | Path | None = None,
    threshold_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_SUBMISSION_OUTPUT,
    model_name: str = "efficientnet_b0",
    synthetic_smoke: bool = False,
    device: str = "auto",
    batch_size: int = 1,
    image_size: int = 384,
) -> SubmissionReport:
    """Generate a sample-aligned V1 submission CSV."""

    if _looks_like_v5_submission(artifact_root, output_path):
        resolved_artifacts = resolve_v5_artifact_paths(artifact_root)
    elif _looks_like_v2_submission(artifact_root, output_path):
        resolved_artifacts = resolve_v2_artifact_paths(artifact_root)
    else:
        resolved_artifacts = resolve_v1_artifact_paths(artifact_root)
    sample_path, image_dir = _resolve_dataset_inputs(dataset_root, sample_submission_path, test_images_dir)
    sample_rows = _read_sample_submission(sample_path)
    image_ids = [row["image_id"] for row in sample_rows]
    try:
        image_paths = resolve_image_paths(image_ids, image_dir)
        predictions = predict_images(
            image_paths,
            artifact_root=resolved_artifacts.artifact_root,
            model_path=model_path,
            threshold_path=threshold_path,
            model_name=model_name,
            synthetic_smoke=synthetic_smoke,
            device=device,
            batch_size=batch_size,
            image_size=image_size,
        )
    except V1InferenceError as exc:
        raise SubmissionError(str(exc)) from exc
    by_id = {prediction.image_id: prediction for prediction in predictions}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "target"])
        writer.writeheader()
        for image_id in image_ids:
            writer.writerow({"image_id": image_id, "target": by_id[image_id].target})
    if _looks_like_v5_submission(artifact_root, output_path):
        _validate_strict_submission(output)
        _update_v5_metrics_for_submission(
            artifact_root=resolved_artifacts.artifact_root,
            row_count=len(image_ids),
        )
    elif _looks_like_v2_submission(artifact_root, output_path):
        _update_metrics_for_submission(
            artifact_root=resolved_artifacts.artifact_root,
            row_count=len(image_ids),
        )
    return SubmissionReport(output_path=output, row_count=len(image_ids), artifact_root=resolved_artifacts.artifact_root)


def export_classifier_test_predictions(
    *,
    dataset_root: str | Path | None = None,
    sample_submission_path: str | Path | None = None,
    test_images_dir: str | Path | None = None,
    artifact_root: str | Path = DEFAULT_V2_ARTIFACT_ROOT,
    model_path: str | Path | None = None,
    threshold_path: str | Path | None = None,
    output_path: str | Path = Path("outputs/hybrid/v4/input/test_classifier_predictions_v2b.csv"),
    model_name: str = "efficientnet_b1",
    synthetic_smoke: bool = False,
    device: str = "auto",
    batch_size: int = 1,
    image_size: int = 384,
) -> SubmissionReport:
    """Export sample-aligned classifier probabilities for hybrid inference."""

    resolved_artifacts = resolve_v5_artifact_paths(artifact_root) if _looks_like_v5_submission(artifact_root, output_path) else resolve_v2_artifact_paths(artifact_root)
    sample_path, image_dir = _resolve_dataset_inputs(dataset_root, sample_submission_path, test_images_dir)
    sample_rows = _read_sample_submission(sample_path)
    image_ids = [row["image_id"] for row in sample_rows]
    try:
        image_paths = resolve_image_paths(image_ids, image_dir)
        predictions = predict_images(
            image_paths,
            artifact_root=resolved_artifacts.artifact_root,
            model_path=model_path,
            threshold_path=threshold_path,
            model_name=model_name,
            synthetic_smoke=synthetic_smoke,
            device=device,
            batch_size=batch_size,
            image_size=image_size,
        )
    except V1InferenceError as exc:
        raise SubmissionError(str(exc)) from exc
    by_id = {prediction.image_id: prediction for prediction in predictions}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "prob_bad", "classifier_prediction", "target"])
        writer.writeheader()
        for image_id in image_ids:
            prediction = by_id[image_id]
            writer.writerow(
                {
                    "image_id": image_id,
                    "prob_bad": prediction.probability,
                    "classifier_prediction": prediction.target,
                    "target": prediction.target,
                }
            )
    if _looks_like_v5_submission(artifact_root, output_path):
        validate_v5_test_predictions(output)
        _update_v5_metrics_for_submission(
            artifact_root=resolved_artifacts.artifact_root,
            row_count=len(image_ids),
            test_prediction_distribution=_prediction_distribution_from_test_csv(output),
        )
    else:
        _update_metrics_for_submission(
            artifact_root=resolved_artifacts.artifact_root,
            row_count=len(image_ids),
            test_prediction_distribution=_prediction_distribution_from_test_csv(output),
        )
    return SubmissionReport(output_path=output, row_count=len(image_ids), artifact_root=resolved_artifacts.artifact_root)


def _looks_like_v2_submission(artifact_root: str | Path, output_path: str | Path) -> bool:
    root = Path(artifact_root)
    out = Path(output_path)
    normalized = f"{root}|{out}".replace("\\", "/").lower()
    return (
        "kaggle_v2" in root.parts
        or "kaggle_v2" in out.parts
        or root == DEFAULT_V2_ARTIFACT_ROOT
        or "controlled_phase3" in normalized
        or "kaggle_phase3" in normalized
    )


def _looks_like_v5_submission(artifact_root: str | Path, output_path: str | Path) -> bool:
    root = Path(artifact_root)
    out = Path(output_path)
    normalized = f"{root}|{out}".replace("\\", "/").lower()
    return "kaggle_v5" in normalized or "v5_strong_classifier" in normalized or root == DEFAULT_V5_ARTIFACT_ROOT


def _resolve_dataset_inputs(
    dataset_root: str | Path | None,
    sample_submission_path: str | Path | None,
    test_images_dir: str | Path | None,
) -> tuple[Path, Path]:
    if dataset_root is not None:
        root = Path(dataset_root)
        sample = Path(sample_submission_path) if sample_submission_path is not None else root / "sample_submission.csv"
        images = Path(test_images_dir) if test_images_dir is not None else root / "test_images"
    else:
        if sample_submission_path is None or test_images_dir is None:
            raise SubmissionError("dataset_root or both sample_submission_path and test_images_dir are required")
        sample = Path(sample_submission_path)
        images = Path(test_images_dir)
    if not sample.exists():
        raise SubmissionError(f"sample_submission.csv not found: {sample}")
    if not images.exists():
        raise SubmissionError(f"test_images directory not found: {images}")
    return sample, images


def _read_sample_submission(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_id" not in reader.fieldnames:
            raise SubmissionError("sample_submission.csv must contain image_id column")
        rows = list(reader)
    if not rows:
        raise SubmissionError("sample_submission.csv contains no rows")
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate classifier submissions and prediction exports.")
    parser.add_argument("--export-test-predictions", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--sample-submission-path", type=Path, default=None)
    parser.add_argument("--test-images-dir", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--threshold-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_SUBMISSION_OUTPUT)
    parser.add_argument("--save-test-predictions", type=Path, default=None)
    parser.add_argument("--model-name", default="efficientnet_b0")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args(argv)

    if args.export_test_predictions:
        report = export_classifier_test_predictions(
            dataset_root=args.dataset_root,
            sample_submission_path=args.sample_submission_path,
            test_images_dir=args.test_images_dir,
            artifact_root=args.artifact_root,
            model_path=args.model_path,
            threshold_path=args.threshold_path,
            output_path=args.output_path,
            model_name=args.model_name,
            synthetic_smoke=args.synthetic_smoke,
            device=args.device,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
        print(f"Generated {report.row_count} classifier prediction rows at {report.output_path}")
        return 0
    if args.save_test_predictions is not None:
        export_classifier_test_predictions(
            dataset_root=args.dataset_root,
            sample_submission_path=args.sample_submission_path,
            test_images_dir=args.test_images_dir,
            artifact_root=args.artifact_root,
            model_path=args.model_path,
            threshold_path=args.threshold_path,
            output_path=args.save_test_predictions,
            model_name=args.model_name,
            synthetic_smoke=args.synthetic_smoke,
            device=args.device,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )

    report = generate_submission(
        dataset_root=args.dataset_root,
        sample_submission_path=args.sample_submission_path,
        test_images_dir=args.test_images_dir,
        artifact_root=args.artifact_root,
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        output_path=args.output_path,
        model_name=args.model_name,
        synthetic_smoke=args.synthetic_smoke,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    print(f"Generated {report.row_count} submission rows at {report.output_path}")
    return 0


def _validate_strict_submission(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if fieldnames != ["image_id", "target"]:
        raise SubmissionError("V5 submission must contain exactly image_id,target")
    for row in rows:
        if int(row["target"]) not in (0, 1):
            raise SubmissionError("V5 submission targets must be binary")


def _prediction_distribution_from_test_csv(path: Path) -> dict[str, int]:
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    return {"0": sum(1 for row in rows if int(row["target"]) == 0), "1": sum(1 for row in rows if int(row["target"]) == 1)}


def _update_v5_metrics_for_submission(
    *,
    artifact_root: Path,
    row_count: int,
    test_prediction_distribution: Optional[dict[str, int]] = None,
) -> None:
    metrics_path = artifact_root / V5_METRICS_OUTPUT.relative_to("outputs/kaggle_v5/v5_strong_classifier")
    if not metrics_path.exists():
        return
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["submission_row_count"] = row_count
    if test_prediction_distribution is not None:
        payload["test_prediction_distribution"] = test_prediction_distribution
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _update_metrics_for_submission(
    *,
    artifact_root: Path,
    row_count: int,
    test_prediction_distribution: Optional[dict[str, int]] = None,
) -> None:
    metrics_path = artifact_root / "reports" / "classifier_metrics.json"
    if not metrics_path.exists():
        return
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["submission_row_count"] = row_count
    if test_prediction_distribution is not None:
        payload["test_prediction_distribution"] = test_prediction_distribution
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
