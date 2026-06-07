"""V1 classifier prediction helpers for SPEC-006."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch

from src.data.preprocessing import preprocess_image
from src.data.roi import RoiCropRequest
from src.models.classifier import create_classifier
from src.training.train_classifier import select_training_device


DEFAULT_ARTIFACT_ROOT = Path("artifacts/kaggle_v1_artifacts/outputs/kaggle_v1")
DEFAULT_MODEL_RELATIVE = Path("models/classifier_effnet_b0_best.pth")
DEFAULT_METRICS_RELATIVE = Path("reports/classifier_metrics.json")
DEFAULT_THRESHOLD_RELATIVE = Path("reports/best_threshold.json")
DEFAULT_VALIDATION_PREDICTIONS_RELATIVE = Path("predictions/val_classifier_predictions.csv")
DEFAULT_V2_ARTIFACT_ROOT = Path("outputs/kaggle_v2")
DEFAULT_V2_MODEL_RELATIVE = Path("models/classifier_best.pth")
DEFAULT_V5_ARTIFACT_ROOT = Path("outputs/kaggle_v5/v5_strong_classifier")
DEFAULT_V5_MODEL_RELATIVE = Path("models/classifier_best.pth")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class V1InferenceError(ValueError):
    """Raised when SPEC-006 inference inputs or artifacts are invalid."""


@dataclass(frozen=True)
class V1ArtifactPaths:
    artifact_root: Path
    model_path: Path
    metrics_path: Path
    threshold_path: Path
    validation_predictions_path: Path


@dataclass(frozen=True)
class V1Prediction:
    image_id: str
    image_path: Path
    probability: float
    threshold: float
    target: int
    inference_seconds: float = 0.0


def resolve_v1_artifact_paths(artifact_root: str | Path | None = None) -> V1ArtifactPaths:
    """Resolve and validate the concrete V1 artifact root."""

    root = Path(artifact_root) if artifact_root is not None else DEFAULT_ARTIFACT_ROOT
    root = root.expanduser()
    concrete_root = root / "kaggle_v1" if (root / "kaggle_v1").is_dir() else root
    paths = V1ArtifactPaths(
        artifact_root=concrete_root,
        model_path=concrete_root / DEFAULT_MODEL_RELATIVE,
        metrics_path=concrete_root / DEFAULT_METRICS_RELATIVE,
        threshold_path=concrete_root / DEFAULT_THRESHOLD_RELATIVE,
        validation_predictions_path=concrete_root / DEFAULT_VALIDATION_PREDICTIONS_RELATIVE,
    )
    missing = [
        path
        for path in (
            paths.model_path,
            paths.metrics_path,
            paths.threshold_path,
            paths.validation_predictions_path,
        )
        if not path.exists()
    ]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise V1InferenceError(f"Missing required V1 artifact files: {names}")
    return paths


def resolve_v2_artifact_paths(artifact_root: str | Path | None = None) -> V1ArtifactPaths:
    """Resolve selected V2 classifier artifacts without requiring V1 naming."""

    root = Path(artifact_root) if artifact_root is not None else DEFAULT_V2_ARTIFACT_ROOT
    root = root.expanduser()
    concrete_root = root / "kaggle_v2" if (root / "kaggle_v2").is_dir() else root
    paths = V1ArtifactPaths(
        artifact_root=concrete_root,
        model_path=concrete_root / DEFAULT_V2_MODEL_RELATIVE,
        metrics_path=concrete_root / DEFAULT_METRICS_RELATIVE,
        threshold_path=concrete_root / DEFAULT_THRESHOLD_RELATIVE,
        validation_predictions_path=concrete_root / DEFAULT_VALIDATION_PREDICTIONS_RELATIVE,
    )
    missing = [path for path in (paths.model_path, paths.metrics_path, paths.threshold_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise V1InferenceError(f"Missing required V2 artifact files: {names}")
    return paths


def resolve_v5_artifact_paths(artifact_root: str | Path | None = None) -> V1ArtifactPaths:
    """Resolve selected V5 classifier artifacts."""

    root = Path(artifact_root) if artifact_root is not None else DEFAULT_V5_ARTIFACT_ROOT
    root = root.expanduser()
    concrete_root = root / "kaggle_v5" / "v5_strong_classifier" if (root / "kaggle_v5" / "v5_strong_classifier").is_dir() else root
    paths = V1ArtifactPaths(
        artifact_root=concrete_root,
        model_path=concrete_root / DEFAULT_V5_MODEL_RELATIVE,
        metrics_path=concrete_root / DEFAULT_METRICS_RELATIVE,
        threshold_path=concrete_root / DEFAULT_THRESHOLD_RELATIVE,
        validation_predictions_path=concrete_root / DEFAULT_VALIDATION_PREDICTIONS_RELATIVE,
    )
    missing = [path for path in (paths.model_path, paths.metrics_path, paths.threshold_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise V1InferenceError(f"Missing required V5 artifact files: {names}")
    return paths


def load_threshold(threshold_path: str | Path) -> float:
    """Load and validate the saved V1 threshold."""

    path = Path(threshold_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise V1InferenceError(f"Threshold file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise V1InferenceError(f"Threshold file is not valid JSON: {path}") from exc
    try:
        threshold = float(raw["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise V1InferenceError(f"Threshold file must contain numeric 'threshold': {path}") from exc
    if not 0.0 <= threshold <= 1.0:
        raise V1InferenceError("Threshold must be between 0.0 and 1.0")
    return threshold


def discover_images(image_dir: str | Path) -> list[Path]:
    """Return supported images in stable order."""

    root = Path(image_dir)
    if not root.exists():
        raise V1InferenceError(f"Image directory not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def resolve_image_paths(image_ids: Sequence[str], image_dir: str | Path) -> list[Path]:
    """Resolve sample-submission image IDs to existing image paths."""

    root = Path(image_dir)
    paths = [root / image_id for image_id in image_ids]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise V1InferenceError("Missing sample submission images: " + ", ".join(missing))
    return paths


def predict_images(
    image_paths: Sequence[str | Path],
    *,
    artifact_root: str | Path | None = None,
    model_path: str | Path | None = None,
    threshold_path: str | Path | None = None,
    model_name: str = "efficientnet_b0",
    synthetic_smoke: bool = False,
    device: str = "auto",
    batch_size: int = 1,
    image_size: int = 384,
) -> list[V1Prediction]:
    """Predict class-1 probabilities and binary targets for images."""

    if batch_size < 1:
        raise V1InferenceError("batch_size must be at least 1")
    artifacts: Optional[V1ArtifactPaths] = None
    if artifact_root is not None or model_path is None or threshold_path is None:
        if _looks_like_v5_root(artifact_root):
            artifacts = resolve_v5_artifact_paths(artifact_root)
        elif _looks_like_v2_root(artifact_root):
            artifacts = resolve_v2_artifact_paths(artifact_root)
        else:
            artifacts = resolve_v1_artifact_paths(artifact_root)
    active_model_path = Path(model_path) if model_path is not None else artifacts.model_path
    active_threshold_path = Path(threshold_path) if threshold_path is not None else artifacts.threshold_path
    threshold = load_threshold(active_threshold_path)
    resolved_device = select_training_device(device)
    model = create_classifier(
        model_name=model_name,
        num_classes=1,
        synthetic_smoke=synthetic_smoke,
    ).to(resolved_device)
    _load_checkpoint(model, active_model_path, resolved_device)
    model.eval()

    resolved_paths = [Path(path) for path in image_paths]
    missing = [str(path) for path in resolved_paths if not path.exists()]
    if missing:
        raise V1InferenceError("Missing image files: " + ", ".join(missing))

    predictions: list[V1Prediction] = []
    with torch.no_grad():
        for start in range(0, len(resolved_paths), batch_size):
            batch_paths = resolved_paths[start : start + batch_size]
            inputs = [
                torch.from_numpy(
                    preprocess_image(
                        RoiCropRequest(
                            image_id=path.name,
                            image_path=path,
                            target_size=(image_size, image_size),
                        ),
                        split="test",
                    ).normalized
                ).to(dtype=torch.float32)
                for path in batch_paths
            ]
            batch = torch.stack(inputs).to(resolved_device)
            logits = model(batch)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
            for path, probability in zip(batch_paths, probabilities):
                prob = float(probability)
                predictions.append(
                    V1Prediction(
                        image_id=path.name,
                        image_path=path,
                        probability=prob,
                        threshold=threshold,
                        target=int(prob >= threshold),
                    )
                )
    return predictions


def _looks_like_v2_root(artifact_root: str | Path | None) -> bool:
    if artifact_root is None:
        return False
    root = Path(artifact_root)
    return "kaggle_v2" in root.parts or (root / "models" / "classifier_best.pth").exists()


def _looks_like_v5_root(artifact_root: str | Path | None) -> bool:
    if artifact_root is None:
        return False
    root = Path(artifact_root)
    normalized = str(root).replace("\\", "/").lower()
    return "kaggle_v5" in normalized or "v5_strong_classifier" in normalized


def _load_checkpoint(model: torch.nn.Module, model_path: Path, device: torch.device) -> None:
    if not model_path.exists():
        raise V1InferenceError(f"Model checkpoint not found: {model_path}")
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    except Exception as exc:
        raise V1InferenceError(f"Unable to load classifier checkpoint: {model_path}") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run SPEC-006 V1 image predictions.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--model-name", default="efficientnet_b0")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args(argv)

    predictions = predict_images(
        discover_images(args.image_dir),
        artifact_root=args.artifact_root,
        model_name=args.model_name,
        synthetic_smoke=args.synthetic_smoke,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    print(f"Generated {len(predictions)} V1 predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
