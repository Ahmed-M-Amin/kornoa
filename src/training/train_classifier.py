"""Training orchestration for SPEC-005 binary classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import DatasetValidationError, audit_dataset
from src.data.preprocessing import preprocess_image
from src.data.roi import RoiCropRequest
from src.models.classifier import create_classifier
from src.training.losses import create_weighted_bce_loss
from src.training.metrics import BinaryMetrics, compute_binary_metrics
from src.training.threshold_search import ThresholdSearchResult, find_best_threshold


DEFAULT_MODEL_OUTPUT = Path("outputs/models/classifier_effnet_b0_best.pth")
DEFAULT_METRICS_OUTPUT = Path("outputs/reports/classifier_metrics.json")
DEFAULT_THRESHOLD_OUTPUT = Path("outputs/reports/best_threshold.json")
DEFAULT_PREDICTIONS_OUTPUT = Path("outputs/predictions/val_classifier_predictions.csv")

class TrainingValidationError(ValueError):
    """Raised when classifier training inputs are invalid."""


@dataclass(frozen=True)
class TrainingExample:
    image_id: str
    image_path: Path
    label: int

    def load_preprocessed(self, *, split: str, seed: Optional[int] = None) -> np.ndarray:
        request = RoiCropRequest(
            image_id=self.image_id,
            image_path=self.image_path,
            split=split,
            annotation_bbox=None,
        )
        return preprocess_image(request, split=split, seed=seed).normalized


@dataclass(frozen=True)
class SplitAssignment:
    train: list[TrainingExample]
    validation: list[TrainingExample]


@dataclass(frozen=True)
class TrainingRunConfig:
    model_name: str = "efficientnet_b0"
    image_size: int = 384
    num_classes: int = 2
    validation_split: float = 0.2
    seed: int = 42
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    num_workers: int = 0
    pin_memory: bool = False
    device: str = "auto"


@dataclass(frozen=True)
class ValidationPrediction:
    image_id: str
    true_label: int
    probability: float
    threshold: float
    predicted_label: int


@dataclass(frozen=True)
class ClassifierMetricsReport:
    f1_score: float
    threshold: float
    train_class_counts: dict[str, int]
    validation_class_counts: dict[str, int]
    confusion_counts: dict[str, int]
    runtime_seconds: float
    best_epoch: int
    device: str
    pos_weight: float
    artifact_paths: dict[str, str]


@dataclass(frozen=True)
class BestThresholdRecord:
    threshold: float
    f1_score: float
    tie_break: str
    candidate_count: int


@dataclass(frozen=True)
class GeneratedArtifactConfidentiality:
    ignored_paths: list[Path] = field(default_factory=list)
    tracked_paths: list[Path] = field(default_factory=list)
    unchecked_paths: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingRunResult:
    split: SplitAssignment
    metrics: BinaryMetrics
    threshold: ThresholdSearchResult
    model_path: Path
    metrics_path: Path
    threshold_path: Path
    predictions_path: Path
    runtime_seconds: float


@dataclass(frozen=True)
class TrainingDataLoaders:
    train: DataLoader
    validation: DataLoader


class ClassifierTrainingDataset(Dataset):
    """PyTorch dataset that applies SPEC-004 preprocessing per sample."""

    def __init__(
        self,
        examples: Sequence[TrainingExample],
        *,
        split_name: str,
        seed: Optional[int] = None,
    ) -> None:
        self.examples = list(examples)
        self.split_name = split_name
        self.seed = seed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        example = self.examples[index]
        sample_seed = None if self.seed is None else self.seed + index
        array = example.load_preprocessed(split=self.split_name, seed=sample_seed)
        inputs = torch.from_numpy(array).to(dtype=torch.float32)
        label = torch.tensor(float(example.label), dtype=torch.float32)
        return inputs, label, example.image_id


def load_classifier_config(config_path: str | Path = "configs/classifier.yaml") -> TrainingRunConfig:
    """Load classifier training config."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = raw.get("classifier", raw)
    config = TrainingRunConfig(
        model_name=str(cfg.get("model_name", "efficientnet_b0")),
        image_size=int(cfg.get("image_size", 384)),
        num_classes=int(cfg.get("num_classes", 2)),
        validation_split=float(cfg.get("validation_split", 0.2)),
        seed=int(cfg.get("seed", 42)),
        epochs=int(cfg.get("epochs", 3)),
        batch_size=int(cfg.get("batch_size", 4)),
        learning_rate=float(cfg.get("learning_rate", 0.001)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", False)),
        device=str(cfg.get("device", "auto")),
    )
    if config.model_name != "efficientnet_b0":
        raise TrainingValidationError("classifier.model_name must be efficientnet_b0 for SPEC-005")
    if config.image_size != 384:
        raise TrainingValidationError("classifier.image_size must be 384")
    if config.num_classes != 2:
        raise TrainingValidationError("classifier.num_classes must be 2")
    if config.batch_size < 1:
        raise TrainingValidationError("classifier.batch_size must be at least 1")
    if config.num_workers < 0:
        raise TrainingValidationError("classifier.num_workers must be non-negative")
    return config


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_training_examples(dataset_root: str | Path) -> list[TrainingExample]:
    """Load and validate labeled training examples from a dataset root."""

    try:
        report = audit_dataset(dataset_root)
    except DatasetValidationError as exc:
        raise TrainingValidationError(str(exc)) from exc

    if report.missing_train_images:
        raise TrainingValidationError(_format_missing_images(report.missing_train_images))

    image_by_id = {image.image_id: image.path for image in report.train_images}
    examples = [
        TrainingExample(
            image_id=row["image_id"],
            image_path=image_by_id[row["image_id"]],
            label=int(row["target"]),
        )
        for row in report.train_rows
    ]
    if not examples:
        raise TrainingValidationError("No labeled training examples found")
    return examples


def make_stratified_split(
    examples: Sequence[TrainingExample],
    *,
    validation_split: float = 0.2,
    seed: int = 42,
) -> SplitAssignment:
    """Create a reproducible 80/20 stratified split."""

    by_label: dict[int, list[TrainingExample]] = {0: [], 1: []}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)
    if not by_label.get(0) or not by_label.get(1):
        raise TrainingValidationError("Training data must contain both binary classes")

    rng = random.Random(seed)
    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    for label in (0, 1):
        items = list(by_label[label])
        rng.shuffle(items)
        validation_count = max(1, int(round(len(items) * validation_split)))
        if len(items) - validation_count < 1:
            raise TrainingValidationError("Not enough examples to preserve both classes in train and validation splits")
        validation.extend(sorted(items[:validation_count], key=lambda item: item.image_id))
        train.extend(sorted(items[validation_count:], key=lambda item: item.image_id))

    return SplitAssignment(
        train=sorted(train, key=lambda item: item.image_id),
        validation=sorted(validation, key=lambda item: item.image_id),
    )


def select_training_device(requested: str = "auto") -> torch.device:
    """Resolve the configured training device."""

    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise TrainingValidationError("CUDA device requested but CUDA is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise TrainingValidationError("MPS device requested but MPS is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise TrainingValidationError("classifier.device must be one of auto, cuda, mps, or cpu")


def build_training_dataloaders(
    split: SplitAssignment,
    *,
    config: TrainingRunConfig,
    device: torch.device,
    synthetic_smoke: bool = False,
) -> TrainingDataLoaders:
    """Build train/validation DataLoaders without preloading the full dataset."""

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    pin_memory = bool(config.pin_memory and device.type == "cuda")
    train_dataset = ClassifierTrainingDataset(
        split.train,
        split_name="train",
        seed=config.seed if synthetic_smoke else config.seed,
    )
    validation_dataset = ClassifierTrainingDataset(split.validation, split_name="validation", seed=config.seed)
    return TrainingDataLoaders(
        train=DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            generator=generator,
        ),
        validation=DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        ),
    )


def run_training(
    *,
    dataset_root: str | Path,
    output_root: str | Path = "outputs",
    config: Optional[TrainingRunConfig] = None,
    synthetic_smoke: bool = False,
    seed: Optional[int] = None,
    epochs: Optional[int] = None,
) -> TrainingRunResult:
    """Run a reproducible binary classifier training pass."""

    started = time.perf_counter()
    active_config = config or TrainingRunConfig()
    if seed is not None:
        active_config = _replace_config(active_config, seed=seed)
    if epochs is not None:
        active_config = _replace_config(active_config, epochs=epochs)

    set_reproducible_seed(active_config.seed)
    examples = load_training_examples(dataset_root)
    split = make_stratified_split(examples, validation_split=active_config.validation_split, seed=active_config.seed)
    train_counts = _class_counts(split.train)
    validation_counts = _class_counts(split.validation)
    device = select_training_device(active_config.device)
    non_blocking = bool(active_config.pin_memory and device.type == "cuda")

    model = create_classifier(
        model_name="tiny_cnn" if synthetic_smoke else active_config.model_name,
        num_classes=1,
        synthetic_smoke=synthetic_smoke,
    ).to(device)
    loss_fn = create_weighted_bce_loss(
        negative_count=train_counts["0"],
        positive_count=train_counts["1"],
        device=device,
    ).to(device)
    pos_weight = train_counts["0"] / train_counts["1"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=active_config.learning_rate, weight_decay=active_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(active_config.epochs, 1))
    loaders = build_training_dataloaders(split, config=active_config, device=device, synthetic_smoke=synthetic_smoke)

    best_state = None
    best_metrics = None
    best_threshold = None
    best_probs: list[float] = []
    best_epoch = 0
    for epoch in range(1, active_config.epochs + 1):
        model.train()
        for batch_inputs, batch_labels, _image_ids in loaders.train:
            batch_inputs = batch_inputs.to(device, non_blocking=non_blocking)
            batch_labels = batch_labels.to(device, non_blocking=non_blocking)
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        validation_labels, probabilities = _collect_validation_predictions(model, loaders.validation, device=device)
        threshold = find_best_threshold(y_true=validation_labels, probabilities=probabilities)
        metrics = compute_binary_metrics(
            y_true=validation_labels,
            probabilities=probabilities,
            threshold=threshold.threshold,
        )
        if best_metrics is None or metrics.f1_score > best_metrics.f1_score:
            best_metrics = metrics
            best_threshold = threshold
            best_probs = probabilities
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None and best_metrics is not None and best_threshold is not None
    output_root = Path(output_root)
    model_path = _resolve_output_path(output_root, DEFAULT_MODEL_OUTPUT)
    metrics_path = _resolve_output_path(output_root, DEFAULT_METRICS_OUTPUT)
    threshold_path = _resolve_output_path(output_root, DEFAULT_THRESHOLD_OUTPUT)
    predictions_path = _resolve_output_path(output_root, DEFAULT_PREDICTIONS_OUTPUT)
    for path in (model_path, metrics_path, threshold_path, predictions_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, model_path)
    predictions = _build_predictions(split.validation, best_probs, best_threshold.threshold)
    _write_predictions(predictions_path, predictions)
    runtime_seconds = round(time.perf_counter() - started, 6)
    _write_threshold(threshold_path, best_threshold)
    _write_metrics(
        metrics_path,
        report=ClassifierMetricsReport(
            f1_score=best_metrics.f1_score,
            threshold=best_threshold.threshold,
            train_class_counts=train_counts,
            validation_class_counts=validation_counts,
            confusion_counts=best_metrics.confusion_counts,
            runtime_seconds=runtime_seconds,
            best_epoch=best_epoch,
            device=device.type,
            pos_weight=pos_weight,
            artifact_paths={
                "model": str(model_path),
                "metrics": str(metrics_path),
                "threshold": str(threshold_path),
                "predictions": str(predictions_path),
            },
        ),
    )

    return TrainingRunResult(
        split=split,
        metrics=best_metrics,
        threshold=best_threshold,
        model_path=model_path,
        metrics_path=metrics_path,
        threshold_path=threshold_path,
        predictions_path=predictions_path,
        runtime_seconds=runtime_seconds,
    )


def validate_generated_artifact_confidentiality(paths: Iterable[Path | str]) -> GeneratedArtifactConfidentiality:
    ignored: list[Path] = []
    tracked: list[Path] = []
    unchecked: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        git_path = _normalize_git_path(path)
        if _git_is_tracked(git_path):
            tracked.append(path)
        if _git_is_ignored(git_path):
            ignored.append(path)
        else:
            unchecked.append(path)
    return GeneratedArtifactConfidentiality(ignored_paths=ignored, tracked_paths=tracked, unchecked_paths=unchecked)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the SPEC-005 binary classifier.")
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_classifier_config(args.config)
    result = run_training(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        config=config,
        synthetic_smoke=args.synthetic_smoke,
        epochs=args.epochs,
    )
    print(
        "Classifier training complete: "
        f"f1={result.metrics.f1_score:.4f}, "
        f"threshold={result.threshold.threshold:.2f}, "
        f"model={result.model_path}"
    )
    return 0


def _replace_config(config: TrainingRunConfig, **changes: object) -> TrainingRunConfig:
    values = config.__dict__.copy()
    values.update(changes)
    return TrainingRunConfig(**values)


def _resolve_output_path(output_root: Path, default_path: Path) -> Path:
    if output_root in (Path("."), Path("")):
        return default_path
    try:
        relative = default_path.relative_to("outputs")
    except ValueError:
        relative = default_path
    return output_root / relative


def _class_counts(examples: Sequence[TrainingExample]) -> dict[str, int]:
    return {
        "0": sum(1 for example in examples if example.label == 0),
        "1": sum(1 for example in examples if example.label == 1),
    }


def _format_missing_images(missing_images: Sequence[str], *, limit: int = 20) -> str:
    shown = list(missing_images[:limit])
    extra_count = max(0, len(missing_images) - len(shown))
    message = f"Missing {len(missing_images)} training images: " + ", ".join(shown)
    if extra_count:
        message += f", and {extra_count} more"
    return message


def _collect_validation_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels, _image_ids in loader:
            batch_inputs = batch_inputs.to(device)
            logits = model(batch_inputs)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().astype(float).tolist())
            labels.extend(batch_labels.cpu().numpy().astype(int).tolist())
    return labels, probabilities


def _build_predictions(
    examples: Sequence[TrainingExample],
    probabilities: Sequence[float],
    threshold: float,
) -> list[ValidationPrediction]:
    return [
        ValidationPrediction(
            image_id=example.image_id,
            true_label=example.label,
            probability=float(probability),
            threshold=float(threshold),
            predicted_label=int(float(probability) >= threshold),
        )
        for example, probability in zip(examples, probabilities)
    ]


def _write_predictions(path: Path, predictions: Sequence[ValidationPrediction]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "true_label", "probability", "threshold", "predicted_label"])
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(prediction.__dict__)


def _write_threshold(path: Path, threshold: ThresholdSearchResult) -> None:
    path.write_text(
        json.dumps(
            {
                "threshold": threshold.threshold,
                "f1_score": threshold.f1_score,
                "tie_break": threshold.tie_break,
                "candidate_count": threshold.candidate_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_metrics(path: Path, *, report: ClassifierMetricsReport) -> None:
    path.write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")


def _normalize_git_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _run_git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], check=False, capture_output=True, text=True)


def _git_is_tracked(path: str) -> bool:
    return _run_git(["ls-files", "--error-unmatch", path]).returncode == 0


def _git_is_ignored(path: str) -> bool:
    return _run_git(["check-ignore", path]).returncode == 0


if __name__ == "__main__":
    raise SystemExit(main())
