"""Training orchestration for SPEC-005 binary classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.dataset import DatasetValidationError, audit_dataset
from src.data.preprocessing import preprocess_image
from src.data.roi import RoiCropRequest
from src.models.classifier import create_classifier, validate_v5_classifier_scope
from src.training.losses import create_binary_focal_loss, create_weighted_bce_loss
from src.training.metrics import BinaryMetrics, V1BaselineRecord, V2CandidateResult, compute_binary_metrics, generate_v1_vs_v2_comparison
from src.training.threshold_search import ThresholdSearchResult, find_best_threshold

if TYPE_CHECKING:
    from src.training.hard_example_mining import HardExampleSourceReport


DEFAULT_MODEL_OUTPUT = Path("outputs/models/classifier_effnet_b0_best.pth")
DEFAULT_METRICS_OUTPUT = Path("outputs/reports/classifier_metrics.json")
DEFAULT_THRESHOLD_OUTPUT = Path("outputs/reports/best_threshold.json")
DEFAULT_LABEL_DISTRIBUTION_OUTPUT = Path("outputs/reports/label_distribution.json")
DEFAULT_SPLIT_DISTRIBUTION_OUTPUT = Path("outputs/reports/split_distribution.json")
DEFAULT_PREDICTIONS_OUTPUT = Path("outputs/predictions/val_classifier_predictions.csv")
V2_OUTPUT_ROOT = Path("outputs/kaggle_v2")
V2_MODEL_OUTPUT = V2_OUTPUT_ROOT / "models/classifier_best.pth"
V2_METRICS_OUTPUT = V2_OUTPUT_ROOT / "reports/classifier_metrics.json"
V2_THRESHOLD_OUTPUT = V2_OUTPUT_ROOT / "reports/best_threshold.json"
V2_PREDICTIONS_OUTPUT = V2_OUTPUT_ROOT / "predictions/val_classifier_predictions.csv"
V2_COMPARISON_OUTPUT = V2_OUTPUT_ROOT / "reports/v1_vs_v2_comparison.json"
V5_OUTPUT_ROOT = Path("outputs/kaggle_v5/v5_strong_classifier")
V5_MODEL_OUTPUT = V5_OUTPUT_ROOT / "models/classifier_best.pth"
V5_METRICS_OUTPUT = V5_OUTPUT_ROOT / "reports/classifier_metrics.json"
V5_THRESHOLD_OUTPUT = V5_OUTPUT_ROOT / "reports/best_threshold.json"
V5_PREDICTIONS_OUTPUT = V5_OUTPUT_ROOT / "predictions/val_classifier_predictions.csv"
V5_TEST_PREDICTIONS_OUTPUT = V5_OUTPUT_ROOT / "predictions/test_classifier_predictions.csv"
V5_SUBMISSION_OUTPUT = V5_OUTPUT_ROOT / "submissions/submission_v5.csv"
V5_BENCHMARK_OUTPUT = V5_OUTPUT_ROOT / "benchmarks/v5_inference_benchmark.json"
V2_ALLOWED_HARD_EXAMPLE_STRATEGIES = {"none", "analysis_only", "oversample"}
V2_ALLOWED_BACKBONES = {"efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "convnext_tiny", "tiny_cnn"}
V5_ALLOWED_HARD_EXAMPLE_STRATEGIES = {"analysis_only", "oversample"}
V2_FORBIDDEN_FLAGS = {
    "detector",
    "segmentation",
    "grad_cam",
    "gradcam",
    "dashboard",
    "feature_memory_bank",
    "memory_bank",
    "default_ensemble",
    "ensemble",
    "distillation",
    "hybrid_inference",
    "hybrid",
}

class TrainingValidationError(ValueError):
    """Raised when classifier training inputs are invalid."""


@dataclass(frozen=True)
class TrainingExample:
    image_id: str
    image_path: Path
    label: int

    def load_preprocessed(
        self,
        *,
        split: str,
        seed: Optional[int] = None,
        target_size: tuple[int, int] = (384, 384),
        augmentation_recipe: str = "v1",
    ) -> np.ndarray:
        request = RoiCropRequest(
            image_id=self.image_id,
            image_path=self.image_path,
            split=split,
            annotation_bbox=None,
            target_size=target_size,
        )
        return preprocess_image(
            request,
            split=split,
            seed=seed,
            augmentation_recipe=augmentation_recipe,
        ).normalized


@dataclass(frozen=True)
class SplitAssignment:
    train: list[TrainingExample]
    validation: list[TrainingExample]


@dataclass(frozen=True)
class TrainingExampleLoadResult:
    examples: list[TrainingExample]
    resolved_dataset_root: Path
    train_csv_row_count: int
    train_image_count: int


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
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    log_every_n_batches: int = 25
    experiment_name: str = "v1_effnet_b0"
    imbalance_strategy: str = "weighted_bce"
    augmentation_recipe: str = "v1"
    hard_example_strategy: str = "none"
    hard_example_source: str = "auto"
    weighted_sampler: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    speed_ceiling_multiplier: float = 2.0
    close_f1_tolerance: float = 0.002
    v2: bool = False
    v5: bool = False
    fallback_model_name: str = "efficientnet_b2"
    split_source: str = "v2b_compatible"
    public_score_baseline: float = 0.92181
    v3_detector_public_score: float = 0.74169
    v2b_public_score: float = 0.92121
    v2b_average_time_per_image: Optional[float] = None
    public_score_decision: str = "analysis_only_pending_manual_review"


@dataclass(frozen=True)
class ValidationPrediction:
    image_id: str
    true_label: int
    probability: float
    threshold: float
    predicted_label: int
    prob_bad: float
    classifier_prediction: int
    target: int


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
    class_weights: dict[str, float]
    artifact_paths: dict[str, str]
    experiment_name: str = "v1_effnet_b0"
    hard_example_strategy: str = "none"
    imbalance_strategy: str = "weighted_bce"
    hard_example_source_used: str = ""
    split_disjointness: dict[str, object] = field(default_factory=dict)
    selected_model_name: str = ""
    fallback_model_name: str = ""
    split_source: str = ""
    validation_prediction_distribution: dict[str, int] = field(default_factory=dict)
    validation_target_distribution: dict[str, int] = field(default_factory=dict)
    test_prediction_distribution: dict[str, int] = field(default_factory=dict)
    submission_row_count: int = 0
    v2b_benchmark_reference: dict[str, object] = field(default_factory=dict)
    speed_ratio_vs_v2b: Optional[float] = None
    within_v5_speed_ceiling: Optional[bool] = None
    public_score_baseline: float = 0.92181
    v2b_public_score: float = 0.92121
    v3_detector_public_score: float = 0.74169
    public_score_decision: str = "analysis_only_pending_manual_review"
    analysis_only: bool = True
    selection_source: str = "validation_only"


@dataclass(frozen=True)
class BestThresholdRecord:
    threshold: float
    f1_score: float
    tie_break: str
    candidate_count: int
    split_source: str = ""
    selection_source: str = "validation_only"
    confusion_counts: dict[str, int] = field(default_factory=dict)


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
    label_distribution_path: Path
    split_distribution_path: Path
    predictions_path: Path
    runtime_seconds: float
    hard_example_report: Optional["HardExampleSourceReport"] = None


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
        target_size: tuple[int, int] = (384, 384),
        augmentation_recipe: str = "v1",
    ) -> None:
        self.examples = list(examples)
        self.split_name = split_name
        self.seed = seed
        self.target_size = target_size
        self.augmentation_recipe = augmentation_recipe

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        example = self.examples[index]
        sample_seed = None if self.seed is None else self.seed + index
        array = example.load_preprocessed(
            split=self.split_name,
            seed=sample_seed,
            target_size=self.target_size,
            augmentation_recipe=self.augmentation_recipe,
        )
        inputs = torch.from_numpy(array).to(dtype=torch.float32)
        label = torch.tensor(float(example.label), dtype=torch.float32)
        return inputs, label, example.image_id


def load_classifier_config(config_path: str | Path = "configs/classifier.yaml") -> TrainingRunConfig:
    """Load classifier training config."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = raw.get("classifier", raw)
    experiment_cfg = raw.get("experiment", {})
    model_cfg = raw.get("model", {})
    training_cfg = raw.get("training", {})
    data_cfg = raw.get("data", {})
    benchmark_cfg = raw.get("benchmark", {})
    is_v5 = bool(cfg.get("v5", False) or raw.get("v5", False) or "v5_strong_classifier" in Path(config_path).name)
    if is_v5:
        cfg = {
            **cfg,
            "v5": True,
            "experiment_name": experiment_cfg.get("name", cfg.get("experiment_name")),
            "seed": experiment_cfg.get("seed", cfg.get("seed")),
            "model_name": model_cfg.get("model_name", cfg.get("model_name")),
            "fallback_model_name": model_cfg.get("fallback_model_name", cfg.get("fallback_model_name")),
            "image_size": model_cfg.get("image_size", cfg.get("image_size")),
            "imbalance_strategy": training_cfg.get("loss", cfg.get("imbalance_strategy")),
            "weighted_sampler": training_cfg.get("sampler", cfg.get("weighted_sampler")),
            "hard_example_strategy": training_cfg.get("hard_example_strategy", cfg.get("hard_example_strategy")),
            "augmentation_recipe": training_cfg.get("augmentation_recipe", cfg.get("augmentation_recipe")),
            "split_source": data_cfg.get("split_source", cfg.get("split_source")),
            "speed_ceiling_multiplier": benchmark_cfg.get("speed_ceiling_multiplier", cfg.get("speed_ceiling_multiplier")),
            "v2b_average_time_per_image": benchmark_cfg.get("v2b_average_time_per_image", cfg.get("v2b_average_time_per_image")),
        }
    is_v2 = bool(cfg.get("v2", False) or raw.get("v2", False) or "classifier_v2" in Path(config_path).name)
    if is_v2:
        validate_v2_scope_guards({**raw, **cfg})
    if is_v5:
        validate_v5_scope_guards({**raw, **cfg})
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
        max_train_samples=_optional_positive_int(cfg.get("max_train_samples"), "classifier.max_train_samples"),
        max_val_samples=_optional_positive_int(cfg.get("max_val_samples"), "classifier.max_val_samples"),
        log_every_n_batches=int(cfg.get("log_every_n_batches", 25)),
        experiment_name=str(cfg.get("experiment_name", "v2a_effnet_b0_recipe" if is_v2 else "v1_effnet_b0")),
        imbalance_strategy=str(cfg.get("imbalance_strategy", "focal_loss_weighted_sampler" if is_v2 else "weighted_bce")),
        augmentation_recipe=str(cfg.get("augmentation_recipe", "v2_safe" if is_v2 else "v1")),
        hard_example_strategy=str(cfg.get("hard_example_strategy", "analysis_only" if is_v2 else "none")),
        hard_example_source=str(cfg.get("hard_example_source", "auto")),
        weighted_sampler=bool(cfg.get("weighted_sampler", is_v2)),
        focal_alpha=float(cfg.get("focal_alpha", 0.25)),
        focal_gamma=float(cfg.get("focal_gamma", 2.0)),
        speed_ceiling_multiplier=float(cfg.get("speed_ceiling_multiplier", 2.0)),
        close_f1_tolerance=float(cfg.get("close_f1_tolerance", 0.002)),
        v2=is_v2,
        v5=is_v5,
        fallback_model_name=str(cfg.get("fallback_model_name", "efficientnet_b2")),
        split_source=str(cfg.get("split_source", "v2b_compatible")),
        public_score_baseline=float(cfg.get("public_score_baseline", 0.92181)),
        v3_detector_public_score=float(cfg.get("v3_detector_public_score", 0.74169)),
        v2b_public_score=float(cfg.get("v2b_public_score", 0.92121)),
        v2b_average_time_per_image=(
            None
            if cfg.get("v2b_average_time_per_image") in (None, "")
            else float(cfg.get("v2b_average_time_per_image"))
        ),
        public_score_decision=str(cfg.get("public_score_decision", "analysis_only_pending_manual_review")),
    )
    _validate_training_config(config)
    return config


def _validate_training_config(config: TrainingRunConfig) -> None:
    if config.v5:
        _validate_v5_training_config(config)
        return
    if config.v2:
        _validate_v2_training_config(config)
        return
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
    if config.log_every_n_batches < 1:
        raise TrainingValidationError("classifier.log_every_n_batches must be at least 1")
    if config.epochs < 1:
        raise TrainingValidationError("classifier.epochs must be at least 1")


def _validate_v2_training_config(config: TrainingRunConfig) -> None:
    if config.model_name not in V2_ALLOWED_BACKBONES:
        raise TrainingValidationError("V2 classifier.model_name must be EfficientNet-B0/B1/B2 or optional ConvNeXt-Tiny")
    if config.image_size not in {384, 448}:
        raise TrainingValidationError("V2 classifier.image_size must be 384 or optional 448")
    if config.image_size == 448 and "448" not in config.experiment_name:
        raise TrainingValidationError("V2 448x448 candidates must be explicitly named and run after 384 baseline candidates")
    if config.num_classes != 2:
        raise TrainingValidationError("classifier.num_classes must be 2")
    if config.hard_example_strategy not in V2_ALLOWED_HARD_EXAMPLE_STRATEGIES:
        raise TrainingValidationError("V2 hard_example_strategy must be none, analysis_only, or oversample")
    if not config.hard_example_source:
        raise TrainingValidationError("V2 hard_example_source must be a path or auto")
    if config.augmentation_recipe not in {"v1", "v2_safe"}:
        raise TrainingValidationError("V2 augmentation_recipe must be v1 or v2_safe")
    if config.speed_ceiling_multiplier > 2.0 or config.speed_ceiling_multiplier <= 0:
        raise TrainingValidationError("V2 speed_ceiling_multiplier must be > 0 and <= 2.0")
    if config.close_f1_tolerance != 0.002:
        raise TrainingValidationError("V2 close_f1_tolerance must remain 0.002")
    if config.batch_size < 1:
        raise TrainingValidationError("classifier.batch_size must be at least 1")
    if config.num_workers < 0:
        raise TrainingValidationError("classifier.num_workers must be non-negative")
    if config.log_every_n_batches < 1:
        raise TrainingValidationError("classifier.log_every_n_batches must be at least 1")
    if config.epochs < 1:
        raise TrainingValidationError("classifier.epochs must be at least 1")


def _validate_v5_training_config(config: TrainingRunConfig) -> None:
    try:
        validate_v5_classifier_scope(config.model_name, config.fallback_model_name)
    except ValueError as exc:
        raise TrainingValidationError(str(exc)) from exc
    if config.image_size != 512:
        raise TrainingValidationError("V5 classifier.image_size must be 512")
    if config.num_classes != 2:
        raise TrainingValidationError("classifier.num_classes must be 2")
    if config.hard_example_strategy not in V5_ALLOWED_HARD_EXAMPLE_STRATEGIES:
        raise TrainingValidationError("V5 hard_example_strategy must be analysis_only or oversample")
    if config.augmentation_recipe not in {"v2_safe", "v5_safe"}:
        raise TrainingValidationError("V5 augmentation_recipe must be v2_safe or v5_safe")
    if config.speed_ceiling_multiplier > 2.0 or config.speed_ceiling_multiplier <= 0:
        raise TrainingValidationError("V5 speed_ceiling_multiplier must be > 0 and <= 2.0")
    if config.public_score_baseline != 0.92181:
        raise TrainingValidationError("V5 public_score_baseline must remain 0.92181")
    if config.split_source not in {"v2b", "v2b_compatible"} and not Path(config.split_source).exists():
        raise TrainingValidationError("V5 split_source must be v2b_compatible, v2b, or an existing split path")
    if config.batch_size < 1:
        raise TrainingValidationError("classifier.batch_size must be at least 1")
    if config.num_workers < 0:
        raise TrainingValidationError("classifier.num_workers must be non-negative")
    if config.log_every_n_batches < 1:
        raise TrainingValidationError("classifier.log_every_n_batches must be at least 1")
    if config.epochs < 1:
        raise TrainingValidationError("classifier.epochs must be at least 1")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_training_examples(dataset_root: str | Path) -> list[TrainingExample]:
    """Load and validate labeled training examples from a dataset root."""

    return _load_training_examples_with_report(dataset_root).examples


def _load_training_examples_with_report(dataset_root: str | Path) -> TrainingExampleLoadResult:
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
    return TrainingExampleLoadResult(
        examples=examples,
        resolved_dataset_root=report.paths.root.resolve(),
        train_csv_row_count=len(report.train_rows),
        train_image_count=len(report.train_images),
    )


def make_stratified_split(
    examples: Sequence[TrainingExample],
    *,
    validation_split: float = 0.2,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
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

    train = _limit_stratified_examples(train, max_train_samples, seed=seed + 101, split_name="train")
    validation = _limit_stratified_examples(validation, max_val_samples, seed=seed + 202, split_name="validation")

    return SplitAssignment(train=train, validation=validation)


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
    hard_example_weights: Optional[dict[str, float]] = None,
) -> TrainingDataLoaders:
    """Build train/validation DataLoaders without preloading the full dataset."""

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    pin_memory = bool(config.pin_memory and device.type == "cuda")
    train_dataset = ClassifierTrainingDataset(
        split.train,
        split_name="train",
        seed=config.seed if synthetic_smoke else config.seed,
        target_size=(config.image_size, config.image_size),
        augmentation_recipe=config.augmentation_recipe,
    )
    validation_dataset = ClassifierTrainingDataset(
        split.validation,
        split_name="validation",
        seed=config.seed,
        target_size=(config.image_size, config.image_size),
        augmentation_recipe=config.augmentation_recipe,
    )
    sampler = None
    shuffle = True
    if config.weighted_sampler:
        sampler = build_weighted_random_sampler(
            split.train,
            seed=config.seed,
            hard_example_weights=hard_example_weights,
        )
        shuffle = False
    return TrainingDataLoaders(
        train=DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
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
    hard_examples_root: str | Path | None = None,
    hard_example_summary_path: str | Path | None = None,
    v1_baseline: Optional[V1BaselineRecord] = None,
) -> TrainingRunResult:
    """Run a reproducible binary classifier training pass."""

    started = time.perf_counter()
    active_config = config or TrainingRunConfig()
    if seed is not None:
        active_config = _replace_config(active_config, seed=seed)
    if epochs is not None:
        active_config = _replace_config(active_config, epochs=epochs)
    _validate_training_config(active_config)

    set_reproducible_seed(active_config.seed)
    output_root = Path(output_root)
    _log("Training start")
    load_result = _load_training_examples_with_report(dataset_root)
    examples = load_result.examples
    _log(f"resolved_dataset_root={load_result.resolved_dataset_root}")
    _log(f"train_csv_row_count={load_result.train_csv_row_count}")
    _log(f"train_image_count={load_result.train_image_count}")
    _log(f"label_distribution={_class_counts(examples)}")
    _log_examples(examples)

    split = make_stratified_split(
        examples,
        validation_split=active_config.validation_split,
        seed=active_config.seed,
        max_train_samples=active_config.max_train_samples,
        max_val_samples=active_config.max_val_samples,
    )
    from src.training.hard_example_mining import prepare_hard_example_report

    if active_config.hard_example_strategy == "oversample":
        initial_hard_example_report = prepare_hard_example_report(
            hard_example_source=hard_examples_root if hard_examples_root is not None else active_config.hard_example_source,
            summary_path=hard_example_summary_path,
            train_image_ids=[example.image_id for example in split.train],
            validation_image_ids=[example.image_id for example in split.validation],
            strategy="analysis_only",
        )
        split = exclude_hard_examples_from_validation(
            split,
            hard_example_image_ids=initial_hard_example_report.validation_excluded_image_ids,
        )

    train_counts = _class_counts(split.train)
    validation_counts = _class_counts(split.validation)
    split_disjointness = report_split_disjointness(split)
    hard_example_report = prepare_hard_example_report(
        hard_example_source=hard_examples_root if hard_examples_root is not None else active_config.hard_example_source,
        summary_path=hard_example_summary_path,
        train_image_ids=[example.image_id for example in split.train],
        validation_image_ids=[example.image_id for example in split.validation],
        strategy=active_config.hard_example_strategy,
    )
    hard_example_weights = (
        build_hard_example_weight_map(hard_example_report, strategy=active_config.hard_example_strategy)
        if active_config.hard_example_strategy == "oversample"
        else None
    )
    device = select_training_device(active_config.device)
    non_blocking = bool(active_config.pin_memory and device.type == "cuda")
    _log(f"train_split_size={len(split.train)} validation_split_size={len(split.validation)}")
    _log(f"train_label_distribution={train_counts}")
    _log(f"validation_label_distribution={validation_counts}")
    _log(f"split_disjointness={split_disjointness}")
    _log(f"hard_example_strategy={active_config.hard_example_strategy}")
    _log(f"selected_device={device}")
    _log(f"model_name={'tiny_cnn' if synthetic_smoke else active_config.model_name}")
    _log(f"image_size={active_config.image_size}")
    _log(f"batch_size={active_config.batch_size}")
    _log(f"num_workers={active_config.num_workers}")
    _log(f"epochs={active_config.epochs}")

    selected_model_name = active_config.model_name
    try:
        model = create_classifier(
            model_name="tiny_cnn" if synthetic_smoke else active_config.model_name,
            num_classes=1,
            synthetic_smoke=synthetic_smoke,
        ).to(device)
    except RuntimeError:
        if not active_config.v5:
            raise
        selected_model_name = active_config.fallback_model_name
        model = create_classifier(
            model_name="tiny_cnn" if synthetic_smoke else active_config.fallback_model_name,
            num_classes=1,
            synthetic_smoke=synthetic_smoke,
        ).to(device)
    if active_config.imbalance_strategy.startswith("focal"):
        loss_fn = create_binary_focal_loss(
            negative_count=train_counts["0"],
            positive_count=train_counts["1"],
            alpha=active_config.focal_alpha,
            gamma=active_config.focal_gamma,
            use_pos_weight="weighted" in active_config.imbalance_strategy,
            device=device,
        ).to(device)
    else:
        loss_fn = create_weighted_bce_loss(
            negative_count=train_counts["0"],
            positive_count=train_counts["1"],
            device=device,
        ).to(device)
    pos_weight = train_counts["0"] / train_counts["1"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=active_config.learning_rate, weight_decay=active_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(active_config.epochs, 1))
    loaders = build_training_dataloaders(
        split,
        config=active_config,
        device=device,
        synthetic_smoke=synthetic_smoke,
        hard_example_weights=hard_example_weights,
    )
    _log(f"DataLoader created train_batches={len(loaders.train)} validation_batches={len(loaders.validation)}")

    best_state = None
    best_metrics = None
    best_threshold = None
    best_probs: list[float] = []
    best_epoch = 0
    for epoch in range(1, active_config.epochs + 1):
        model.train()
        for batch_index, (batch_inputs, batch_labels, _image_ids) in enumerate(loaders.train, start=1):
            batch_inputs = batch_inputs.to(device, non_blocking=non_blocking)
            batch_labels = batch_labels.to(device, non_blocking=non_blocking)
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            if batch_index == 1 or batch_index % active_config.log_every_n_batches == 0:
                elapsed = time.perf_counter() - started
                _log(f"epoch={epoch} batch={batch_index}/{len(loaders.train)} loss={loss.item():.6f} elapsed_seconds={elapsed:.2f}")
        scheduler.step()

        validation_labels, probabilities = _collect_validation_predictions(model, loaders.validation, device=device)
        threshold = find_best_threshold(y_true=validation_labels, probabilities=probabilities)
        metrics = compute_binary_metrics(
            y_true=validation_labels,
            probabilities=probabilities,
            threshold=threshold.threshold,
        )
        _log(f"epoch={epoch} validation_f1={metrics.f1_score:.6f} threshold={threshold.threshold:.2f}")
        if best_metrics is None or metrics.f1_score > best_metrics.f1_score:
            best_metrics = metrics
            best_threshold = threshold
            best_probs = probabilities
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None and best_metrics is not None and best_threshold is not None
    model_output = V5_MODEL_OUTPUT if active_config.v5 else V2_MODEL_OUTPUT if active_config.v2 else DEFAULT_MODEL_OUTPUT
    metrics_output = V5_METRICS_OUTPUT if active_config.v5 else V2_METRICS_OUTPUT if active_config.v2 else DEFAULT_METRICS_OUTPUT
    threshold_output = V5_THRESHOLD_OUTPUT if active_config.v5 else V2_THRESHOLD_OUTPUT if active_config.v2 else DEFAULT_THRESHOLD_OUTPUT
    predictions_output = V5_PREDICTIONS_OUTPUT if active_config.v5 else V2_PREDICTIONS_OUTPUT if active_config.v2 else DEFAULT_PREDICTIONS_OUTPUT
    model_path = _resolve_output_path(output_root, model_output)
    metrics_path = _resolve_output_path(output_root, metrics_output)
    threshold_path = _resolve_output_path(output_root, threshold_output)
    label_distribution_path = _resolve_output_path(output_root, DEFAULT_LABEL_DISTRIBUTION_OUTPUT)
    split_distribution_path = _resolve_output_path(output_root, DEFAULT_SPLIT_DISTRIBUTION_OUTPUT)
    predictions_path = _resolve_output_path(output_root, predictions_output)
    for path in (model_path, metrics_path, threshold_path, label_distribution_path, split_distribution_path, predictions_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, model_path)
    predictions = _build_predictions(split.validation, best_probs, best_threshold.threshold)
    _write_predictions(predictions_path, predictions, v5=active_config.v5)
    runtime_seconds = round(time.perf_counter() - started, 6)
    speed_ratio_vs_v2b = (
        runtime_seconds / active_config.v2b_average_time_per_image
        if active_config.v5 and active_config.v2b_average_time_per_image and active_config.v2b_average_time_per_image > 0
        else None
    )
    _write_threshold(
        threshold_path,
        best_threshold,
        split_source=active_config.split_source,
        selection_source="validation_only",
        confusion_counts=best_metrics.confusion_counts if active_config.v5 else None,
    )
    _write_label_distribution(
        label_distribution_path,
        total_rows=load_result.train_csv_row_count,
        train_image_count=load_result.train_image_count,
        counts=_class_counts(examples),
    )
    _write_split_distribution(split_distribution_path, split=split)
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
            class_weights={"0": 1.0, "1": pos_weight},
            artifact_paths={
                "model": str(model_path),
                "metrics": str(metrics_path),
                "threshold": str(threshold_path),
                "label_distribution": str(label_distribution_path),
                "split_distribution": str(split_distribution_path),
                "predictions": str(predictions_path),
            },
            experiment_name=active_config.experiment_name,
            hard_example_strategy=active_config.hard_example_strategy,
            imbalance_strategy=active_config.imbalance_strategy,
            hard_example_source_used=hard_example_report.hard_example_source_used,
            split_disjointness={
                **split_disjointness,
                "hard_examples": asdict(hard_example_report),
            },
            selected_model_name=selected_model_name,
            fallback_model_name=active_config.fallback_model_name if active_config.v5 else "",
            split_source=active_config.split_source if active_config.v5 else "",
            validation_prediction_distribution=_prediction_distribution_from_probabilities(
                best_probs,
                best_threshold.threshold,
            ),
            validation_target_distribution=validation_counts,
            test_prediction_distribution={},
            submission_row_count=0,
            v2b_benchmark_reference=(
                {
                    "reference_name": "v2b",
                    "average_time_per_image": active_config.v2b_average_time_per_image,
                    "speed_ceiling_multiplier": active_config.speed_ceiling_multiplier,
                }
                if active_config.v5
                else {}
            ),
            speed_ratio_vs_v2b=None if speed_ratio_vs_v2b is None else round(speed_ratio_vs_v2b, 6),
            within_v5_speed_ceiling=(
                None
                if speed_ratio_vs_v2b is None
                else speed_ratio_vs_v2b <= active_config.speed_ceiling_multiplier
            ),
            public_score_baseline=active_config.public_score_baseline,
            v2b_public_score=active_config.v2b_public_score,
            v3_detector_public_score=active_config.v3_detector_public_score,
            public_score_decision=active_config.public_score_decision,
            analysis_only=active_config.public_score_decision != "accepted_best_public_submission",
            selection_source="validation_only",
        ),
    )
    if active_config.v5:
        validate_v5_metrics_report(metrics_path)
        validate_v5_validation_predictions(predictions_path, expected_validation_ids=[example.image_id for example in split.validation])
    if active_config.v2 and v1_baseline is not None:
        comparison_path = _resolve_output_path(output_root, V2_COMPARISON_OUTPUT)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        v2_candidate = V2CandidateResult(
            experiment_name=active_config.experiment_name,
            backbone=active_config.model_name,
            image_size=active_config.image_size,
            validation_f1=best_metrics.f1_score,
            best_threshold=best_threshold.threshold,
            false_positives=best_metrics.confusion_counts["fp"],
            false_negatives=best_metrics.confusion_counts["fn"],
            uncertain_samples=0,
            average_time_per_image=0.0,
            speed_multiplier_vs_v1=1.0,
            model_size_bytes=model_path.stat().st_size if model_path.exists() else 0,
        )
        comparison = generate_v1_vs_v2_comparison(
            v1=v1_baseline,
            candidates=[v2_candidate],
            speed_ceiling_multiplier=active_config.speed_ceiling_multiplier,
            close_f1_tolerance=active_config.close_f1_tolerance,
        )
        comparison_path.write_text(json.dumps(asdict(comparison), indent=2), encoding="utf-8")
    _log(f"total_runtime_seconds={runtime_seconds:.2f}")

    return TrainingRunResult(
        split=split,
        metrics=best_metrics,
        threshold=best_threshold,
        model_path=model_path,
        metrics_path=metrics_path,
        threshold_path=threshold_path,
        label_distribution_path=label_distribution_path,
        split_distribution_path=split_distribution_path,
        predictions_path=predictions_path,
        runtime_seconds=runtime_seconds,
        hard_example_report=hard_example_report,
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


def build_weighted_random_sampler(
    examples: Sequence[TrainingExample],
    *,
    seed: int = 42,
    hard_example_weights: Optional[dict[str, float]] = None,
) -> WeightedRandomSampler:
    """Build a deterministic weighted sampler from class counts and optional hard-example weights."""

    counts = _class_counts(examples)
    if counts["0"] <= 0 or counts["1"] <= 0:
        raise TrainingValidationError("weighted sampler requires both classes")
    class_weight = {0: 1.0 / counts["0"], 1: 1.0 / counts["1"]}
    hard_example_weights = hard_example_weights or {}
    weights = [
        float(class_weight[example.label] * hard_example_weights.get(example.image_id, 1.0))
        for example in examples
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def build_hard_example_weight_map(
    report: "HardExampleSourceReport",
    *,
    strategy: str,
    oversample_multiplier: float = 2.0,
) -> dict[str, float]:
    """Return per-image multipliers for eligible hard examples only when oversampling is explicit."""

    if strategy != "oversample":
        return {}
    return {image_id: oversample_multiplier for image_id in report.oversampled_image_ids}


def report_split_disjointness(split: SplitAssignment) -> dict[str, object]:
    """Report train/validation image-ID disjointness for leakage checks."""

    train_ids = {example.image_id for example in split.train}
    validation_ids = {example.image_id for example in split.validation}
    overlap = sorted(train_ids & validation_ids)
    return {
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "train_validation_disjoint": not overlap,
        "overlap_image_ids": overlap,
    }


def exclude_hard_examples_from_validation(
    split: SplitAssignment,
    *,
    hard_example_image_ids: Sequence[str],
) -> SplitAssignment:
    """Move hard-example rows out of validation before explicit oversampling."""

    hard_ids = set(hard_example_image_ids)
    if not hard_ids:
        return split
    moved_to_train = [example for example in split.validation if example.image_id in hard_ids]
    kept_validation = [example for example in split.validation if example.image_id not in hard_ids]
    existing_train_ids = {example.image_id for example in split.train}
    train = [
        *split.train,
        *(example for example in moved_to_train if example.image_id not in existing_train_ids),
    ]
    return SplitAssignment(
        train=sorted(train, key=lambda item: item.image_id),
        validation=sorted(kept_validation, key=lambda item: item.image_id),
    )


def validate_v2_scope_guards(options: dict[str, object]) -> None:
    """Reject SPEC-007 out-of-scope feature flags when enabled."""

    enabled = sorted(key for key, value in options.items() if key in V2_FORBIDDEN_FLAGS and bool(value))
    if enabled:
        raise TrainingValidationError("SPEC-007 V2 classifier forbids: " + ", ".join(enabled))


def validate_v5_scope_guards(options: dict[str, object]) -> None:
    """Reject detector/fusion/dashboard scope changes for SPEC-010."""

    enabled = sorted(key for key, value in options.items() if key in V2_FORBIDDEN_FLAGS and bool(value))
    if enabled:
        raise TrainingValidationError("SPEC-010 V5 classifier forbids: " + ", ".join(enabled))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the SPEC-005 binary classifier.")
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--log-every-n-batches", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_classifier_config(args.config)
    overrides: dict[str, object] = {}
    if args.max_train_samples is not None:
        overrides["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        overrides["max_val_samples"] = args.max_val_samples
    if args.log_every_n_batches is not None:
        overrides["log_every_n_batches"] = args.log_every_n_batches
    if overrides:
        config = _replace_config(config, **overrides)
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


def _optional_positive_int(value: object, field_name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 1:
        raise TrainingValidationError(f"{field_name} must be at least 1 when set")
    return parsed


def _resolve_output_path(output_root: Path, default_path: Path) -> Path:
    try:
        default_path.relative_to(output_root)
        return default_path
    except ValueError:
        pass
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


def _limit_stratified_examples(
    examples: Sequence[TrainingExample],
    max_samples: Optional[int],
    *,
    seed: int,
    split_name: str,
) -> list[TrainingExample]:
    examples = list(examples)
    if max_samples is None or max_samples >= len(examples):
        return sorted(examples, key=lambda item: item.image_id)
    if max_samples < 2:
        raise TrainingValidationError(f"{split_name} debug limit must be at least 2 to preserve both classes")

    by_label: dict[int, list[TrainingExample]] = {0: [], 1: []}
    for example in examples:
        by_label[example.label].append(example)
    if not by_label[0] or not by_label[1]:
        raise TrainingValidationError(f"{split_name} split must contain both classes before applying debug limits")

    rng = random.Random(seed)
    label_counts = {label: len(items) for label, items in by_label.items()}
    selected_counts = {
        label: max(1, int(round(label_counts[label] / len(examples) * max_samples)))
        for label in (0, 1)
    }

    while sum(selected_counts.values()) > max_samples:
        candidates = [label for label in (0, 1) if selected_counts[label] > 1]
        if not candidates:
            break
        label = max(candidates, key=lambda item: selected_counts[item])
        selected_counts[label] -= 1
    while sum(selected_counts.values()) < max_samples:
        candidates = [label for label in (0, 1) if selected_counts[label] < label_counts[label]]
        if not candidates:
            break
        label = max(candidates, key=lambda item: label_counts[item] - selected_counts[item])
        selected_counts[label] += 1

    limited: list[TrainingExample] = []
    for label in (0, 1):
        items = list(by_label[label])
        rng.shuffle(items)
        limited.extend(items[: selected_counts[label]])
    return sorted(limited, key=lambda item: item.image_id)


def _log(message: str) -> None:
    print(message, flush=True)


def _log_examples(examples: Sequence[TrainingExample], *, limit: int = 5) -> None:
    for example in examples[:limit]:
        _log(f"example image_id={example.image_id} target={example.label} image_path={example.image_path}")


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
            batch_labels = batch_labels.to(device)
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
            prob_bad=float(probability),
            classifier_prediction=int(float(probability) >= threshold),
            target=int(float(probability) >= threshold),
        )
        for example, probability in zip(examples, probabilities)
    ]


def _write_predictions(path: Path, predictions: Sequence[ValidationPrediction], *, v5: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            ["image_id", "true_label", "prob_bad", "classifier_prediction", "target"]
            if v5
            else ["image_id", "true_label", "probability", "threshold", "predicted_label"]
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            row = prediction.__dict__
            writer.writerow({fieldname: row[fieldname] for fieldname in fieldnames})


def _write_threshold(
    path: Path,
    threshold: ThresholdSearchResult,
    *,
    split_source: str = "",
    selection_source: str = "validation_only",
    confusion_counts: Optional[dict[str, int]] = None,
) -> None:
    payload = {
        "threshold": threshold.threshold,
        "f1_score": threshold.f1_score,
        "tie_break": threshold.tie_break,
        "candidate_count": threshold.candidate_count,
    }
    if split_source:
        payload["split_source"] = split_source
        payload["selection_source"] = selection_source
    if confusion_counts is not None:
        payload["confusion_counts"] = confusion_counts
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_label_distribution(
    path: Path,
    *,
    total_rows: int,
    train_image_count: int,
    counts: dict[str, int],
) -> None:
    path.write_text(
        json.dumps(
            {
                "total_rows": total_rows,
                "train_image_count": train_image_count,
                "label_counts": counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_split_distribution(path: Path, *, split: SplitAssignment) -> None:
    path.write_text(
        json.dumps(
            {
                "train": {
                    "count": len(split.train),
                    "label_counts": _class_counts(split.train),
                },
                "validation": {
                    "count": len(split.validation),
                    "label_counts": _class_counts(split.validation),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_metrics(path: Path, *, report: ClassifierMetricsReport) -> None:
    path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")


def _prediction_distribution_from_probabilities(probabilities: Sequence[float], threshold: float) -> dict[str, int]:
    predictions = [int(float(probability) >= threshold) for probability in probabilities]
    return {"0": sum(1 for value in predictions if value == 0), "1": sum(1 for value in predictions if value == 1)}


def validate_v5_validation_predictions(path: str | Path, *, expected_validation_ids: Sequence[str] | None = None) -> None:
    rows = _read_csv_rows(path, required_columns=["image_id", "true_label", "prob_bad", "classifier_prediction", "target"])
    _validate_probability_rows(rows, allow_true_label=True)
    if expected_validation_ids is not None:
        actual = {row["image_id"] for row in rows}
        expected = set(expected_validation_ids)
        if actual != expected:
            raise TrainingValidationError("V5 validation predictions must match the V2B-compatible validation split")


def validate_v5_test_predictions(path: str | Path) -> None:
    rows = _read_csv_rows(path, required_columns=["image_id", "prob_bad", "classifier_prediction", "target"])
    forbidden = {"true_label", "label", "public_score", "sample_label"} & set(rows[0]) if rows else set()
    if forbidden:
        raise TrainingValidationError("V5 test predictions contain forbidden label or review columns")
    _validate_probability_rows(rows, allow_true_label=False)


def validate_v5_threshold_report(path: str | Path) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"threshold", "f1_score", "candidate_count", "split_source", "selection_source", "confusion_counts"}
    missing = sorted(required - set(payload))
    if missing:
        raise TrainingValidationError("V5 threshold report missing: " + ", ".join(missing))


def validate_v5_metrics_report(path: str | Path) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "f1_score",
        "threshold",
        "validation_target_distribution",
        "validation_prediction_distribution",
        "submission_row_count",
        "v2b_benchmark_reference",
        "public_score_baseline",
        "v3_detector_public_score",
        "public_score_decision",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise TrainingValidationError("V5 metrics report missing: " + ", ".join(missing))


def _read_csv_rows(path: str | Path, *, required_columns: Sequence[str]) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        raise TrainingValidationError("Missing required CSV columns: " + ", ".join(missing))
    if not rows:
        raise TrainingValidationError(f"CSV contains no rows: {path}")
    return rows


def _validate_probability_rows(rows: Sequence[dict[str, str]], *, allow_true_label: bool) -> None:
    image_ids = [row["image_id"] for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise TrainingValidationError("Probability export contains duplicate image_id rows")
    for row in rows:
        try:
            prob_bad = float(row["prob_bad"])
        except (TypeError, ValueError) as exc:
            raise TrainingValidationError("prob_bad must be numeric") from exc
        if not 0.0 <= prob_bad <= 1.0:
            raise TrainingValidationError("prob_bad must be between 0 and 1")
        for key in ("classifier_prediction", "target"):
            if int(row[key]) not in (0, 1):
                raise TrainingValidationError(f"{key} must be binary")
        if allow_true_label and int(row["true_label"]) not in (0, 1):
            raise TrainingValidationError("true_label must be binary")


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
