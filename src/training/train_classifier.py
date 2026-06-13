"""Training orchestration for SPEC-005 binary classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
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
V2_2_OUTPUT_ROOT = Path("outputs/kaggle_v2_2/v2_2_hard_examples")
V2_2_MODEL_OUTPUT = V2_2_OUTPUT_ROOT / "models/classifier_best.pth"
V2_2_METRICS_OUTPUT = V2_2_OUTPUT_ROOT / "reports/classifier_metrics.json"
V2_2_THRESHOLD_OUTPUT = V2_2_OUTPUT_ROOT / "reports/best_threshold.json"
V2_2_PREDICTIONS_OUTPUT = V2_2_OUTPUT_ROOT / "predictions/val_classifier_predictions.csv"
V2_2_COMPARISON_OUTPUT = V2_2_OUTPUT_ROOT / "reports/v1_vs_v2_comparison.json"
V5_OUTPUT_ROOT = Path("outputs/kaggle_v5/v5_strong_classifier")
V5_MODEL_OUTPUT = V5_OUTPUT_ROOT / "models/classifier_best.pth"
V5_METRICS_OUTPUT = V5_OUTPUT_ROOT / "reports/classifier_metrics.json"
V5_THRESHOLD_OUTPUT = V5_OUTPUT_ROOT / "reports/best_threshold.json"
V5_PREDICTIONS_OUTPUT = V5_OUTPUT_ROOT / "predictions/val_classifier_predictions.csv"
V5_TEST_PREDICTIONS_OUTPUT = V5_OUTPUT_ROOT / "predictions/test_classifier_predictions.csv"
V5_SUBMISSION_OUTPUT = V5_OUTPUT_ROOT / "submissions/submission_v5.csv"
V5_BENCHMARK_OUTPUT = V5_OUTPUT_ROOT / "benchmarks/v5_inference_benchmark.json"
V2_ALLOWED_HARD_EXAMPLE_STRATEGIES = {"none", "analysis_only", "oversample", "conservative_loss_weighting"}
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
CONSERVATIVE_HARD_EXAMPLE_STRATEGY = "conservative_loss_weighting"

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
    dataset_root: str = ""
    output_root: str = ""
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
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
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
    start_checkpoint: str = ""
    split_source: str = "v2b_compatible"
    early_stopping: bool = False
    public_score_baseline: float = 0.92181
    v3_detector_public_score: float = 0.74169
    v2b_public_score: float = 0.92121
    v2b_average_time_per_image: Optional[float] = None
    public_score_decision: str = "analysis_only_pending_manual_review"
    hard_examples_enabled: bool = False
    hard_negatives_path: str = ""
    hard_positives_path: str = ""
    uncertain_examples_path: str = ""
    hard_negative_weight: float = 1.5
    hard_positive_weight: float = 1.5
    uncertain_weight: float = 1.0
    max_extra_sampling_multiplier: float = 2.0
    controlled_phase3: bool = False
    cleaned_replay_enabled: bool = False
    baseline_name: str = ""
    baseline_checkpoint_path: str = ""
    locked_baseline_predictions_path: str = ""
    locked_baseline_threshold_path: str = ""
    locked_baseline_metrics_path: str = ""
    locked_baseline_runtime_report_path: str = ""
    approved_candidate_package_path: str = ""
    blocked_roi_pipeline_bug_rows_path: str = ""
    blocked_suspected_mislabel_rows_path: str = ""
    blocked_other_high_risk_rows_path: str = ""
    approved_manifest_path: str = ""
    auto_exclude_rows_path: str = ""
    needs_adjudication_rows_path: str = ""
    deferred_uncertain_rows_path: str = ""
    decision_lock_summary_path: str = ""
    candidate_hypothesis: str = ""
    candidate_family: str = ""
    intended_tradeoff: str = ""
    allow_test_labels: bool = False
    public_leaderboard_input: bool = False
    generate_submission: bool = False
    apply_relabels: bool = False


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
    hard_example_weighting: dict[str, object] = field(default_factory=dict)
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
class LockedBaselineReport:
    baseline_name: str
    baseline_checkpoint_path: str
    validation_predictions_path: str
    best_threshold_path: str
    validation_f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int
    validation_target_distribution: dict[str, int]
    test_target_distribution: dict[str, int]
    runtime_reference: dict[str, object]


@dataclass(frozen=True)
class GovernedPhase3Row:
    image_id: str
    target: int
    approved_candidate_action: str
    phase3_use_allowed: bool
    source_candidate_package_path: str
    baseline_membership_context: str
    hard_example_group: str


@dataclass(frozen=True)
class CandidateHypothesisRecord:
    candidate_name: str
    hypothesis: str
    candidate_family: str
    image_size: int
    hard_example_strategy: str
    baseline_reference: str
    intended_tradeoff: str


@dataclass(frozen=True)
class CandidateEvaluationRecord:
    candidate_name: str
    validation_f1: float
    hard_example_f1: float
    non_hard_example_f1: float
    precision: float
    recall: float
    fp_count: int
    fn_count: int
    best_threshold: float
    threshold_stability_summary: dict[str, object]
    changed_rows_vs_baseline: list[dict[str, object]]
    validation_target_distribution: dict[str, int]
    prediction_target_distribution: dict[str, int]
    runtime_report_path: str
    artifact_paths: dict[str, str]
    ablation_status: str
    decision: str
    decision_reason: str
    baseline_reference: str = ""
    artifact_completeness_status: str = "complete"
    submission_path: str = ""
    finalist_ready: bool = False
    best_of_n_runtime: Optional[float] = None


@dataclass(frozen=True)
class RuntimeEvidencePack:
    candidate_name: str
    timing_run_count: int
    best_runtime_seconds: float
    average_runtime_seconds: float
    seconds_per_image: float
    images_per_second: float
    device: str
    batch_size: int
    image_size: int
    benchmark_reference: str


@dataclass(frozen=True)
class InsightEvidencePack:
    baseline_summary: dict[str, object]
    hard_example_failure_modes: list[dict[str, object]]
    candidate_outcomes: list[dict[str, object]]
    runtime_tradeoff_summary: dict[str, object]
    final_recommendation: dict[str, object]
    rejected_alternatives: list[dict[str, object]]


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
        sample_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.examples = list(examples)
        self.split_name = split_name
        self.seed = seed
        self.target_size = target_size
        self.augmentation_recipe = augmentation_recipe
        self.sample_weights = sample_weights or {}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str] | tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
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
        if self.sample_weights:
            weight = torch.tensor(float(self.sample_weights.get(example.image_id, 1.0)), dtype=torch.float32)
            return inputs, label, example.image_id, weight
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
    output_cfg = raw.get("output", {})
    benchmark_cfg = raw.get("benchmark", {})
    hard_examples_cfg = raw.get("hard_examples", {})
    phase3_cfg = raw.get("phase3_controlled", {})
    cleaned_replay_cfg = raw.get("cleaned_replay", {})
    baseline_cfg = raw.get("baseline", {})
    safety_cfg = raw.get("safety", {})
    config_name = Path(config_path).name
    is_v5 = bool(cfg.get("v5", False) or raw.get("v5", False) or "v5_strong_classifier" in config_name)
    is_v1r = bool("v1r_repaired_baseline" in Path(config_path).name or raw.get("v1r", False))
    is_v5b = bool("v5b_safe_finetune" in Path(config_path).name or raw.get("v5b", False))
    is_v2b_enhanced = bool("v2b_enhanced" in config_name)
    is_v2_2 = bool("v2_2_hard_examples" in config_name or raw.get("v2_2", False))
    is_controlled_phase3 = bool("phase3_controlled_training" in config_name or phase3_cfg.get("enabled", False))
    is_cleaned_replay = bool("v2b_cleaned_replay_training" in config_name or cleaned_replay_cfg.get("enabled", False))
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
    elif is_v2_2:
        cfg = {
            **cfg,
            "v2": True,
            "experiment_name": experiment_cfg.get("name", cfg.get("experiment_name", "v2_2_hard_examples")),
            "seed": experiment_cfg.get("seed", cfg.get("seed")),
            "dataset_root": data_cfg.get("dataset_root", cfg.get("dataset_root")),
            "output_root": output_cfg.get("root", cfg.get("output_root")),
            "split_source": data_cfg.get("split_source", cfg.get("split_source")),
            "validation_split": data_cfg.get("validation_split", cfg.get("validation_split")),
            "model_name": model_cfg.get("model_name", cfg.get("model_name")),
            "image_size": model_cfg.get("image_size", cfg.get("image_size")),
            "num_classes": model_cfg.get("num_classes", cfg.get("num_classes")),
            "start_checkpoint": model_cfg.get("start_checkpoint", cfg.get("start_checkpoint")),
            "epochs": training_cfg.get("epochs", cfg.get("epochs")),
            "batch_size": training_cfg.get("batch_size", cfg.get("batch_size")),
            "learning_rate": training_cfg.get("learning_rate", cfg.get("learning_rate")),
            "weight_decay": training_cfg.get("weight_decay", cfg.get("weight_decay")),
            "imbalance_strategy": training_cfg.get("loss", cfg.get("imbalance_strategy")),
            "weighted_sampler": training_cfg.get("weighted_sampler", cfg.get("weighted_sampler")),
            "augmentation_recipe": training_cfg.get("augmentation_recipe", cfg.get("augmentation_recipe")),
            "hard_example_strategy": hard_examples_cfg.get(
                "strategy",
                training_cfg.get("hard_example_strategy", cfg.get("hard_example_strategy", "analysis_only")),
            ),
            "num_workers": training_cfg.get("num_workers", cfg.get("num_workers")),
            "pin_memory": training_cfg.get("pin_memory", cfg.get("pin_memory")),
            "device": training_cfg.get("device", cfg.get("device")),
        }
    elif is_controlled_phase3:
        cfg = {
            **cfg,
            "v2": True,
            "experiment_name": experiment_cfg.get("name", cfg.get("experiment_name", "phase3_controlled_candidate")),
            "seed": experiment_cfg.get("seed", cfg.get("seed")),
            "dataset_root": data_cfg.get("dataset_root", cfg.get("dataset_root")),
            "output_root": output_cfg.get("root", cfg.get("output_root")),
            "split_source": data_cfg.get("split_source", cfg.get("split_source")),
            "model_name": cfg.get("model_name", model_cfg.get("model_name", "efficientnet_b1")),
            "image_size": cfg.get("image_size", model_cfg.get("image_size", 384)),
            "num_classes": cfg.get("num_classes", model_cfg.get("num_classes", 2)),
            "batch_size": cfg.get("batch_size", training_cfg.get("batch_size", 4)),
            "epochs": cfg.get("epochs", training_cfg.get("epochs", 3)),
            "learning_rate": cfg.get("learning_rate", training_cfg.get("learning_rate", 0.001)),
            "weight_decay": cfg.get("weight_decay", training_cfg.get("weight_decay", 0.01)),
            "imbalance_strategy": cfg.get("imbalance_strategy", training_cfg.get("loss", "focal_loss_weighted_sampler")),
            "weighted_sampler": cfg.get("weighted_sampler", training_cfg.get("weighted_sampler", True)),
            "augmentation_recipe": cfg.get("augmentation_recipe", training_cfg.get("augmentation_recipe", "v2_safe")),
            "hard_example_strategy": hard_examples_cfg.get("strategy", cfg.get("hard_example_strategy", "oversample")),
        }
    elif is_cleaned_replay:
        cfg = {
            **cfg,
            "v2": True,
            "experiment_name": experiment_cfg.get("name", cfg.get("experiment_name", "cleaned_replay_v2b")),
            "seed": experiment_cfg.get("seed", cfg.get("seed")),
            "dataset_root": data_cfg.get("dataset_root", cfg.get("dataset_root")),
            "output_root": output_cfg.get("root", cfg.get("output_root")),
            "split_source": data_cfg.get("split_source", cfg.get("split_source", "v2b_compatible")),
            "model_name": cfg.get("model_name", model_cfg.get("model_name", "efficientnet_b1")),
            "image_size": cfg.get("image_size", model_cfg.get("image_size", 384)),
            "num_classes": cfg.get("num_classes", model_cfg.get("num_classes", 2)),
            "batch_size": cfg.get("batch_size", training_cfg.get("batch_size", 4)),
            "epochs": cfg.get("epochs", training_cfg.get("epochs", 3)),
            "learning_rate": cfg.get("learning_rate", training_cfg.get("learning_rate", 0.001)),
            "weight_decay": cfg.get("weight_decay", training_cfg.get("weight_decay", 0.01)),
            "imbalance_strategy": cfg.get("imbalance_strategy", training_cfg.get("loss", "focal_loss_weighted_sampler")),
            "weighted_sampler": cfg.get("weighted_sampler", training_cfg.get("weighted_sampler", True)),
            "augmentation_recipe": cfg.get("augmentation_recipe", training_cfg.get("augmentation_recipe", "v2_safe")),
            "hard_example_strategy": hard_examples_cfg.get("strategy", cfg.get("hard_example_strategy", "none")),
        }
    elif is_v5b or is_v2b_enhanced:
        default_experiment = "v2b_enhanced_448" if "448" in config_name else "v2b_enhanced_384"
        cfg = {
            **cfg,
            "v2": True,
            "experiment_name": cfg.get("experiment_name", default_experiment if is_v2b_enhanced else "v5b_safe_finetune"),
            "model_name": model_cfg.get("model_name", cfg.get("model_name")),
            "image_size": model_cfg.get("image_size", cfg.get("image_size")),
            "start_checkpoint": model_cfg.get("start_checkpoint", cfg.get("start_checkpoint")),
            "epochs": training_cfg.get("epochs", cfg.get("epochs")),
            "learning_rate": training_cfg.get("learning_rate", cfg.get("learning_rate")),
            "imbalance_strategy": training_cfg.get("loss", cfg.get("imbalance_strategy")),
            "weighted_sampler": training_cfg.get("weighted_sampler", cfg.get("weighted_sampler")),
            "augmentation_recipe": training_cfg.get("augmentation_recipe", cfg.get("augmentation_recipe")),
            "split_source": training_cfg.get("split_source", cfg.get("split_source")),
            "hard_example_strategy": training_cfg.get(
                "hard_examples",
                cfg.get("hard_example_strategy", "none" if is_v2b_enhanced else None),
            ),
            "early_stopping": training_cfg.get("early_stopping", cfg.get("early_stopping")),
        }
    elif is_v1r:
        cfg = {
            **cfg,
            "experiment_name": cfg.get("experiment_name", "v1r_repaired_baseline"),
            "model_name": model_cfg.get("model_name", cfg.get("model_name")),
            "image_size": model_cfg.get("image_size", cfg.get("image_size")),
            "epochs": training_cfg.get("epochs", cfg.get("epochs")),
            "learning_rate": training_cfg.get("learning_rate", cfg.get("learning_rate")),
            "imbalance_strategy": training_cfg.get("loss", cfg.get("imbalance_strategy")),
            "weighted_sampler": training_cfg.get("weighted_sampler", cfg.get("weighted_sampler")),
            "augmentation_recipe": training_cfg.get("augmentation_recipe", cfg.get("augmentation_recipe")),
            "split_source": training_cfg.get("split_source", cfg.get("split_source")),
        }
    is_v2 = bool(cfg.get("v2", False) or raw.get("v2", False) or "classifier_v2" in Path(config_path).name)
    if is_v2:
        validate_v2_scope_guards({**raw, **cfg})
    if is_v5:
        validate_v5_scope_guards({**raw, **cfg})
    config = TrainingRunConfig(
        dataset_root=str(cfg.get("dataset_root", "")),
        output_root=str(cfg.get("output_root", "")),
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
        limit_train_batches=_optional_positive_int(cfg.get("limit_train_batches"), "classifier.limit_train_batches"),
        limit_val_batches=_optional_positive_int(cfg.get("limit_val_batches"), "classifier.limit_val_batches"),
        log_every_n_batches=int(cfg.get("log_every_n_batches", 25)),
        experiment_name=str(cfg.get("experiment_name", "v2a_effnet_b0_recipe" if is_v2 else "v1_effnet_b0")),
        imbalance_strategy=str(cfg.get("imbalance_strategy", "focal_loss_weighted_sampler" if is_v2 else "weighted_bce")),
        augmentation_recipe=str(cfg.get("augmentation_recipe", "v2_safe" if is_v2 else "v1")),
        hard_example_strategy=str(hard_examples_cfg.get("strategy", cfg.get("hard_example_strategy", "analysis_only" if is_v2 else "none"))),
        hard_example_source=str(cfg.get("hard_example_source", "auto")),
        weighted_sampler=bool(cfg.get("weighted_sampler", is_v2)),
        focal_alpha=float(cfg.get("focal_alpha", 0.25)),
        focal_gamma=float(cfg.get("focal_gamma", 2.0)),
        speed_ceiling_multiplier=float(cfg.get("speed_ceiling_multiplier", 2.0)),
        close_f1_tolerance=float(cfg.get("close_f1_tolerance", 0.002)),
        v2=is_v2,
        v5=is_v5,
        fallback_model_name=str(cfg.get("fallback_model_name", "efficientnet_b2")),
        start_checkpoint=str(cfg.get("start_checkpoint", "")),
        split_source=str(cfg.get("split_source", "v2b_compatible")),
        early_stopping=bool(cfg.get("early_stopping", False)),
        public_score_baseline=float(cfg.get("public_score_baseline", 0.92181)),
        v3_detector_public_score=float(cfg.get("v3_detector_public_score", 0.74169)),
        v2b_public_score=float(cfg.get("v2b_public_score", 0.92121)),
        v2b_average_time_per_image=(
            None
            if cfg.get("v2b_average_time_per_image") in (None, "")
            else float(cfg.get("v2b_average_time_per_image"))
        ),
        public_score_decision=str(cfg.get("public_score_decision", "analysis_only_pending_manual_review")),
        hard_examples_enabled=bool(hard_examples_cfg.get("enabled", False)),
        hard_negatives_path=str(hard_examples_cfg.get("hard_negatives", "")),
        hard_positives_path=str(hard_examples_cfg.get("hard_positives", "")),
        uncertain_examples_path=str(hard_examples_cfg.get("uncertain_examples", "")),
        hard_negative_weight=float(hard_examples_cfg.get("hard_negative_weight", 1.5)),
        hard_positive_weight=float(hard_examples_cfg.get("hard_positive_weight", 1.5)),
        uncertain_weight=float(hard_examples_cfg.get("uncertain_weight", 1.0)),
        max_extra_sampling_multiplier=float(hard_examples_cfg.get("max_extra_sampling_multiplier", 2.0)),
        controlled_phase3=is_controlled_phase3,
        cleaned_replay_enabled=is_cleaned_replay,
        baseline_name=str((baseline_cfg if is_cleaned_replay else phase3_cfg).get("baseline_name", "")),
        baseline_checkpoint_path=str((baseline_cfg if is_cleaned_replay else phase3_cfg).get("baseline_checkpoint_path", "")),
        locked_baseline_predictions_path=str((baseline_cfg if is_cleaned_replay else phase3_cfg).get("locked_baseline_predictions_path", "")),
        locked_baseline_threshold_path=str((baseline_cfg if is_cleaned_replay else phase3_cfg).get("locked_baseline_threshold_path", "")),
        locked_baseline_metrics_path=str((baseline_cfg if is_cleaned_replay else phase3_cfg).get("locked_baseline_metrics_path", "")),
        locked_baseline_runtime_report_path=str((baseline_cfg if is_cleaned_replay else phase3_cfg).get("locked_baseline_runtime_report_path", "")),
        approved_candidate_package_path=str(phase3_cfg.get("approved_candidate_package_path", "")),
        blocked_roi_pipeline_bug_rows_path=str(phase3_cfg.get("blocked_roi_pipeline_bug_rows_path", "")),
        blocked_suspected_mislabel_rows_path=str(phase3_cfg.get("blocked_suspected_mislabel_rows_path", "")),
        blocked_other_high_risk_rows_path=str(phase3_cfg.get("blocked_other_high_risk_rows_path", "")),
        approved_manifest_path=str(cleaned_replay_cfg.get("approved_manifest_path", "")),
        auto_exclude_rows_path=str(cleaned_replay_cfg.get("auto_exclude_rows_path", "")),
        needs_adjudication_rows_path=str(cleaned_replay_cfg.get("needs_adjudication_rows_path", "")),
        deferred_uncertain_rows_path=str(cleaned_replay_cfg.get("deferred_uncertain_rows_path", "")),
        decision_lock_summary_path=str(cleaned_replay_cfg.get("decision_lock_summary_path", "")),
        candidate_hypothesis=str(phase3_cfg.get("candidate_hypothesis", "")),
        candidate_family=str(phase3_cfg.get("candidate_family", "")),
        intended_tradeoff=str(phase3_cfg.get("intended_tradeoff", "")),
        allow_test_labels=bool(safety_cfg.get("allow_test_labels", False)),
        public_leaderboard_input=bool(safety_cfg.get("public_leaderboard_input", False)),
        generate_submission=bool(safety_cfg.get("generate_submission", False)),
        apply_relabels=bool(safety_cfg.get("apply_relabels", False)),
    )
    _validate_training_config(config)
    return config


def _validate_training_config(config: TrainingRunConfig) -> None:
    _validate_common_training_config(config)
    if config.v5:
        _validate_v5_training_config(config)
        return
    if config.v2:
        _validate_v2_training_config(config)
        return
    if config.model_name != "efficientnet_b0":
        raise TrainingValidationError("classifier.model_name must be efficientnet_b0 for SPEC-005")
    if config.image_size not in {320, 384}:
        raise TrainingValidationError("classifier.image_size must be 320 or 384")
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


def _validate_common_training_config(config: TrainingRunConfig) -> None:
    if config.limit_train_batches is not None and config.limit_train_batches < 1:
        raise TrainingValidationError("classifier.limit_train_batches must be at least 1 when set")
    if config.limit_val_batches is not None and config.limit_val_batches < 1:
        raise TrainingValidationError("classifier.limit_val_batches must be at least 1 when set")
    if config.hard_examples_enabled:
        if not config.hard_negatives_path:
            raise TrainingValidationError("V2.2 hard_examples.hard_negatives is required when enabled")
        if not config.hard_positives_path:
            raise TrainingValidationError("V2.2 hard_examples.hard_positives is required when enabled")
        if config.hard_negative_weight > 2.0 or config.hard_positive_weight > 2.0 or config.uncertain_weight > 2.0:
            raise TrainingValidationError("V2.2 hard-example weights must remain conservative and <= 2.0")
        if config.max_extra_sampling_multiplier > 2.0:
            raise TrainingValidationError("V2.2 max_extra_sampling_multiplier must be <= 2.0")
    if config.hard_example_strategy == CONSERVATIVE_HARD_EXAMPLE_STRATEGY and config.imbalance_strategy != "bce":
        raise TrainingValidationError("V2.2 conservative_loss_weighting requires bce loss, not focal loss")
    if config.controlled_phase3:
        if not config.baseline_name:
            raise TrainingValidationError("phase3_controlled.baseline_name is required")
        if not config.approved_candidate_package_path:
            raise TrainingValidationError("phase3_controlled.approved_candidate_package_path is required")
        if not config.blocked_roi_pipeline_bug_rows_path:
            raise TrainingValidationError("phase3_controlled.blocked_roi_pipeline_bug_rows_path is required")
        if not config.blocked_suspected_mislabel_rows_path:
            raise TrainingValidationError("phase3_controlled.blocked_suspected_mislabel_rows_path is required")
        if not config.blocked_other_high_risk_rows_path:
            raise TrainingValidationError("phase3_controlled.blocked_other_high_risk_rows_path is required")
        if not config.candidate_hypothesis:
            raise TrainingValidationError("phase3_controlled.candidate_hypothesis is required")
    if config.cleaned_replay_enabled:
        if not config.baseline_name:
            raise TrainingValidationError("baseline.baseline_name is required")
        if not config.approved_manifest_path:
            raise TrainingValidationError("cleaned_replay.approved_manifest_path is required")
        if not config.auto_exclude_rows_path:
            raise TrainingValidationError("cleaned_replay.auto_exclude_rows_path is required")
        if not config.needs_adjudication_rows_path:
            raise TrainingValidationError("cleaned_replay.needs_adjudication_rows_path is required")
        if not config.deferred_uncertain_rows_path:
            raise TrainingValidationError("cleaned_replay.deferred_uncertain_rows_path is required")
        if not config.decision_lock_summary_path:
            raise TrainingValidationError("cleaned_replay.decision_lock_summary_path is required")
        if config.allow_test_labels:
            raise TrainingValidationError("safety.allow_test_labels must remain false")
        if config.public_leaderboard_input:
            raise TrainingValidationError("safety.public_leaderboard_input must remain false")
        if config.generate_submission:
            raise TrainingValidationError("safety.generate_submission must remain false")
        if config.apply_relabels:
            raise TrainingValidationError("safety.apply_relabels must remain false")


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
        raise TrainingValidationError("V2 hard_example_strategy must be none, analysis_only, or oversample, or conservative_loss_weighting")
    if not config.hard_example_source:
        raise TrainingValidationError("V2 hard_example_source must be a path or auto")
    if config.augmentation_recipe not in {"v1", "v2_safe", "mild_safe"}:
        raise TrainingValidationError("V2 augmentation_recipe must be v1, mild_safe, or v2_safe")
    if config.imbalance_strategy not in {"weighted_bce", "bce", "focal_loss_weighted_sampler"}:
        raise TrainingValidationError("V2 imbalance_strategy must be weighted_bce, bce, or focal_loss_weighted_sampler")
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
    sample_loss_weights: Optional[dict[str, float]] = None,
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
        sample_weights=sample_loss_weights,
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
    cleaned_replay_report_path: Optional[Path] = None
    cleaned_replay_approved_ids: set[str] = set()
    cleaned_replay_validation_payload: dict[str, object] = {}
    if active_config.cleaned_replay_enabled:
        cleaned_replay_report_path = run_cleaned_replay_dry_run_validation(
            active_config,
            dataset_root_override=dataset_root,
            output_root_override=output_root,
        )
        cleaned_replay_validation_payload = json.loads(cleaned_replay_report_path.read_text(encoding="utf-8"))
        cleaned_replay_approved_ids = _load_required_image_ids(
            active_config.approved_manifest_path,
            label="Cleaned replay approved manifest",
        )

    set_reproducible_seed(active_config.seed)
    output_root = Path(output_root)
    _log("Training start")
    load_result = _load_training_examples_with_report(dataset_root)
    examples = load_result.examples
    if active_config.cleaned_replay_enabled:
        examples = [example for example in examples if example.image_id in cleaned_replay_approved_ids]
        if not examples:
            raise TrainingValidationError("Cleaned replay approved manifest removed all training examples")
    _log(f"resolved_dataset_root={load_result.resolved_dataset_root}")
    _log(f"train_csv_row_count={load_result.train_csv_row_count}")
    _log(f"train_image_count={load_result.train_image_count}")
    _log(f"label_distribution={_class_counts(examples)}")
    _log_examples(examples)
    v2_2_hard_examples = None
    if active_config.hard_examples_enabled:
        from src.data.hard_examples import HardExampleError, load_hard_examples, validate_hard_example_images

        try:
            v2_2_hard_examples = load_hard_examples(
                hard_negatives=active_config.hard_negatives_path,
                hard_positives=active_config.hard_positives_path,
                uncertain_examples=active_config.uncertain_examples_path,
            )
            validate_hard_example_images(
                v2_2_hard_examples,
                available_image_ids=[example.image_id for example in examples],
                fail_on_missing=True,
            )
        except HardExampleError as exc:
            raise TrainingValidationError(str(exc)) from exc
        _log(f"hard_negative_file_found={Path(active_config.hard_negatives_path).exists()}")
        _log(f"hard_positive_file_found={Path(active_config.hard_positives_path).exists()}")
        _log(f"hard_negative_count={v2_2_hard_examples.hard_negative_count}")
        _log(f"hard_positive_count={v2_2_hard_examples.hard_positive_count}")
        _log(f"uncertain_count={v2_2_hard_examples.uncertain_count}")
        _log(f"first_5_hard_negative_image_ids={v2_2_hard_examples.hard_negative_image_ids[:5]}")
        _log(f"first_5_hard_positive_image_ids={v2_2_hard_examples.hard_positive_image_ids[:5]}")

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
    elif (
        active_config.hard_examples_enabled
        and active_config.hard_example_strategy == CONSERVATIVE_HARD_EXAMPLE_STRATEGY
    ):
        assert v2_2_hard_examples is not None
        split = exclude_hard_examples_from_validation(
            split,
            hard_example_image_ids=v2_2_hard_examples.strong_hard_example_image_ids,
        )

    train_counts = _class_counts(split.train)
    validation_counts = _class_counts(split.validation)
    split_disjointness = report_split_disjointness(split)
    hard_example_report = None
    v2_2_hard_example_summary: dict[str, object] = {}
    sample_loss_weights: dict[str, float] = {}
    if active_config.hard_examples_enabled and active_config.hard_example_strategy != "oversample":
        assert v2_2_hard_examples is not None
        if active_config.hard_example_strategy == CONSERVATIVE_HARD_EXAMPLE_STRATEGY:
            sample_loss_weights = build_conservative_hard_example_weight_map(
                v2_2_hard_examples,
                train_image_ids=[example.image_id for example in split.train],
                config=active_config,
            )
        v2_2_hard_example_summary = {
            "source_type": "v2_1_auto_triage",
            "usage": (
                "conservative_loss_weighting"
                if active_config.hard_example_strategy == CONSERVATIVE_HARD_EXAMPLE_STRATEGY
                else "validated_only"
            ),
            "hard_negatives_path": str(v2_2_hard_examples.hard_negatives_path),
            "hard_positives_path": str(v2_2_hard_examples.hard_positives_path),
            "uncertain_examples_path": (
                "" if v2_2_hard_examples.uncertain_examples_path is None else str(v2_2_hard_examples.uncertain_examples_path)
            ),
            "hard_negative_count": v2_2_hard_examples.hard_negative_count,
            "hard_positive_count": v2_2_hard_examples.hard_positive_count,
            "uncertain_count": v2_2_hard_examples.uncertain_count,
            "used_for_oversampling_count": 0,
            "weighted_training_samples": len(sample_loss_weights),
        }
    else:
        hard_example_report = prepare_hard_example_report(
            hard_example_source=hard_examples_root if hard_examples_root is not None else active_config.hard_example_source,
            summary_path=hard_example_summary_path,
            train_image_ids=[example.image_id for example in split.train],
            validation_image_ids=[example.image_id for example in split.validation],
            strategy=active_config.hard_example_strategy,
        )
    hard_example_weights = (
        build_hard_example_weight_map(hard_example_report, strategy=active_config.hard_example_strategy)
        if hard_example_report is not None and active_config.hard_example_strategy == "oversample"
        else None
    )
    device = select_training_device(active_config.device)
    non_blocking = bool(active_config.pin_memory and device.type == "cuda")
    _log(f"train_split_size={len(split.train)} validation_split_size={len(split.validation)}")
    _log(f"train_label_distribution={train_counts}")
    _log(f"validation_label_distribution={validation_counts}")
    _log(f"split_disjointness={split_disjointness}")
    _log(f"hard_example_strategy={active_config.hard_example_strategy}")
    if active_config.hard_examples_enabled:
        _log(f"hard_negative_weight={active_config.hard_negative_weight}")
        _log(f"hard_positive_weight={active_config.hard_positive_weight}")
        _log(f"uncertain_weight={active_config.uncertain_weight}")
        _log(f"max_extra_sampling_multiplier={active_config.max_extra_sampling_multiplier}")
        _log(f"weighted_training_samples={len(sample_loss_weights)}")
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
    load_start_checkpoint_if_configured(model, active_config, device)
    if active_config.imbalance_strategy == "bce":
        loss_fn = nn.BCEWithLogitsLoss(reduction="none" if sample_loss_weights else "mean").to(device)
        pos_weight = 1.0
    elif active_config.imbalance_strategy.startswith("focal"):
        loss_fn = create_binary_focal_loss(
            negative_count=train_counts["0"],
            positive_count=train_counts["1"],
            alpha=active_config.focal_alpha,
            gamma=active_config.focal_gamma,
            use_pos_weight="weighted" in active_config.imbalance_strategy,
            device=device,
        ).to(device)
        pos_weight = train_counts["0"] / train_counts["1"]
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
        sample_loss_weights=sample_loss_weights if sample_loss_weights else None,
    )
    _log(f"DataLoader created train_batches={len(loaders.train)} validation_batches={len(loaders.validation)}")

    best_state = None
    best_metrics = None
    best_threshold = None
    best_probs: list[float] = []
    best_epoch = 0
    for epoch in range(1, active_config.epochs + 1):
        model.train()
        for batch_index, batch in enumerate(loaders.train, start=1):
            if active_config.limit_train_batches is not None and batch_index > active_config.limit_train_batches:
                break
            batch_inputs, batch_labels, _image_ids, batch_weights = _unpack_training_batch(batch)
            batch_inputs = batch_inputs.to(device, non_blocking=non_blocking)
            batch_labels = batch_labels.to(device, non_blocking=non_blocking)
            batch_weights = None if batch_weights is None else batch_weights.to(device, non_blocking=non_blocking)
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss_values = loss_fn(logits, batch_labels)
            loss = _reduce_training_loss(loss_values, batch_weights)
            loss.backward()
            optimizer.step()
            if batch_index == 1 or batch_index % active_config.log_every_n_batches == 0:
                elapsed = time.perf_counter() - started
                _log(f"epoch={epoch} batch={batch_index}/{len(loaders.train)} loss={loss.item():.6f} elapsed_seconds={elapsed:.2f}")
        scheduler.step()

        validation_labels, probabilities = _collect_validation_predictions(
            model,
            loaders.validation,
            device=device,
            limit_batches=active_config.limit_val_batches,
        )
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
    model_output = _default_model_output_for_config(active_config)
    metrics_output = _default_metrics_output_for_config(active_config)
    threshold_output = _default_threshold_output_for_config(active_config)
    predictions_output = _default_predictions_output_for_config(active_config)
    model_path = _resolve_output_path(output_root, model_output)
    metrics_path = _resolve_output_path(output_root, metrics_output)
    threshold_path = _resolve_output_path(output_root, threshold_output)
    label_distribution_path = _resolve_output_path(output_root, DEFAULT_LABEL_DISTRIBUTION_OUTPUT)
    split_distribution_path = _resolve_output_path(output_root, DEFAULT_SPLIT_DISTRIBUTION_OUTPUT)
    predictions_path = _resolve_output_path(output_root, predictions_output)
    for path in (model_path, metrics_path, threshold_path, label_distribution_path, split_distribution_path, predictions_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_replay_snapshot_path = output_root / "reports" / "approved_manifest_snapshot.csv"
    cleaned_replay_run_manifest_path = output_root / "reports" / "cleaned_replay_run_manifest.json"
    cleaned_replay_config_snapshot_path = output_root / "reports" / "cleaned_replay_config_snapshot.json"
    cleaned_replay_runtime_path = output_root / "benchmarks" / "runtime_metadata.json"
    cleaned_replay_comparison_path = output_root / "reports" / "cleaned_replay_comparison.json"
    if active_config.cleaned_replay_enabled:
        for path in (
            cleaned_replay_snapshot_path,
            cleaned_replay_run_manifest_path,
            cleaned_replay_config_snapshot_path,
            cleaned_replay_runtime_path,
            cleaned_replay_comparison_path,
        ):
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
            hard_example_weighting={
                "hard_example_strategy": active_config.hard_example_strategy,
                "hard_negative_count": int(v2_2_hard_example_summary.get("hard_negative_count", 0)),
                "hard_positive_count": int(v2_2_hard_example_summary.get("hard_positive_count", 0)),
                "uncertain_count": int(v2_2_hard_example_summary.get("uncertain_count", 0)),
                "hard_negative_weight": active_config.hard_negative_weight,
                "hard_positive_weight": active_config.hard_positive_weight,
                "uncertain_weight": active_config.uncertain_weight,
                "weighted_training_samples": len(sample_loss_weights),
                "outputs_isolated_under": str(output_root),
                "used_test_labels": False,
                "submission_created": False,
            },
            imbalance_strategy=active_config.imbalance_strategy,
            hard_example_source_used=(
                hard_example_report.hard_example_source_used
                if hard_example_report is not None
                else str(v2_2_hard_example_summary.get("hard_negatives_path", ""))
            ),
            split_disjointness={
                **split_disjointness,
                "hard_examples": (
                    asdict(hard_example_report) if hard_example_report is not None else v2_2_hard_example_summary
                ),
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
        comparison_path = _resolve_output_path(output_root, _default_comparison_output_for_config(active_config))
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
    if active_config.controlled_phase3:
        emit_phase3_post_training_reports(
            config=active_config,
            output_root=output_root,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
            threshold_path=threshold_path,
            runtime_seconds=runtime_seconds,
        )
    if active_config.cleaned_replay_enabled:
        approved_rows = _read_csv_rows(active_config.approved_manifest_path, required_columns=["image_id"])
        with cleaned_replay_snapshot_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(approved_rows[0].keys()))
            writer.writeheader()
            writer.writerows(approved_rows)
        cleaned_replay_config_snapshot_path.write_text(json.dumps(asdict(active_config), indent=2), encoding="utf-8")
        cleaned_replay_runtime_path.write_text(
            json.dumps(
                {
                    "runtime_seconds": runtime_seconds,
                    "train_row_count": len(split.train),
                    "validation_row_count": len(split.validation),
                    "device": device.type,
                    "model_name": selected_model_name,
                    "synthetic_smoke": synthetic_smoke,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cleaned_replay_run_manifest_path.write_text(
            json.dumps(
                {
                    "spec_id": "018",
                    "experiment_name": active_config.experiment_name,
                    "approved_manifest_path": active_config.approved_manifest_path,
                    "approved_manifest_count": len(cleaned_replay_approved_ids),
                    "train_row_count": len(split.train),
                    "validation_row_count": len(split.validation),
                    "seed": active_config.seed,
                    "split_source": active_config.split_source,
                    "model_checkpoint_path": str(model_path),
                    "validation_predictions_path": str(predictions_path),
                    "best_threshold_path": str(threshold_path),
                    "metrics_path": str(metrics_path),
                    "config_snapshot_path": str(cleaned_replay_config_snapshot_path),
                    "manifest_snapshot_path": str(cleaned_replay_snapshot_path),
                    "runtime_metadata_path": str(cleaned_replay_runtime_path),
                    "dry_run_report_path": "" if cleaned_replay_report_path is None else str(cleaned_replay_report_path),
                    "safety": cleaned_replay_validation_payload,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        comparison_payload = build_cleaned_replay_comparison_record(
            baseline_name=active_config.baseline_name,
            baseline_checkpoint_path=active_config.baseline_checkpoint_path,
            validation_predictions_path=active_config.locked_baseline_predictions_path,
            best_threshold_path=active_config.locked_baseline_threshold_path,
            metrics_path=active_config.locked_baseline_metrics_path,
            runtime_report_path=active_config.locked_baseline_runtime_report_path,
            cleaned_metrics_path=metrics_path,
            cleaned_threshold_path=threshold_path,
            cleaned_predictions_path=predictions_path,
            cleaned_runtime_path=cleaned_replay_runtime_path,
            dry_run_report_path=cleaned_replay_report_path if cleaned_replay_report_path is not None else cleaned_replay_run_manifest_path,
        )
        write_cleaned_replay_comparison_report(comparison_payload, cleaned_replay_comparison_path)
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


def load_locked_baseline_report(
    *,
    baseline_name: str,
    baseline_checkpoint_path: str | Path,
    validation_predictions_path: str | Path,
    best_threshold_path: str | Path,
    metrics_path: str | Path,
    runtime_report_path: str | Path,
) -> LockedBaselineReport:
    checkpoint = Path(baseline_checkpoint_path)
    predictions = Path(validation_predictions_path)
    threshold = Path(best_threshold_path)
    metrics = Path(metrics_path)
    runtime = Path(runtime_report_path)
    for required_path in (checkpoint, predictions, threshold, metrics, runtime):
        if not required_path.exists():
            raise TrainingValidationError(f"Locked baseline artifact missing: {required_path}")
    metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
    threshold_payload = json.loads(threshold.read_text(encoding="utf-8"))
    runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
    confusion = metrics_payload.get("confusion_counts", {})
    report = LockedBaselineReport(
        baseline_name=baseline_name,
        baseline_checkpoint_path=str(checkpoint),
        validation_predictions_path=str(predictions),
        best_threshold_path=str(threshold),
        validation_f1=float(metrics_payload["f1_score"]),
        precision=float(metrics_payload["confusion_counts"]["tp"]) / max(
            1,
            int(metrics_payload["confusion_counts"]["tp"]) + int(metrics_payload["confusion_counts"]["fp"]),
        ),
        recall=float(metrics_payload["confusion_counts"]["tp"]) / max(
            1,
            int(metrics_payload["confusion_counts"]["tp"]) + int(metrics_payload["confusion_counts"]["fn"]),
        ),
        tp=int(confusion["tp"]),
        fp=int(confusion["fp"]),
        fn=int(confusion["fn"]),
        tn=int(confusion["tn"]),
        validation_target_distribution={
            str(key): int(value) for key, value in metrics_payload.get("validation_target_distribution", {}).items()
        },
        test_target_distribution={
            str(key): int(value) for key, value in metrics_payload.get("test_prediction_distribution", {}).items()
        },
        runtime_reference={
            "runtime_report_path": str(runtime),
            "best_threshold": float(threshold_payload["threshold"]),
            **runtime_payload,
        },
    )
    return report


def write_locked_baseline_report(report: LockedBaselineReport, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return output


def build_governed_phase3_training_manifest(
    candidate_package_path: str | Path,
    *,
    output_path: str | Path | None = None,
    default_source_candidate_package_path: str | Path | None = None,
) -> list[GovernedPhase3Row]:
    rows = list(csv.DictReader(Path(candidate_package_path).open("r", newline="", encoding="utf-8")))
    if not rows:
        raise TrainingValidationError("Approved candidate package contains no rows")
    governed: list[GovernedPhase3Row] = []
    rejected_blocked = 0
    for row in rows:
        if "test" in row:
            raise TrainingValidationError("Phase 3 candidate package must not include test labels")
        allowed = _csv_bool(row.get("phase3_use_allowed"))
        blocked = _csv_bool(row.get("blocked")) or str(row.get("approved_candidate_action", "")).strip().lower() in {
            "blocked",
            "reject",
            "rejected",
            "exclude",
        }
        if blocked:
            rejected_blocked += 1
            continue
        if not allowed:
            continue
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            raise TrainingValidationError("Governed Phase 3 rows require image_id")
        source_candidate_package_path = str(row.get("source_candidate_package_path", "")).strip()
        if not source_candidate_package_path and default_source_candidate_package_path is not None:
            source_candidate_package_path = str(default_source_candidate_package_path).strip()
        if not source_candidate_package_path:
            raise TrainingValidationError("Governed Phase 3 rows must remain traceable to a source candidate package path")
        governed.append(
            GovernedPhase3Row(
                image_id=image_id,
                target=int(row.get("target", "0")),
                approved_candidate_action=str(row.get("approved_candidate_action", "")).strip(),
                phase3_use_allowed=True,
                source_candidate_package_path=source_candidate_package_path,
                baseline_membership_context=str(row.get("baseline_membership_context", "")).strip(),
                hard_example_group=str(row.get("hard_example_group", "")).strip(),
            )
        )
    if not governed:
        raise TrainingValidationError("No governed Phase 3 rows remained after applying allow/block rules")
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image_id",
                    "target",
                    "approved_candidate_action",
                    "phase3_use_allowed",
                    "source_candidate_package_path",
                    "baseline_membership_context",
                    "hard_example_group",
                ],
            )
            writer.writeheader()
            for item in governed:
                writer.writerow(asdict(item))
    return governed


def validate_baseline_lock_targets(
    baseline_report: LockedBaselineReport,
    *,
    candidate_output_root: str | Path,
) -> None:
    candidate_root = Path(candidate_output_root).resolve()
    baseline_paths = [
        Path(baseline_report.baseline_checkpoint_path).resolve(),
        Path(baseline_report.validation_predictions_path).resolve(),
        Path(baseline_report.best_threshold_path).resolve(),
    ]
    for baseline_path in baseline_paths:
        if baseline_path == candidate_root or candidate_root in baseline_path.parents or baseline_path in candidate_root.parents:
            raise TrainingValidationError("Candidate outputs must not overwrite or nest locked baseline artifacts")


def validate_locked_baseline_artifacts(
    *,
    config: TrainingRunConfig,
    candidate_output_root: str | Path,
) -> LockedBaselineReport:
    baseline_report = load_locked_baseline_report(
        baseline_name=config.baseline_name,
        baseline_checkpoint_path=config.baseline_checkpoint_path,
        validation_predictions_path=config.locked_baseline_predictions_path,
        best_threshold_path=config.locked_baseline_threshold_path,
        metrics_path=config.locked_baseline_metrics_path,
        runtime_report_path=config.locked_baseline_runtime_report_path,
    )
    validate_baseline_lock_targets(baseline_report, candidate_output_root=candidate_output_root)
    return baseline_report


def _load_required_image_ids(path: str | Path, *, label: str, allow_empty: bool = False) -> set[str]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise TrainingValidationError(f"{label} CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if "image_id" not in fieldnames:
        raise TrainingValidationError(f"{label} CSV must contain image_id column: {csv_path}")
    image_ids = {str(row["image_id"]).strip() for row in rows if str(row["image_id"]).strip()}
    if not image_ids and not allow_empty:
        raise TrainingValidationError(f"{label} CSV contains no image_id rows: {csv_path}")
    return image_ids


def _fail_on_overlap(
    primary_ids: set[str],
    secondary_ids: set[str],
    *,
    overlap_label: str,
) -> list[str]:
    overlap = sorted(primary_ids & secondary_ids)
    if overlap:
        raise TrainingValidationError(f"Cleaned replay {overlap_label}: {', '.join(overlap[:10])}")
    return overlap


def _validate_cleaned_replay_dataset_membership(
    approved_ids: set[str],
    *,
    train_csv: Path,
    train_images_dir: Path,
) -> tuple[list[str], list[str]]:
    rows = _read_csv_rows(train_csv, required_columns=["image_id", "target"])
    train_ids = {str(row["image_id"]).strip() for row in rows if str(row["image_id"]).strip()}
    missing_labels = sorted(approved_ids - train_ids)
    missing_images = sorted(image_id for image_id in approved_ids if not (train_images_dir / image_id).exists())
    return missing_labels, missing_images


def build_cleaned_replay_comparison_record(
    *,
    baseline_name: str,
    baseline_checkpoint_path: str | Path,
    validation_predictions_path: str | Path,
    best_threshold_path: str | Path,
    metrics_path: str | Path,
    runtime_report_path: str | Path,
    cleaned_metrics_path: str | Path,
    cleaned_threshold_path: str | Path,
    cleaned_predictions_path: str | Path,
    cleaned_runtime_path: str | Path,
    dry_run_report_path: str | Path,
) -> dict[str, object]:
    baseline_report = load_locked_baseline_report(
        baseline_name=baseline_name,
        baseline_checkpoint_path=baseline_checkpoint_path,
        validation_predictions_path=validation_predictions_path,
        best_threshold_path=best_threshold_path,
        metrics_path=metrics_path,
        runtime_report_path=runtime_report_path,
    )
    cleaned_metrics = json.loads(Path(cleaned_metrics_path).read_text(encoding="utf-8"))
    cleaned_threshold = json.loads(Path(cleaned_threshold_path).read_text(encoding="utf-8"))
    cleaned_predictions = _read_prediction_rows(cleaned_predictions_path)
    dry_run_payload = json.loads(Path(dry_run_report_path).read_text(encoding="utf-8"))
    safety_pass = (
        dry_run_payload.get("validation_split_status") == "passed"
        and not bool(dry_run_payload.get("training_started", False))
        and bool(dry_run_payload.get("no_test_labels_used", True))
        and bool(dry_run_payload.get("no_submission_created", True))
        and bool(dry_run_payload.get("no_leaderboard_tuning", True))
    )
    cleaned_f1 = float(cleaned_metrics["f1_score"])
    baseline_f1 = float(baseline_report.validation_f1)
    f1_delta = round(cleaned_f1 - baseline_f1, 6)
    if not safety_pass:
        decision = "inconclusive"
        reason = "Safety gate failed for cleaned replay dry-run validation."
    elif cleaned_f1 > baseline_f1:
        decision = "accepted"
        reason = "Cleaned replay beats the locked V2B validation F1 and safety gates passed."
    else:
        decision = "rejected"
        reason = "Cleaned replay did not beat the locked V2B validation F1."
    return {
        "baseline_name": baseline_report.baseline_name,
        "baseline_validation_f1": baseline_f1,
        "cleaned_replay_validation_f1": cleaned_f1,
        "f1_delta": f1_delta,
        "baseline_threshold": float(baseline_report.runtime_reference["best_threshold"]),
        "cleaned_replay_threshold": float(cleaned_threshold["threshold"]),
        "validation_row_count": len(cleaned_predictions),
        "decision": decision,
        "decision_reason": reason,
        "safety_gate_status": "passed" if safety_pass else "failed",
        "artifact_paths": {
            "baseline_predictions": str(validation_predictions_path),
            "cleaned_predictions": str(cleaned_predictions_path),
            "baseline_metrics": str(metrics_path),
            "cleaned_metrics": str(cleaned_metrics_path),
            "baseline_threshold": str(best_threshold_path),
            "cleaned_threshold": str(cleaned_threshold_path),
            "cleaned_runtime": str(cleaned_runtime_path),
            "dry_run_report": str(dry_run_report_path),
        },
    }


def write_cleaned_replay_comparison_report(payload: dict[str, object], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def validate_candidate_hypothesis(record: CandidateHypothesisRecord) -> None:
    if not record.candidate_name.strip():
        raise TrainingValidationError("Controlled Phase 3 candidates require a candidate_name")
    if not record.hypothesis.strip():
        raise TrainingValidationError("Controlled Phase 3 candidates require an explicit hypothesis")
    if not record.baseline_reference.strip():
        raise TrainingValidationError("Controlled Phase 3 candidates require a locked baseline reference")


def validate_candidate_artifact_package(
    artifact_paths: dict[str, str | Path],
    *,
    require_runtime: bool = True,
    require_submission: bool = True,
) -> str:
    required = {"validation_predictions", "test_probabilities", "best_threshold", "metrics", "target_distribution"}
    if require_runtime:
        required.add("runtime_report")
    if require_submission:
        required.add("submission")
    missing = sorted(name for name in required if name not in artifact_paths or not Path(artifact_paths[name]).exists())
    return "complete" if not missing else f"incomplete:{','.join(missing)}"


def build_candidate_evaluation_record(
    *,
    candidate_name: str,
    baseline_report: LockedBaselineReport,
    metrics_path: str | Path,
    validation_predictions_path: str | Path,
    test_probabilities_path: str | Path,
    best_threshold_path: str | Path,
    runtime_report_path: str | Path,
    target_distribution_path: str | Path,
    submission_path: str | Path,
    hard_example_image_ids: Sequence[str],
    ablation_status: str,
    decision: str,
    decision_reason: str,
    require_submission: bool = True,
) -> CandidateEvaluationRecord:
    metrics_payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    threshold_payload = json.loads(Path(best_threshold_path).read_text(encoding="utf-8"))
    candidate_predictions = _read_prediction_rows(validation_predictions_path)
    baseline_predictions = _read_prediction_rows(baseline_report.validation_predictions_path)
    changed_rows = _build_changed_rows_vs_baseline(candidate_predictions, baseline_predictions)
    hard_ids = set(hard_example_image_ids)
    hard_metrics = _compute_subset_metrics(candidate_predictions, hard_ids)
    non_hard_metrics = _compute_subset_metrics(candidate_predictions, set(candidate_predictions) - hard_ids)
    artifact_paths = {
        "validation_predictions": str(Path(validation_predictions_path)),
        "test_probabilities": str(Path(test_probabilities_path)),
        "best_threshold": str(Path(best_threshold_path)),
        "metrics": str(Path(metrics_path)),
        "runtime_report": str(Path(runtime_report_path)),
        "target_distribution": str(Path(target_distribution_path)),
        "submission": str(Path(submission_path)),
    }
    completeness = validate_candidate_artifact_package(artifact_paths, require_submission=require_submission)
    finalist_ready = completeness == "complete" and decision == "accepted"
    runtime_payload = json.loads(Path(runtime_report_path).read_text(encoding="utf-8"))
    return CandidateEvaluationRecord(
        candidate_name=candidate_name,
        validation_f1=float(metrics_payload["f1_score"]),
        hard_example_f1=hard_metrics.f1_score,
        non_hard_example_f1=non_hard_metrics.f1_score,
        precision=_precision_from_confusion(metrics_payload["confusion_counts"]),
        recall=_recall_from_confusion(metrics_payload["confusion_counts"]),
        fp_count=int(metrics_payload["confusion_counts"]["fp"]),
        fn_count=int(metrics_payload["confusion_counts"]["fn"]),
        best_threshold=float(threshold_payload["threshold"]),
        threshold_stability_summary={
            "candidate_count": int(threshold_payload.get("candidate_count", 0)),
            "selection_source": str(threshold_payload.get("selection_source", "validation_only")),
            "threshold": float(threshold_payload["threshold"]),
        },
        changed_rows_vs_baseline=changed_rows,
        validation_target_distribution={
            str(key): int(value) for key, value in metrics_payload.get("validation_target_distribution", {}).items()
        },
        prediction_target_distribution=_prediction_distribution_from_prediction_rows(candidate_predictions),
        runtime_report_path=str(Path(runtime_report_path)),
        artifact_paths=artifact_paths,
        ablation_status=ablation_status,
        decision=decision,
        decision_reason=decision_reason,
        baseline_reference=baseline_report.baseline_name,
        artifact_completeness_status=completeness,
        submission_path=str(Path(submission_path)),
        finalist_ready=finalist_ready,
        best_of_n_runtime=float(runtime_payload.get("best_runtime_seconds", runtime_payload.get("total_time_seconds", 0.0))),
    )


def write_candidate_comparison_table(
    evaluations: Sequence[CandidateEvaluationRecord],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_name",
        "baseline_reference",
        "validation_f1",
        "hard_example_f1",
        "non_hard_example_f1",
        "precision",
        "recall",
        "fp_count",
        "fn_count",
        "best_threshold",
        "best_of_n_runtime",
        "artifact_completeness_status",
        "decision",
        "decision_reason",
        "submission_path",
        "finalist_ready",
        "ablation_status",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        seen: set[str] = set()
        for evaluation in evaluations:
            if evaluation.candidate_name in seen:
                raise TrainingValidationError("Every serious candidate must appear exactly once in the comparison table")
            seen.add(evaluation.candidate_name)
            writer.writerow({name: getattr(evaluation, name) for name in fieldnames})
    return output


def aggregate_runtime_evidence(
    candidate_name: str,
    runtime_report_paths: Sequence[str | Path],
) -> RuntimeEvidencePack:
    if not runtime_report_paths:
        raise TrainingValidationError("Runtime evidence requires repeated benchmark runs")
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in runtime_report_paths]
    best_runtime = min(float(item["total_time_seconds"]) for item in payloads)
    average_runtime = sum(float(item["total_time_seconds"]) for item in payloads) / len(payloads)
    best_payload = min(payloads, key=lambda item: float(item["total_time_seconds"]))
    return RuntimeEvidencePack(
        candidate_name=candidate_name,
        timing_run_count=len(payloads),
        best_runtime_seconds=round(best_runtime, 6),
        average_runtime_seconds=round(average_runtime, 6),
        seconds_per_image=float(best_payload["average_time_per_image"]),
        images_per_second=float(best_payload["images_per_second"]),
        device=str(best_payload["device"]),
        batch_size=int(best_payload["batch_size"]),
        image_size=int(best_payload["image_size"]),
        benchmark_reference=str(best_payload.get("benchmark_reference", "")),
    )


def build_insight_evidence_pack(
    *,
    baseline_report: LockedBaselineReport,
    evaluations: Sequence[CandidateEvaluationRecord],
    runtime_pack: RuntimeEvidencePack,
    final_recommendation_candidate: str,
) -> InsightEvidencePack:
    by_name = {item.candidate_name: item for item in evaluations}
    if final_recommendation_candidate not in by_name:
        raise TrainingValidationError("Final recommendation must reference a known controlled candidate")
    recommended = by_name[final_recommendation_candidate]
    rejected = [item for item in evaluations if item.candidate_name != final_recommendation_candidate]
    return InsightEvidencePack(
        baseline_summary={
            "baseline_name": baseline_report.baseline_name,
            "validation_f1": baseline_report.validation_f1,
            "precision": baseline_report.precision,
            "recall": baseline_report.recall,
        },
        hard_example_failure_modes=[
            {
                "candidate_name": item.candidate_name,
                "hard_example_f1": item.hard_example_f1,
                "decision": item.decision,
                "decision_reason": item.decision_reason,
            }
            for item in evaluations
        ],
        candidate_outcomes=[
            {
                "candidate_name": item.candidate_name,
                "validation_f1": item.validation_f1,
                "hard_example_f1": item.hard_example_f1,
                "runtime_seconds": item.best_of_n_runtime,
                "decision": item.decision,
            }
            for item in evaluations
        ],
        runtime_tradeoff_summary={
            "candidate_name": runtime_pack.candidate_name,
            "timing_run_count": runtime_pack.timing_run_count,
            "best_runtime_seconds": runtime_pack.best_runtime_seconds,
            "average_runtime_seconds": runtime_pack.average_runtime_seconds,
        },
        final_recommendation={
            "candidate_name": recommended.candidate_name,
            "decision": recommended.decision,
            "decision_reason": recommended.decision_reason,
            "validation_f1": recommended.validation_f1,
            "hard_example_f1": recommended.hard_example_f1,
        },
        rejected_alternatives=[
            {
                "candidate_name": item.candidate_name,
                "decision": item.decision,
                "decision_reason": item.decision_reason,
            }
            for item in rejected
        ],
    )


def emit_phase3_post_training_reports(
    *,
    config: TrainingRunConfig,
    output_root: str | Path,
    predictions_path: str | Path,
    metrics_path: str | Path,
    threshold_path: str | Path,
    runtime_seconds: float,
) -> dict[str, Path]:
    if not config.controlled_phase3:
        return {}

    reports_root = Path(output_root) / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    governed_manifest_path = reports_root / "phase3_governed_training_manifest.csv"
    baseline_lock_path = reports_root / "phase3_baseline_lock.json"
    run_manifest_path = reports_root / "phase3_run_manifest.json"
    clean_vs_hard_metrics_path = reports_root / "phase3_clean_vs_hard_metrics.json"
    changed_row_evidence_path = reports_root / "phase3_changed_row_evidence.csv"
    comparison_table_path = reports_root / "phase3_comparison_table.csv"
    runtime_summary_path = reports_root / "phase3_runtime_summary.json"
    insight_pack_path = reports_root / "phase3_insight_pack.json"

    governed_rows = build_governed_phase3_training_manifest(
        config.approved_candidate_package_path,
        output_path=governed_manifest_path,
        default_source_candidate_package_path=config.approved_candidate_package_path,
    )
    allowed_ids = {row.image_id for row in governed_rows}
    blocked_sources = {
        "blocked_roi_pipeline_bug_rows_path": Path(config.blocked_roi_pipeline_bug_rows_path),
        "blocked_suspected_mislabel_rows_path": Path(config.blocked_suspected_mislabel_rows_path),
        "blocked_other_high_risk_rows_path": Path(config.blocked_other_high_risk_rows_path),
    }
    blocked_row_counts: dict[str, int] = {}
    blocked_union: set[str] = set()
    for name, path in blocked_sources.items():
        if not path.exists():
            raise TrainingValidationError(f"Phase 3 blocked-row CSV not found: {path}")
        ids = _load_normalized_image_ids(path)
        blocked_row_counts[name] = len(ids)
        blocked_union.update(ids)
    overlap = sorted(allowed_ids & blocked_union)
    if overlap:
        raise TrainingValidationError(f"Phase 3 allowed/blocked overlap detected: {', '.join(overlap[:10])}")

    baseline_report = load_locked_baseline_report(
        baseline_name=config.baseline_name,
        baseline_checkpoint_path=config.baseline_checkpoint_path,
        validation_predictions_path=config.locked_baseline_predictions_path,
        best_threshold_path=config.locked_baseline_threshold_path,
        metrics_path=config.locked_baseline_metrics_path,
        runtime_report_path=config.locked_baseline_runtime_report_path,
    )
    write_locked_baseline_report(baseline_report, baseline_lock_path)

    candidate_predictions = _read_prediction_rows(predictions_path)
    baseline_predictions = _read_prediction_rows(baseline_report.validation_predictions_path)
    hard_metrics = _compute_subset_metrics(candidate_predictions, allowed_ids)
    clean_metrics = _compute_subset_metrics(candidate_predictions, set(candidate_predictions) - allowed_ids)
    blocked_metrics = _compute_subset_metrics(candidate_predictions, blocked_union & set(candidate_predictions))
    metrics_payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    threshold_payload = json.loads(Path(threshold_path).read_text(encoding="utf-8"))

    clean_vs_hard_payload = {
        "full_validation_f1": float(metrics_payload["f1_score"]),
        "allowed_hard_f1": hard_metrics.f1_score,
        "clean_validation_f1": clean_metrics.f1_score,
        "blocked_audit_f1": blocked_metrics.f1_score,
        "allowed_hard_count": sum(1 for image_id in candidate_predictions if image_id in allowed_ids),
        "clean_validation_count": sum(1 for image_id in candidate_predictions if image_id not in allowed_ids),
        "blocked_audit_count": sum(1 for image_id in candidate_predictions if image_id in blocked_union),
    }
    clean_vs_hard_metrics_path.write_text(json.dumps(clean_vs_hard_payload, indent=2), encoding="utf-8")

    changed_rows = _build_changed_rows_vs_baseline(candidate_predictions, baseline_predictions)
    with changed_row_evidence_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "target",
                "baseline_probability",
                "baseline_prediction",
                "phase3_probability",
                "phase3_prediction",
                "change_type",
                "baseline_correct",
                "phase3_correct",
                "allowed_hard_row",
                "blocked_row",
            ],
        )
        writer.writeheader()
        for row in changed_rows:
            target = int(row["true_label"])
            baseline_prediction = int(row["baseline_prediction"])
            candidate_prediction = int(row["candidate_prediction"])
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "target": target,
                    "baseline_probability": row["baseline_probability"],
                    "baseline_prediction": baseline_prediction,
                    "phase3_probability": row["candidate_probability"],
                    "phase3_prediction": candidate_prediction,
                    "change_type": f"{baseline_prediction}_to_{candidate_prediction}",
                    "baseline_correct": baseline_prediction == target,
                    "phase3_correct": candidate_prediction == target,
                    "allowed_hard_row": row["image_id"] in allowed_ids,
                    "blocked_row": row["image_id"] in blocked_union,
                }
            )

    runtime_pack = RuntimeEvidencePack(
        candidate_name=config.experiment_name,
        timing_run_count=1,
        best_runtime_seconds=round(runtime_seconds, 6),
        average_runtime_seconds=round(runtime_seconds, 6),
        seconds_per_image=round(runtime_seconds / max(1, len(candidate_predictions)), 6),
        images_per_second=round(len(candidate_predictions) / max(runtime_seconds, 1e-9), 6),
        device=str(metrics_payload.get("device", "cpu")),
        batch_size=int(config.batch_size),
        image_size=int(config.image_size),
        benchmark_reference="validation_runtime",
    )
    runtime_summary_path.write_text(json.dumps(asdict(runtime_pack), indent=2), encoding="utf-8")

    evaluation = build_candidate_evaluation_record(
        candidate_name=config.experiment_name,
        baseline_report=baseline_report,
        metrics_path=metrics_path,
        validation_predictions_path=predictions_path,
        test_probabilities_path=predictions_path,
        best_threshold_path=threshold_path,
        runtime_report_path=runtime_summary_path,
        target_distribution_path=clean_vs_hard_metrics_path,
        submission_path=reports_root / "submission_not_created.csv",
        hard_example_image_ids=sorted(allowed_ids),
        ablation_status="not_run",
        decision="analysis_only",
        decision_reason="Controlled Phase 3 post-training report package emitted without submission export.",
        require_submission=False,
    )
    write_candidate_comparison_table([evaluation], comparison_table_path)

    insight_pack = build_insight_evidence_pack(
        baseline_report=baseline_report,
        evaluations=[evaluation],
        runtime_pack=runtime_pack,
        final_recommendation_candidate=config.experiment_name,
    )
    insight_pack_path.write_text(json.dumps(asdict(insight_pack), indent=2), encoding="utf-8")

    run_manifest = {
        "spec_id": "015",
        "experiment_name": config.experiment_name,
        "allowed_hard_row_count": len(allowed_ids),
        "blocked_row_counts": blocked_row_counts,
        "blocked_overlap_count": 0,
        "used_test_labels": False,
        "submission_enabled": False,
        "leaderboard_tuning_enabled": False,
        "training_completed": True,
        "output_root": str(output_root),
        "normal_artifacts": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "threshold": str(threshold_path),
        },
        "phase3_report_paths": {
            "governed_training_manifest": str(governed_manifest_path),
            "baseline_lock": str(baseline_lock_path),
            "clean_vs_hard_metrics": str(clean_vs_hard_metrics_path),
            "changed_row_evidence": str(changed_row_evidence_path),
            "comparison_table": str(comparison_table_path),
            "runtime_summary": str(runtime_summary_path),
            "insight_pack": str(insight_pack_path),
        },
        "selected_threshold": float(threshold_payload["threshold"]),
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return {
        "governed_training_manifest": governed_manifest_path,
        "baseline_lock": baseline_lock_path,
        "run_manifest": run_manifest_path,
        "clean_vs_hard_metrics": clean_vs_hard_metrics_path,
        "changed_row_evidence": changed_row_evidence_path,
        "comparison_table": comparison_table_path,
        "runtime_summary": runtime_summary_path,
        "insight_pack": insight_pack_path,
    }


def run_phase3_dry_run_validation(
    config_path: str | Path,
    *,
    dataset_root_override: str | Path | None = None,
    output_root_override: str | Path | None = None,
) -> Path:
    config = load_classifier_config(config_path)
    if not config.controlled_phase3:
        raise TrainingValidationError("Phase 3 dry-run requires a controlled Phase 3 config")

    dataset_root = Path(dataset_root_override) if dataset_root_override is not None else Path(config.dataset_root)
    if not config.dataset_root and dataset_root_override is None:
        raise TrainingValidationError("Phase 3 dry-run requires data.dataset_root")
    if not dataset_root.exists():
        raise TrainingValidationError(f"Phase 3 dataset root not found: {dataset_root}")
    train_csv = dataset_root / "train.csv"
    train_images_dir = dataset_root / "train_images"
    if not train_csv.exists():
        raise TrainingValidationError(f"Phase 3 dry-run requires train.csv: {train_csv}")
    if not train_images_dir.exists():
        raise TrainingValidationError(f"Phase 3 dry-run requires train_images directory: {train_images_dir}")

    allowed_path = Path(config.approved_candidate_package_path)
    if not allowed_path.exists():
        raise TrainingValidationError(f"Phase 3 allowed candidate CSV not found: {allowed_path}")
    allowed_ids = _load_normalized_image_ids(allowed_path)
    if len(allowed_ids) != 422:
        raise TrainingValidationError(f"Phase 3 allowed candidate CSV must contain 422 unique rows, found {len(allowed_ids)}")

    blocked_sources = {
        "blocked_roi_pipeline_bug_rows_path": Path(config.blocked_roi_pipeline_bug_rows_path),
        "blocked_suspected_mislabel_rows_path": Path(config.blocked_suspected_mislabel_rows_path),
        "blocked_other_high_risk_rows_path": Path(config.blocked_other_high_risk_rows_path),
    }
    blocked_counts: dict[str, int] = {}
    blocked_examples: list[str] = []
    blocked_union: set[str] = set()
    for name, path in blocked_sources.items():
        if not path.exists():
            raise TrainingValidationError(f"Phase 3 blocked-row CSV not found: {path}")
        blocked_ids = _load_normalized_image_ids(path)
        blocked_counts[name] = len(blocked_ids)
        blocked_union.update(blocked_ids)
    overlap = sorted(allowed_ids & blocked_union)
    if overlap:
        blocked_examples = overlap[:10]
        raise TrainingValidationError(f"Phase 3 allowed/blocked overlap detected: {', '.join(blocked_examples)}")

    baseline_report = load_locked_baseline_report(
        baseline_name=config.baseline_name,
        baseline_checkpoint_path=config.baseline_checkpoint_path,
        validation_predictions_path=config.locked_baseline_predictions_path,
        best_threshold_path=config.locked_baseline_threshold_path,
        metrics_path=config.locked_baseline_metrics_path,
        runtime_report_path=config.locked_baseline_runtime_report_path,
    )

    output_root = Path(output_root_override) if output_root_override is not None else Path(config.output_root or "outputs")
    report_path = output_root / "reports" / "phase3_dry_run_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_id": "015",
        "config_path": str(Path(config_path)),
        "allowed_hard_rows_csv": str(allowed_path),
        "allowed_hard_row_count": len(allowed_ids),
        "blocked_sources": {name: str(path) for name, path in blocked_sources.items()},
        "blocked_row_counts": blocked_counts,
        "blocked_overlap_count": 0,
        "blocked_overlap_examples": blocked_examples,
        "used_test_labels": False,
        "submission_enabled": False,
        "leaderboard_tuning_enabled": False,
        "training_started": False,
        "output_root": str(output_root),
        "baseline_lock_report_path": str(config.locked_baseline_metrics_path),
        "baseline_name": baseline_report.baseline_name,
        "baseline_threshold_path": baseline_report.best_threshold_path,
        "baseline_predictions_path": baseline_report.validation_predictions_path,
        "dataset_root": str(dataset_root),
        "train_csv": str(train_csv),
        "train_images_dir": str(train_images_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def run_cleaned_replay_dry_run_validation(
    config_or_path: TrainingRunConfig | str | Path,
    *,
    dataset_root_override: str | Path | None = None,
    output_root_override: str | Path | None = None,
) -> Path:
    config = config_or_path if isinstance(config_or_path, TrainingRunConfig) else load_classifier_config(config_or_path)
    if not config.cleaned_replay_enabled:
        raise TrainingValidationError("Cleaned replay dry-run requires a cleaned replay config")

    dataset_root = Path(dataset_root_override) if dataset_root_override is not None else Path(config.dataset_root)
    if not config.dataset_root and dataset_root_override is None:
        raise TrainingValidationError("Cleaned replay dry-run requires data.dataset_root")
    if not dataset_root.exists():
        raise TrainingValidationError(f"Cleaned replay dataset root not found: {dataset_root}")
    train_csv = dataset_root / "train.csv"
    train_images_dir = dataset_root / "train_images"
    if not train_csv.exists():
        raise TrainingValidationError(f"Cleaned replay dry-run requires train.csv: {train_csv}")
    if not train_images_dir.exists():
        raise TrainingValidationError(f"Cleaned replay dry-run requires train_images directory: {train_images_dir}")

    approved_ids = _load_required_image_ids(config.approved_manifest_path, label="Cleaned replay approved manifest")
    auto_exclude_ids = _load_required_image_ids(config.auto_exclude_rows_path, label="Cleaned replay auto-exclude", allow_empty=True)
    adjudication_ids = _load_required_image_ids(
        config.needs_adjudication_rows_path,
        label="Cleaned replay needs-adjudication",
        allow_empty=True,
    )
    deferred_ids = _load_required_image_ids(config.deferred_uncertain_rows_path, label="Cleaned replay deferred", allow_empty=True)
    decision_lock_summary = Path(config.decision_lock_summary_path)
    if not decision_lock_summary.exists():
        raise TrainingValidationError(f"Cleaned replay decision summary not found: {decision_lock_summary}")

    _fail_on_overlap(approved_ids, auto_exclude_ids, overlap_label="auto-excluded overlap")
    _fail_on_overlap(approved_ids, adjudication_ids, overlap_label="needs-adjudication overlap")
    _fail_on_overlap(approved_ids, deferred_ids, overlap_label="deferred overlap")

    missing_labels, missing_images = _validate_cleaned_replay_dataset_membership(
        approved_ids,
        train_csv=train_csv,
        train_images_dir=train_images_dir,
    )
    if missing_labels:
        raise TrainingValidationError("Cleaned replay missing approved training labels: " + ", ".join(missing_labels[:10]))
    if missing_images:
        raise TrainingValidationError("Cleaned replay missing approved training images: " + ", ".join(missing_images[:10]))

    baseline_report = validate_locked_baseline_artifacts(config=config, candidate_output_root=output_root_override or config.output_root or "outputs")
    output_root = Path(output_root_override) if output_root_override is not None else Path(config.output_root or "outputs")
    report_path = output_root / "reports" / "cleaned_replay_dry_run_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_id": "018",
        "config_path": str(config_or_path) if not isinstance(config_or_path, TrainingRunConfig) else config.experiment_name,
        "approved_manifest_path": config.approved_manifest_path,
        "approved_manifest_count": len(approved_ids),
        "auto_exclude_overlap_count": 0,
        "needs_adjudication_overlap_count": 0,
        "deferred_overlap_count": 0,
        "missing_train_label_count": 0,
        "missing_train_image_count": 0,
        "baseline_artifacts_available": True,
        "validation_split_status": "passed",
        "training_started": False,
        "no_test_labels_used": not config.allow_test_labels,
        "no_leaderboard_tuning": not config.public_leaderboard_input,
        "no_submission_created": not config.generate_submission,
        "warnings": [],
        "baseline_name": baseline_report.baseline_name,
        "dataset_root": str(dataset_root),
        "train_csv": str(train_csv),
        "train_images_dir": str(train_images_dir),
        "decision_lock_summary_path": str(decision_lock_summary),
        "output_root": str(output_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


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


def _csv_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_prediction_rows(path: str | Path) -> dict[str, dict[str, object]]:
    rows = list(csv.DictReader(Path(path).open("r", newline="", encoding="utf-8")))
    parsed: dict[str, dict[str, object]] = {}
    for row in rows:
        image_id = str(row["image_id"])
        parsed[image_id] = {
            "image_id": image_id,
            "true_label": int(row["true_label"]),
            "probability": float(row["probability"]),
            "threshold": float(row["threshold"]),
            "predicted_label": int(row["predicted_label"]),
        }
    return parsed


def _build_changed_rows_vs_baseline(
    candidate_predictions: dict[str, dict[str, object]],
    baseline_predictions: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    changed: list[dict[str, object]] = []
    for image_id, candidate_row in sorted(candidate_predictions.items()):
        baseline_row = baseline_predictions.get(image_id)
        if baseline_row is None:
            continue
        if int(candidate_row["predicted_label"]) != int(baseline_row["predicted_label"]):
            changed.append(
                {
                    "image_id": image_id,
                    "true_label": int(candidate_row["true_label"]),
                    "baseline_prediction": int(baseline_row["predicted_label"]),
                    "candidate_prediction": int(candidate_row["predicted_label"]),
                    "baseline_probability": float(baseline_row["probability"]),
                    "candidate_probability": float(candidate_row["probability"]),
                }
            )
    return changed


def _compute_subset_metrics(
    prediction_rows: dict[str, dict[str, object]],
    subset_image_ids: set[str],
) -> BinaryMetrics:
    filtered = [row for image_id, row in prediction_rows.items() if image_id in subset_image_ids]
    if not filtered:
        return BinaryMetrics(
            f1_score=0.0,
            threshold=0.5,
            confusion_counts={"tp": 0, "fp": 0, "fn": 0, "tn": 0},
            class_counts={"0": 0, "1": 0},
        )
    y_true = [int(row["true_label"]) for row in filtered]
    probabilities = [float(row["probability"]) for row in filtered]
    threshold = float(filtered[0]["threshold"])
    return compute_binary_metrics(y_true=y_true, probabilities=probabilities, threshold=threshold)


def _precision_from_confusion(confusion: dict[str, int]) -> float:
    tp = int(confusion["tp"])
    fp = int(confusion["fp"])
    return tp / max(1, tp + fp)


def _recall_from_confusion(confusion: dict[str, int]) -> float:
    tp = int(confusion["tp"])
    fn = int(confusion["fn"])
    return tp / max(1, tp + fn)


def _prediction_distribution_from_prediction_rows(prediction_rows: dict[str, dict[str, object]]) -> dict[str, int]:
    zeros = sum(1 for row in prediction_rows.values() if int(row["predicted_label"]) == 0)
    ones = sum(1 for row in prediction_rows.values() if int(row["predicted_label"]) == 1)
    return {"0": zeros, "1": ones}


def _load_normalized_image_ids(path: str | Path) -> set[str]:
    rows = list(csv.DictReader(Path(path).open("r", newline="", encoding="utf-8")))
    if not rows:
        return set()
    fieldnames = rows[0].keys()
    if "image_id" not in fieldnames:
        raise TrainingValidationError(f"CSV must contain image_id column: {path}")
    return {str(row["image_id"]).strip() for row in rows if str(row["image_id"]).strip()}


def load_start_checkpoint_if_configured(
    model: torch.nn.Module,
    config: TrainingRunConfig,
    device: torch.device,
) -> Optional[Path]:
    """Load a configured fine-tuning checkpoint into an initialized model."""

    checkpoint_value = config.start_checkpoint.strip()
    if not checkpoint_value:
        if _requires_start_checkpoint(config):
            raise TrainingValidationError("V2B-Enhanced training requires start_checkpoint to be configured and loaded")
        return None

    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Configured start_checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_model_state_dict(checkpoint, checkpoint_path)
    model.load_state_dict(state_dict)
    _log(f"loaded_start_checkpoint={checkpoint_path}")
    return checkpoint_path


def _extract_model_state_dict(checkpoint: object, checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise TrainingValidationError(f"Unsupported start_checkpoint format: {checkpoint_path}")


def _requires_start_checkpoint(config: TrainingRunConfig) -> bool:
    return "v2b_enhanced" in config.experiment_name.lower()


def _is_v2_2_config(config: TrainingRunConfig) -> bool:
    return config.experiment_name == "v2_2_hard_examples"


def _default_model_output_for_config(config: TrainingRunConfig) -> Path:
    if config.cleaned_replay_enabled:
        return Path("outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/models/classifier_best.pth")
    if config.v5:
        return V5_MODEL_OUTPUT
    if _is_v2_2_config(config):
        return V2_2_MODEL_OUTPUT
    if config.v2:
        return V2_MODEL_OUTPUT
    return DEFAULT_MODEL_OUTPUT


def _default_metrics_output_for_config(config: TrainingRunConfig) -> Path:
    if config.cleaned_replay_enabled:
        return Path("outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/classifier_metrics.json")
    if config.v5:
        return V5_METRICS_OUTPUT
    if _is_v2_2_config(config):
        return V2_2_METRICS_OUTPUT
    if config.v2:
        return V2_METRICS_OUTPUT
    return DEFAULT_METRICS_OUTPUT


def _default_threshold_output_for_config(config: TrainingRunConfig) -> Path:
    if config.cleaned_replay_enabled:
        return Path("outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/best_threshold.json")
    if config.v5:
        return V5_THRESHOLD_OUTPUT
    if _is_v2_2_config(config):
        return V2_2_THRESHOLD_OUTPUT
    if config.v2:
        return V2_THRESHOLD_OUTPUT
    return DEFAULT_THRESHOLD_OUTPUT


def _default_predictions_output_for_config(config: TrainingRunConfig) -> Path:
    if config.cleaned_replay_enabled:
        return Path("outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/predictions/val_classifier_predictions.csv")
    if config.v5:
        return V5_PREDICTIONS_OUTPUT
    if _is_v2_2_config(config):
        return V2_2_PREDICTIONS_OUTPUT
    if config.v2:
        return V2_PREDICTIONS_OUTPUT
    return DEFAULT_PREDICTIONS_OUTPUT


def _default_comparison_output_for_config(config: TrainingRunConfig) -> Path:
    if config.cleaned_replay_enabled:
        return Path("outputs/kaggle_v2b_cleaned_replay/cleaned_replay_v2b/reports/cleaned_replay_comparison.json")
    if _is_v2_2_config(config):
        return V2_2_COMPARISON_OUTPUT
    return V2_COMPARISON_OUTPUT


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


def build_conservative_hard_example_weight_map(
    report: object,
    *,
    train_image_ids: Sequence[str],
    config: TrainingRunConfig,
) -> dict[str, float]:
    """Return conservative per-image loss weights for V2.2 strong hard examples."""

    if config.hard_example_strategy != CONSERVATIVE_HARD_EXAMPLE_STRATEGY:
        return {}
    train_ids = set(train_image_ids)
    weights: dict[str, float] = {}
    for image_id in getattr(report, "hard_negative_image_ids", []):
        if image_id in train_ids:
            weights[image_id] = float(config.hard_negative_weight)
    for image_id in getattr(report, "hard_positive_image_ids", []):
        if image_id in train_ids:
            weights[image_id] = float(config.hard_positive_weight)
    return weights


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
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run-phase3", action="store_true")
    parser.add_argument("--dry-run-cleaned-replay", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--epochs", "--max-epochs", dest="epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--log-every-n-batches", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_classifier_config(args.config)
    overrides: dict[str, object] = {}
    if args.max_train_samples is not None:
        overrides["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        overrides["max_val_samples"] = args.max_val_samples
    if args.limit_train_batches is not None:
        overrides["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        overrides["limit_val_batches"] = args.limit_val_batches
    if args.log_every_n_batches is not None:
        overrides["log_every_n_batches"] = args.log_every_n_batches
    if overrides:
        config = _replace_config(config, **overrides)
    if args.dry_run_phase3:
        report_path = run_phase3_dry_run_validation(
            args.config,
            dataset_root_override=args.dataset_root,
            output_root_override=args.output_root,
        )
        print(f"Phase 3 dry-run validation complete: {report_path}")
        return 0
    if args.dry_run_cleaned_replay:
        report_path = run_cleaned_replay_dry_run_validation(
            args.config,
            dataset_root_override=args.dataset_root,
            output_root_override=args.output_root,
        )
        print(f"Cleaned replay dry-run validation complete: {report_path}")
        return 0
    dataset_root = args.dataset_root or config.dataset_root
    if not dataset_root:
        parser.error("--dataset-root is required unless data.dataset_root is configured")
    output_root = args.output_root or config.output_root or "outputs"
    result = run_training(
        dataset_root=dataset_root,
        output_root=output_root,
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
    limit_batches: Optional[int] = None,
) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_index, (batch_inputs, batch_labels, _image_ids) in enumerate(loader, start=1):
            if limit_batches is not None and batch_index > limit_batches:
                break
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().astype(float).tolist())
            labels.extend(batch_labels.cpu().numpy().astype(int).tolist())
    return labels, probabilities


def _unpack_training_batch(
    batch: Sequence[object],
) -> tuple[torch.Tensor, torch.Tensor, Sequence[str], Optional[torch.Tensor]]:
    if len(batch) == 4:
        batch_inputs, batch_labels, image_ids, batch_weights = batch
        return batch_inputs, batch_labels, image_ids, batch_weights
    batch_inputs, batch_labels, image_ids = batch
    return batch_inputs, batch_labels, image_ids, None


def _reduce_training_loss(loss_values: torch.Tensor, batch_weights: Optional[torch.Tensor]) -> torch.Tensor:
    if batch_weights is None:
        return loss_values.mean() if loss_values.ndim > 0 else loss_values
    return (loss_values * batch_weights).mean()


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
