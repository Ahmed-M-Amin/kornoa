"""SPEC-008 detector command wrapper for conversion, manual training, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from src.data.yolo_converter import run_audit as run_yolo_audit
from src.data.yolo_converter import run_convert as run_yolo_convert
from src.data.yolo_converter import run_visualize as run_yolo_visualize
from src.data.yolo_converter import _jsonable


DEFAULT_DETECTOR_CONFIG = Path("configs/detector.yaml")


class DetectorConfigError(ValueError):
    """Raised when detector configuration is invalid."""


def load_detector_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise DetectorConfigError(f"Detector config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = raw.get("detector", raw)
    required = ("annotation_path", "train_images", "output_root")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise DetectorConfigError(f"Detector config missing required keys: {', '.join(missing)}")
    train_images = Path(config["train_images"])
    if any(part.lower() == "test_images" for part in train_images.parts):
        raise DetectorConfigError("Detector training source must not use test_images")
    return config


def run_audit(config: dict[str, Any]) -> int:
    return run_yolo_audit(
        annotation_path=config["annotation_path"],
        image_dir=config["train_images"],
        output_root=config["output_root"],
    )


def run_convert(config: dict[str, Any]) -> int:
    return run_yolo_convert(
        annotation_path=config["annotation_path"],
        image_dir=config["train_images"],
        output_root=config["output_root"],
        seed=int(config.get("seed", 42)),
        val_fraction=float(config.get("val_fraction", 0.2)),
    )


def run_visualize(config: dict[str, Any], *, max_samples: int) -> int:
    return run_yolo_visualize(
        annotation_path=config["annotation_path"],
        image_dir=config["train_images"],
        output_root=config["output_root"],
        max_samples=max_samples,
    )


def run_train(config: dict[str, Any], *, model: str, dry_run: bool) -> int:
    output_root = Path(config["output_root"])
    dataset_yaml = output_root / "dataset" / "data.yaml"
    report = {
        "model": model,
        "dataset_yaml": str(dataset_yaml),
        "image_size": int(config.get("image_size", 640)),
        "output_dir": str(output_root / "models"),
        "dry_run": dry_run,
        "command": (
            f"yolo detect train model={model}.pt data={dataset_yaml} "
            f"imgsz={int(config.get('image_size', 640))} project={output_root / 'models'}"
        ),
    }
    if not dry_run:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorConfigError("ultralytics is required for detector training") from exc
        YOLO(f"{model}.pt").train(
            data=str(dataset_yaml),
            imgsz=int(config.get("image_size", 640)),
            project=str(output_root / "models"),
        )
    _write_json(output_root / "reports" / "detector_training.json", report)
    return 0


def run_evaluate(config: dict[str, Any], *, model_path: str | Path, dry_run: bool) -> int:
    output_root = Path(config["output_root"])
    dataset_yaml = output_root / "dataset" / "data.yaml"
    report: dict[str, Any] = {
        "model_path": str(model_path),
        "dataset_yaml": str(dataset_yaml),
        "dry_run": dry_run,
        "evaluation_mode": "dry_run" if dry_run else "ultralytics",
        "map": "unavailable",
        "precision": "unavailable",
        "recall": "unavailable",
        "per_category_metrics": "unavailable",
        "per_category_metrics_available": False,
        "validation_prediction_examples": [],
        "validation_prediction_examples_available": False,
    }
    if not dry_run:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorConfigError("ultralytics is required for detector evaluation") from exc
        metrics = YOLO(str(model_path)).val(data=str(dataset_yaml))
        box_metrics = getattr(metrics, "box", None)
        report.update(
            {
                "map": getattr(box_metrics, "map", "unavailable"),
                "precision": getattr(box_metrics, "mp", "unavailable"),
                "recall": getattr(box_metrics, "mr", "unavailable"),
            }
        )
    _write_json(output_root / "reports" / "detector_evaluation.json", report)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SPEC-008 detector preparation, manual training, and evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("audit", "convert", "visualize", "train", "evaluate"):
        sub = subparsers.add_parser(name, help=f"Run detector {name}.")
        sub.add_argument("--config", type=Path, default=DEFAULT_DETECTOR_CONFIG)
        if name == "visualize":
            sub.add_argument("--max-samples", type=int, default=8)
        if name == "train":
            sub.add_argument("--model", choices=("yolo11n", "yolov8n"), default="yolo11n")
            sub.add_argument("--dry-run", action="store_true")
        if name == "evaluate":
            sub.add_argument("--model-path", type=Path, required=True)
            sub.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    config = load_detector_config(args.config)
    if args.command == "audit":
        return run_audit(config)
    if args.command == "convert":
        return run_convert(config)
    if args.command == "visualize":
        return run_visualize(config, max_samples=args.max_samples)
    if args.command == "train":
        return run_train(config, model=args.model, dry_run=args.dry_run)
    if args.command == "evaluate":
        return run_evaluate(config, model_path=args.model_path, dry_run=args.dry_run)
    raise DetectorConfigError(f"Unsupported detector command: {args.command}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
