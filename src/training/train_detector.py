"""SPEC-008 detector dataset, visualization, training, and evaluation commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.data.coco_parser import load_coco_annotations
from src.data.yolo_converter import (
    audit_detector_annotations,
    build_category_mapping,
    convert_coco_to_yolo_dataset,
    create_detector_visualizations,
)


DEFAULT_DETECTOR_CONFIG = Path("configs/detector.yaml")


class DetectorConfigError(ValueError):
    """Raised when detector configuration is invalid."""


def load_detector_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise DetectorConfigError(f"Detector config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    detector = raw.get("detector", raw)
    required = ("annotation_path", "train_images", "output_root")
    missing = [key for key in required if not detector.get(key)]
    if missing:
        raise DetectorConfigError(f"Detector config missing required keys: {', '.join(missing)}")
    train_images = Path(detector["train_images"])
    if any(part.lower() == "test_images" for part in train_images.parts):
        raise DetectorConfigError("Detector training source must not use test_images")
    return detector


def run_audit(config: dict[str, Any]) -> int:
    output_root = Path(config["output_root"])
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    coco = load_coco_annotations(config["annotation_path"])
    mapping = build_category_mapping(coco)
    audit = audit_detector_annotations(coco, image_dir=config["train_images"])
    _write_json(reports_dir / "category_mapping.json", mapping)
    _write_json(reports_dir / "annotation_audit.json", audit)
    return 0


def run_convert(config: dict[str, Any]) -> int:
    convert_coco_to_yolo_dataset(
        annotation_path=config["annotation_path"],
        image_dir=config["train_images"],
        output_root=config["output_root"],
        seed=int(config.get("seed", 42)),
        val_fraction=float(config.get("val_fraction", 0.2)),
    )
    return 0


def run_visualize(config: dict[str, Any], *, max_samples: int) -> int:
    output_root = Path(config["output_root"])
    report = create_detector_visualizations(
        annotation_path=config["annotation_path"],
        image_dir=config["train_images"],
        output_dir=output_root / "figures",
        max_samples=max_samples,
    )
    _write_json(output_root / "reports" / "visualization_report.json", report)
    return 0


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
        yolo = YOLO(f"{model}.pt")
        yolo.train(data=str(dataset_yaml), imgsz=int(config.get("image_size", 640)), project=str(output_root / "models"))
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
        report.update(
            {
                "map": getattr(getattr(metrics, "box", None), "map", "unavailable"),
                "precision": getattr(getattr(metrics, "box", None), "mp", "unavailable"),
                "recall": getattr(getattr(metrics, "box", None), "mr", "unavailable"),
            }
        )
    _write_json(output_root / "reports" / "detector_evaluation.json", report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SPEC-008 detector preparation and training commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("audit", "convert", "visualize", "train", "evaluate"):
        sub = subparsers.add_parser(name)
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
    from src.data.yolo_converter import _jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
