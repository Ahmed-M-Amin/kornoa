"""SPEC-008 COCO audit, YOLO conversion, visualization, and CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml
from PIL import Image, ImageDraw

from src.data.coco_parser import AnnotationRecord, CocoParseResult, load_coco_annotations


GENERATED_SPLIT_DIRS = [
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels/train",
    "dataset/labels/val",
]


@dataclass(frozen=True)
class CategoryMappingItem:
    category_id: int | str
    category_name: str
    class_index: int
    annotation_count: int


@dataclass(frozen=True)
class CategoryMapping:
    items: list[CategoryMappingItem]

    @property
    def id_to_index(self) -> dict[int | str, int]:
        return {item.category_id: item.class_index for item in self.items}

    @property
    def names(self) -> list[str]:
        return [item.category_name for item in self.items]


@dataclass(frozen=True)
class InvalidAnnotation:
    annotation_id: int | str
    image_id: int | str
    reason: str
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class DetectorAuditReport:
    image_count: int
    annotation_count: int
    category_distribution: dict[str, int]
    bbox_size_distribution: dict[str, float | int]
    images_without_annotations: list[str]
    images_with_multiple_defects: list[str]
    missing_image_references: list[str]
    invalid_boxes: list[InvalidAnnotation]
    invalid_masks: list[int | str]
    segmentation_available_count: int


@dataclass(frozen=True)
class DetectorSplit:
    seed: int
    train_image_ids: list[str]
    validation_image_ids: list[str]
    stratification_key: str
    stratification_fallback_reason: str
    train_validation_disjoint: bool
    split_strategy_used: str


@dataclass(frozen=True)
class YoloConversionResult:
    dataset_root: Path
    data_yaml_path: Path
    split: DetectorSplit
    category_mapping: CategoryMapping
    label_files: list[Path]
    conversion_report: dict[str, Any]


@dataclass(frozen=True)
class VisualizationReport:
    figure_paths: list[Path]
    mask_visualizations_available: bool


def build_category_mapping(coco: CocoParseResult) -> CategoryMapping:
    distribution = coco.category_distribution()
    items = [
        CategoryMappingItem(
            category_id=category.category_id,
            category_name=category.name,
            class_index=index,
            annotation_count=distribution.get(category.category_id, 0),
        )
        for index, category in enumerate(sorted(coco.categories, key=lambda item: str(item.category_id)))
    ]
    return CategoryMapping(items=items)


def audit_detector_annotations(coco: CocoParseResult, *, image_dir: str | Path) -> DetectorAuditReport:
    image_dir = Path(image_dir)
    images_by_id = {str(image.image_id): image for image in coco.images}
    category_names = {category.category_id: category.name for category in coco.categories}
    category_distribution: dict[str, int] = {category.name: 0 for category in coco.categories}
    invalid_boxes: list[InvalidAnnotation] = []
    missing_refs: list[str] = []
    valid_areas: list[float] = []

    for annotation in coco.annotations:
        category_name = category_names.get(annotation.category_id, str(annotation.category_id))
        category_distribution[category_name] = category_distribution.get(category_name, 0) + 1
        image = images_by_id.get(str(annotation.image_id))
        if image is None or not (image_dir / image.file_name).exists():
            missing_refs.append(str(annotation.image_id))
        invalid_reason = _invalid_bbox_reason(annotation, images_by_id)
        if invalid_reason is not None:
            invalid_boxes.append(
                InvalidAnnotation(
                    annotation_id=annotation.annotation_id,
                    image_id=annotation.image_id,
                    reason=invalid_reason,
                    bbox=annotation.bbox,
                )
            )
        elif annotation.bbox is not None:
            _, _, width, height = annotation.bbox
            valid_areas.append(float(width * height))

    images_without_annotations = [
        str(image.image_id) for image in coco.images if image.image_id not in coco.image_to_annotations
    ]
    images_with_multiple_defects = [
        str(image_id) for image_id, annotations in coco.image_to_annotations.items() if len(annotations) > 1
    ]
    return DetectorAuditReport(
        image_count=len(coco.images),
        annotation_count=len(coco.annotations),
        category_distribution=category_distribution,
        bbox_size_distribution={
            "valid_count": len(valid_areas),
            "min_area": min(valid_areas) if valid_areas else 0,
            "max_area": max(valid_areas) if valid_areas else 0,
        },
        images_without_annotations=images_without_annotations,
        images_with_multiple_defects=images_with_multiple_defects,
        missing_image_references=_ordered_unique(missing_refs),
        invalid_boxes=invalid_boxes,
        invalid_masks=[],
        segmentation_available_count=coco.segmentation_available_count(),
    )


def convert_coco_to_yolo_dataset(
    *,
    annotation_path: str | Path,
    image_dir: str | Path,
    output_root: str | Path,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> YoloConversionResult:
    coco = load_coco_annotations(annotation_path)
    image_dir = Path(image_dir)
    output_root = Path(output_root)
    dataset_root = output_root / "dataset"
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    mapping = build_category_mapping(coco)
    audit = audit_detector_annotations(coco, image_dir=image_dir)
    split = _make_split(coco, image_dir=image_dir, seed=seed, val_fraction=val_fraction)
    cleaned_paths = _clean_generated_dataset_dirs(dataset_root)

    image_by_id = {str(image.image_id): image for image in coco.images}
    label_lines = {image_id: [] for image_id in [*split.train_image_ids, *split.validation_image_ids]}
    skipped_invalid = 0
    skipped_missing = 0

    for annotation in coco.annotations:
        image = image_by_id.get(str(annotation.image_id))
        if image is None or str(annotation.image_id) not in label_lines:
            skipped_missing += 1
            continue
        if _invalid_bbox_reason(annotation, image_by_id) is not None:
            skipped_invalid += 1
            continue
        if annotation.category_id not in mapping.id_to_index or annotation.bbox is None:
            skipped_invalid += 1
            continue
        label_lines[str(annotation.image_id)].append(
            _to_yolo_line(mapping.id_to_index[annotation.category_id], annotation.bbox, image.width, image.height)
        )

    label_files: list[Path] = []
    for image_id in split.train_image_ids:
        label_files.append(_copy_image_and_write_label(image_by_id[image_id], image_dir, dataset_root, "train", label_lines[image_id]))
    for image_id in split.validation_image_ids:
        label_files.append(_copy_image_and_write_label(image_by_id[image_id], image_dir, dataset_root, "val", label_lines[image_id]))

    data_yaml_path = dataset_root / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "nc": len(mapping.items),
                "names": mapping.names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    no_annotation_ids = set(audit.images_without_annotations)
    conversion_report = {
        "converted_annotations": sum(len(lines) for lines in label_lines.values()),
        "skipped_invalid_annotations": skipped_invalid,
        "skipped_missing_images": skipped_missing,
        "no_annotation_train_count": sum(1 for image_id in split.train_image_ids if image_id in no_annotation_ids),
        "no_annotation_val_count": sum(1 for image_id in split.validation_image_ids if image_id in no_annotation_ids),
        "total_empty_label_files": sum(1 for lines in label_lines.values() if not lines),
        "output_dataset_cleaned": True,
        "cleaned_paths": cleaned_paths,
        "split_strategy_used": split.split_strategy_used,
        "stratification_fallback_reason": split.stratification_fallback_reason,
        "class_names": mapping.names,
    }
    _write_json(reports_dir / "category_mapping.json", mapping)
    _write_json(reports_dir / "annotation_audit.json", audit)
    _write_json(reports_dir / "detector_split.json", split)
    _write_json(reports_dir / "yolo_conversion_report.json", conversion_report)

    return YoloConversionResult(
        dataset_root=dataset_root,
        data_yaml_path=data_yaml_path,
        split=split,
        category_mapping=mapping,
        label_files=label_files,
        conversion_report=conversion_report,
    )


def create_detector_visualizations(
    *,
    annotation_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    max_samples: int = 8,
) -> VisualizationReport:
    coco = load_coco_annotations(annotation_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = build_category_mapping(coco)
    image_by_id = {str(image.image_id): image for image in coco.images}
    colors = ["red", "lime", "blue", "yellow", "magenta", "cyan"]
    figure_paths: list[Path] = []

    for image_id, annotations in coco.image_to_annotations.items():
        if len(figure_paths) >= max_samples:
            break
        image = image_by_id.get(str(image_id))
        if image is None:
            continue
        image_path = image_dir / image.file_name
        if not image_path.exists():
            continue
        with Image.open(image_path).convert("RGB") as opened:
            draw = ImageDraw.Draw(opened)
            for annotation in annotations:
                if annotation.bbox is None or _invalid_bbox_reason(annotation, image_by_id) is not None:
                    continue
                x, y, width, height = annotation.bbox
                class_index = mapping.id_to_index.get(annotation.category_id, 0)
                draw.rectangle([x, y, x + width, y + height], outline=colors[class_index % len(colors)], width=2)
            output_path = output_dir / f"{image_id}_detector_overlay.png"
            opened.save(output_path)
            figure_paths.append(output_path)

    return VisualizationReport(
        figure_paths=figure_paths,
        mask_visualizations_available=coco.segmentation_available_count() > 0,
    )


def run_audit(*, annotation_path: str | Path, image_dir: str | Path, output_root: str | Path) -> int:
    output_root = Path(output_root)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    coco = load_coco_annotations(annotation_path)
    mapping = build_category_mapping(coco)
    audit = audit_detector_annotations(coco, image_dir=image_dir)
    _write_json(reports_dir / "category_mapping.json", mapping)
    _write_json(reports_dir / "annotation_audit.json", audit)
    return 0


def run_convert(
    *,
    annotation_path: str | Path,
    image_dir: str | Path,
    output_root: str | Path,
    seed: int,
    val_fraction: float,
) -> int:
    convert_coco_to_yolo_dataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        output_root=output_root,
        seed=seed,
        val_fraction=val_fraction,
    )
    return 0


def run_visualize(
    *,
    annotation_path: str | Path,
    image_dir: str | Path,
    output_root: str | Path,
    max_samples: int,
) -> int:
    output_root = Path(output_root)
    report = create_detector_visualizations(
        annotation_path=annotation_path,
        image_dir=image_dir,
        output_dir=output_root / "figures",
        max_samples=max_samples,
    )
    _write_json(output_root / "reports" / "visualization_report.json", report)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SPEC-008 COCO audit and YOLO bbox dataset conversion.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit COCO annotations and write detector reports.")
    _add_source_args(audit)

    convert = subparsers.add_parser("convert", help="Convert COCO bbox annotations to a YOLO dataset.")
    _add_source_args(convert)
    convert.add_argument("--seed", type=int, default=42)
    convert.add_argument("--val-fraction", type=float, default=0.2)

    visualize = subparsers.add_parser("visualize", help="Save category-colored bbox overlay samples.")
    _add_source_args(visualize)
    visualize.add_argument("--max-samples", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command == "audit":
        return run_audit(annotation_path=args.annotation_path, image_dir=args.image_dir, output_root=args.output_root)
    if args.command == "convert":
        return run_convert(
            annotation_path=args.annotation_path,
            image_dir=args.image_dir,
            output_root=args.output_root,
            seed=args.seed,
            val_fraction=args.val_fraction,
        )
    if args.command == "visualize":
        return run_visualize(
            annotation_path=args.annotation_path,
            image_dir=args.image_dir,
            output_root=args.output_root,
            max_samples=args.max_samples,
        )
    raise ValueError(f"Unsupported command: {args.command}")


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--annotation-path", type=Path, required=True, help="Path to train_annotations.json.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing training images.")
    parser.add_argument("--output-root", type=Path, required=True, help="Detector output root.")


def _make_split(coco: CocoParseResult, *, image_dir: Path, seed: int, val_fraction: float) -> DetectorSplit:
    eligible = [str(image.image_id) for image in coco.images if (image_dir / image.file_name).exists()]
    strata = _primary_defect_strata(coco, eligible)
    if _can_stratify_by_primary_category(strata):
        train_ids, validation_ids = _stratified_split(strata, seed=seed, val_fraction=val_fraction)
        split_strategy_used = "primary_defect_category"
        fallback_reason = ""
        stratification_key = "primary_defect_category"
    else:
        train_ids, validation_ids = _deterministic_random_split(eligible, seed=seed, val_fraction=val_fraction)
        split_strategy_used = "deterministic_random"
        fallback_reason = "too few images per primary defect category for disjoint stratified split"
        stratification_key = "deterministic"
    return DetectorSplit(
        seed=seed,
        train_image_ids=train_ids,
        validation_image_ids=validation_ids,
        stratification_key=stratification_key,
        stratification_fallback_reason=fallback_reason,
        train_validation_disjoint=set(train_ids).isdisjoint(validation_ids),
        split_strategy_used=split_strategy_used,
    )


def _primary_defect_strata(coco: CocoParseResult, eligible_image_ids: Sequence[str]) -> dict[str, list[str]]:
    category_names = {category.category_id: category.name for category in coco.categories}
    annotations_by_image_id: dict[str, list[AnnotationRecord]] = {}
    for annotation in coco.annotations:
        annotations_by_image_id.setdefault(str(annotation.image_id), []).append(annotation)
    strata: dict[str, list[str]] = {}
    for image_id in eligible_image_ids:
        annotations = annotations_by_image_id.get(image_id, [])
        primary = "no_annotation"
        if annotations:
            primary = sorted(category_names.get(annotation.category_id, str(annotation.category_id)) for annotation in annotations)[0]
        strata.setdefault(primary, []).append(image_id)
    return strata


def _can_stratify_by_primary_category(strata: dict[str, list[str]]) -> bool:
    return bool(strata) and all(len(image_ids) >= 2 for image_ids in strata.values())


def _stratified_split(
    strata: dict[str, list[str]],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    train_ids: list[str] = []
    validation_ids: list[str] = []
    for _, image_ids in sorted(strata.items()):
        shuffled = sorted(image_ids)
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_fraction))
        val_count = min(max(1, val_count), len(shuffled) - 1)
        validation_ids.extend(shuffled[:val_count])
        train_ids.extend(shuffled[val_count:])
    return sorted(train_ids), sorted(validation_ids)


def _deterministic_random_split(eligible: Sequence[str], *, seed: int, val_fraction: float) -> tuple[list[str], list[str]]:
    shuffled = sorted(eligible)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    validation_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])
    return train_ids, validation_ids


def _clean_generated_dataset_dirs(dataset_root: Path) -> list[str]:
    for relative in GENERATED_SPLIT_DIRS:
        path = dataset_root.parent / relative
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    return list(GENERATED_SPLIT_DIRS)


def _copy_image_and_write_label(
    image: Any,
    image_dir: Path,
    dataset_root: Path,
    split_name: str,
    lines: Sequence[str],
) -> Path:
    source = image_dir / image.file_name
    destination_image = dataset_root / "images" / split_name / image.file_name
    destination_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination_image)
    label_path = dataset_root / "labels" / split_name / f"{Path(image.file_name).stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return label_path


def _invalid_bbox_reason(annotation: AnnotationRecord, images_by_id: dict[Any, Any]) -> str | None:
    if annotation.bbox is None:
        return "missing_or_malformed_bbox"
    image = images_by_id.get(str(annotation.image_id))
    x, y, width, height = annotation.bbox
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        return "non_positive_or_negative_bbox"
    if image is not None and (x + width > image.width or y + height > image.height):
        return "bbox_outside_image"
    return None


def _to_yolo_line(class_index: int, bbox: tuple[float, float, float, float], image_width: int, image_height: int) -> str:
    x, y, width, height = bbox
    return (
        f"{class_index} "
        f"{(x + width / 2) / image_width:.6f} {(y + height / 2) / image_height:.6f} "
        f"{width / image_width:.6f} {height / image_height:.6f}"
    )


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
