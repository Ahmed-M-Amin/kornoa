"""Dataset audit utilities for SPEC-002."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def run_dataset_audit(
    dataset_root: str | Path,
    output_root: str | Path = "outputs",
    generate_outputs: bool = True,
) -> dict[str, Any]:
    """Audit a dataset root and optionally write report/figure outputs."""
    started = time.perf_counter()
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)

    train_rows = _read_csv_rows(dataset_root / "train.csv")
    annotations = _read_coco_annotations(dataset_root / "train_annotations.json")
    train_images = _discover_images(dataset_root / "train_images", "train")
    test_images = _discover_images(dataset_root / "test_images", "test")

    class_distribution = Counter(str(row["label"]) for row in train_rows)
    annotation_summary = _summarize_annotations(annotations)
    image_size_summary = train_images + test_images

    missing_train_images = [
        row["image_id"]
        for row in train_rows
        if f"{row['image_id']}.jpg" not in {image["file_name"] for image in train_images}
    ]

    result: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "train_label_count": len(train_rows),
        "train_image_count": len(train_images),
        "test_image_count": len(test_images),
        "missing_train_images": missing_train_images,
        "annotation_coverage": annotation_summary["coverage"],
        "orphaned_annotations": annotation_summary["orphaned_annotations"],
        "roi_availability": annotation_summary["roi_availability"],
        "class_distribution": dict(class_distribution),
        "defect_distribution": annotation_summary["defect_distribution"],
        "image_size_summary": image_size_summary,
    }
    result["runtime_seconds"] = round(time.perf_counter() - started, 6)

    if generate_outputs:
        _write_reports(result, output_root / "reports")
        _write_figures(result, dataset_root, output_root / "figures")

    return result


def validate_generated_artifact_confidentiality(
    artifact_paths: list[str | Path],
) -> dict[str, dict[str, bool]]:
    """Check that generated report/figure files are ignored and untracked by git."""
    paths = [Path(path) for path in artifact_paths]
    tracked = {_normalize_git_path(path) for path in _run_git(["ls-files", "--", *[str(path) for path in paths]])}
    ignored = {_normalize_git_path(path) for path in _run_git(["check-ignore", "--", *[str(path) for path in paths]])}

    return {
        str(path): {
            "ignored": _normalize_git_path(path) in ignored,
            "tracked": _normalize_git_path(path) in tracked,
        }
        for path in paths
    }


def _normalize_git_path(path: str | Path) -> str:
    normalized = str(path).strip('"')
    normalized = normalized.replace("\\\\", "/")
    return normalized.replace("\\", "/")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_coco_annotations(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _discover_images(image_dir: Path, split: str) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "split": split,
                "image_id": image_path.stem,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "path": str(image_path),
            }
        )
    return images


def _summarize_annotations(coco: dict[str, Any]) -> dict[str, Any]:
    images_by_id = {image["id"]: image for image in coco.get("images", [])}
    categories_by_id = {
        category["id"]: category["name"] for category in coco.get("categories", [])
    }

    image_ids_with_annotations: set[Any] = set()
    image_ids_with_roi: set[Any] = set()
    orphaned_annotations: list[dict[str, Any]] = []
    defect_distribution: Counter[str] = Counter()

    for annotation in coco.get("annotations", []):
        image_id = annotation.get("image_id")
        category_name = categories_by_id.get(annotation.get("category_id"), "unknown")
        has_roi = bool(annotation.get("bbox") or annotation.get("segmentation"))

        if image_id not in images_by_id:
            orphaned_annotations.append(
                {
                    "annotation_id": annotation.get("id"),
                    "image_id": image_id,
                    "category_name": category_name,
                    "has_roi": has_roi,
                }
            )
            continue

        image_ids_with_annotations.add(image_id)
        if has_roi:
            image_ids_with_roi.add(image_id)
        defect_distribution[category_name] += 1

    image_count = len(images_by_id)
    annotated_count = len(image_ids_with_annotations)
    roi_count = len(image_ids_with_roi)

    return {
        "coverage": {
            "annotation_image_count": image_count,
            "annotated_image_count": annotated_count,
            "unannotated_image_count": image_count - annotated_count,
            "coverage_ratio": annotated_count / image_count if image_count else 0.0,
        },
        "roi_availability": {
            "images_with_roi": roi_count,
            "images_without_roi": image_count - roi_count,
            "roi_ratio": roi_count / image_count if image_count else 0.0,
        },
        "orphaned_annotations": orphaned_annotations,
        "defect_distribution": dict(defect_distribution),
    }


def _write_reports(result: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    _write_key_value_csv(
        reports_dir / "dataset_summary.csv",
        {
            "train_label_count": result["train_label_count"],
            "train_image_count": result["train_image_count"],
            "test_image_count": result["test_image_count"],
            "orphaned_annotation_count": len(result["orphaned_annotations"]),
            "annotated_image_count": result["annotation_coverage"][
                "annotated_image_count"
            ],
            "images_with_roi": result["roi_availability"]["images_with_roi"],
            "runtime_seconds": result["runtime_seconds"],
        },
    )
    _write_distribution_csv(
        reports_dir / "class_distribution.csv",
        "label",
        result["class_distribution"],
    )
    _write_distribution_csv(
        reports_dir / "defect_distribution.csv",
        "defect_label",
        result["defect_distribution"],
    )
    _write_rows_csv(
        reports_dir / "orphaned_annotations.csv",
        ["annotation_id", "image_id", "category_name", "has_roi"],
        result["orphaned_annotations"],
    )
    _write_rows_csv(
        reports_dir / "missing_files.csv",
        ["image_id"],
        [{"image_id": image_id} for image_id in result["missing_train_images"]],
    )
    _write_rows_csv(
        reports_dir / "image_size_summary.csv",
        ["split", "image_id", "file_name", "width", "height", "path"],
        result["image_size_summary"],
    )


def _write_key_value_csv(path: Path, values: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in values.items():
            writer.writerow({"metric": metric, "value": value})


def _write_distribution_csv(path: Path, key_name: str, distribution: dict[str, int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key_name, "count"])
        writer.writeheader()
        for key, count in sorted(distribution.items()):
            writer.writerow({key_name: key, "count": count})


def _write_rows_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(result: dict[str, Any], dataset_root: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    _write_bar_figure(
        figures_dir / "class_distribution.png",
        "Class Distribution",
        result["class_distribution"],
    )
    _write_bar_figure(
        figures_dir / "defect_distribution.png",
        "Defect Distribution",
        result["defect_distribution"],
    )
    _write_sample_grid(dataset_root, figures_dir / "sample_grid.png")


def _write_bar_figure(path: Path, title: str, distribution: dict[str, int]) -> None:
    labels = list(distribution.keys()) or ["none"]
    counts = list(distribution.values()) or [0]
    max_count = max(counts) or 1
    image = Image.new("RGB", (420, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), title, fill="black")

    chart_left = 48
    chart_bottom = 250
    chart_height = 170
    bar_width = max(24, min(70, 260 // len(labels)))
    gap = 20

    for index, (label, count) in enumerate(zip(labels, counts)):
        left = chart_left + index * (bar_width + gap)
        top = chart_bottom - int((count / max_count) * chart_height)
        draw.rectangle((left, top, left + bar_width, chart_bottom), fill="#2f6f9f")
        draw.text((left, chart_bottom + 8), str(label), fill="black")
        draw.text((left, max(top - 16, 40)), str(count), fill="black")

    image.save(path)


def _write_sample_grid(dataset_root: Path, path: Path) -> None:
    sample_paths = sorted((dataset_root / "train_images").glob("*.jpg"))[:4]
    cell_size = 120
    image = Image.new("RGB", (cell_size * max(len(sample_paths), 1), cell_size), "white")
    draw = ImageDraw.Draw(image)

    for index, image_path in enumerate(sample_paths):
        with Image.open(image_path) as sample:
            sample = sample.resize((100, 100))
            image.paste(sample, (index * cell_size + 10, 10))
        draw.text((index * cell_size + 10, 104), image_path.stem, fill="black")

    image.save(path)


def _run_git(args: list[str]) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.strip())
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]
