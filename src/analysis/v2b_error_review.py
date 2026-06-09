"""Read-only V2B error review and safe fine-tune preparation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml
from PIL import Image, ImageDraw


ERROR_GROUPS = [
    "v1_correct_v2b_wrong",
    "both_wrong",
    "high_confidence_wrong",
    "uncertain",
]
REQUIRED_PATH_KEYS = {"train_csv", "train_images", *ERROR_GROUPS}
OPTIONAL_PATH_KEYS = {"train_annotations"}
EMPTY_BREAKDOWN_COLUMNS = ["annotation_category", "error_group", "count", "ratio"]


class V2BErrorReviewError(ValueError):
    """Raised when V2B error review inputs are invalid."""


def load_review_config(path: str | Path) -> dict:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise V2BErrorReviewError(f"V2B error review config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise V2BErrorReviewError(f"V2B error review config is not valid YAML: {config_path}") from exc
    if not isinstance(raw, dict):
        raise V2BErrorReviewError("V2B error review config must be a mapping")
    paths = raw.get("paths")
    if not isinstance(paths, dict):
        raise V2BErrorReviewError("V2B error review config must contain a paths mapping")
    missing = sorted(REQUIRED_PATH_KEYS - set(paths))
    if missing:
        raise V2BErrorReviewError("V2B error review config missing required paths: " + ", ".join(missing))
    return raw


def load_error_groups(paths: dict[str, object]) -> dict[str, pd.DataFrame]:
    grouped: dict[str, pd.DataFrame] = {}
    for group in ERROR_GROUPS:
        path = Path(str(paths[group]))
        if not path.exists():
            raise V2BErrorReviewError(f"Missing required error CSV: {path}")
        df = pd.read_csv(path)
        image_col = _find_column(df, ["image_id", "id", "filename", "image", "path"])
        df = df.copy()
        df["image_id"] = df[image_col].map(_normalize_image_id)
        df["error_group"] = group
        df["source_csv"] = path.name
        grouped[group] = df
    return grouped


def run_review(config_path: str | Path) -> dict[str, object]:
    config = load_review_config(config_path)
    paths = config["paths"]
    _assert_required_inputs_exist(paths)

    output_root = Path(config.get("output", {}).get("root", "outputs/analysis/v2b_error_review"))
    reports_dir = output_root / "reports"
    hard_examples_dir = output_root / "hard_examples"
    sheets_dir = output_root / "contact_sheets"
    reports_dir.mkdir(parents=True, exist_ok=True)
    hard_examples_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_train_labels(Path(str(paths["train_csv"])))
    grouped = load_error_groups(paths)
    annotations, missing_optional = _load_optional_annotations(paths.get("train_annotations"))
    combined = _build_review_frame(grouped, labels, annotations)

    counts = _build_error_group_counts(combined)
    counts_path = reports_dir / "error_group_counts.csv"
    counts.to_csv(counts_path, index=False)

    breakdown, missing_annotation_warning = _build_annotation_breakdown(combined)
    if missing_annotation_warning:
        missing_optional.append(missing_annotation_warning)
    breakdown_path = reports_dir / "annotation_category_breakdown.csv"
    breakdown.to_csv(breakdown_path, index=False)

    hard_examples = _build_hard_examples(combined)
    hard_examples_path = hard_examples_dir / "v2b_hard_examples.csv"
    hard_examples.to_csv(hard_examples_path, index=False)

    contact_sheet_outputs, missing_images = _build_contact_sheets(
        combined,
        train_images_root=Path(str(paths["train_images"])),
        output_dir=sheets_dir,
        review_config=config.get("review", {}),
    )

    summary_path = reports_dir / "v2b_improvement_summary.json"
    summary = _build_summary(
        paths=paths,
        counts=counts,
        breakdown=breakdown,
        hard_examples=hard_examples,
        contact_sheet_outputs=contact_sheet_outputs,
        missing_images=missing_images,
        missing_optional=missing_optional,
    )
    _write_json(summary_path, summary)

    return {
        "summary": summary_path,
        "error_group_counts": counts_path,
        "annotation_breakdown": breakdown_path,
        "hard_examples": hard_examples_path,
        "contact_sheets": contact_sheet_outputs,
    }


def _assert_required_inputs_exist(paths: dict[str, object]) -> None:
    missing = [key for key in sorted(REQUIRED_PATH_KEYS) if not Path(str(paths[key])).exists()]
    if missing:
        details = ", ".join(f"{key}={paths[key]}" for key in missing)
        raise V2BErrorReviewError("Missing required V2B review input: " + details)


def _load_train_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    image_col = _find_column(df, ["image_id", "id", "filename", "image", "path"])
    target_col = _find_column(df, ["target", "label", "true_label", "y_true"])
    out = pd.DataFrame(
        {
            "image_id": df[image_col].map(_normalize_image_id),
            "target": pd.to_numeric(df[target_col], errors="raise").astype(int),
        }
    )
    if not set(out["target"]).issubset({0, 1}):
        raise V2BErrorReviewError("train.csv labels must be binary")
    return out.drop_duplicates("image_id")


def _load_optional_annotations(path_value: object) -> tuple[dict[str, list[str]], list[str]]:
    if not path_value:
        return {}, ["train_annotations"]
    path = Path(str(path_value))
    if not path.exists():
        return {}, ["train_annotations"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_image: dict[str, set[str]] = {}
    if isinstance(payload, dict):
        for annotation in payload.get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            image_id = annotation.get("image_id")
            if image_id in (None, ""):
                continue
            category = str(annotation.get("category_name", annotation.get("class_name", ""))).strip()
            if not category:
                continue
            by_image.setdefault(_normalize_image_id(image_id), set()).add(category)
    return {image_id: sorted(values) for image_id, values in by_image.items()}, []


def _build_review_frame(
    grouped: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    annotations: dict[str, list[str]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group, df in grouped.items():
        working = df.copy()
        working = working.merge(labels, on="image_id", how="left", suffixes=("", "_train"))
        if working["target"].isna().any():
            missing_ids = sorted(working.loc[working["target"].isna(), "image_id"].astype(str).unique().tolist())
            raise V2BErrorReviewError("train.csv is missing labels for: " + ", ".join(missing_ids))
        working["target"] = working["target"].astype(int)
        working["annotation_categories"] = working["image_id"].map(lambda item: "|".join(annotations.get(item, [])))
        working["v1_prediction"] = _extract_numeric(working, ["v1_target", "v1_prediction", "predicted_v1"], default=0).astype(int)
        working["v2b_prediction"] = _extract_numeric(working, ["v2b_target", "v2b_prediction", "predicted_v2b"], default=0).astype(int)
        working["v1_prob_bad"] = _extract_numeric(working, ["v1_score", "v1_prob_bad", "v1_probability"], default=0.0)
        working["v2b_prob_bad"] = _extract_numeric(working, ["v2b_score", "v2b_prob_bad", "v2b_probability"], default=0.0)
        working["v2b_confidence"] = (working["v2b_prob_bad"].astype(float) - 0.5).abs().round(6)
        working["review_priority"] = group_to_review_priority(group)
        working["error_group"] = group
        working["source_csv"] = working.get("source_csv", pd.Series([f"{group}.csv"] * len(working), index=working.index))
        frames.append(
            working[
                [
                    "image_id",
                    "target",
                    "error_group",
                    "v1_prediction",
                    "v2b_prediction",
                    "v1_prob_bad",
                    "v2b_prob_bad",
                    "v2b_confidence",
                    "annotation_categories",
                    "source_csv",
                    "review_priority",
                ]
            ].copy()
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_error_group_counts(review: pd.DataFrame) -> pd.DataFrame:
    total = len(review)
    rows = []
    for group in ERROR_GROUPS:
        count = int((review["error_group"] == group).sum())
        rows.append({"error_group": group, "count": count, "ratio": (count / total) if total else 0.0})
    return pd.DataFrame(rows)


def _build_annotation_breakdown(review: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if review.empty or review["annotation_categories"].eq("").all():
        return pd.DataFrame(columns=EMPTY_BREAKDOWN_COLUMNS), "train_annotations"
    exploded = review.copy()
    exploded["annotation_category"] = exploded["annotation_categories"].map(lambda value: str(value).split("|") if value else [])
    exploded = exploded.explode("annotation_category")
    exploded = exploded[exploded["annotation_category"].astype(str) != ""]
    if exploded.empty:
        return pd.DataFrame(columns=EMPTY_BREAKDOWN_COLUMNS), "train_annotations"
    rows = []
    total = len(review)
    for (category, group), subset in exploded.groupby(["annotation_category", "error_group"], dropna=False):
        rows.append(
            {
                "annotation_category": str(category),
                "error_group": str(group),
                "count": int(len(subset)),
                "ratio": len(subset) / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "annotation_category"], ascending=[False, True]), None


def _build_hard_examples(review: pd.DataFrame) -> pd.DataFrame:
    return review.sort_values(
        ["review_priority", "v2b_confidence", "image_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _build_contact_sheets(
    review: pd.DataFrame,
    *,
    train_images_root: Path,
    output_dir: Path,
    review_config: dict[str, object],
) -> tuple[dict[str, list[Path]], dict[str, list[str]]]:
    columns = max(1, int(review_config.get("contact_sheet_columns", 4)))
    thumb_size = max(32, int(review_config.get("contact_sheet_image_size", 160)))
    rows_per_sheet = max(1, int(review_config.get("contact_sheet_rows", 4)))
    page_size = columns * rows_per_sheet
    created: dict[str, list[Path]] = {}
    missing: dict[str, list[str]] = {}
    for group in ERROR_GROUPS:
        subset = review[review["error_group"] == group]
        images: list[tuple[str, Image.Image]] = []
        missing_ids: list[str] = []
        for image_id in subset["image_id"].astype(str).tolist():
            image_path = train_images_root / image_id
            if not image_path.exists():
                missing_ids.append(image_id)
                continue
            images.append((image_id, Image.open(image_path).convert("RGB")))
        missing[group] = missing_ids
        pages: list[Path] = []
        for page_index in range(max(1, math.ceil(len(images) / page_size))):
            page_items = images[page_index * page_size : (page_index + 1) * page_size]
            sheet = _render_contact_sheet(page_items, columns=columns, thumb_size=thumb_size, title=group)
            sheet_path = output_dir / f"{group}_sheet_{page_index + 1:03d}.jpg"
            sheet.save(sheet_path, format="JPEG", quality=90)
            pages.append(sheet_path)
        created[group] = pages
    return created, missing


def _render_contact_sheet(
    items: list[tuple[str, Image.Image]],
    *,
    columns: int,
    thumb_size: int,
    title: str,
) -> Image.Image:
    header_height = 32
    label_height = 18
    tile_height = thumb_size + label_height
    rows = max(1, math.ceil(max(1, len(items)) / columns))
    sheet = Image.new("RGB", (columns * thumb_size, header_height + rows * tile_height), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for index, (image_id, opened) in enumerate(items):
        row = index // columns
        col = index % columns
        x = col * thumb_size
        y = header_height + row * tile_height
        thumb = opened.copy()
        thumb.thumbnail((thumb_size - 8, thumb_size - 8))
        thumb_x = x + (thumb_size - thumb.width) // 2
        thumb_y = y + (thumb_size - thumb.height) // 2
        sheet.paste(thumb, (thumb_x, thumb_y))
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline=(120, 120, 120))
        draw.text((x + 4, y + thumb_size + 2), image_id[:18], fill=(0, 0, 0))
    return sheet


def _build_summary(
    *,
    paths: dict[str, object],
    counts: pd.DataFrame,
    breakdown: pd.DataFrame,
    hard_examples: pd.DataFrame,
    contact_sheet_outputs: dict[str, list[Path]],
    missing_images: dict[str, list[str]],
    missing_optional: list[str],
) -> dict[str, object]:
    breakdown_top = []
    if not breakdown.empty:
        breakdown_top = breakdown.sort_values(["count", "annotation_category"], ascending=[False, True]).head(10).to_dict("records")
    return {
        "input_files": {key: str(value) for key, value in paths.items()},
        "missing_optional_files": sorted(set(missing_optional)),
        "error_group_counts": {row["error_group"]: int(row["count"]) for row in counts.to_dict("records")},
        "annotation_category_top_errors": breakdown_top,
        "hard_example_count": int(len(hard_examples)),
        "contact_sheets_created": {group: [str(path) for path in paths] for group, paths in contact_sheet_outputs.items()},
        "contact_sheets_missing_images": missing_images,
        "safety_flags": {
            "used_test_labels": False,
            "trained_model": False,
            "submitted": False,
            "hard_examples_mode": "analysis_only",
            "oversampling_enabled": False,
            "focal_loss_enabled": False,
            "weighted_sampler_enabled": False,
        },
        "recommended_next_action": "Review V2B hard examples manually before any smoke fine-tune; keep hard examples analysis-only.",
    }


def group_to_review_priority(group: str) -> str:
    if group in {"high_confidence_wrong", "v1_correct_v2b_wrong"}:
        return "high"
    if group == "both_wrong":
        return "medium"
    return "low"


def _extract_numeric(df: pd.DataFrame, candidates: Sequence[str], *, default: float | int) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def _find_column(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> str | None:
    lookup = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise V2BErrorReviewError("Missing required column; expected one of: " + ", ".join(candidates))
    return None


def _normalize_image_id(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        raise V2BErrorReviewError("image_id cannot be empty")
    return Path(text).name


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V2B error review analysis.")
    parser.add_argument("--config", default="configs/v2b_error_review.yaml")
    args = parser.parse_args(argv)
    try:
        outputs = run_review(args.config)
    except V2BErrorReviewError as exc:
        print(f"V2B error review failed: {exc}", file=sys.stderr)
        return 2
    print(f"V2B error review complete: {outputs['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
