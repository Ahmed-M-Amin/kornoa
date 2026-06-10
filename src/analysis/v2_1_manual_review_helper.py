"""Build offline manual-review helper artifacts for V2.1 error review."""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml


FORBIDDEN_REVIEW_FLAGS = {"allow_test_labels", "generate_submission", "train_model"}
REQUIRED_TEMPLATE_COLUMNS = [
    "image_id",
    "error_type",
    "true_label",
    "v2b_prediction",
    "v2b_probability",
    "threshold_distance",
    "review_group",
    "visual_tag",
    "reviewer_note",
    "recommended_action",
]
ALLOWED_VISUAL_TAGS = [
    "acceptable_reflection",
    "acceptable_water_drop",
    "embossing",
    "edge_ring",
    "small_dark_defect",
    "center_contamination",
    "low_contrast_defect",
    "blur",
    "overexposed",
    "ambiguous_label",
    "unknown",
]
ALLOWED_RECOMMENDED_ACTIONS = [
    "add_hard_negative",
    "add_hard_positive",
    "raise_threshold",
    "lower_threshold",
    "detector_needed",
    "data_label_review",
    "ignore_ambiguous",
    "keep_threshold",
]
OUTPUT_FILENAMES = {
    "html_preview": "manual_review_preview.html",
    "prefilled_csv": "manual_review_prefilled.csv",
    "instructions": "manual_review_instructions.md",
    "tag_counts_template": "manual_review_tag_counts_template.csv",
}


class V21ManualReviewHelperError(ValueError):
    """Raised when manual-review helper inputs are invalid."""


def load_manual_review_helper_config(path: str | Path) -> dict:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise V21ManualReviewHelperError(f"Manual review helper config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise V21ManualReviewHelperError(f"Manual review helper config is not valid YAML: {config_path}") from exc
    if not isinstance(raw, dict):
        raise V21ManualReviewHelperError("Manual review helper config must be a mapping")
    input_cfg = raw.get("input")
    dataset_cfg = raw.get("dataset")
    if not isinstance(input_cfg, dict) or "manual_review_template" not in input_cfg:
        raise V21ManualReviewHelperError("Manual review helper config must contain input.manual_review_template")
    if not isinstance(dataset_cfg, dict) or "train_images" not in dataset_cfg:
        raise V21ManualReviewHelperError("Manual review helper config must contain dataset.train_images")
    review_cfg = dict(raw.get("review") or {})
    review_cfg.setdefault("max_rows_per_group", 50)
    review_cfg.setdefault("near_threshold_distance", 0.05)
    raw["review"] = review_cfg
    validate_manual_review_helper_config(raw)
    return raw


def validate_manual_review_helper_config(config: dict) -> None:
    review_cfg = config.get("review") or {}
    for flag in FORBIDDEN_REVIEW_FLAGS:
        if bool(review_cfg.get(flag)):
            raise V21ManualReviewHelperError(f"Manual review helper forbids review.{flag}=true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build V2.1 manual review helper artifacts.")
    parser.add_argument("--config", default="configs/v2_1_manual_review_helper.yaml")
    args = parser.parse_args(argv)
    try:
        outputs = run_manual_review_helper(args.config)
    except V21ManualReviewHelperError as exc:
        print(f"V2.1 manual review helper failed: {exc}", file=sys.stderr)
        return 2
    print(f"V2.1 manual review helper complete: {outputs['html_preview']}")
    return 0


def run_manual_review_helper(config_path: str | Path) -> dict[str, Path]:
    config = load_manual_review_helper_config(config_path)
    template_path = Path(str(config["input"]["manual_review_template"]))
    train_images_root = Path(str(config["dataset"]["train_images"]))
    output_root = Path(str(config.get("output", {}).get("root", "outputs/analysis/v2_1_error_intelligence/manual_review")))
    if not template_path.exists():
        raise V21ManualReviewHelperError(f"Manual review template not found: {template_path}")
    if not train_images_root.exists():
        raise V21ManualReviewHelperError(f"Train images path not found: {train_images_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {key: output_root / filename for key, filename in OUTPUT_FILENAMES.items()}

    template = pd.read_csv(template_path)
    missing_columns = [column for column in REQUIRED_TEMPLATE_COLUMNS if column not in template.columns]
    if missing_columns:
        raise V21ManualReviewHelperError(
            "Manual review template missing required columns: " + ", ".join(missing_columns)
        )

    prefilled = _build_prefilled_review(template, train_images_root, float(config["review"]["near_threshold_distance"]))
    prefilled.to_csv(outputs["prefilled_csv"], index=False)
    outputs["instructions"].write_text(_build_instructions(), encoding="utf-8")
    _build_tag_counts_template().to_csv(outputs["tag_counts_template"], index=False)
    outputs["html_preview"].write_text(
        _build_html_preview(
            prefilled=prefilled,
            html_path=outputs["html_preview"],
            train_images_root=train_images_root,
            max_rows_per_group=int(config["review"]["max_rows_per_group"]),
        ),
        encoding="utf-8",
    )
    return outputs


def _build_prefilled_review(template: pd.DataFrame, train_images_root: Path, near_threshold_distance: float) -> pd.DataFrame:
    prefilled = template.copy(deep=True)
    suggested_tags: list[str] = []
    suggested_actions: list[str] = []
    reviewer_notes: list[str] = []
    final_visual_tags: list[str] = []
    final_actions: list[str] = []

    for row in prefilled.to_dict("records"):
        image_id = str(row["image_id"])
        image_exists = (train_images_root / image_id).exists()
        threshold_distance = float(row["threshold_distance"])
        error_type = str(row["error_type"])
        review_group = str(row["review_group"])

        if not image_exists:
            suggested_tag = "missing_image"
            suggested_action = "data_label_review"
        elif error_type == "FN":
            suggested_tag = "small_or_low_contrast_defect"
            suggested_action = "add_hard_positive"
        elif error_type == "FP" or review_group == "over_rejected_reusable":
            suggested_tag = "acceptable_reflection_or_ring"
            suggested_action = "add_hard_negative"
        else:
            suggested_tag = "unknown"
            suggested_action = "keep_threshold"

        reviewer_note = ""
        if threshold_distance <= near_threshold_distance:
            reviewer_note = "near_threshold_ambiguous_check_manually"

        suggested_tags.append(suggested_tag)
        suggested_actions.append(suggested_action)
        reviewer_notes.append(reviewer_note)
        final_visual_tags.append("")
        final_actions.append("")

    prefilled["suggested_visual_tag"] = suggested_tags
    prefilled["suggested_recommended_action"] = suggested_actions
    prefilled["final_visual_tag"] = final_visual_tags
    prefilled["final_recommended_action"] = final_actions
    prefilled["reviewer_note"] = reviewer_notes
    ordered_columns = REQUIRED_TEMPLATE_COLUMNS + [
        "suggested_visual_tag",
        "suggested_recommended_action",
        "final_visual_tag",
        "final_recommended_action",
    ]
    return prefilled[ordered_columns]


def _build_html_preview(
    *,
    prefilled: pd.DataFrame,
    html_path: Path,
    train_images_root: Path,
    max_rows_per_group: int,
) -> str:
    sections: list[str] = []
    grouped = prefilled.groupby("review_group", sort=True)
    for review_group, frame in grouped:
        cards: list[str] = []
        for row in frame.head(max_rows_per_group).to_dict("records"):
            image_path = train_images_root / str(row["image_id"])
            if image_path.exists():
                image_src = _relative_html_path(html_path.parent, image_path)
                image_html = f'<img src="{html.escape(image_src)}" alt="{html.escape(str(row["image_id"]))}">'
            else:
                image_html = '<div class="missing-image">missing_image</div>'
            cards.append(
                "\n".join(
                    [
                        '<article class="card">',
                        f'  <div class="thumb">{image_html}</div>',
                        f'  <div><strong>image_id</strong>: {html.escape(str(row["image_id"]))}</div>',
                        f'  <div><strong>error_type</strong>: {html.escape(str(row["error_type"]))}</div>',
                        f'  <div><strong>true_label</strong>: {html.escape(str(row["true_label"]))}</div>',
                        f'  <div><strong>v2b_prediction</strong>: {html.escape(str(row["v2b_prediction"]))}</div>',
                        f'  <div><strong>v2b_probability</strong>: {html.escape(str(row["v2b_probability"]))}</div>',
                        f'  <div><strong>threshold_distance</strong>: {html.escape(str(row["threshold_distance"]))}</div>',
                        f'  <div><strong>review_group</strong>: {html.escape(str(row["review_group"]))}</div>',
                        f'  <div><strong>suggested_visual_tag</strong>: {html.escape(str(row["suggested_visual_tag"]))}</div>',
                        f'  <div><strong>suggested_recommended_action</strong>: {html.escape(str(row["suggested_recommended_action"]))}</div>',
                        f'  <div><strong>reviewer_note</strong>: {html.escape(str(row["reviewer_note"])) or "&nbsp;"}</div>',
                        '  <div><strong>final_visual_tag</strong>: ________</div>',
                        '  <div><strong>final_recommended_action</strong>: ________</div>',
                        '</article>',
                    ]
                )
            )
        sections.append(
            "\n".join(
                [
                    f"<section><h2>{html.escape(str(review_group))}</h2>",
                    f"<p>Previewing up to {max_rows_per_group} rows for this group.</p>",
                    '<div class="grid">',
                    *cards,
                    "</div></section>",
                ]
            )
        )
    allowed_visual_tags = ", ".join(ALLOWED_VISUAL_TAGS)
    allowed_actions = ", ".join(ALLOWED_RECOMMENDED_ACTIONS)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8"><title>V2.1 Manual Review Preview</title>',
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; color: #222; }",
            "h1, h2 { margin-bottom: 8px; }",
            ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }",
            ".card { background: #fff; border: 1px solid #d0d0d0; padding: 12px; border-radius: 6px; }",
            ".thumb { margin-bottom: 10px; }",
            "img { width: 100%; max-width: 240px; height: 180px; object-fit: contain; background: #eee; display: block; }",
            ".missing-image { width: 240px; height: 180px; background: #eee; display: flex; align-items: center; justify-content: center; color: #666; }",
            ".note { background: #fff4d6; border: 1px solid #f0d48a; padding: 10px; border-radius: 6px; margin-bottom: 18px; }",
            "</style></head><body>",
            "<h1>V2.1 Manual Review Preview</h1>",
            '<div class="note">Suggested tags and suggested actions are review hints only, not ground truth.</div>',
            f"<p><strong>Allowed final visual tags</strong>: {html.escape(allowed_visual_tags)}</p>",
            f"<p><strong>Allowed final recommended actions</strong>: {html.escape(allowed_actions)}</p>",
            *sections,
            "</body></html>",
        ]
    )


def _build_instructions() -> str:
    return "\n".join(
        [
            "# V2.1 Manual Review Instructions",
            "",
            "1. Open `manual_review_preview.html` in a browser from the local filesystem.",
            "2. Review each image together with the displayed error metadata.",
            "3. Treat `suggested_visual_tag` and `suggested_recommended_action` as hints, not ground truth.",
            "4. Fill `final_visual_tag` with one allowed final tag after visual inspection.",
            "5. Fill `final_recommended_action` with one allowed final action after review.",
            "6. Use `reviewer_note` for ambiguity, labeling concerns, or follow-up evidence needs.",
            "7. This workflow does not train, submit, or change model predictions.",
            "",
            "Allowed final visual tags:",
            f"- {', '.join(ALLOWED_VISUAL_TAGS)}",
            "",
            "Allowed final recommended actions:",
            f"- {', '.join(ALLOWED_RECOMMENDED_ACTIONS)}",
        ]
    )


def _build_tag_counts_template() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "review_group",
            "suggested_visual_tag",
            "final_visual_tag",
            "suggested_recommended_action",
            "final_recommended_action",
            "count",
            "review_status",
        ]
    )


def _relative_html_path(base_dir: Path, target_path: Path) -> str:
    try:
        path = Path(os.path.relpath(target_path.resolve(), start=base_dir.resolve()))
    except ValueError:
        path = target_path.resolve()
    return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
