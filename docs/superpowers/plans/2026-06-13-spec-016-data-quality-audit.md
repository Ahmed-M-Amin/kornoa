# Spec 016 Data Quality Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe, data-first audit pipeline that ranks label, ROI/crop, duplicate, outlier, and hard-row risks, then produces controlled cleaned training manifests before any new model training.

**Architecture:** Add a focused analysis module, `src/analysis/data_quality_audit.py`, driven by `configs/data_quality_audit.yaml`. The module ingests train labels, optional V2B/OOF predictions, Spec 014 reports, optional annotations, and image files, then emits a master evidence table, review buckets, reports, and a cleaned manifest. Training remains out of scope; the next feature should use an automated decision-lock and active-review reduction workflow before any cleaned-data training begins.

**Tech Stack:** Python 3.11, pandas, numpy, PyYAML, Pillow, pytest, optional cleanlab, optional FiftyOne, optional imagehash/sklearn embeddings. Tests must not require cleanlab, FiftyOne, internet access, GPU, or real Kaggle training.

---

## Non-Negotiable Scope

- Do not train a model in Spec 016.
- Do not create a Kaggle submission.
- Do not use test labels.
- Do not tune thresholds from the public leaderboard.
- Do not overwrite V2B baseline artifacts.
- Do not modify original `train.csv` labels.
- Do not automatically relabel data; only generate evidence and review decisions.
- Treat missing optional files as reportable warnings, not fatal errors.
- Require all final training rows to come from `clean_train` or `hard_valid_train`.
- Exclude `manual_review_required` rows from `cleaned_training_manifest.csv` unless a later explicit approval file is introduced.

## Why This Plan Exists

The current reliable model is V2B:

- Public score: `0.92121`
- Validation F1: `0.926917`

Controlled Phase 3 hard-example training failed:

- Phase 3 validation F1: `0.8549`
- Phase 3 hard-example F1: `0.2541`

This means hard rows are not automatically useful. Some are likely mislabeled, ambiguous, weak-evidence, bad-crop, annotation-related, duplicate-conflict, or harmful. Spec 016 must separate trustworthy training signal from noisy/harmful rows before any stronger model or retraining attempt.

## File Structure

### Create

- `configs/data_quality_audit.yaml`
  - Runtime configuration, input paths, optional dependency switches, bucket thresholds, output paths.

- `src/analysis/data_quality_audit.py`
  - CLI and implementation for master table generation, prediction ingestion, Spec 014 integration, ROI/crop audit, duplicate audit, fallback label scoring, optional cleanlab/FiftyOne exports, buckets, reports, and cleaned manifest.

- `tests/test_data_quality_audit.py`
  - Unit and integration-style tests using tiny synthetic data and generated images.

- `docs/superpowers/plans/2026-06-13-spec-016-data-quality-audit.md`
  - This implementation plan.

### Modify

- `docs/file-path-index.md`
  - Add all new config, source, test, and output paths.

### Primary Outputs

- `outputs/analysis/data_quality_audit/data_quality_master.csv`
- `outputs/analysis/data_quality_audit/clean_train_rows.csv`
- `outputs/analysis/data_quality_audit/hard_valid_train_rows.csv`
- `outputs/analysis/data_quality_audit/exclude_from_training_rows.csv`
- `outputs/analysis/data_quality_audit/manual_review_required_rows.csv`
- `outputs/analysis/data_quality_audit/cleaned_training_manifest.csv`
- `outputs/analysis/data_quality_audit/review_decision_template.csv`
- `outputs/analysis/data_quality_audit/reports/data_quality_summary.json`
- `outputs/analysis/data_quality_audit/reports/bucket_counts.csv`
- `outputs/analysis/data_quality_audit/reports/exclusion_reason_counts.csv`
- `outputs/analysis/data_quality_audit/reports/top_suspicious_rows.csv`
- `outputs/analysis/data_quality_audit/reports/top_label_issue_rows.csv`
- `outputs/analysis/data_quality_audit/reports/top_confident_wrong_rows.csv`
- `outputs/analysis/data_quality_audit/reports/top_anomaly_rows.csv`
- `outputs/analysis/data_quality_audit/reports/duplicate_conflicts.csv`
- `outputs/analysis/data_quality_audit/reports/roi_quality_issues.csv`
- `outputs/analysis/data_quality_audit/reports/manual_review_plan.json`
- `outputs/analysis/data_quality_audit/reports/input_file_inventory.csv`
- `outputs/analysis/data_quality_audit/reports/validation_alignment_summary.json`
- `outputs/analysis/data_quality_audit/contact_sheets/`

---

## Data Contract

### Required Inputs

- `1st-krones-vision-ai-challenge/train.csv`
- `1st-krones-vision-ai-challenge/train_images/`

### Strongly Recommended Inputs

- `artifacts/kaggle_v2b_artifacts/kaggle_v2/v2b_effnet_b1/kaggle_v2/predictions/val_classifier_predictions.csv`
- `artifacts/kaggle_v2b_artifacts/kaggle_v2/v2b_effnet_b1/kaggle_v2/reports/classifier_metrics.json`
- `outputs/analysis/hard_row_visual_review/reports/phase3_allowed_hard_training_rows.csv`
- `outputs/analysis/hard_row_visual_review/reports/blocked_suspected_mislabel_rows.csv`
- `outputs/analysis/hard_row_visual_review/reports/blocked_roi_pipeline_bug_rows.csv`
- `outputs/analysis/hard_row_visual_review/reports/blocked_other_high_risk_rows.csv`

### High-End Upgrade Inputs

- `outputs/analysis/data_quality_audit/oof_train_predictions.csv`
  - Preferred full-train out-of-fold predicted probabilities.
  - If missing, the audit still runs, but summary JSON must set `oof_prediction_status = "missing"` and `label_quality_reliability = "limited"`.

- `outputs/analysis/data_quality_audit/image_embeddings.csv`
  - Optional image-level feature vectors from DINO/CLIP/V2B penultimate layer.
  - If missing, the audit uses deterministic image heuristics and reports `embedding_status = "missing"`.

### Optional Inputs

- `bottletypes.csv`
- `train_annotations.json`
- `outputs/analysis/hard_row_quality_audit/audit/hard_row_audit.csv`
- `outputs/analysis/hard_row_visual_review/review_manifest.csv`
- `outputs/analysis/hard_row_visual_review/reports/final_hard_row_action_plan.csv`
- `outputs/analysis/hard_row_visual_review/reports/phase3_training_candidates.csv`
- `outputs/analysis/hard_row_visual_review/reports/excluded_ambiguous_or_noisy_rows.csv`
- `outputs/analysis/hard_row_visual_review/reports/engineering_fix_candidates.csv`
- `outputs/analysis/hard_row_visual_review/reports/human_verification_queue.csv`

### Master Table Columns

`data_quality_master.csv` must contain these columns even when values are unavailable:

```text
image_id
target
bottle_type
source_split
v2b_probability
v2b_prediction
v2b_correct
v2b_confidence
v2b_loss_or_risk_score
decision_boundary_distance
prediction_source
oof_probability
oof_prediction
oof_correct
oof_confidence
oof_loss_or_risk_score
label_quality_reliability
hard_row_status
hard_example_type
spec014_final_failure_mode
spec014_recommended_action
spec014_phase3_use_allowed
blocked_reason
roi_quality_flag
crop_quality_score
annotation_evidence_flag
image_width
image_height
image_mean_brightness
image_contrast
edge_strength
background_heavy_score
cluster_id
anomaly_score
label_issue_score
duplicate_group_id
near_duplicate_score
duplicate_conflict_flag
manual_review_status
data_quality_bucket
exclude_from_training
exclusion_reason
review_priority
evidence_summary
audit_run_id
created_at
source_config_hash
```

---

## Bucket Rules

Use deterministic precedence. A row can have many risk flags, but one final `data_quality_bucket`.

1. Missing or unreadable image:
   - `data_quality_bucket = "exclude_from_training"`
   - `exclude_from_training = true`
   - `exclusion_reason = "missing_or_unreadable_image"`

2. Spec 014 suspected mislabel blocked row:
   - `data_quality_bucket = "exclude_from_training"`
   - `exclude_from_training = true`
   - `exclusion_reason = "spec014_suspected_mislabel"`

3. Spec 014 ROI pipeline bug blocked row:
   - `data_quality_bucket = "manual_review_required"`
   - `exclude_from_training = true`
   - `exclusion_reason = "spec014_roi_pipeline_bug_requires_review"`

4. Spec 014 other high-risk blocked row:
   - `data_quality_bucket = "manual_review_required"`
   - `exclude_from_training = true`
   - `exclusion_reason = "spec014_other_high_risk_requires_review"`

5. Duplicate label conflict:
   - `data_quality_bucket = "manual_review_required"`
   - `exclude_from_training = true`
   - `exclusion_reason = "duplicate_label_conflict"`

6. High-confidence opposite-label prediction:
   - `data_quality_bucket = "manual_review_required"`
   - `exclude_from_training = true`
   - `exclusion_reason = "confident_wrong_prediction_requires_review"`

7. High label issue score:
   - `data_quality_bucket = "manual_review_required"`
   - `exclude_from_training = true`
   - `exclusion_reason = "high_label_issue_score"`

8. ROI/crop issue:
   - `data_quality_bucket = "manual_review_required"`
   - `exclude_from_training = true`
   - `exclusion_reason = "roi_or_crop_problem_requires_review"`

9. Spec 014 allowed hard row with acceptable label/ROI/duplicate evidence:
   - `data_quality_bucket = "hard_valid_train"`
   - `exclude_from_training = false`
   - `exclusion_reason = ""`

10. Normal row with no major risk:
   - `data_quality_bucket = "clean_train"`
   - `exclude_from_training = false`
   - `exclusion_reason = ""`

The cleaned training manifest may include only:

- `clean_train`
- `hard_valid_train`

It must exclude:

- `manual_review_required`
- `exclude_from_training`

---

## Task 1: Configuration Contract

**Files:**

- Create: `configs/data_quality_audit.yaml`
- Test: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Write failing config-load test**

Add this test to `tests/test_data_quality_audit.py`:

```python
from pathlib import Path

from src.analysis.data_quality_audit import DataQualityConfig, load_config


def test_load_data_quality_config_defaults(tmp_path):
    config_path = tmp_path / "data_quality_audit.yaml"
    config_path.write_text(
        """
dataset:
  train_csv_path: train.csv
  train_images_dir: train_images
outputs:
  output_root: outputs/analysis/data_quality_audit
predictions:
  v2b_validation_predictions_path: ""
  oof_train_predictions_path: ""
spec014:
  allowed_hard_rows_path: ""
  blocked_suspected_mislabel_rows_path: ""
  blocked_roi_pipeline_bug_rows_path: ""
  blocked_other_high_risk_rows_path: ""
quality:
  confident_wrong_probability_threshold: 0.90
  label_issue_threshold: 0.80
  dark_brightness_threshold: 20.0
  bright_brightness_threshold: 235.0
  low_contrast_threshold: 8.0
  weak_edge_threshold: 2.0
optional_tools:
  use_cleanlab_if_available: true
  use_fiftyone_if_available: false
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, DataQualityConfig)
    assert config.train_csv_path == Path("train.csv")
    assert config.train_images_dir == Path("train_images")
    assert config.output_root == Path("outputs/analysis/data_quality_audit")
    assert config.confident_wrong_probability_threshold == 0.90
    assert config.use_cleanlab_if_available is True
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
python -m pytest tests/test_data_quality_audit.py::test_load_data_quality_config_defaults -q
```

Expected:

```text
ModuleNotFoundError: No module named 'src.analysis.data_quality_audit'
```

- [ ] **Step 3: Create config implementation**

Create `src/analysis/data_quality_audit.py` with:

```python
"""Data quality audit pipeline for controlled cleaned training manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFilter, UnidentifiedImageError


DEFAULT_CONFIG_PATH = Path("configs/data_quality_audit.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/data_quality_audit")


@dataclass(frozen=True)
class DataQualityConfig:
    train_csv_path: Path
    train_images_dir: Path
    output_root: Path
    v2b_validation_predictions_path: Path | None
    oof_train_predictions_path: Path | None
    allowed_hard_rows_path: Path | None
    blocked_suspected_mislabel_rows_path: Path | None
    blocked_roi_pipeline_bug_rows_path: Path | None
    blocked_other_high_risk_rows_path: Path | None
    confident_wrong_probability_threshold: float
    label_issue_threshold: float
    dark_brightness_threshold: float
    bright_brightness_threshold: float
    low_contrast_threshold: float
    weak_edge_threshold: float
    use_cleanlab_if_available: bool
    use_fiftyone_if_available: bool


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def load_config(config_path: str | Path) -> DataQualityConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    dataset = raw.get("dataset", {})
    outputs = raw.get("outputs", {})
    predictions = raw.get("predictions", {})
    spec014 = raw.get("spec014", {})
    quality = raw.get("quality", {})
    optional_tools = raw.get("optional_tools", {})

    return DataQualityConfig(
        train_csv_path=Path(dataset["train_csv_path"]),
        train_images_dir=Path(dataset["train_images_dir"]),
        output_root=Path(outputs.get("output_root", DEFAULT_OUTPUT_ROOT)),
        v2b_validation_predictions_path=_optional_path(
            predictions.get("v2b_validation_predictions_path")
        ),
        oof_train_predictions_path=_optional_path(predictions.get("oof_train_predictions_path")),
        allowed_hard_rows_path=_optional_path(spec014.get("allowed_hard_rows_path")),
        blocked_suspected_mislabel_rows_path=_optional_path(
            spec014.get("blocked_suspected_mislabel_rows_path")
        ),
        blocked_roi_pipeline_bug_rows_path=_optional_path(
            spec014.get("blocked_roi_pipeline_bug_rows_path")
        ),
        blocked_other_high_risk_rows_path=_optional_path(
            spec014.get("blocked_other_high_risk_rows_path")
        ),
        confident_wrong_probability_threshold=float(
            quality.get("confident_wrong_probability_threshold", 0.90)
        ),
        label_issue_threshold=float(quality.get("label_issue_threshold", 0.80)),
        dark_brightness_threshold=float(quality.get("dark_brightness_threshold", 20.0)),
        bright_brightness_threshold=float(quality.get("bright_brightness_threshold", 235.0)),
        low_contrast_threshold=float(quality.get("low_contrast_threshold", 8.0)),
        weak_edge_threshold=float(quality.get("weak_edge_threshold", 2.0)),
        use_cleanlab_if_available=bool(optional_tools.get("use_cleanlab_if_available", True)),
        use_fiftyone_if_available=bool(optional_tools.get("use_fiftyone_if_available", False)),
    )
```

- [ ] **Step 4: Create repository config**

Create `configs/data_quality_audit.yaml`:

```yaml
dataset:
  train_csv_path: 1st-krones-vision-ai-challenge/train.csv
  train_images_dir: 1st-krones-vision-ai-challenge/train_images

outputs:
  output_root: outputs/analysis/data_quality_audit

predictions:
  v2b_validation_predictions_path: artifacts/kaggle_v2b_artifacts/kaggle_v2/v2b_effnet_b1/kaggle_v2/predictions/val_classifier_predictions.csv
  oof_train_predictions_path: outputs/analysis/data_quality_audit/oof_train_predictions.csv
  image_embeddings_path: outputs/analysis/data_quality_audit/image_embeddings.csv

optional_inputs:
  bottletypes_path: bottletypes.csv
  train_annotations_path: train_annotations.json
  hard_row_quality_audit_path: outputs/analysis/hard_row_quality_audit/audit/hard_row_audit.csv
  hard_row_review_manifest_path: outputs/analysis/hard_row_visual_review/review_manifest.csv
  final_hard_row_action_plan_path: outputs/analysis/hard_row_visual_review/reports/final_hard_row_action_plan.csv
  phase3_training_candidates_path: outputs/analysis/hard_row_visual_review/reports/phase3_training_candidates.csv
  excluded_ambiguous_or_noisy_rows_path: outputs/analysis/hard_row_visual_review/reports/excluded_ambiguous_or_noisy_rows.csv
  engineering_fix_candidates_path: outputs/analysis/hard_row_visual_review/reports/engineering_fix_candidates.csv
  human_verification_queue_path: outputs/analysis/hard_row_visual_review/reports/human_verification_queue.csv

spec014:
  allowed_hard_rows_path: outputs/analysis/hard_row_visual_review/reports/phase3_allowed_hard_training_rows.csv
  blocked_suspected_mislabel_rows_path: outputs/analysis/hard_row_visual_review/reports/blocked_suspected_mislabel_rows.csv
  blocked_roi_pipeline_bug_rows_path: outputs/analysis/hard_row_visual_review/reports/blocked_roi_pipeline_bug_rows.csv
  blocked_other_high_risk_rows_path: outputs/analysis/hard_row_visual_review/reports/blocked_other_high_risk_rows.csv

quality:
  confident_wrong_probability_threshold: 0.90
  label_issue_threshold: 0.80
  dark_brightness_threshold: 20.0
  bright_brightness_threshold: 235.0
  low_contrast_threshold: 8.0
  weak_edge_threshold: 2.0
  min_image_width: 32
  min_image_height: 32

manual_review:
  top_suspicious_limit: 800
  top_contact_sheet_limit_per_bucket: 64

optional_tools:
  use_cleanlab_if_available: true
  use_fiftyone_if_available: false
  use_imagehash_if_available: true
```

- [ ] **Step 5: Run config test**

Run:

```powershell
python -m pytest tests/test_data_quality_audit.py::test_load_data_quality_config_defaults -q
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

```powershell
git add configs/data_quality_audit.yaml src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: add data quality audit config contract"
```

---

## Task 2: Master Table From Train CSV and Predictions

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing master-table test**

```python
import pandas as pd

from src.analysis.data_quality_audit import build_master_table


def test_build_master_table_from_train_and_v2b_predictions(tmp_path):
    train_csv = tmp_path / "train.csv"
    predictions_csv = tmp_path / "val_predictions.csv"
    images_dir = tmp_path / "train_images"
    images_dir.mkdir()

    pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0},
            {"image_id": "b.jpg", "target": 1},
            {"image_id": "c.jpg", "target": 0},
        ]
    ).to_csv(train_csv, index=False)

    pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0, "probability": 0.10, "prediction": 0},
            {"image_id": "b.jpg", "target": 1, "probability": 0.95, "prediction": 1},
        ]
    ).to_csv(predictions_csv, index=False)

    config = DataQualityConfig(
        train_csv_path=train_csv,
        train_images_dir=images_dir,
        output_root=tmp_path / "out",
        v2b_validation_predictions_path=predictions_csv,
        oof_train_predictions_path=None,
        allowed_hard_rows_path=None,
        blocked_suspected_mislabel_rows_path=None,
        blocked_roi_pipeline_bug_rows_path=None,
        blocked_other_high_risk_rows_path=None,
        confident_wrong_probability_threshold=0.90,
        label_issue_threshold=0.80,
        dark_brightness_threshold=20.0,
        bright_brightness_threshold=235.0,
        low_contrast_threshold=8.0,
        weak_edge_threshold=2.0,
        use_cleanlab_if_available=False,
        use_fiftyone_if_available=False,
    )

    master, warnings = build_master_table(config)

    assert len(master) == 3
    assert set(master["image_id"]) == {"a.jpg", "b.jpg", "c.jpg"}
    assert master.loc[master["image_id"] == "a.jpg", "v2b_probability"].item() == 0.10
    assert master.loc[master["image_id"] == "b.jpg", "v2b_correct"].item() is True
    assert pd.isna(master.loc[master["image_id"] == "c.jpg", "v2b_probability"]).item()
    assert any("OOF train predictions missing" in warning for warning in warnings)
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_build_master_table_from_train_and_v2b_predictions -q
```

Expected:

```text
ImportError: cannot import name 'build_master_table'
```

- [ ] **Step 3: Implement master table**

Add to `src/analysis/data_quality_audit.py`:

```python
MASTER_COLUMNS = [
    "image_id",
    "target",
    "bottle_type",
    "source_split",
    "v2b_probability",
    "v2b_prediction",
    "v2b_correct",
    "v2b_confidence",
    "v2b_loss_or_risk_score",
    "decision_boundary_distance",
    "prediction_source",
    "oof_probability",
    "oof_prediction",
    "oof_correct",
    "oof_confidence",
    "oof_loss_or_risk_score",
    "label_quality_reliability",
    "hard_row_status",
    "hard_example_type",
    "spec014_final_failure_mode",
    "spec014_recommended_action",
    "spec014_phase3_use_allowed",
    "blocked_reason",
    "roi_quality_flag",
    "crop_quality_score",
    "annotation_evidence_flag",
    "image_width",
    "image_height",
    "image_mean_brightness",
    "image_contrast",
    "edge_strength",
    "background_heavy_score",
    "cluster_id",
    "anomaly_score",
    "label_issue_score",
    "duplicate_group_id",
    "near_duplicate_score",
    "duplicate_conflict_flag",
    "manual_review_status",
    "data_quality_bucket",
    "exclude_from_training",
    "exclusion_reason",
    "review_priority",
    "evidence_summary",
    "audit_run_id",
    "created_at",
    "source_config_hash",
]


def _normalize_predictions(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = df.copy()
    probability_col = "probability" if "probability" in result.columns else "target_probability"
    prediction_col = "prediction" if "prediction" in result.columns else "predicted_label"
    if probability_col not in result.columns:
        raise ValueError(f"Prediction file is missing probability column for {prefix}")
    if prediction_col not in result.columns:
        result[prediction_col] = (result[probability_col].astype(float) >= 0.5).astype(int)
    keep = ["image_id", probability_col, prediction_col]
    if "target" in result.columns:
        keep.append("target")
    result = result[keep].copy()
    result = result.rename(
        columns={
            probability_col: f"{prefix}_probability",
            prediction_col: f"{prefix}_prediction",
            "target": f"{prefix}_target_from_predictions",
        }
    )
    return result


def _add_prediction_derived_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    probability_col = f"{prefix}_probability"
    prediction_col = f"{prefix}_prediction"
    correct_col = f"{prefix}_correct"
    confidence_col = f"{prefix}_confidence"
    risk_col = f"{prefix}_loss_or_risk_score"

    if probability_col not in df.columns:
        df[correct_col] = pd.NA
        df[confidence_col] = pd.NA
        df[risk_col] = pd.NA
        return df

    prob = pd.to_numeric(df[probability_col], errors="coerce")
    pred = pd.to_numeric(df[prediction_col], errors="coerce")
    target = pd.to_numeric(df["target"], errors="coerce")
    df[correct_col] = pred.eq(target)
    df.loc[prob.isna(), correct_col] = pd.NA
    df[confidence_col] = np.maximum(prob, 1.0 - prob)
    target_prob = np.where(target == 1, prob, 1.0 - prob)
    df[risk_col] = 1.0 - target_prob
    return df


def _source_config_hash(config: DataQualityConfig) -> str:
    payload = json.dumps(
        {
            "train_csv_path": str(config.train_csv_path),
            "train_images_dir": str(config.train_images_dir),
            "v2b_validation_predictions_path": str(config.v2b_validation_predictions_path),
            "oof_train_predictions_path": str(config.oof_train_predictions_path),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_master_table(config: DataQualityConfig) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    train = pd.read_csv(config.train_csv_path)
    if "image_id" not in train.columns or "target" not in train.columns:
        raise ValueError("train.csv must contain image_id and target columns")

    master = train[["image_id", "target"]].copy()
    master["image_id"] = master["image_id"].astype(str)
    master["target"] = master["target"].astype(int)

    if config.v2b_validation_predictions_path and config.v2b_validation_predictions_path.exists():
        preds = _normalize_predictions(pd.read_csv(config.v2b_validation_predictions_path), "v2b")
        master = master.merge(preds.drop(columns=["v2b_target_from_predictions"], errors="ignore"), on="image_id", how="left")
        master = _add_prediction_derived_columns(master, "v2b")
        master["prediction_source"] = np.where(master["v2b_probability"].notna(), "v2b_validation", "")
    else:
        warnings.append("V2B validation predictions missing")
        master["prediction_source"] = ""

    if config.oof_train_predictions_path and config.oof_train_predictions_path.exists():
        oof = _normalize_predictions(pd.read_csv(config.oof_train_predictions_path), "oof")
        master = master.merge(oof.drop(columns=["oof_target_from_predictions"], errors="ignore"), on="image_id", how="left")
        master = _add_prediction_derived_columns(master, "oof")
        master["label_quality_reliability"] = np.where(master["oof_probability"].notna(), "oof", "limited")
    else:
        warnings.append("OOF train predictions missing; label-quality scoring is limited")
        master["oof_probability"] = pd.NA
        master["oof_prediction"] = pd.NA
        master["oof_correct"] = pd.NA
        master["oof_confidence"] = pd.NA
        master["oof_loss_or_risk_score"] = pd.NA
        master["label_quality_reliability"] = "limited"

    if "v2b_probability" not in master.columns:
        master["v2b_probability"] = pd.NA
        master["v2b_prediction"] = pd.NA
        master["v2b_correct"] = pd.NA
        master["v2b_confidence"] = pd.NA
        master["v2b_loss_or_risk_score"] = pd.NA

    master["decision_boundary_distance"] = (
        pd.to_numeric(master["oof_probability"], errors="coerce")
        .fillna(pd.to_numeric(master["v2b_probability"], errors="coerce"))
        .sub(0.5)
        .abs()
    )
    master["created_at"] = datetime.now(timezone.utc).isoformat()
    master["audit_run_id"] = master["created_at"].str.replace(":", "", regex=False).str.replace("+", "_", regex=False)
    master["source_config_hash"] = _source_config_hash(config)

    for column in MASTER_COLUMNS:
        if column not in master.columns:
            master[column] = pd.NA

    return master[MASTER_COLUMNS], warnings
```

- [ ] **Step 4: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_build_master_table_from_train_and_v2b_predictions -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: build data quality master table"
```

---

## Task 3: Spec 014 Integration and Blocked-Row Safety

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing Spec 014 bucket test**

```python
from src.analysis.data_quality_audit import apply_spec014_evidence


def test_spec014_blocked_rows_map_to_safe_buckets(tmp_path):
    master = pd.DataFrame(
        [
            {"image_id": "allowed.jpg", "target": 1},
            {"image_id": "mislabel.jpg", "target": 0},
            {"image_id": "roi.jpg", "target": 1},
            {"image_id": "risk.jpg", "target": 0},
        ]
    )
    allowed = tmp_path / "allowed.csv"
    mislabel = tmp_path / "mislabel.csv"
    roi = tmp_path / "roi.csv"
    risk = tmp_path / "risk.csv"

    pd.DataFrame([{"image_id": "allowed.jpg", "hard_example_type": "correct_but_visually_hard"}]).to_csv(allowed, index=False)
    pd.DataFrame([{"image_id": "mislabel.jpg"}]).to_csv(mislabel, index=False)
    pd.DataFrame([{"image_id": "roi.jpg"}]).to_csv(roi, index=False)
    pd.DataFrame([{"image_id": "risk.jpg"}]).to_csv(risk, index=False)

    config = DataQualityConfig(
        train_csv_path=tmp_path / "train.csv",
        train_images_dir=tmp_path / "images",
        output_root=tmp_path / "out",
        v2b_validation_predictions_path=None,
        oof_train_predictions_path=None,
        allowed_hard_rows_path=allowed,
        blocked_suspected_mislabel_rows_path=mislabel,
        blocked_roi_pipeline_bug_rows_path=roi,
        blocked_other_high_risk_rows_path=risk,
        confident_wrong_probability_threshold=0.90,
        label_issue_threshold=0.80,
        dark_brightness_threshold=20.0,
        bright_brightness_threshold=235.0,
        low_contrast_threshold=8.0,
        weak_edge_threshold=2.0,
        use_cleanlab_if_available=False,
        use_fiftyone_if_available=False,
    )

    updated, warnings = apply_spec014_evidence(master, config)

    buckets = dict(zip(updated["image_id"], updated["hard_row_status"]))
    reasons = dict(zip(updated["image_id"], updated["blocked_reason"]))

    assert buckets["allowed.jpg"] == "spec014_allowed_hard"
    assert buckets["mislabel.jpg"] == "spec014_blocked"
    assert reasons["mislabel.jpg"] == "spec014_suspected_mislabel"
    assert reasons["roi.jpg"] == "spec014_roi_pipeline_bug"
    assert reasons["risk.jpg"] == "spec014_other_high_risk"
    assert warnings == []
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_spec014_blocked_rows_map_to_safe_buckets -q
```

Expected:

```text
ImportError: cannot import name 'apply_spec014_evidence'
```

- [ ] **Step 3: Implement Spec 014 evidence merge**

Add to `src/analysis/data_quality_audit.py`:

```python
def _read_image_id_set(path: Path | None, label: str, warnings: list[str]) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        warnings.append(f"{label} missing: {path}")
        return set()
    frame = pd.read_csv(path)
    if "image_id" not in frame.columns:
        warnings.append(f"{label} has no image_id column: {path}")
        return set()
    return set(frame["image_id"].astype(str))


def apply_spec014_evidence(
    master: pd.DataFrame, config: DataQualityConfig
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    df = master.copy()
    for column in [
        "hard_row_status",
        "hard_example_type",
        "spec014_final_failure_mode",
        "spec014_recommended_action",
        "spec014_phase3_use_allowed",
        "blocked_reason",
    ]:
        if column not in df.columns:
            df[column] = pd.NA

    if config.allowed_hard_rows_path and config.allowed_hard_rows_path.exists():
        allowed = pd.read_csv(config.allowed_hard_rows_path)
        if "image_id" in allowed.columns:
            allowed = allowed.copy()
            allowed["image_id"] = allowed["image_id"].astype(str)
            rename_map = {
                "final_failure_mode": "spec014_final_failure_mode",
                "recommended_action": "spec014_recommended_action",
                "phase3_use_allowed": "spec014_phase3_use_allowed",
            }
            allowed = allowed.rename(columns=rename_map)
            keep = [
                col
                for col in [
                    "image_id",
                    "hard_example_type",
                    "spec014_final_failure_mode",
                    "spec014_recommended_action",
                    "spec014_phase3_use_allowed",
                ]
                if col in allowed.columns
            ]
            df = df.merge(allowed[keep].drop_duplicates("image_id"), on="image_id", how="left", suffixes=("", "_spec014"))
            allowed_ids = set(allowed["image_id"])
            df.loc[df["image_id"].isin(allowed_ids), "hard_row_status"] = "spec014_allowed_hard"
        else:
            warnings.append(f"Allowed hard rows file has no image_id column: {config.allowed_hard_rows_path}")
    elif config.allowed_hard_rows_path:
        warnings.append(f"Allowed hard rows missing: {config.allowed_hard_rows_path}")

    blocked_specs = [
        (config.blocked_suspected_mislabel_rows_path, "spec014_suspected_mislabel"),
        (config.blocked_roi_pipeline_bug_rows_path, "spec014_roi_pipeline_bug"),
        (config.blocked_other_high_risk_rows_path, "spec014_other_high_risk"),
    ]
    for path, reason in blocked_specs:
        ids = _read_image_id_set(path, reason, warnings)
        if not ids:
            continue
        mask = df["image_id"].isin(ids)
        df.loc[mask, "hard_row_status"] = "spec014_blocked"
        df.loc[mask, "blocked_reason"] = reason

    return df, warnings
```

- [ ] **Step 4: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_spec014_blocked_rows_map_to_safe_buckets -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: integrate spec014 data quality evidence"
```

---

## Task 4: ROI and Crop Quality Audit

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing ROI test**

```python
from PIL import Image

from src.analysis.data_quality_audit import apply_roi_quality_audit


def test_roi_quality_audit_marks_missing_and_low_contrast_images(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(images_dir / "flat.jpg")

    master = pd.DataFrame(
        [
            {"image_id": "flat.jpg", "target": 0},
            {"image_id": "missing.jpg", "target": 1},
        ]
    )

    updated = apply_roi_quality_audit(
        master,
        images_dir=images_dir,
        dark_threshold=20.0,
        bright_threshold=235.0,
        low_contrast_threshold=8.0,
        weak_edge_threshold=2.0,
    )

    flags = dict(zip(updated["image_id"], updated["roi_quality_flag"]))

    assert flags["flat.jpg"] == "low_contrast_or_weak_texture"
    assert flags["missing.jpg"] == "missing_image"
    assert updated.loc[updated["image_id"] == "flat.jpg", "image_width"].item() == 64
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_roi_quality_audit_marks_missing_and_low_contrast_images -q
```

Expected:

```text
ImportError: cannot import name 'apply_roi_quality_audit'
```

- [ ] **Step 3: Implement ROI audit**

Add to `src/analysis/data_quality_audit.py`:

```python
def _image_quality_metrics(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            arr = np.asarray(gray, dtype=np.float32)
            edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
            return {
                "roi_quality_flag": "ok",
                "image_width": int(image.width),
                "image_height": int(image.height),
                "image_mean_brightness": float(arr.mean()),
                "image_contrast": float(arr.std()),
                "edge_strength": float(edges.mean()),
                "background_heavy_score": float((arr > 245).mean()),
            }
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return {
            "roi_quality_flag": "missing_image",
            "image_width": pd.NA,
            "image_height": pd.NA,
            "image_mean_brightness": pd.NA,
            "image_contrast": pd.NA,
            "edge_strength": pd.NA,
            "background_heavy_score": pd.NA,
        }


def apply_roi_quality_audit(
    master: pd.DataFrame,
    images_dir: Path,
    dark_threshold: float,
    bright_threshold: float,
    low_contrast_threshold: float,
    weak_edge_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_id in master["image_id"].astype(str):
        metrics = _image_quality_metrics(images_dir / image_id)
        if metrics["roi_quality_flag"] == "ok":
            brightness = metrics["image_mean_brightness"]
            contrast = metrics["image_contrast"]
            edge = metrics["edge_strength"]
            if brightness < dark_threshold:
                metrics["roi_quality_flag"] = "mostly_dark"
            elif brightness > bright_threshold:
                metrics["roi_quality_flag"] = "mostly_bright"
            elif contrast < low_contrast_threshold or edge < weak_edge_threshold:
                metrics["roi_quality_flag"] = "low_contrast_or_weak_texture"
        metrics["image_id"] = image_id
        rows.append(metrics)

    quality = pd.DataFrame(rows)
    df = master.drop(
        columns=[
            "roi_quality_flag",
            "image_width",
            "image_height",
            "image_mean_brightness",
            "image_contrast",
            "edge_strength",
            "background_heavy_score",
        ],
        errors="ignore",
    )
    return df.merge(quality, on="image_id", how="left")
```

- [ ] **Step 4: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_roi_quality_audit_marks_missing_and_low_contrast_images -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: add roi and crop quality audit"
```

---

## Task 5: Duplicate and Conflict Detection

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing duplicate test**

```python
from src.analysis.data_quality_audit import apply_duplicate_audit


def test_duplicate_audit_flags_same_file_hash_with_label_conflict(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    content = b"same-bytes"
    (images_dir / "a.jpg").write_bytes(content)
    (images_dir / "b.jpg").write_bytes(content)

    master = pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0},
            {"image_id": "b.jpg", "target": 1},
        ]
    )

    updated, conflicts = apply_duplicate_audit(master, images_dir)

    assert updated["duplicate_conflict_flag"].tolist() == [True, True]
    assert len(conflicts) == 2
    assert conflicts["duplicate_group_id"].nunique() == 1
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_duplicate_audit_flags_same_file_hash_with_label_conflict -q
```

Expected:

```text
ImportError: cannot import name 'apply_duplicate_audit'
```

- [ ] **Step 3: Implement duplicate audit**

Add to `src/analysis/data_quality_audit.py`:

```python
def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_duplicate_audit(master: pd.DataFrame, images_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = master.copy()
    hashes = []
    for image_id in df["image_id"].astype(str):
        hashes.append(_file_hash(images_dir / image_id))
    df["_file_hash"] = hashes
    df["duplicate_group_id"] = ""
    df["near_duplicate_score"] = 0.0
    df["duplicate_conflict_flag"] = False

    valid_hashes = df["_file_hash"].astype(str) != ""
    duplicate_hashes = df.loc[valid_hashes, "_file_hash"].value_counts()
    duplicate_hashes = set(duplicate_hashes[duplicate_hashes > 1].index)

    for index, file_hash in enumerate(sorted(duplicate_hashes), start=1):
        mask = df["_file_hash"] == file_hash
        group_id = f"exact_dup_{index:05d}"
        df.loc[mask, "duplicate_group_id"] = group_id
        df.loc[mask, "near_duplicate_score"] = 1.0
        if df.loc[mask, "target"].nunique(dropna=True) > 1:
            df.loc[mask, "duplicate_conflict_flag"] = True

    conflicts = df.loc[df["duplicate_conflict_flag"]].drop(columns=["_file_hash"]).copy()
    return df.drop(columns=["_file_hash"]), conflicts
```

- [ ] **Step 4: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_duplicate_audit_flags_same_file_hash_with_label_conflict -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: detect duplicate label conflicts"
```

---

## Task 6: Label Issue and Suspicion Ranking

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing ranking test**

```python
from src.analysis.data_quality_audit import apply_label_issue_scoring


def test_confident_wrong_rows_receive_high_label_issue_score():
    master = pd.DataFrame(
        [
            {"image_id": "wrong.jpg", "target": 0, "oof_probability": 0.96, "v2b_probability": pd.NA, "blocked_reason": pd.NA},
            {"image_id": "right.jpg", "target": 1, "oof_probability": 0.97, "v2b_probability": pd.NA, "blocked_reason": pd.NA},
            {"image_id": "boundary.jpg", "target": 1, "oof_probability": 0.51, "v2b_probability": pd.NA, "blocked_reason": pd.NA},
        ]
    )

    scored = apply_label_issue_scoring(master)

    scores = dict(zip(scored["image_id"], scored["label_issue_score"]))
    priorities = dict(zip(scored["image_id"], scored["review_priority"]))

    assert scores["wrong.jpg"] > scores["right.jpg"]
    assert scores["wrong.jpg"] >= 0.90
    assert priorities["wrong.jpg"] == 1
    assert priorities["boundary.jpg"] == 7
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_confident_wrong_rows_receive_high_label_issue_score -q
```

Expected:

```text
ImportError: cannot import name 'apply_label_issue_scoring'
```

- [ ] **Step 3: Implement fallback label issue scoring**

Add to `src/analysis/data_quality_audit.py`:

```python
def apply_label_issue_scoring(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    prob = pd.to_numeric(df.get("oof_probability"), errors="coerce")
    prob = prob.fillna(pd.to_numeric(df.get("v2b_probability"), errors="coerce"))
    target = pd.to_numeric(df["target"], errors="coerce")

    predicted = (prob >= 0.5).astype("Int64")
    confidence = np.maximum(prob, 1.0 - prob)
    correct = predicted.eq(target)
    target_probability = np.where(target == 1, prob, 1.0 - prob)
    fallback_score = 1.0 - target_probability
    fallback_score = pd.Series(fallback_score, index=df.index).clip(lower=0.0, upper=1.0)
    fallback_score = fallback_score.fillna(0.0)

    blocked_bonus = df.get("blocked_reason", pd.Series(pd.NA, index=df.index)).notna().astype(float) * 0.15
    df["label_issue_score"] = (fallback_score + blocked_bonus).clip(upper=1.0)
    df["review_priority"] = 99

    confident_wrong = (~correct.fillna(False)) & (confidence >= 0.90)
    boundary = prob.sub(0.5).abs() <= 0.05
    high_label_issue = df["label_issue_score"] >= 0.80
    blocked = df.get("blocked_reason", pd.Series(pd.NA, index=df.index)).notna()

    df.loc[boundary, "review_priority"] = 7
    df.loc[blocked, "review_priority"] = 3
    df.loc[high_label_issue, "review_priority"] = 2
    df.loc[confident_wrong, "review_priority"] = 1

    df["v2b_prediction"] = df.get("v2b_prediction", predicted)
    df["v2b_confidence"] = df.get("v2b_confidence", confidence)
    df["decision_boundary_distance"] = prob.sub(0.5).abs()
    return df
```

- [ ] **Step 4: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_confident_wrong_rows_receive_high_label_issue_score -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: score suspicious label issues"
```

---

## Task 7: Optional Cleanlab and Embedding Hooks

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing optional dependency test**

```python
from src.analysis.data_quality_audit import apply_optional_cleanlab_scoring


def test_cleanlab_missing_keeps_fallback_scores_and_reports_warning(monkeypatch):
    master = pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0, "label_issue_score": 0.2},
        ]
    )

    def fake_import(name):
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", fake_import)

    updated, warnings = apply_optional_cleanlab_scoring(master, enabled=True)

    assert updated["label_issue_score"].tolist() == [0.2]
    assert any("Cleanlab not installed" in warning for warning in warnings)
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_cleanlab_missing_keeps_fallback_scores_and_reports_warning -q
```

Expected:

```text
ImportError: cannot import name 'apply_optional_cleanlab_scoring'
```

- [ ] **Step 3: Implement optional Cleanlab hook**

Add `import importlib` near the top of `src/analysis/data_quality_audit.py`, then add:

```python
def apply_optional_cleanlab_scoring(
    master: pd.DataFrame, enabled: bool
) -> tuple[pd.DataFrame, list[str]]:
    df = master.copy()
    warnings: list[str] = []
    if not enabled:
        warnings.append("Cleanlab disabled by config")
        return df, warnings

    try:
        importlib.import_module("cleanlab")
    except ImportError:
        warnings.append("Cleanlab not installed; using deterministic fallback label_issue_score")
        return df, warnings

    # Keep this hook deliberately conservative. A future enhancement can call
    # cleanlab.rank.get_label_quality_scores when full OOF pred_probs are present.
    warnings.append("Cleanlab installed but fallback scoring retained unless OOF pred_probs matrix is available")
    return df, warnings
```

- [ ] **Step 4: Add embedding status helper**

Add:

```python
def load_optional_embeddings(path: Path | None) -> tuple[pd.DataFrame | None, str]:
    if path is None:
        return None, "not_configured"
    if not path.exists():
        return None, "missing"
    frame = pd.read_csv(path)
    if "image_id" not in frame.columns:
        return None, "invalid_missing_image_id"
    return frame, "available"
```

- [ ] **Step 5: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_cleanlab_missing_keeps_fallback_scores_and_reports_warning -q
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: add optional cleanlab and embedding hooks"
```

---

## Task 8: Bucket Assignment and Cleaned Manifest

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing bucket/manifest test**

```python
from src.analysis.data_quality_audit import assign_data_quality_buckets, build_cleaned_training_manifest


def test_cleaned_manifest_excludes_blocked_and_manual_review_rows():
    master = pd.DataFrame(
        [
            {"image_id": "clean.jpg", "target": 0, "hard_row_status": pd.NA, "blocked_reason": pd.NA, "roi_quality_flag": "ok", "duplicate_conflict_flag": False, "label_issue_score": 0.1},
            {"image_id": "hard.jpg", "target": 1, "hard_row_status": "spec014_allowed_hard", "blocked_reason": pd.NA, "roi_quality_flag": "ok", "duplicate_conflict_flag": False, "label_issue_score": 0.2},
            {"image_id": "blocked.jpg", "target": 0, "hard_row_status": "spec014_blocked", "blocked_reason": "spec014_suspected_mislabel", "roi_quality_flag": "ok", "duplicate_conflict_flag": False, "label_issue_score": 0.2},
            {"image_id": "manual.jpg", "target": 1, "hard_row_status": pd.NA, "blocked_reason": pd.NA, "roi_quality_flag": "low_contrast_or_weak_texture", "duplicate_conflict_flag": False, "label_issue_score": 0.2},
        ]
    )

    bucketed = assign_data_quality_buckets(master, label_issue_threshold=0.80)
    manifest = build_cleaned_training_manifest(bucketed)

    assert set(manifest["image_id"]) == {"clean.jpg", "hard.jpg"}
    assert dict(zip(bucketed["image_id"], bucketed["data_quality_bucket"]))["blocked.jpg"] == "exclude_from_training"
    assert dict(zip(bucketed["image_id"], bucketed["data_quality_bucket"]))["manual.jpg"] == "manual_review_required"
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_cleaned_manifest_excludes_blocked_and_manual_review_rows -q
```

Expected:

```text
ImportError: cannot import name 'assign_data_quality_buckets'
```

- [ ] **Step 3: Implement bucket assignment**

Add to `src/analysis/data_quality_audit.py`:

```python
def assign_data_quality_buckets(master: pd.DataFrame, label_issue_threshold: float) -> pd.DataFrame:
    df = master.copy()
    df["data_quality_bucket"] = "clean_train"
    df["exclude_from_training"] = False
    df["exclusion_reason"] = ""

    def set_bucket(mask: pd.Series, bucket: str, reason: str, exclude: bool = True) -> None:
        df.loc[mask, "data_quality_bucket"] = bucket
        df.loc[mask, "exclude_from_training"] = exclude
        df.loc[mask, "exclusion_reason"] = reason

    roi = df.get("roi_quality_flag", pd.Series("ok", index=df.index)).fillna("ok").astype(str)
    blocked_reason = df.get("blocked_reason", pd.Series(pd.NA, index=df.index)).astype("string")
    duplicate_conflict = df.get("duplicate_conflict_flag", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    label_issue = pd.to_numeric(df.get("label_issue_score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    hard_status = df.get("hard_row_status", pd.Series(pd.NA, index=df.index)).astype("string")

    set_bucket(roi.eq("missing_image"), "exclude_from_training", "missing_or_unreadable_image")
    set_bucket(blocked_reason.eq("spec014_suspected_mislabel"), "exclude_from_training", "spec014_suspected_mislabel")
    set_bucket(blocked_reason.eq("spec014_roi_pipeline_bug"), "manual_review_required", "spec014_roi_pipeline_bug_requires_review")
    set_bucket(blocked_reason.eq("spec014_other_high_risk"), "manual_review_required", "spec014_other_high_risk_requires_review")
    set_bucket(duplicate_conflict, "manual_review_required", "duplicate_label_conflict")
    set_bucket(label_issue.ge(label_issue_threshold), "manual_review_required", "high_label_issue_score")
    set_bucket(roi.ne("ok") & roi.ne("missing_image"), "manual_review_required", "roi_or_crop_problem_requires_review")

    eligible_hard = (
        hard_status.eq("spec014_allowed_hard")
        & df["exclusion_reason"].eq("")
        & roi.eq("ok")
        & ~duplicate_conflict
        & label_issue.lt(label_issue_threshold)
    )
    df.loc[eligible_hard, "data_quality_bucket"] = "hard_valid_train"
    df.loc[eligible_hard, "exclude_from_training"] = False
    df.loc[eligible_hard, "exclusion_reason"] = ""

    df["manual_review_status"] = np.where(
        df["data_quality_bucket"].eq("manual_review_required"), "pending", ""
    )
    df["evidence_summary"] = df.apply(_build_evidence_summary, axis=1)
    return df


def _build_evidence_summary(row: pd.Series) -> str:
    parts = []
    for column in [
        "data_quality_bucket",
        "exclusion_reason",
        "blocked_reason",
        "roi_quality_flag",
        "hard_row_status",
    ]:
        value = row.get(column)
        if pd.notna(value) and str(value):
            parts.append(f"{column}={value}")
    return "; ".join(parts)


def build_cleaned_training_manifest(master: pd.DataFrame) -> pd.DataFrame:
    allowed = master["data_quality_bucket"].isin(["clean_train", "hard_valid_train"])
    not_excluded = ~master["exclude_from_training"].fillna(True).astype(bool)
    manifest = master.loc[allowed & not_excluded].copy()
    manifest["hard_training_allowed"] = manifest["data_quality_bucket"].eq("hard_valid_train")
    columns = [
        "image_id",
        "target",
        "data_quality_bucket",
        "hard_training_allowed",
        "exclude_from_training",
        "exclusion_reason",
        "review_priority",
        "evidence_summary",
    ]
    for column in columns:
        if column not in manifest.columns:
            manifest[column] = pd.NA
    return manifest[columns]
```

- [ ] **Step 4: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_cleaned_manifest_excludes_blocked_and_manual_review_rows -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: assign data quality buckets and cleaned manifest"
```

---

## Task 9: Reports and Review Decision Template

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing report test**

```python
from src.analysis.data_quality_audit import write_reports


def test_write_reports_creates_summary_and_ranked_outputs(tmp_path):
    master = pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0, "data_quality_bucket": "clean_train", "exclude_from_training": False, "exclusion_reason": "", "label_issue_score": 0.1, "review_priority": 99, "roi_quality_flag": "ok", "duplicate_conflict_flag": False},
            {"image_id": "b.jpg", "target": 1, "data_quality_bucket": "manual_review_required", "exclude_from_training": True, "exclusion_reason": "high_label_issue_score", "label_issue_score": 0.95, "review_priority": 1, "roi_quality_flag": "ok", "duplicate_conflict_flag": False},
        ]
    )
    warnings = ["OOF train predictions missing; label-quality scoring is limited"]

    write_reports(master, output_root=tmp_path, warnings=warnings, duplicate_conflicts=pd.DataFrame())

    assert (tmp_path / "reports" / "data_quality_summary.json").exists()
    assert (tmp_path / "reports" / "bucket_counts.csv").exists()
    assert (tmp_path / "reports" / "top_suspicious_rows.csv").exists()
    assert (tmp_path / "review_decision_template.csv").exists()

    summary = json.loads((tmp_path / "reports" / "data_quality_summary.json").read_text(encoding="utf-8"))
    assert summary["total_rows"] == 2
    assert summary["manual_review_required_count"] == 1
    assert summary["no_training_started"] is True
    assert summary["no_submission_created"] is True
    assert summary["no_test_labels_used"] is True
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_write_reports_creates_summary_and_ranked_outputs -q
```

Expected:

```text
ImportError: cannot import name 'write_reports'
```

- [ ] **Step 3: Implement report writer**

Add to `src/analysis/data_quality_audit.py`:

```python
def write_reports(
    master: pd.DataFrame,
    output_root: Path,
    warnings: list[str],
    duplicate_conflicts: pd.DataFrame,
) -> None:
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    bucket_counts = (
        master["data_quality_bucket"].value_counts(dropna=False).rename_axis("data_quality_bucket").reset_index(name="count")
    )
    bucket_counts.to_csv(reports_dir / "bucket_counts.csv", index=False)

    exclusion_counts = (
        master.loc[master["exclusion_reason"].fillna("").astype(str) != "", "exclusion_reason"]
        .value_counts()
        .rename_axis("exclusion_reason")
        .reset_index(name="count")
    )
    exclusion_counts.to_csv(reports_dir / "exclusion_reason_counts.csv", index=False)

    ranked = master.sort_values(["review_priority", "label_issue_score"], ascending=[True, False])
    ranked.to_csv(reports_dir / "top_suspicious_rows.csv", index=False)
    ranked.head(800).to_csv(output_root / "review_decision_template.csv", index=False)

    master.sort_values("label_issue_score", ascending=False).head(800).to_csv(
        reports_dir / "top_label_issue_rows.csv", index=False
    )
    confident_wrong = master.loc[master["review_priority"].eq(1)].copy()
    confident_wrong.to_csv(reports_dir / "top_confident_wrong_rows.csv", index=False)

    anomaly_col = "anomaly_score" if "anomaly_score" in master.columns else "label_issue_score"
    master.sort_values(anomaly_col, ascending=False).head(800).to_csv(
        reports_dir / "top_anomaly_rows.csv", index=False
    )
    duplicate_conflicts.to_csv(reports_dir / "duplicate_conflicts.csv", index=False)
    master.loc[master["roi_quality_flag"].fillna("ok").ne("ok")].to_csv(
        reports_dir / "roi_quality_issues.csv", index=False
    )

    summary = {
        "total_rows": int(len(master)),
        "clean_train_count": int(master["data_quality_bucket"].eq("clean_train").sum()),
        "hard_valid_train_count": int(master["data_quality_bucket"].eq("hard_valid_train").sum()),
        "exclude_from_training_count": int(master["data_quality_bucket"].eq("exclude_from_training").sum()),
        "manual_review_required_count": int(master["data_quality_bucket"].eq("manual_review_required").sum()),
        "warnings": warnings,
        "oof_prediction_status": "available" if master["oof_probability"].notna().any() else "missing",
        "label_quality_reliability": "oof" if master["oof_probability"].notna().any() else "limited",
        "no_training_started": True,
        "no_submission_created": True,
        "no_test_labels_used": True,
        "no_leaderboard_tuning": True,
    }
    (reports_dir / "data_quality_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    manual_review_plan = {
        "priority_1": "confident wrong predictions",
        "priority_2": "top label issue rows",
        "priority_3": "Spec 014 blocked or ambiguous rows",
        "priority_4": "duplicate label conflicts",
        "priority_5": "cluster outliers or anomalies",
        "priority_6": "ROI/crop issues",
        "priority_7": "decision-boundary unclear rows",
        "default_review_limit": 800,
    }
    (reports_dir / "manual_review_plan.json").write_text(
        json.dumps(manual_review_plan, indent=2, sort_keys=True), encoding="utf-8"
    )
```

- [ ] **Step 4: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_write_reports_creates_summary_and_ranked_outputs -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: write data quality audit reports"
```

---

## Task 10: Pipeline Orchestration and CLI

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing end-to-end test**

```python
from src.analysis.data_quality_audit import run_audit


def test_run_audit_writes_master_buckets_and_cleaned_manifest(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(images_dir / "clean.jpg")
    Image.new("RGB", (64, 64), color=(10, 10, 10)).save(images_dir / "dark.jpg")

    train_csv = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {"image_id": "clean.jpg", "target": 0},
            {"image_id": "dark.jpg", "target": 1},
        ]
    ).to_csv(train_csv, index=False)

    config = DataQualityConfig(
        train_csv_path=train_csv,
        train_images_dir=images_dir,
        output_root=tmp_path / "out",
        v2b_validation_predictions_path=None,
        oof_train_predictions_path=None,
        allowed_hard_rows_path=None,
        blocked_suspected_mislabel_rows_path=None,
        blocked_roi_pipeline_bug_rows_path=None,
        blocked_other_high_risk_rows_path=None,
        confident_wrong_probability_threshold=0.90,
        label_issue_threshold=0.80,
        dark_brightness_threshold=20.0,
        bright_brightness_threshold=235.0,
        low_contrast_threshold=8.0,
        weak_edge_threshold=2.0,
        use_cleanlab_if_available=False,
        use_fiftyone_if_available=False,
    )

    summary = run_audit(config)

    assert (tmp_path / "out" / "data_quality_master.csv").exists()
    assert (tmp_path / "out" / "clean_train_rows.csv").exists()
    assert (tmp_path / "out" / "manual_review_required_rows.csv").exists()
    assert (tmp_path / "out" / "cleaned_training_manifest.csv").exists()
    assert summary["total_rows"] == 2
    assert summary["no_training_started"] is True
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_run_audit_writes_master_buckets_and_cleaned_manifest -q
```

Expected:

```text
ImportError: cannot import name 'run_audit'
```

- [ ] **Step 3: Implement orchestration**

Add to `src/analysis/data_quality_audit.py`:

```python
def run_audit(config: DataQualityConfig) -> dict[str, Any]:
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    master, master_warnings = build_master_table(config)
    warnings.extend(master_warnings)

    master, spec_warnings = apply_spec014_evidence(master, config)
    warnings.extend(spec_warnings)

    master = apply_roi_quality_audit(
        master,
        images_dir=config.train_images_dir,
        dark_threshold=config.dark_brightness_threshold,
        bright_threshold=config.bright_brightness_threshold,
        low_contrast_threshold=config.low_contrast_threshold,
        weak_edge_threshold=config.weak_edge_threshold,
    )

    master, duplicate_conflicts = apply_duplicate_audit(master, config.train_images_dir)
    master = apply_label_issue_scoring(master)
    master, cleanlab_warnings = apply_optional_cleanlab_scoring(
        master, enabled=config.use_cleanlab_if_available
    )
    warnings.extend(cleanlab_warnings)
    master = assign_data_quality_buckets(master, label_issue_threshold=config.label_issue_threshold)
    manifest = build_cleaned_training_manifest(master)

    master.to_csv(output_root / "data_quality_master.csv", index=False)
    master.loc[master["data_quality_bucket"].eq("clean_train")].to_csv(
        output_root / "clean_train_rows.csv", index=False
    )
    master.loc[master["data_quality_bucket"].eq("hard_valid_train")].to_csv(
        output_root / "hard_valid_train_rows.csv", index=False
    )
    master.loc[master["data_quality_bucket"].eq("exclude_from_training")].to_csv(
        output_root / "exclude_from_training_rows.csv", index=False
    )
    master.loc[master["data_quality_bucket"].eq("manual_review_required")].to_csv(
        output_root / "manual_review_required_rows.csv", index=False
    )
    manifest.to_csv(output_root / "cleaned_training_manifest.csv", index=False)

    write_reports(master, output_root=output_root, warnings=warnings, duplicate_conflicts=duplicate_conflicts)

    summary_path = output_root / "reports" / "data_quality_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Add CLI**

Add to `src/analysis/data_quality_audit.py`:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run data quality audit pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Build data quality audit outputs")
    run_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))

    contact_parser = subparsers.add_parser(
        "build-contact-sheets", help="Build contact sheets from existing audit outputs"
    )
    contact_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "run":
        summary = run_audit(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "build-contact-sheets":
        raise SystemExit("Contact sheet generation is planned as an optional follow-up task")

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_run_audit_writes_master_buckets_and_cleaned_manifest -q
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Run compile check**

```powershell
python -m py_compile src/analysis/data_quality_audit.py
```

Expected: no output and exit code `0`.

- [ ] **Step 7: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: orchestrate data quality audit cli"
```

---

## Task 11: Contact Sheets for Manual Review

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing contact-sheet test**

```python
from src.analysis.data_quality_audit import build_contact_sheet


def test_build_contact_sheet_creates_image_grid(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(images_dir / "a.jpg")
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(images_dir / "b.jpg")
    rows = pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0, "evidence_summary": "clean"},
            {"image_id": "b.jpg", "target": 1, "evidence_summary": "manual"},
        ]
    )

    output_path = tmp_path / "sheet.jpg"
    build_contact_sheet(rows, images_dir=images_dir, output_path=output_path, max_rows=2)

    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.width > 32
        assert image.height > 32
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_build_contact_sheet_creates_image_grid -q
```

Expected:

```text
ImportError: cannot import name 'build_contact_sheet'
```

- [ ] **Step 3: Implement contact sheet generation**

Add `ImageDraw` import:

```python
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError
```

Add:

```python
def build_contact_sheet(
    rows: pd.DataFrame,
    images_dir: Path,
    output_path: Path,
    max_rows: int = 64,
    thumb_size: tuple[int, int] = (160, 160),
    columns: int = 4,
) -> None:
    selected = rows.head(max_rows).copy()
    if selected.empty:
        return

    caption_height = 56
    cell_width = thumb_size[0]
    cell_height = thumb_size[1] + caption_height
    grid_rows = int(np.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (columns * cell_width, grid_rows * cell_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for idx, (_, row) in enumerate(selected.iterrows()):
        x = (idx % columns) * cell_width
        y = (idx // columns) * cell_height
        image_path = images_dir / str(row["image_id"])
        try:
            with Image.open(image_path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail(thumb_size)
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            thumb = Image.new("RGB", thumb_size, color=(230, 230, 230))
        sheet.paste(thumb, (x, y))
        caption = f'{row.get("image_id", "")}\nt={row.get("target", "")} {row.get("data_quality_bucket", "")}'
        draw.text((x + 4, y + thumb_size[1] + 4), caption[:120], fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
```

- [ ] **Step 4: Wire contact sheets into CLI**

Add:

```python
def build_contact_sheets_from_outputs(config: DataQualityConfig) -> list[Path]:
    output_root = config.output_root
    master_path = output_root / "data_quality_master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing data quality master table: {master_path}")
    master = pd.read_csv(master_path)
    contact_root = output_root / "contact_sheets"
    contact_root.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    buckets = [
        "manual_review_required",
        "exclude_from_training",
        "hard_valid_train",
        "clean_train",
    ]
    for bucket in buckets:
        rows = master.loc[master["data_quality_bucket"].eq(bucket)].sort_values(
            ["review_priority", "label_issue_score"], ascending=[True, False]
        )
        path = contact_root / f"{bucket}.jpg"
        build_contact_sheet(rows, config.train_images_dir, path)
        if path.exists():
            outputs.append(path)
    return outputs
```

Modify the `main()` contact-sheet branch:

```python
    if args.command == "build-contact-sheets":
        paths = build_contact_sheets_from_outputs(config)
        print(json.dumps({"contact_sheets": [str(path) for path in paths]}, indent=2))
        return 0
```

- [ ] **Step 5: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_build_contact_sheet_creates_image_grid -q
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: add data quality contact sheets"
```

---

## Task 12: Validation Alignment and Input Inventory

**Files:**

- Modify: `src/analysis/data_quality_audit.py`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add failing inventory test**

```python
from src.analysis.data_quality_audit import build_input_file_inventory


def test_input_file_inventory_records_missing_and_present_files(tmp_path):
    present = tmp_path / "present.csv"
    missing = tmp_path / "missing.csv"
    present.write_text("image_id,target\nx.jpg,0\n", encoding="utf-8")

    inventory = build_input_file_inventory({"present": present, "missing": missing})

    assert dict(zip(inventory["input_name"], inventory["exists"])) == {
        "present": True,
        "missing": False,
    }
    assert inventory.loc[inventory["input_name"] == "present", "size_bytes"].item() > 0
```

- [ ] **Step 2: Run test and verify it fails**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_input_file_inventory_records_missing_and_present_files -q
```

Expected:

```text
ImportError: cannot import name 'build_input_file_inventory'
```

- [ ] **Step 3: Implement inventory and alignment summary**

Add:

```python
def build_input_file_inventory(paths: dict[str, Path | None]) -> pd.DataFrame:
    rows = []
    for name, path in paths.items():
        exists = bool(path and path.exists())
        rows.append(
            {
                "input_name": name,
                "path": "" if path is None else str(path),
                "exists": exists,
                "size_bytes": int(path.stat().st_size) if exists and path else 0,
            }
        )
    return pd.DataFrame(rows)


def build_validation_alignment_summary(master: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "has_source_split": "source_split" in master.columns and master["source_split"].notna().any(),
        "note": "Validation alignment requires source_split or fold metadata. Missing metadata limits split-drift analysis.",
    }
    if not summary["has_source_split"]:
        return summary

    grouped = (
        master.groupby(["source_split", "data_quality_bucket"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    summary["bucket_counts_by_split"] = grouped.to_dict(orient="records")
    return summary
```

- [ ] **Step 4: Wire into `run_audit`**

Inside `run_audit`, before returning summary, add:

```python
    inventory = build_input_file_inventory(
        {
            "train_csv": config.train_csv_path,
            "train_images_dir": config.train_images_dir,
            "v2b_validation_predictions": config.v2b_validation_predictions_path,
            "oof_train_predictions": config.oof_train_predictions_path,
            "spec014_allowed_hard_rows": config.allowed_hard_rows_path,
            "spec014_blocked_suspected_mislabel": config.blocked_suspected_mislabel_rows_path,
            "spec014_blocked_roi_pipeline_bug": config.blocked_roi_pipeline_bug_rows_path,
            "spec014_blocked_other_high_risk": config.blocked_other_high_risk_rows_path,
        }
    )
    inventory.to_csv(output_root / "reports" / "input_file_inventory.csv", index=False)

    alignment = build_validation_alignment_summary(master)
    (output_root / "reports" / "validation_alignment_summary.json").write_text(
        json.dumps(alignment, indent=2, sort_keys=True), encoding="utf-8"
    )
```

- [ ] **Step 5: Run test**

```powershell
python -m pytest tests/test_data_quality_audit.py::test_input_file_inventory_records_missing_and_present_files -q
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

```powershell
git add src/analysis/data_quality_audit.py tests/test_data_quality_audit.py
git commit -m "feat: record audit inputs and validation alignment"
```

---

## Task 13: Full Test Suite and Docs Index

**Files:**

- Modify: `docs/file-path-index.md`
- Modify: `tests/test_data_quality_audit.py`

- [ ] **Step 1: Add final safety test**

Add:

```python
def test_summary_confirms_no_training_submission_or_leaderboard_usage(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    master = pd.DataFrame(
        [
            {"image_id": "a.jpg", "target": 0, "data_quality_bucket": "clean_train", "exclude_from_training": False, "exclusion_reason": "", "label_issue_score": 0.1, "review_priority": 99, "roi_quality_flag": "ok", "duplicate_conflict_flag": False, "oof_probability": pd.NA},
        ]
    )
    write_reports(master, output_root=tmp_path, warnings=[], duplicate_conflicts=pd.DataFrame())

    summary = json.loads((reports_dir / "data_quality_summary.json").read_text(encoding="utf-8"))

    assert summary["no_training_started"] is True
    assert summary["no_submission_created"] is True
    assert summary["no_test_labels_used"] is True
    assert summary["no_leaderboard_tuning"] is True
```

- [ ] **Step 2: Run all Spec 016 tests**

```powershell
python -m pytest tests/test_data_quality_audit.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 3: Run compile check**

```powershell
python -m py_compile src/analysis/data_quality_audit.py
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Update path index**

Add these entries to `docs/file-path-index.md` in the same style as existing project entries:

```markdown
### Spec 016 Data Quality Audit

- `configs/data_quality_audit.yaml` - Configures train inputs, V2B/OOF predictions, Spec 014 evidence, quality thresholds, optional tools, and output root for the data-quality audit.
- `src/analysis/data_quality_audit.py` - Builds the master data-quality table, ROI/crop audit, duplicate audit, label-issue ranking, controlled buckets, reports, contact sheets, and cleaned training manifest.
- `tests/test_data_quality_audit.py` - Covers config loading, master-table construction, optional-file handling, Spec 014 blocked-row safety, ROI/crop flags, duplicate conflicts, suspicious-row ranking, cleaned manifest exclusions, report generation, and safety invariants.
- `outputs/analysis/data_quality_audit/data_quality_master.csv` - Full per-row audit table for all training rows.
- `outputs/analysis/data_quality_audit/cleaned_training_manifest.csv` - Training-safe manifest containing only clean and verified hard-valid rows.
- `outputs/analysis/data_quality_audit/reports/data_quality_summary.json` - Summary counts, warnings, OOF/embedding status, and safety confirmations.
```

- [ ] **Step 5: Run docs sanity command**

```powershell
Select-String -Path docs/file-path-index.md -Pattern "Spec 016 Data Quality Audit"
```

Expected:

```text
docs/file-path-index.md:...:### Spec 016 Data Quality Audit
```

- [ ] **Step 6: Commit**

```powershell
git add docs/file-path-index.md tests/test_data_quality_audit.py src/analysis/data_quality_audit.py configs/data_quality_audit.yaml
git commit -m "docs: index spec016 data quality audit paths"
```

---

## Task 14: Real Artifact Dry Run

**Files:**

- No source edits expected unless the dry run reveals a bug.

- [ ] **Step 1: Run compile**

```powershell
python -m py_compile src/analysis/data_quality_audit.py
```

Expected: no output and exit code `0`.

- [ ] **Step 2: Run focused tests**

```powershell
python -m pytest tests/test_data_quality_audit.py -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 3: Run audit on real project paths**

```powershell
python -m src.analysis.data_quality_audit run --config configs/data_quality_audit.yaml
```

Expected:

```text
JSON summary prints to stdout.
no_training_started: true
no_submission_created: true
no_test_labels_used: true
no_leaderboard_tuning: true
```

- [ ] **Step 4: Build contact sheets**

```powershell
python -m src.analysis.data_quality_audit build-contact-sheets --config configs/data_quality_audit.yaml
```

Expected:

```text
JSON list of generated contact sheet paths.
```

- [ ] **Step 5: Inspect key output counts**

Run:

```powershell
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path("outputs/analysis/data_quality_audit")
summary = json.loads((root / "reports" / "data_quality_summary.json").read_text())
print(json.dumps(summary, indent=2))
for name in [
    "clean_train_rows.csv",
    "hard_valid_train_rows.csv",
    "exclude_from_training_rows.csv",
    "manual_review_required_rows.csv",
    "cleaned_training_manifest.csv",
]:
    path = root / name
    print(name, len(pd.read_csv(path)) if path.exists() else "MISSING")
PY
```

Expected:

```text
All five CSVs exist and row counts are printed.
```

- [ ] **Step 6: Commit if real dry run required code fixes**

Only commit if source, tests, config, or docs changed during dry-run bug fixing:

```powershell
git add configs/data_quality_audit.yaml src/analysis/data_quality_audit.py tests/test_data_quality_audit.py docs/file-path-index.md
git commit -m "fix: stabilize spec016 data quality audit dry run"
```

---

## Task 15: Execution Report

When implementation finishes, return this exact information:

```text
Changed files:
- configs/data_quality_audit.yaml
- src/analysis/data_quality_audit.py
- tests/test_data_quality_audit.py
- docs/file-path-index.md

Exact run command:
python -m src.analysis.data_quality_audit run --config configs/data_quality_audit.yaml

Optional contact-sheet command:
python -m src.analysis.data_quality_audit build-contact-sheets --config configs/data_quality_audit.yaml

Output paths:
- outputs/analysis/data_quality_audit/data_quality_master.csv
- outputs/analysis/data_quality_audit/cleaned_training_manifest.csv
- outputs/analysis/data_quality_audit/reports/data_quality_summary.json
- outputs/analysis/data_quality_audit/contact_sheets/

Test results:
- python -m pytest tests/test_data_quality_audit.py -q
- python -m py_compile src/analysis/data_quality_audit.py

Counts:
- total rows: <from summary>
- clean_train count: <from summary>
- hard_valid_train count: <from summary>
- exclude_from_training count: <from summary>
- manual_review_required count: <from summary>
- top exclusion reasons: <from exclusion_reason_counts.csv>

Safety:
- no training started
- no submission created
- no test labels used
- no leaderboard tuning
- original labels unchanged
```

---

## Spec Coverage Self-Review

- Master table for all train rows: covered by Tasks 2 and 10.
- V2B and OOF prediction handling: covered by Tasks 2, 6, 7, and summary reporting.
- Cleanlab-style label ranking with deterministic fallback: covered by Tasks 6 and 7.
- FiftyOne-style workflow support: covered by optional visual/contact-sheet outputs and future embedding hook in Tasks 7 and 11.
- ROI/crop audit: covered by Task 4.
- Duplicate conflict detection: covered by Task 5.
- Spec 014 blocked/allowed row integration: covered by Task 3.
- Controlled buckets and cleaned manifest: covered by Task 8.
- Reports and manual review plan: covered by Task 9.
- Contact sheets: covered by Task 11.
- Validation alignment and input file inventory: covered by Task 12.
- No training/submission/test-label/leaderboard usage: covered by Tasks 9, 13, and 15.
- Docs path index update: covered by Task 13.

## Recommended Next Spec After This

Do not train directly from raw Spec 016 outputs. Also do not make the next feature a broad manual review task, because the current audit has too many candidates for slow row-by-row review and because manual review alone is not a reliable standard.

The recommended next feature is:

```text
Spec 017: Automated Decision Lock and Active Review Reduction
```

### Current Spec 016 Evidence State

Spec 016 produced a useful first-pass audit, but the evidence is not strong enough to train from blindly:

- `data_quality_master.csv`: 35,342 train rows.
- `clean_train_rows.csv`: 31,559 rows.
- `hard_valid_train_rows.csv`: 422 rows from Spec 014 approved hard rows.
- `cleaned_training_manifest.csv`: 31,981 rows, combining clean train rows and hard-valid rows.
- `manual_review_required_rows.csv`: 3,361 rows.
- Top review reasons: 3,230 ROI/crop review rows, 126 high label-issue score rows, 5 Spec 014 other high-risk review rows.
- Spec 014 blocked rows are present and must stay excluded: 5 ROI pipeline bug rows, 5 suspected mislabel rows, and 10 other high-risk rows.
- OOF prediction evidence is missing, so label-quality scoring is currently limited.
- Embeddings are missing, so duplicate, near-duplicate, cluster, and outlier evidence is currently weak.
- Cleanlab is not installed, so current label issue scores use a deterministic fallback.

### Spec 017 Goal

Build a controlled, evidence-driven decision-lock pipeline that turns Spec 016 outputs into a safer approved training manifest while reducing manual review to a small high-impact adjudication queue.

The goal is not to relabel the dataset. The goal is to decide which rows are safe to train on, which rows must be excluded, and which rows must remain deferred because the evidence is not strong enough.

### Spec 017 Architecture

Spec 017 should add a new command that consumes the existing Spec 016 artifacts, upgrades evidence where possible, applies deterministic decision rules, and writes locked decision outputs.

Recommended command:

```powershell
python -m src.analysis.data_quality_decision_lock run --config configs/data_quality_decision_lock.yaml
```

Recommended module:

```text
src/analysis/data_quality_decision_lock.py
```

Recommended tests:

```text
tests/test_data_quality_decision_lock.py
```

Recommended config:

```text
configs/data_quality_decision_lock.yaml
```

Recommended output root:

```text
outputs/analysis/data_quality_decision_lock/
```

### Required Inputs

Spec 017 should require these existing artifacts:

- `outputs/analysis/data_quality_audit/data_quality_master.csv`
- `outputs/analysis/data_quality_audit/cleaned_training_manifest.csv`
- `outputs/analysis/data_quality_audit/manual_review_required_rows.csv`
- `outputs/analysis/data_quality_audit/review_decision_template.csv`
- `outputs/analysis/data_quality_audit/reports/data_quality_summary.json`
- `outputs/analysis/data_quality_audit/reports/exclusion_reason_counts.csv`
- `outputs/analysis/hard_row_visual_review/reports/phase3_allowed_hard_training_rows.csv`
- `outputs/analysis/hard_row_visual_review/reports/blocked_roi_pipeline_bug_rows.csv`
- `outputs/analysis/hard_row_visual_review/reports/blocked_suspected_mislabel_rows.csv`
- `outputs/analysis/hard_row_visual_review/reports/blocked_other_high_risk_rows.csv`
- `data/train.csv`
- `data/train_images/`

Optional evidence inputs should be supported but not required for the first pass:

- OOF prediction probabilities for V2B or later safe baselines.
- V2B train prediction probabilities.
- V2B penultimate-layer features.
- DINOv2 image embeddings.
- CLIP image embeddings.
- Cleanlab Datalab scores if OOF probabilities and features exist.

If optional evidence is missing, Spec 017 must record the missing evidence in the summary and downgrade decisions to conservative outputs instead of silently trusting weak evidence.

### Evidence Upgrade Requirements

Spec 017 should add or support these evidence columns in a decision table:

- `embedding_status`: `missing`, `available`, or `partial`.
- `embedding_source`: `none`, `v2b_penultimate`, `dinov2`, `clip`, or `external_csv`.
- `nearest_neighbor_image_id`: nearest training neighbor when embeddings exist.
- `nearest_neighbor_distance`: numeric distance for nearest-neighbor checks.
- `duplicate_group_id`: stable group id for exact or near duplicate groups.
- `duplicate_conflict_flag`: true when a duplicate or near-duplicate group contains conflicting labels.
- `cluster_id`: stable cluster id when embeddings exist.
- `cluster_outlier_score`: numeric score for visual outlier detection.
- `cleanlab_label_quality_score`: optional score when Cleanlab evidence exists.
- `roi_quality_group`: grouped ROI/crop issue type.
- `roi_severity`: `none`, `low`, `medium`, `high`, or `critical`.
- `prediction_risk_level`: `none`, `low`, `medium`, `high`, or `critical`.
- `decision`: final controlled decision.
- `decision_reason`: compact machine-readable reason.
- `decision_confidence`: `low`, `medium`, or `high`.
- `evidence_sources`: pipe-separated list of evidence sources used for the decision.

### Decision Classes

Spec 017 should produce exactly these decision classes:

- `auto_keep`: row is safe enough for the next cleaned training manifest.
- `auto_exclude`: row is unsafe and must not be used for training.
- `needs_adjudication`: row is important but evidence is not enough for an automatic decision.
- `defer`: row remains unresolved and must not enter the next training manifest.

No row from `manual_review_required_rows.csv` should enter the approved manifest unless Spec 017 writes an explicit `auto_keep` decision or a later adjudication file gives an explicit keep decision.

### Automated Decision Rules

Spec 017 should implement conservative rules first:

- Always `auto_exclude` every image_id in any Spec 014 blocked file.
- Always `auto_exclude` rows in duplicate-conflict groups until the conflict is resolved.
- Always `defer` rows with missing image files, unreadable images, or malformed labels.
- Keep existing Spec 016 `clean_train_rows.csv` rows as `auto_keep` only if they do not overlap blocked rows, duplicate conflicts, severe ROI failures, or critical prediction-risk evidence.
- Keep Spec 014 allowed hard rows as `auto_keep` only if they do not overlap blocked rows, duplicate conflicts, severe ROI failures, or critical prediction-risk evidence.
- Convert high-confidence severe ROI/crop failures to `auto_exclude` only when the ROI issue is confirmed by multiple evidence signals, not by a single broad heuristic.
- Convert high label-risk rows to `needs_adjudication` unless optional OOF/Cleanlab/embedding evidence confirms a high-confidence exclude decision.
- Convert ambiguous rows to `defer`, not keep.

### ROI Heuristic Recalibration

Spec 016 marked 3,230 rows for ROI/crop review, so Spec 017 must not assume all are bad.

Spec 017 should group ROI rows by:

- ROI flag source.
- brightness statistics.
- contrast statistics.
- edge strength.
- image size and crop geometry.
- V2B prediction risk when available.
- embedding cluster when available.
- Spec 014 category when available.

For each ROI group, Spec 017 should produce:

- count of rows.
- positive and negative label counts.
- representative image_ids.
- group-level risk score.
- proposed group action: `keep_candidate`, `exclude_candidate`, or `adjudicate_candidate`.

Rows from over-broad ROI rules should be downgraded from manual review to `auto_keep` only if other evidence is clean. Rows from confirmed severe ROI groups should become `auto_exclude` or `needs_adjudication`, depending on confidence.

### Active Review Reduction

Manual review should be reduced to a small, evidence-ranked queue instead of a full review of 3,361 rows.

Spec 017 should rank `needs_adjudication` rows by:

- duplicate conflict risk.
- label issue risk.
- prediction disagreement risk.
- ROI severity.
- cluster outlier score.
- expected training impact.
- whether the row belongs to a larger risk group.

The adjudication queue should prioritize group representatives first, not random individual rows. This lets one decision resolve a group when evidence supports it.

Recommended adjudication file:

```text
outputs/analysis/data_quality_decision_lock/adjudication_queue.csv
```

Recommended adjudication columns:

- `image_id`
- `target`
- `risk_rank`
- `risk_bucket`
- `risk_group_id`
- `primary_issue`
- `secondary_issue`
- `suggested_decision`
- `suggested_reason`
- `evidence_sources`
- `reviewer_decision`
- `reviewer_confidence`
- `reviewer_notes`

Consensus review should be required only for high-impact uncertain rows. If two reviewers disagree, the row must become `defer`, not `auto_keep`.

### Locked Outputs

Spec 017 should write these required outputs:

- `outputs/analysis/data_quality_decision_lock/decision_master.csv`
- `outputs/analysis/data_quality_decision_lock/approved_cleaned_training_manifest.csv`
- `outputs/analysis/data_quality_decision_lock/auto_keep_rows.csv`
- `outputs/analysis/data_quality_decision_lock/auto_exclude_rows.csv`
- `outputs/analysis/data_quality_decision_lock/needs_adjudication_rows.csv`
- `outputs/analysis/data_quality_decision_lock/deferred_uncertain_rows.csv`
- `outputs/analysis/data_quality_decision_lock/adjudication_queue.csv`
- `outputs/analysis/data_quality_decision_lock/reports/decision_lock_summary.json`
- `outputs/analysis/data_quality_decision_lock/reports/decision_rule_audit.csv`
- `outputs/analysis/data_quality_decision_lock/reports/roi_recalibration_summary.csv`
- `outputs/analysis/data_quality_decision_lock/reports/evidence_inventory.json`
- `outputs/analysis/data_quality_decision_lock/contact_sheets/needs_adjudication_top_risk.jpg`
- `outputs/analysis/data_quality_decision_lock/contact_sheets/auto_exclude_representatives.jpg`
- `outputs/analysis/data_quality_decision_lock/contact_sheets/roi_group_representatives.jpg`

### Acceptance Gates

Spec 017 is complete only if all gates pass:

- `approved_cleaned_training_manifest.csv` exists.
- No image_id from Spec 014 blocked files appears in the approved manifest.
- No duplicate-conflict row appears in the approved manifest.
- No row with decision `auto_exclude`, `needs_adjudication`, or `defer` appears in the approved manifest.
- Every approved row exists in `data/train.csv`.
- Every approved row image exists under `data/train_images/`.
- Original labels in `data/train.csv` are not modified.
- Missing optional evidence is reported explicitly.
- `decision_lock_summary.json` reports counts for all decision classes.
- `decision_rule_audit.csv` explains which rule produced each decision class.
- No training starts.
- No submission is created.
- No test labels are read.
- No leaderboard tuning is performed.

### Out of Scope

Spec 017 must not:

- train a model.
- create a Kaggle submission.
- tune thresholds on leaderboard feedback.
- change labels in `data/train.csv`.
- include test labels.
- use Phase 3 rejected model outputs as a source of truth.
- automatically relabel ambiguous rows.

### Recommended Spec Kit Prompt

Use this as the next command:

```text
[$speckit-specify] Spec 017 automated decision lock and active review reduction workflow based on Spec 016 outputs. Consume outputs/analysis/data_quality_audit/data_quality_master.csv, cleaned_training_manifest.csv, manual_review_required_rows.csv, review_decision_template.csv, contact sheets, and reports. Add evidence-upgrade support for embeddings, near-duplicate groups, cluster/outlier scores, optional Cleanlab Datalab scores, and ROI heuristic recalibration. Automatically lock high-confidence keep/exclude decisions, reduce manual review to a small adjudication queue, and produce approved_cleaned_training_manifest.csv for later training. No training, no submission, no test labels, no leaderboard tuning, no original label modification.
```

### Recommended Sequence

1. `Spec 017`: Build automated decision lock and active-review reduction from Spec 016 outputs.
2. `Spec 018`: Retrain a V2B-compatible cleaned-data baseline only from `approved_cleaned_training_manifest.csv`.
3. `Spec 019`: If Spec 018 beats the locked V2B baseline on validation, test stronger architecture, targeted loss, TTA, or small ensemble under the same validation rules.
