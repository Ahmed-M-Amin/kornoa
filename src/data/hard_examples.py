"""V2.2 hard-example loaders for automatic V2.1 triage outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


class HardExampleError(ValueError):
    """Raised when V2.2 hard-example inputs are invalid."""


@dataclass(frozen=True)
class HardExampleReport:
    hard_negatives_path: Path
    hard_positives_path: Path
    uncertain_examples_path: Path | None
    hard_negative_image_ids: list[str]
    hard_positive_image_ids: list[str]
    uncertain_image_ids: list[str]

    @property
    def hard_negative_count(self) -> int:
        return len(self.hard_negative_image_ids)

    @property
    def hard_positive_count(self) -> int:
        return len(self.hard_positive_image_ids)

    @property
    def uncertain_count(self) -> int:
        return len(self.uncertain_image_ids)

    @property
    def strong_hard_example_image_ids(self) -> list[str]:
        return _ordered_unique([*self.hard_negative_image_ids, *self.hard_positive_image_ids])


def load_hard_examples(
    *,
    hard_negatives: str | Path,
    hard_positives: str | Path,
    uncertain_examples: str | Path | None = None,
) -> HardExampleReport:
    """Load and validate V2.2 hard-example CSVs from automatic V2.1 triage."""

    hard_negative_path = Path(hard_negatives)
    hard_positive_path = Path(hard_positives)
    uncertain_path = Path(uncertain_examples) if uncertain_examples not in (None, "") else None
    hard_negative_ids = _load_group(
        hard_negative_path,
        group_name="hard negatives",
        expected_error_type="FP",
        expected_true_label=0,
        expected_v2b_prediction=1,
    )
    hard_positive_ids = _load_group(
        hard_positive_path,
        group_name="hard positives",
        expected_error_type="FN",
        expected_true_label=1,
        expected_v2b_prediction=0,
    )
    uncertain_ids = _load_uncertain(uncertain_path) if uncertain_path is not None else []
    return HardExampleReport(
        hard_negatives_path=hard_negative_path,
        hard_positives_path=hard_positive_path,
        uncertain_examples_path=uncertain_path,
        hard_negative_image_ids=hard_negative_ids,
        hard_positive_image_ids=hard_positive_ids,
        uncertain_image_ids=uncertain_ids,
    )


def validate_hard_example_images(
    report: HardExampleReport,
    *,
    available_image_ids: Sequence[str],
    fail_on_missing: bool = True,
) -> list[str]:
    """Return missing strong hard-example image IDs, optionally failing clearly."""

    available = {str(image_id) for image_id in available_image_ids}
    missing = [image_id for image_id in report.strong_hard_example_image_ids if image_id not in available]
    if missing and fail_on_missing:
        shown = ", ".join(missing[:10])
        extra = max(0, len(missing) - 10)
        suffix = f", and {extra} more" if extra else ""
        raise HardExampleError(f"Missing {len(missing)} hard-example train images: {shown}{suffix}")
    return missing


def _load_group(
    path: Path,
    *,
    group_name: str,
    expected_error_type: str,
    expected_true_label: int,
    expected_v2b_prediction: int,
) -> list[str]:
    frame = _read_required_csv(path)
    if {"error_type", "true_label", "v2b_prediction"} <= set(frame.columns):
        error_type = frame["error_type"].map(lambda value: str(value).strip().upper())
        true_label = pd.to_numeric(frame["true_label"], errors="raise").astype(int)
        v2b_prediction = pd.to_numeric(frame["v2b_prediction"], errors="raise").astype(int)
        valid = (
            (error_type == expected_error_type)
            & (true_label == expected_true_label)
            & (v2b_prediction == expected_v2b_prediction)
        )
        if not bool(valid.all()):
            raise HardExampleError(
                f"{group_name} must be {expected_error_type} rows "
                f"(true_label={expected_true_label}, v2b_prediction={expected_v2b_prediction})"
            )
    return _normalize_image_ids(frame["image_id"])


def _load_uncertain(path: Path) -> list[str]:
    frame = _read_required_csv(path)
    return _normalize_image_ids(frame["image_id"])


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HardExampleError(f"Hard-example CSV not found: {path}")
    frame = pd.read_csv(path, keep_default_na=False)
    missing = [column for column in ("image_id",) if column not in frame.columns]
    if missing:
        raise HardExampleError(f"Hard-example CSV missing required columns: {', '.join(missing)}")
    return frame


def _normalize_image_ids(values: Sequence[object]) -> list[str]:
    return _ordered_unique(str(value) for value in values if str(value).strip())


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
