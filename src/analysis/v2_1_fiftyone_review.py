"""Build and export a V2.1 FiftyOne-style manual review dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml


REQUIRED_CONFIG_KEYS = {
    "input_csv",
    "train_images_dir",
    "dataset_name",
    "output_dir",
    "overwrite_dataset",
}
FORBIDDEN_FLAGS = {"allow_test_labels", "generate_submission", "train_model"}
REQUIRED_SAMPLE_FIELDS = [
    "image_id",
    "filepath",
    "error_type",
    "true_label",
    "v2b_prediction",
    "v2b_probability",
    "threshold_distance",
    "review_group",
    "suggested_visual_tag",
    "suggested_recommended_action",
    "final_visual_tag",
    "final_recommended_action",
    "reviewer_note",
    "review_status",
]
REQUIRED_INPUT_COLUMNS = [
    "image_id",
    "error_type",
    "true_label",
    "v2b_prediction",
    "v2b_probability",
    "threshold_distance",
    "review_group",
    "suggested_visual_tag",
    "suggested_recommended_action",
]
DATASET_SUMMARY_NAME = "fiftyone_dataset_summary.json"
DATASET_RECORDS_NAME = "fiftyone_dataset_records.csv"
EXPORT_OUTPUTS = {
    "reviewed_csv": "reviewed_manual_tags.csv",
    "summary": "reviewed_tag_summary.json",
    "hard_negatives": "reviewed_hard_negatives.csv",
    "hard_positives": "reviewed_hard_positives.csv",
    "uncertain_examples": "reviewed_uncertain_examples.csv",
}
STRING_FIELDS = [
    "image_id",
    "error_type",
    "review_group",
    "suggested_visual_tag",
    "suggested_recommended_action",
    "final_visual_tag",
    "final_recommended_action",
    "reviewer_note",
    "review_status",
]
NUMERIC_INT_FIELDS = ["true_label", "v2b_prediction"]
NUMERIC_FLOAT_FIELDS = ["v2b_probability", "threshold_distance"]


class V21FiftyOneReviewError(ValueError):
    """Raised when FiftyOne review inputs are invalid."""


class _LocalSample(dict):
    def __init__(self, filepath: str, **fields: object) -> None:
        super().__init__(fields)
        self["filepath"] = filepath
        self.filepath = filepath


class _LocalDataset:
    def __init__(self, name: str) -> None:
        self.name = name
        self.samples: list[_LocalSample] = []

    def add_samples(self, samples: list[_LocalSample]) -> None:
        self.samples.extend(samples)

    def iter_samples(self) -> list[_LocalSample]:
        return list(self.samples)

    def ensure_string_fields(self, fields: list[str]) -> None:
        return None


class _LocalBackend:
    """Fallback backend when FiftyOne is unavailable in the local environment."""

    class Sample(_LocalSample):
        pass

    def __init__(self) -> None:
        self.datasets: dict[str, _LocalDataset] = {}

    def dataset_exists(self, name: str) -> bool:
        return name in self.datasets

    def delete_dataset(self, name: str) -> None:
        self.datasets.pop(name, None)

    def Dataset(self, name: str) -> _LocalDataset:  # noqa: N802
        dataset = _LocalDataset(name)
        self.datasets[name] = dataset
        return dataset

    def load_dataset(self, name: str) -> _LocalDataset:
        if name not in self.datasets:
            raise V21FiftyOneReviewError(f"Dataset not built in current session: {name}")
        return self.datasets[name]

    def launch_app(self, dataset: _LocalDataset) -> None:
        raise V21FiftyOneReviewError(
            "FiftyOne is not installed in this environment. Build and export still work via local artifacts, "
            "but launch requires installing the `fiftyone` package."
        )


def load_fiftyone_review_config(path: str | Path) -> dict:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise V21FiftyOneReviewError(f"FiftyOne review config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise V21FiftyOneReviewError(f"FiftyOne review config is not valid YAML: {config_path}") from exc
    if not isinstance(raw, dict):
        raise V21FiftyOneReviewError("FiftyOne review config must be a mapping")
    missing = sorted(REQUIRED_CONFIG_KEYS - set(raw))
    if missing:
        raise V21FiftyOneReviewError("FiftyOne review config missing keys: " + ", ".join(missing))
    validate_fiftyone_review_config(raw)
    return raw


def validate_fiftyone_review_config(config: dict) -> None:
    for flag in FORBIDDEN_FLAGS:
        if bool(config.get(flag)):
            raise V21FiftyOneReviewError(f"FiftyOne review forbids {flag}=true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V2.1 FiftyOne manual review workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "launch", "export"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", default="configs/v2_1_fiftyone_review.yaml")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            outputs = build_review_dataset(args.config)
            print(f"V2.1 FiftyOne review dataset built: {outputs['summary']}")
        elif args.command == "launch":
            launch_review_app(args.config)
        else:
            outputs = export_review_results(args.config)
            print(f"V2.1 FiftyOne review export complete: {outputs['summary']}")
    except V21FiftyOneReviewError as exc:
        print(f"V2.1 FiftyOne review failed: {exc}", file=sys.stderr)
        return 2
    return 0


def build_review_dataset(config_path: str | Path, *, backend: object | None = None) -> dict[str, Path]:
    config = load_fiftyone_review_config(config_path)
    backend = _resolve_backend(backend)
    output_dir = _ensure_output_dir(config)
    rows = _prepare_rows(config)

    dataset_name = str(config["dataset_name"])
    if bool(config["overwrite_dataset"]) and backend.dataset_exists(dataset_name):
        backend.delete_dataset(dataset_name)
    dataset = backend.Dataset(dataset_name)
    _enforce_dataset_schema(dataset)
    samples = [backend.Sample(filepath=str(row["filepath"]), **{key: row[key] for key in REQUIRED_SAMPLE_FIELDS if key != "filepath"}) for row in rows.to_dict("records")]
    try:
        dataset.add_samples(samples)
    except Exception as exc:
        if backend.dataset_exists(dataset_name):
            backend.delete_dataset(dataset_name)
        raise V21FiftyOneReviewError(f"FiftyOne build failed while adding samples: {exc}") from exc
    persistent = _mark_dataset_persistent(dataset, backend)

    records_path = output_dir / DATASET_RECORDS_NAME
    rows.to_csv(records_path, index=False)
    summary_path = output_dir / DATASET_SUMMARY_NAME
    summary = _build_dataset_summary(rows, dataset_name, backend, persistent=persistent)
    _write_json(summary_path, summary)
    return {"summary": summary_path, "records": records_path}


def launch_review_app(config_path: str | Path, *, backend: object | None = None) -> str:
    config = load_fiftyone_review_config(config_path)
    backend = _resolve_backend(backend)
    dataset_name = str(config["dataset_name"])
    if isinstance(backend, _LocalBackend):
        records_path = Path(str(config["output_dir"])) / DATASET_RECORDS_NAME
        if not records_path.exists():
            raise V21FiftyOneReviewError(
                f"Dataset records not found: {records_path}. Run the build command before launch."
            )
        raise V21FiftyOneReviewError(
            "FiftyOne is not installed in this environment. Build and export still work via local artifacts, "
            "but launch requires installing the `fiftyone` package."
        )
    if hasattr(backend, "dataset_exists") and not backend.dataset_exists(dataset_name):
        raise V21FiftyOneReviewError(
            f"FiftyOne dataset '{dataset_name}' was not found.\n"
            "Run build first:\n"
            "python -m src.analysis.v2_1_fiftyone_review build --config configs/v2_1_fiftyone_review.yaml"
        )
    dataset = backend.load_dataset(dataset_name)
    session = backend.launch_app(dataset)
    url = _session_url(session)
    instructions = "\n".join(
        [
            f"FiftyOne dataset launched: {dataset_name}",
            f"URL: {url}",
            "Keep this terminal open while reviewing; closing it will end the FiftyOne session.",
            "Filter by review_group values:",
            "- high_confidence_false_positives",
            "- near_threshold_false_positives",
            "- near_threshold_false_negatives",
            "- over_rejected_reusable",
            "Suggested tags are not ground truth. Write final decisions into final_visual_tag, final_recommended_action, reviewer_note, and review_status.",
        ]
    )
    print(instructions)
    _wait_for_session(session)
    return instructions


def export_review_results(config_path: str | Path, *, backend: object | None = None) -> dict[str, Path]:
    config = load_fiftyone_review_config(config_path)
    output_dir = _ensure_output_dir(config)
    records = _load_review_records(config, backend=backend)
    reviewed_csv = output_dir / EXPORT_OUTPUTS["reviewed_csv"]
    summary_path = output_dir / EXPORT_OUTPUTS["summary"]
    hard_negatives_path = output_dir / EXPORT_OUTPUTS["hard_negatives"]
    hard_positives_path = output_dir / EXPORT_OUTPUTS["hard_positives"]
    uncertain_path = output_dir / EXPORT_OUTPUTS["uncertain_examples"]

    records.to_csv(reviewed_csv, index=False)
    hard_negatives = records.loc[records["final_recommended_action"] == "add_hard_negative"].copy()
    hard_positives = records.loc[records["final_recommended_action"] == "add_hard_positive"].copy()
    uncertain = records.loc[
        records["final_visual_tag"].isin(["ambiguous_label", "unknown"])
        | (records["final_recommended_action"] == "data_label_review")
    ].copy()
    hard_negatives.to_csv(hard_negatives_path, index=False)
    hard_positives.to_csv(hard_positives_path, index=False)
    uncertain.to_csv(uncertain_path, index=False)
    summary = _build_export_summary(records, hard_negatives, hard_positives, uncertain)
    _write_json(summary_path, summary)
    return {
        "reviewed_csv": reviewed_csv,
        "summary": summary_path,
        "hard_negatives": hard_negatives_path,
        "hard_positives": hard_positives_path,
        "uncertain_examples": uncertain_path,
    }


def _resolve_backend(backend: object | None) -> object:
    if backend is not None:
        return backend
    try:
        import fiftyone as fo  # type: ignore
    except ImportError:
        return _LocalBackend()
    return fo


def _session_url(session: object) -> str:
    for attr in ("url", "server_url"):
        value = getattr(session, attr, None)
        if value:
            return str(value)
    return "http://localhost:5151"


def _wait_for_session(session: object) -> None:
    wait = getattr(session, "wait", None)
    if callable(wait):
        wait()
        return
    input("Press Enter to close FiftyOne session...")


def _ensure_output_dir(config: dict) -> Path:
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _prepare_rows(config: dict) -> pd.DataFrame:
    input_csv = Path(str(config["input_csv"]))
    train_images_dir = Path(str(config["train_images_dir"]))
    if not input_csv.exists():
        raise V21FiftyOneReviewError(f"Input CSV not found: {input_csv}")
    if not train_images_dir.exists():
        raise V21FiftyOneReviewError(f"Train images directory not found: {train_images_dir}")
    frame = pd.read_csv(input_csv, keep_default_na=False)
    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise V21FiftyOneReviewError("Input CSV missing columns: " + ", ".join(missing_columns))
    if "final_visual_tag" not in frame.columns:
        frame["final_visual_tag"] = ""
    if "final_recommended_action" not in frame.columns:
        frame["final_recommended_action"] = ""
    if "reviewer_note" not in frame.columns:
        frame["reviewer_note"] = ""
    if "review_status" not in frame.columns:
        frame["review_status"] = ""
    for field in STRING_FIELDS:
        if field not in frame.columns:
            frame[field] = ""
        frame[field] = frame[field].map(_safe_text)
    for field in NUMERIC_INT_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="raise").astype(int)
    for field in NUMERIC_FLOAT_FIELDS:
        frame[field] = pd.to_numeric(frame[field], errors="raise").astype(float)

    filepaths: list[str] = []
    statuses: list[str] = []
    for row in frame.to_dict("records"):
        image_path = train_images_dir / str(row["image_id"])
        filepaths.append(str(image_path))
        if image_path.exists():
            existing_status = _safe_text(row.get("review_status", "")).strip()
            statuses.append(existing_status or "pending_review")
        else:
            statuses.append("missing_image")
    frame["filepath"] = filepaths
    frame["review_status"] = statuses
    for field in REQUIRED_SAMPLE_FIELDS:
        if field not in frame.columns:
            frame[field] = ""
    return frame[REQUIRED_SAMPLE_FIELDS].copy()


def _load_review_records(config: dict, *, backend: object | None) -> pd.DataFrame:
    dataset_name = str(config["dataset_name"])
    if backend is not None:
        resolved = _resolve_backend(backend)
        dataset = resolved.load_dataset(dataset_name)
        records = []
        for sample in dataset.iter_samples():
            records.append({field: sample.get(field, "") for field in REQUIRED_SAMPLE_FIELDS})
        return pd.DataFrame(records, columns=REQUIRED_SAMPLE_FIELDS)
    records_path = Path(str(config["output_dir"])) / DATASET_RECORDS_NAME
    if not records_path.exists():
        raise V21FiftyOneReviewError(
            f"Dataset records not found: {records_path}. Run the build command before export."
        )
    return pd.read_csv(records_path, keep_default_na=False)


def _enforce_dataset_schema(dataset: object) -> None:
    if hasattr(dataset, "ensure_string_fields"):
        dataset.ensure_string_fields(STRING_FIELDS)
    try:
        import fiftyone.core.fields as fof  # type: ignore
    except ImportError:
        return
    add_field = getattr(dataset, "add_sample_field", None)
    get_schema = getattr(dataset, "get_field_schema", None)
    if add_field is None or get_schema is None:
        return
    schema = get_schema() or {}
    for field in STRING_FIELDS:
        if field not in schema:
            add_field(field, fof.StringField)
    for field in NUMERIC_INT_FIELDS:
        if field not in schema:
            add_field(field, fof.IntField)
    for field in NUMERIC_FLOAT_FIELDS:
        if field not in schema:
            add_field(field, fof.FloatField)


def _mark_dataset_persistent(dataset: object, backend: object) -> bool:
    if isinstance(backend, _LocalBackend):
        return False
    try:
        setattr(dataset, "persistent", True)
    except Exception as exc:
        raise V21FiftyOneReviewError(f"Could not mark FiftyOne dataset as persistent: {exc}") from exc
    save = getattr(dataset, "save", None)
    if callable(save):
        try:
            save()
        except Exception as exc:
            raise V21FiftyOneReviewError(f"Could not save persistent FiftyOne dataset: {exc}") from exc
    return bool(getattr(dataset, "persistent", False))


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _build_dataset_summary(
    rows: pd.DataFrame,
    dataset_name: str,
    backend: object,
    *,
    persistent: bool,
) -> dict[str, object]:
    sample_count = int(len(rows))
    missing_image_count = int((rows["review_status"] == "missing_image").sum())
    return {
        "dataset_name": dataset_name,
        "sample_count": sample_count,
        "missing_image_count": missing_image_count,
        "samples": sample_count,
        "missing_images": missing_image_count,
        "review_group_counts": {
            str(key): int(value) for key, value in rows["review_group"].value_counts().to_dict().items()
        },
        "backend": getattr(backend, "__name__", backend.__class__.__name__),
        "persistent": persistent,
        "safety_flags": {
            "used_test_labels": False,
            "trained_model": False,
            "generated_submission": False,
            "changed_predictions": False,
        },
    }


def _build_export_summary(
    records: pd.DataFrame,
    hard_negatives: pd.DataFrame,
    hard_positives: pd.DataFrame,
    uncertain: pd.DataFrame,
) -> dict[str, object]:
    return {
        "reviewed_row_count": int(len(records)),
        "hard_negative_count": int(len(hard_negatives)),
        "hard_positive_count": int(len(hard_positives)),
        "uncertain_example_count": int(len(uncertain)),
        "review_status_counts": {
            str(key): int(value) for key, value in records["review_status"].fillna("").value_counts().to_dict().items()
        },
        "final_visual_tag_counts": {
            str(key): int(value) for key, value in records["final_visual_tag"].fillna("").value_counts().to_dict().items()
        },
        "final_recommended_action_counts": {
            str(key): int(value)
            for key, value in records["final_recommended_action"].fillna("").value_counts().to_dict().items()
        },
        "safety_flags": {
            "used_test_labels": False,
            "trained_model": False,
            "generated_submission": False,
            "changed_predictions": False,
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
