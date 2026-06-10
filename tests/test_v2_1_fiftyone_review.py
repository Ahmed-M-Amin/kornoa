"""Tests for V2.1 FiftyOne manual review app."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image


class _FakeSample(dict):
    filepath: str

    def __init__(self, filepath: str, **fields: object) -> None:
        super().__init__(fields)
        self.filepath = filepath
        self["filepath"] = filepath


class _FakeDataset:
    def __init__(self, name: str) -> None:
        self.name = name
        self.samples: list[_FakeSample] = []
        self.app_launched = False
        self.string_fields: set[str] = set()
        self.persistent = False
        self.save_called = False

    def add_samples(self, samples: list[_FakeSample]) -> None:
        for sample in samples:
            for key in self.string_fields:
                value = sample.get(key, "")
                if not isinstance(value, str):
                    raise ValueError(f"Invalid value for field '{key}'. Reason: {value} could not be converted to string")
        self.samples.extend(samples)

    def iter_samples(self) -> list[_FakeSample]:
        return list(self.samples)

    def ensure_string_fields(self, fields: list[str]) -> None:
        self.string_fields.update(fields)

    def save(self) -> None:
        self.save_called = True


class _FakeSession:
    def __init__(self) -> None:
        self.url = "http://localhost:5151"
        self.wait_called = False

    def wait(self) -> None:
        self.wait_called = True
        return None


class _FakeFiftyOne:
    def __init__(self) -> None:
        self.datasets: dict[str, _FakeDataset] = {}
        self.deleted: list[str] = []
        self.last_session: _FakeSession | None = None

    class Sample(_FakeSample):
        pass

    def dataset_exists(self, name: str) -> bool:
        return name in self.datasets

    def delete_dataset(self, name: str) -> None:
        self.deleted.append(name)
        self.datasets.pop(name, None)

    def Dataset(self, name: str) -> _FakeDataset:  # noqa: N802
        dataset = _FakeDataset(name)
        self.datasets[name] = dataset
        return dataset

    def load_dataset(self, name: str) -> _FakeDataset:
        return self.datasets[name]

    def launch_app(self, dataset: _FakeDataset) -> _FakeSession:
        dataset.app_launched = True
        self.last_session = _FakeSession()
        return self.last_session


def _write_prefilled_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "image_id": "fp_a.jpg",
                "error_type": "FP",
                "true_label": 0,
                "v2b_prediction": 1,
                "v2b_probability": 0.72,
                "threshold_distance": 0.40,
                "review_group": "high_confidence_false_positives",
                "visual_tag": "unknown",
                "reviewer_note": "",
                "recommended_action": "",
                "suggested_visual_tag": "acceptable_reflection_or_ring",
                "suggested_recommended_action": "add_hard_negative",
                "final_visual_tag": "acceptable_reflection",
                "final_recommended_action": "add_hard_negative",
            },
            {
                "image_id": "fn_a.jpg",
                "error_type": "FN",
                "true_label": 1,
                "v2b_prediction": 0,
                "v2b_probability": 0.28,
                "threshold_distance": 0.04,
                "review_group": "near_threshold_false_negatives",
                "visual_tag": "unknown",
                "reviewer_note": "near_threshold_ambiguous_check_manually",
                "recommended_action": "",
                "suggested_visual_tag": "small_or_low_contrast_defect",
                "suggested_recommended_action": "add_hard_positive",
                "final_visual_tag": "small_dark_defect",
                "final_recommended_action": "add_hard_positive",
            },
            {
                "image_id": "uncertain.jpg",
                "error_type": "FP",
                "true_label": 0,
                "v2b_prediction": 1,
                "v2b_probability": 0.51,
                "threshold_distance": 0.01,
                "review_group": "near_threshold_false_positives",
                "visual_tag": "unknown",
                "reviewer_note": "near_threshold_ambiguous_check_manually",
                "recommended_action": "",
                "suggested_visual_tag": "acceptable_reflection_or_ring",
                "suggested_recommended_action": "add_hard_negative",
                "final_visual_tag": "ambiguous_label",
                "final_recommended_action": "data_label_review",
            },
            {
                "image_id": "missing.jpg",
                "error_type": "FP",
                "true_label": 0,
                "v2b_prediction": 1,
                "v2b_probability": 0.66,
                "threshold_distance": 0.34,
                "review_group": "over_rejected_reusable",
                "visual_tag": "unknown",
                "reviewer_note": "",
                "recommended_action": "",
                "suggested_visual_tag": "missing_image",
                "suggested_recommended_action": "data_label_review",
                "final_visual_tag": "",
                "final_recommended_action": "",
            },
        ]
    ).to_csv(path, index=False)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    input_csv = tmp_path / "manual_review_prefilled.csv"
    train_images = tmp_path / "train_images"
    output_dir = tmp_path / "outputs" / "analysis" / "v2_1_error_intelligence" / "fiftyone_review"
    train_images.mkdir(parents=True)
    _write_prefilled_csv(input_csv)
    for image_id, color in {
        "fp_a.jpg": (200, 30, 30),
        "fn_a.jpg": (30, 200, 30),
        "uncertain.jpg": (30, 30, 200),
    }.items():
        Image.new("RGB", (80, 80), color).save(train_images / image_id)
    return {
        "input_csv": input_csv,
        "train_images": train_images,
        "output_dir": output_dir,
    }


def _write_config(tmp_path: Path, paths: dict[str, Path], **overrides: object) -> Path:
    payload = {
        "input_csv": str(paths["input_csv"]),
        "train_images_dir": str(paths["train_images"]),
        "dataset_name": "krones_v2_1_error_review",
        "output_dir": str(paths["output_dir"]),
        "overwrite_dataset": True,
        "train_model": False,
        "generate_submission": False,
        "allow_test_labels": False,
    } | overrides
    config_path = tmp_path / "v2_1_fiftyone_review.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_build_review_dataset_creates_dataset_records_and_summary(tmp_path):
    from src.analysis.v2_1_fiftyone_review import build_review_dataset

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)
    fake_backend = _FakeFiftyOne()

    outputs = build_review_dataset(config_path, backend=fake_backend)

    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    dataset = fake_backend.load_dataset("krones_v2_1_error_review")
    assert len(dataset.samples) == 4
    assert summary["dataset_name"] == "krones_v2_1_error_review"
    assert summary["review_group_counts"]["high_confidence_false_positives"] == 1


def test_build_review_dataset_marks_dataset_persistent(tmp_path):
    from src.analysis.v2_1_fiftyone_review import build_review_dataset

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)
    fake_backend = _FakeFiftyOne()

    outputs = build_review_dataset(config_path, backend=fake_backend)

    dataset = fake_backend.load_dataset("krones_v2_1_error_review")
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert dataset.persistent is True
    assert dataset.save_called is True
    assert summary["persistent"] is True


def test_build_review_dataset_marks_missing_images(tmp_path):
    from src.analysis.v2_1_fiftyone_review import build_review_dataset

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)
    fake_backend = _FakeFiftyOne()

    build_review_dataset(config_path, backend=fake_backend)

    dataset = fake_backend.load_dataset("krones_v2_1_error_review")
    missing = next(sample for sample in dataset.samples if sample["image_id"] == "missing.jpg")
    assert missing["review_status"] == "missing_image"


def test_build_review_dataset_preserves_existing_final_fields(tmp_path):
    from src.analysis.v2_1_fiftyone_review import build_review_dataset

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)
    fake_backend = _FakeFiftyOne()

    build_review_dataset(config_path, backend=fake_backend)

    dataset = fake_backend.load_dataset("krones_v2_1_error_review")
    sample = next(sample for sample in dataset.samples if sample["image_id"] == "fp_a.jpg")
    assert sample["final_visual_tag"] == "acceptable_reflection"
    assert sample["final_recommended_action"] == "add_hard_negative"


def test_build_review_dataset_normalizes_text_fields_to_strings(tmp_path):
    from src.analysis.v2_1_fiftyone_review import build_review_dataset

    paths = _write_inputs(tmp_path)
    frame = pd.read_csv(paths["input_csv"])
    frame.loc[0, "reviewer_note"] = pd.NA
    frame.loc[0, "final_visual_tag"] = pd.NA
    frame.loc[0, "final_recommended_action"] = pd.NA
    frame.to_csv(paths["input_csv"], index=False)
    config_path = _write_config(tmp_path, paths)
    fake_backend = _FakeFiftyOne()

    build_review_dataset(config_path, backend=fake_backend)

    dataset = fake_backend.load_dataset("krones_v2_1_error_review")
    first = next(sample for sample in dataset.samples if sample["image_id"] == "fp_a.jpg")
    second = next(sample for sample in dataset.samples if sample["image_id"] == "fn_a.jpg")
    assert first["reviewer_note"] == ""
    assert first["final_visual_tag"] == ""
    assert first["final_recommended_action"] == ""
    assert isinstance(second["reviewer_note"], str)
    assert second["reviewer_note"] == "near_threshold_ambiguous_check_manually"


def test_build_review_dataset_cleans_partial_dataset_when_overwriting(tmp_path):
    from src.analysis.v2_1_fiftyone_review import V21FiftyOneReviewError, build_review_dataset

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)
    fake_backend = _FakeFiftyOne()
    partial = fake_backend.Dataset("krones_v2_1_error_review")
    partial.add_samples([fake_backend.Sample(filepath="partial.jpg", image_id="partial.jpg")])

    class _CrashBackend(_FakeFiftyOne):
        def __init__(self, source: _FakeFiftyOne) -> None:
            super().__init__()
            self.datasets = source.datasets
            self.deleted = source.deleted
            self.fail_once = True

        class Sample(_FakeSample):
            pass

        def Dataset(self, name: str) -> _FakeDataset:  # noqa: N802
            dataset = super().Dataset(name)
            original_add = dataset.add_samples

            def crashing_add(samples: list[_FakeSample]) -> None:
                if self.fail_once:
                    self.fail_once = False
                    raise ValueError("simulated add_samples crash")
                original_add(samples)

            dataset.add_samples = crashing_add  # type: ignore[method-assign]
            return dataset

    crash_backend = _CrashBackend(fake_backend)

    with pytest.raises(V21FiftyOneReviewError, match="simulated add_samples crash"):
        build_review_dataset(config_path, backend=crash_backend)

    assert crash_backend.deleted == ["krones_v2_1_error_review", "krones_v2_1_error_review"]
    build_review_dataset(config_path, backend=crash_backend)
    rebuilt = crash_backend.load_dataset("krones_v2_1_error_review")
    assert len(rebuilt.samples) == 4


def test_export_review_results_creates_hard_negative_positive_and_uncertain_outputs(tmp_path):
    from src.analysis.v2_1_fiftyone_review import build_review_dataset, export_review_results

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)
    fake_backend = _FakeFiftyOne()
    build_review_dataset(config_path, backend=fake_backend)

    outputs = export_review_results(config_path, backend=fake_backend)

    hard_negatives = pd.read_csv(outputs["hard_negatives"])
    hard_positives = pd.read_csv(outputs["hard_positives"])
    uncertain = pd.read_csv(outputs["uncertain_examples"])
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert hard_negatives["image_id"].tolist() == ["fp_a.jpg"]
    assert hard_positives["image_id"].tolist() == ["fn_a.jpg"]
    assert uncertain["image_id"].tolist() == ["uncertain.jpg"]
    assert summary["review_status_counts"]["missing_image"] == 1


def test_launch_review_app_uses_existing_dataset(tmp_path):
    from src.analysis.v2_1_fiftyone_review import build_review_dataset, launch_review_app

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)
    fake_backend = _FakeFiftyOne()
    build_review_dataset(config_path, backend=fake_backend)

    message = launch_review_app(config_path, backend=fake_backend)

    dataset = fake_backend.load_dataset("krones_v2_1_error_review")
    assert dataset.app_launched is True
    assert fake_backend.last_session is not None
    assert fake_backend.last_session.wait_called is True
    assert "http://localhost:5151" in message
    assert "high_confidence_false_positives" in message


def test_launch_review_app_reports_missing_dataset(tmp_path):
    from src.analysis.v2_1_fiftyone_review import V21FiftyOneReviewError, launch_review_app

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)
    fake_backend = _FakeFiftyOne()

    with pytest.raises(V21FiftyOneReviewError, match="Run build first"):
        launch_review_app(config_path, backend=fake_backend)


def test_launch_review_app_without_fiftyone_fails_clearly(tmp_path, monkeypatch):
    import src.analysis.v2_1_fiftyone_review as review

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths)
    review.build_review_dataset(config_path, backend=review._LocalBackend())

    monkeypatch.setattr(review, "_resolve_backend", lambda backend=None: review._LocalBackend())

    with pytest.raises(review.V21FiftyOneReviewError, match="installing the `fiftyone` package"):
        review.launch_review_app(config_path)


def test_config_rejects_training_or_submission_flags(tmp_path):
    from src.analysis.v2_1_fiftyone_review import V21FiftyOneReviewError, load_fiftyone_review_config

    paths = _write_inputs(tmp_path)
    config_path = _write_config(tmp_path, paths, train_model=True)

    with pytest.raises(V21FiftyOneReviewError, match="train_model"):
        load_fiftyone_review_config(config_path)
