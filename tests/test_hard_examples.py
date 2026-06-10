"""Tests for V2.2 hard-example training support."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml


def _write_rows(path: Path, rows: list[dict[str, object]], *, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = columns or ["image_id", "error_type", "true_label", "v2b_prediction"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_sources(root: Path) -> dict[str, Path]:
    hard_negatives = root / "hard_negatives.csv"
    hard_positives = root / "hard_positives.csv"
    uncertain = root / "uncertain.csv"
    _write_rows(
        hard_negatives,
        [
            {"image_id": "fp_a.png", "error_type": "FP", "true_label": 0, "v2b_prediction": 1},
            {"image_id": "fp_a.png", "error_type": "FP", "true_label": 0, "v2b_prediction": 1},
            {"image_id": 123, "error_type": "FP", "true_label": 0, "v2b_prediction": 1},
        ],
    )
    _write_rows(
        hard_positives,
        [{"image_id": "fn_a.png", "error_type": "FN", "true_label": 1, "v2b_prediction": 0}],
    )
    _write_rows(
        uncertain,
        [{"image_id": "uncertain_a.png", "error_type": "FP", "true_label": 0, "v2b_prediction": 1}],
    )
    return {
        "hard_negatives": hard_negatives,
        "hard_positives": hard_positives,
        "uncertain_examples": uncertain,
    }


def test_hard_example_loader_loads_counts_and_deduplicates_image_ids(tmp_path):
    from src.data.hard_examples import load_hard_examples

    paths = _write_sources(tmp_path)

    report = load_hard_examples(**paths)

    assert report.hard_negative_count == 2
    assert report.hard_positive_count == 1
    assert report.uncertain_count == 1
    assert report.hard_negative_image_ids == ["fp_a.png", "123"]
    assert report.hard_positive_image_ids == ["fn_a.png"]
    assert report.strong_hard_example_image_ids == ["fp_a.png", "123", "fn_a.png"]
    assert "uncertain_a.png" not in report.strong_hard_example_image_ids


def test_hard_example_loader_missing_csv_fails_clearly(tmp_path):
    from src.data.hard_examples import HardExampleError, load_hard_examples

    paths = _write_sources(tmp_path)
    paths["hard_negatives"].unlink()

    with pytest.raises(HardExampleError, match="Hard-example CSV not found"):
        load_hard_examples(**paths)


def test_hard_example_loader_missing_image_id_column_fails_clearly(tmp_path):
    from src.data.hard_examples import HardExampleError, load_hard_examples

    paths = _write_sources(tmp_path)
    _write_rows(paths["hard_negatives"], [{"error_type": "FP", "true_label": 0, "v2b_prediction": 1}], columns=["error_type", "true_label", "v2b_prediction"])

    with pytest.raises(HardExampleError, match="missing required columns: image_id"):
        load_hard_examples(**paths)


def test_hard_example_loader_rejects_wrong_hard_negative_logic(tmp_path):
    from src.data.hard_examples import HardExampleError, load_hard_examples

    paths = _write_sources(tmp_path)
    _write_rows(paths["hard_negatives"], [{"image_id": "bad.png", "error_type": "FN", "true_label": 1, "v2b_prediction": 0}])

    with pytest.raises(HardExampleError, match="hard negatives must be FP rows"):
        load_hard_examples(**paths)


def test_hard_example_loader_rejects_wrong_hard_positive_logic(tmp_path):
    from src.data.hard_examples import HardExampleError, load_hard_examples

    paths = _write_sources(tmp_path)
    _write_rows(paths["hard_positives"], [{"image_id": "bad.png", "error_type": "FP", "true_label": 0, "v2b_prediction": 1}])

    with pytest.raises(HardExampleError, match="hard positives must be FN rows"):
        load_hard_examples(**paths)


def test_v2_2_config_is_safe_and_uses_binary_f1_contract():
    config_path = Path("configs/v2_2_hard_examples.yaml")

    assert config_path.exists()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hard_examples = payload["hard_examples"]
    metrics = payload["metrics"]

    assert hard_examples["enabled"] is True
    assert hard_examples["hard_negatives"]
    assert hard_examples["hard_positives"]
    assert hard_examples["uncertain_examples"]
    assert hard_examples["hard_negative_weight"] <= 1.5
    assert hard_examples["hard_positive_weight"] <= 1.5
    assert hard_examples["max_extra_sampling_multiplier"] <= 2.0
    assert payload["safety"]["generate_submission"] is False
    assert payload["safety"]["allow_test_labels"] is False
    assert metrics["positive_class"] == 1
    assert metrics["zero_division"] == 0


def test_training_cli_accepts_v2_2_smoke_aliases_without_starting_real_training(monkeypatch):
    from src.training import train_classifier

    captured = {}

    def _fake_run_training(**kwargs):
        captured.update(kwargs)

        class _Result:
            class metrics:
                f1_score = 0.0

            class threshold:
                threshold = 0.5

            model_path = Path("unused.pth")

        return _Result()

    monkeypatch.setattr(train_classifier, "run_training", _fake_run_training)

    exit_code = train_classifier.main(
        [
            "--config",
            "configs/v2_2_hard_examples.yaml",
            "--max-epochs",
            "1",
            "--limit-train-batches",
            "10",
            "--limit-val-batches",
            "5",
        ]
    )

    assert exit_code == 0
    assert captured["epochs"] == 1
    assert captured["config"].limit_train_batches == 10
    assert captured["config"].limit_val_batches == 5
    assert captured["config"].hard_examples_enabled is True
    assert captured["dataset_root"] == "1st-krones-vision-ai-challenge"
