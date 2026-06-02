"""Tests for SPEC-006 offline hard-example mining."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_predictions(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"image_id": "fp.jpg", "true_label": 0, "probability": 0.90, "threshold": 0.50, "predicted_label": 1},
        {"image_id": "fn.jpg", "true_label": 1, "probability": 0.10, "threshold": 0.50, "predicted_label": 0},
        {"image_id": "uncertain.jpg", "true_label": 1, "probability": 0.52, "threshold": 0.50, "predicted_label": 1},
        {"image_id": "easy.jpg", "true_label": 0, "probability": 0.05, "threshold": 0.50, "predicted_label": 0},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _make_artifact_root(root: Path) -> Path:
    artifact_root = root / "artifacts" / "kaggle_v1_artifacts" / "outputs" / "kaggle_v1"
    (artifact_root / "models").mkdir(parents=True, exist_ok=True)
    (artifact_root / "models" / "classifier_effnet_b0_best.pth").write_bytes(b"placeholder")
    reports = artifact_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "best_threshold.json").write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    (reports / "classifier_metrics.json").write_text(json.dumps({"f1_score": 0.9}), encoding="utf-8")
    _write_predictions(artifact_root / "predictions" / "val_classifier_predictions.csv")
    return artifact_root


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_mine_hard_examples_writes_validation_only_outputs(tmp_path):
    from src.training.hard_example_mining import mine_hard_examples

    predictions_path = _write_predictions(tmp_path / "predictions" / "val_classifier_predictions.csv")
    threshold_path = tmp_path / "reports" / "best_threshold.json"
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    output_dir = tmp_path / "outputs" / "hard_examples"
    summary_path = tmp_path / "outputs" / "reports" / "hard_example_summary.json"

    result = mine_hard_examples(
        predictions_path=predictions_path,
        threshold_path=threshold_path,
        output_dir=output_dir,
        summary_path=summary_path,
        uncertainty_margin=0.05,
        high_loss_limit=2,
    )

    assert [row["image_id"] for row in _read_rows(result.false_positives_path)] == ["fp.jpg"]
    assert [row["image_id"] for row in _read_rows(result.false_negatives_path)] == ["fn.jpg"]
    assert [row["image_id"] for row in _read_rows(result.uncertain_path)] == ["uncertain.jpg"]
    assert [row["image_id"] for row in _read_rows(result.high_loss_samples_path)] == ["fp.jpg", "fn.jpg"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source_predictions"] == str(predictions_path)
    assert summary["counts"] == {
        "false_positives": 1,
        "false_negatives": 1,
        "uncertain": 1,
        "high_loss_samples": 2,
    }


def test_mine_hard_examples_uses_artifact_root_without_test_images(tmp_path):
    from src.training.hard_example_mining import mine_hard_examples

    artifact_root = _make_artifact_root(tmp_path)

    result = mine_hard_examples(
        artifact_root=artifact_root,
        output_dir=tmp_path / "outputs" / "hard_examples",
        summary_path=tmp_path / "outputs" / "reports" / "hard_example_summary.json",
    )

    assert result.false_positives_path.exists()
    assert not (tmp_path / "test_images").exists()


def test_hard_example_outputs_are_ignored_when_under_outputs():
    from src.training.hard_example_mining import DEFAULT_HARD_EXAMPLE_OUTPUTS
    from src.training.train_classifier import validate_generated_artifact_confidentiality

    confidentiality = validate_generated_artifact_confidentiality(DEFAULT_HARD_EXAMPLE_OUTPUTS)

    assert confidentiality.tracked_paths == []
    assert sorted(confidentiality.ignored_paths) == sorted(DEFAULT_HARD_EXAMPLE_OUTPUTS)


def test_hard_example_mining_cli_smoke(tmp_path):
    artifact_root = _make_artifact_root(tmp_path)
    output_dir = tmp_path / "outputs" / "hard_examples"
    summary_path = tmp_path / "outputs" / "reports" / "hard_example_summary.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.training.hard_example_mining",
            "--artifact-root",
            str(artifact_root.parent),
            "--output-dir",
            str(output_dir),
            "--summary-path",
            str(summary_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "false_positives.csv").exists()
    assert summary_path.exists()
