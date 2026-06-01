import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_foundation_root_directories_and_files_exist():
    required_paths = [
        "README.md",
        "requirements.txt",
        "configs/paths.yaml",
        "configs/classifier.yaml",
        "configs/detector.yaml",
        "configs/inference.yaml",
        "configs/dashboard.yaml",
        "notebooks",
        "src/data",
        "src/models",
        "src/training",
        "src/inference",
        "src/explainability",
        "src/dashboard",
        "src/utils",
        "tests/test_dataset.py",
        "tests/test_coco_parser.py",
        "outputs/.gitkeep",
        "outputs/models/.gitkeep",
        "outputs/predictions/.gitkeep",
        "outputs/figures/.gitkeep",
        "outputs/reports/.gitkeep",
        "outputs/submissions/.gitkeep",
        "outputs/hard_examples/.gitkeep",
    ]

    missing = [path for path in required_paths if not (ROOT / path).exists()]

    assert missing == []


def test_source_package_markers_are_importable():
    import src
    import src.dashboard
    import src.data
    import src.explainability
    import src.inference
    import src.models
    import src.training
    import src.utils

    assert src is not None


def test_notebook_placeholders_are_valid_minimal_json():
    notebook_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))

    assert {path.name for path in notebook_paths} == {
        "01_dataset_audit.ipynb",
        "02_train_classifier.ipynb",
        "03_train_detector.ipynb",
        "04_distillation_memory.ipynb",
        "05_kaggle_submission.ipynb",
    }
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert isinstance(notebook["cells"], list)


def test_yaml_configs_parse_and_paths_are_configurable():
    config_paths = sorted((ROOT / "configs").glob("*.yaml"))

    assert config_paths
    for path in config_paths:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), path

    paths = yaml.safe_load((ROOT / "configs/paths.yaml").read_text(encoding="utf-8"))
    assert paths["dataset"]["root"]
    assert paths["dataset"]["train_csv"] == "train.csv"
    assert paths["dataset"]["train_images"] == "train_images"
    assert paths["dataset"]["test_images"] == "test_images"


def test_gitignore_protects_private_data_outputs_and_cache():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in [
        "1st-krones-vision-ai-challenge/",
        "train_images/",
        "test_images/",
        "train_annotations.json",
        "outputs/models/*",
        "!outputs/models/.gitkeep",
        "outputs/reports/*",
        "!outputs/reports/.gitkeep",
        "__pycache__/",
        "*.pyc",
        ".venv/",
        ".ipynb_checkpoints/",
    ]:
        assert pattern in gitignore
