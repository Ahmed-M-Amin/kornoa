import pytest
import os
import yaml
from src.utils.config import load_config, ConfigError

def test_load_valid_config(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "paths.yaml"
    config_data = {
        "dataset": {
            "root": "data/raw",
            "train_csv": "train.csv",
            "sample_submission": "sample.csv"
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    config = load_config(str(config_file))
    assert config["dataset"]["root"] == "data/raw"

def test_load_missing_config():
    with pytest.raises(FileNotFoundError) as excinfo:
        load_config("non_existent_path.yaml")
    assert "not found" in str(excinfo.value).lower()

def test_load_malformed_config(tmp_path):
    config_file = tmp_path / "malformed.yaml"
    with open(config_file, "w") as f:
        f.write("dataset: [unclosed list")
    
    with pytest.raises(ConfigError) as excinfo:
        load_config(str(config_file))
    assert "parse" in str(excinfo.value).lower()
