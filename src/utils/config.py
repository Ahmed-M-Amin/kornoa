"""Config loading utilities for the Krones Vision AI project."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or invalid."""
    pass


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dict containing the parsed configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigError: If the file cannot be parsed or read.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not config_path.is_file():
        raise ConfigError(f"Config path is not a file: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ConfigError(f"Failed to read config file {config_path}: {e}") from e
    
    if not content.strip():
        raise ConfigError(f"Config file is empty: {config_path}")
    
    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config {config_path}: {e}") from e
    
    if config is None:
        raise ConfigError(f"Config file contains no data: {config_path}")
    
    return config


def load_dataset_config(config_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load dataset paths configuration.

    Defaults to configs/paths.yaml relative to project root.
    """
    if config_dir is None:
        # Try to find project root by looking for README.md, requirements.txt, etc.
        current = Path.cwd()
        while current != current.parent:
            if (current / "README.md").exists() or (current / "requirements.txt").exists():
                break
            current = current.parent
        config_path = current / "configs" / "paths.yaml"
    else:
        config_path = Path(config_dir) / "paths.yaml"
    
    return load_config(config_path)
