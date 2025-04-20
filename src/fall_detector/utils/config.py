"""Configuration utilities for the fall detector."""

import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config 