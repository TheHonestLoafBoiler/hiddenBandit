"""
Config Loader
==============

Reads a JSON file and returns a validated FullConfig object.
This is the ONLY way configs enter the system.
"""

import json
from pathlib import Path
from .schema import FullConfig


def load_config(path: str | Path) -> FullConfig:
    """Load and validate a JSON config file.

    Args:
        path: Path to a JSON config file. If fields are missing, Pydantic
              fills in defaults.

    Returns:
        A validated FullConfig instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        pydantic.ValidationError: If the config is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = json.load(f)

    return FullConfig(**raw)
