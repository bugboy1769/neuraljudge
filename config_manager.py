import json
import os
from typing import Dict, Any, List

CONFIG_FILE = "config.json"

def load_config() -> Dict[str, Any]:
    """Load configuration from the JSON file."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to the JSON file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def get_constraints(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper to get constraints list."""
    return config.get("constraints", [])

def get_label_map(config: Dict[str, Any]) -> Dict[str, float]:
    """Helper to get label map."""
    return config.get("label_map", {})
