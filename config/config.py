import json
from pathlib import Path


def load_config(config_file: str = "config\\config.json") -> dict:
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config

if __name__ == "__main__":
    config = load_config()
    print("Loaded configuration:")
    print(json.dumps(config, indent=4))