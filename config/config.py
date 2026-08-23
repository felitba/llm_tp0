import json
from pathlib import Path

# Repo root = parent of the config/ package, so paths work from any cwd and OS.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def load_config(config_file: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    config = load_config()
    print("Loaded configuration:")
    print(json.dumps(config, indent=4))
