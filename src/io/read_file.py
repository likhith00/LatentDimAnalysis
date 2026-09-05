import yaml
from pathlib import Path

def merged_specs(path: Path):
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = registry.get("defaults", {})
    return {
        name: {**defaults, **spec}
        for name, spec in registry["datasets"].items()
    }