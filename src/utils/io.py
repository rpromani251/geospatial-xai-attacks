"""I/O helpers: save/load models, save metrics."""

import json
import joblib
from pathlib import Path


def save_model(model, path: Path) -> None:
    """Persist a fitted model to disk with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    """Load a model saved with save_model."""
    return joblib.load(path)


def save_metrics(metrics: dict, path: Path) -> None:
    """Write a metrics dict as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
