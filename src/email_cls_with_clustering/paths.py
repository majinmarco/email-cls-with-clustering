"""Project locations shared by the pipeline commands."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def notebooks_dir() -> Path:
    return PROJECT_ROOT / "notebooks"
