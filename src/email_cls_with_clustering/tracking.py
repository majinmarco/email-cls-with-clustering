"""MLflow experiment tracking helpers for local and remote runs."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.entities import Experiment

DEFAULT_EXPERIMENT = "email-cls-with-clustering"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def setup_tracking(
    *,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    enable_sklearn_autolog: bool = False,
) -> Experiment:
    """Point MLflow at this project's store and activate the experiment.

    Honors ``MLFLOW_TRACKING_URI`` and ``MLFLOW_EXPERIMENT_NAME`` when set.
    Otherwise uses a SQLite store at ``<project>/mlflow.db`` (MLflow 3.x
    default; file stores are in maintenance mode).

    Sklearn autolog is off by default so exploratory notebooks stay quiet;
    turn it on when you start fitting models.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    name = (
        experiment_name
        or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or DEFAULT_EXPERIMENT
    )

    mlflow.set_tracking_uri(uri)
    experiment = mlflow.set_experiment(name)

    if enable_sklearn_autolog:
        mlflow.sklearn.autolog(log_input_examples=False, silent=True)

    return experiment
