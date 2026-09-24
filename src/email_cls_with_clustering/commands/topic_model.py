"""Fit BERTopic and write a datetime-suffixed clustered CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from email_cls_with_clustering.paths import notebooks_dir
from email_cls_with_clustering.topics import (
    TopicHyperparams,
    fit_topics,
    merge_params,
    write_topic_dataset,
)

DEFAULT_OUTPUT_NAME = "emails_clustered.csv"


def run_topic_model(
    source: Path,
    output: Path,
    params: TopicHyperparams,
    *,
    env_file: Path | None = None,
) -> Path:
    """Load a cleaned CSV, fit topics, and write a timestamped dataset."""
    frame = pd.read_csv(source)
    print(f"loaded {len(frame)} rows from {source}")
    labeled, _model, cache_path = fit_topics(
        frame,
        params,
        cache_dir=source.parent,
        env_file=env_file,
    )
    return write_topic_dataset(
        labeled,
        output,
        params,
        source=source,
        cache_path=cache_path,
    )


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a JSON object")
    return raw


def config_runs(path: Path | None) -> list[dict[str, Any]]:
    """One object is one run. A list is one run per object, for a small sweep."""
    if path is None:
        return [{}]
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list) and raw and all(isinstance(item, dict) for item in raw):
        return raw
    raise ValueError(
        f"Config {path} must be a JSON object or a non-empty list of objects"
    )


def add_hyperparam_args(parser: argparse.ArgumentParser) -> None:
    """Flags that override :class:`TopicHyperparams`. Omitted flags stay at defaults."""
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON object of TopicHyperparams fields, or a list of objects "
        "for one run each. CLI flags override every run.",
    )
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument("--umap-components", type=int, default=None, dest="n_components")
    parser.add_argument("--min-dist", type=float, default=None)
    parser.add_argument("--umap-metric", default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--hdbscan-metric", default=None)
    parser.add_argument(
        "--prediction-data",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--calculate-probabilities",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--embed-backend",
        choices=("sentence_transformer", "openai"),
        default=None,
    )
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="dotenv file for the OpenAI backend. Defaults to notebooks/.env.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit BERTopic and write emails_clustered_<datetime>.csv.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Cleaned CSV. Defaults to notebooks/emails_preprocessed.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path before the datetime suffix. "
        f"Defaults to notebooks/{DEFAULT_OUTPUT_NAME}.",
    )
    add_hyperparam_args(parser)
    return parser


def overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_neighbors": args.n_neighbors,
        "n_components": args.n_components,
        "min_dist": args.min_dist,
        "umap_metric": args.umap_metric,
        "random_state": args.random_state,
        "min_cluster_size": args.min_cluster_size,
        "hdbscan_metric": args.hdbscan_metric,
        "prediction_data": args.prediction_data,
        "calculate_probabilities": args.calculate_probabilities,
        "embed_backend": args.embed_backend,
        "embed_model": args.embed_model,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
    }


def params_from_args(args: argparse.Namespace) -> TopicHyperparams:
    return merge_params(load_config(args.config), overrides_from_args(args))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_dir = notebooks_dir()
    source = args.input or (data_dir / "emails_preprocessed.csv")
    output = args.output or (data_dir / DEFAULT_OUTPUT_NAME)
    env_file = args.env_file or (data_dir / ".env")
    overrides = overrides_from_args(args)
    runs = config_runs(args.config)
    for index, config in enumerate(runs, start=1):
        params = merge_params(config, overrides)
        if len(runs) > 1:
            print(
                f"run {index}/{len(runs)} min_cluster_size={params.min_cluster_size}"
            )
        run_topic_model(source, output, params, env_file=env_file)
