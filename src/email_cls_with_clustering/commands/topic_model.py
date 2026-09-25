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

import mlflow
from dataclasses import asdict
from email_cls_with_clustering.evaluate import score_run
from email_cls_with_clustering.tracking import setup_tracking

DEFAULT_OUTPUT_NAME = "emails_clustered.csv"


def params_for_mlflow(params: TopicHyperparams) -> dict[str, Any]:
    """Hyperparameters plus the concrete embedding model id."""
    logged: dict[str, Any] = {
        key: value for key, value in asdict(params).items() if value is not None
    }
    logged["embed_model_resolved"] = params.resolved_embed_model()
    return logged


def topic_info_frame(topic_model) -> pd.DataFrame:
    """``get_topic_info()`` with list columns serialized for ``mlflow.log_table``."""
    info = topic_model.get_topic_info().copy()
    for column in info.columns:
        if info[column].dtype != object:
            continue
        info[column] = info[column].map(
            lambda value: value if isinstance(value, str) else json.dumps(value, default=str)
        )
    return info


def run_topic_model(
    source: Path,
    output: Path,
    params: TopicHyperparams,
    *,
    env_file: Path | None = None,
    coherence: bool = True,
    dbcv: bool = True,
    nested: bool = False,
) -> Path:
    """Load a cleaned CSV, fit topics, score the run, and write a timestamped dataset."""
    frame = pd.read_csv(source)
    print(f"loaded {len(frame)} rows from {source}")
    setup_tracking()
    with mlflow.start_run(run_name=f"mcs-{params.min_cluster_size}", nested=nested):
        labeled, model, cache_path, probs, embeddings = fit_topics(
            frame,
            params,
            cache_dir=source.parent,
            env_file=env_file,
        )
        texts = labeled["ctfidf_text"].fillna("").astype(str).tolist()
        metrics = score_run(
            model,
            labeled["topic"].to_numpy(),
            texts=texts,
            embeddings=embeddings,
            probabilities=probs,
            include_coherence=coherence,
            include_dbcv=dbcv,
        )
        mlflow.log_params(params_for_mlflow(params))
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_table(topic_info_frame(model), "topic_info.json")
        return write_topic_dataset(
            labeled,
            output,
            params,
            source=source,
            cache_path=cache_path,
            topic_model=model,
            probabilities=probs,
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


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Flags that skip the slow scores. Both scores run unless turned off."""
    parser.add_argument(
        "--no-coherence",
        action="store_true",
        help="Skip c_npmi coherence. It is on by default.",
    )
    parser.add_argument(
        "--no-dbcv",
        action="store_true",
        help="Skip DBCV on the original embeddings. It is on by default.",
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
    add_eval_args(parser)
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
    score_kwargs = {
        "env_file": env_file,
        "coherence": not args.no_coherence,
        "dbcv": not args.no_dbcv,
    }
    if len(runs) == 1:
        run_topic_model(source, output, merge_params(runs[0], overrides), **score_kwargs)
        return
    setup_tracking()
    with mlflow.start_run(run_name="sweep"):
        mlflow.log_param("n_runs", len(runs))
        for index, config in enumerate(runs, start=1):
            params = merge_params(config, overrides)
            print(
                f"run {index}/{len(runs)} min_cluster_size={params.min_cluster_size}"
            )
            run_topic_model(
                source, output, params, nested=True, **score_kwargs
            )
