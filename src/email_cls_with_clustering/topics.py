"""Fit BERTopic on precomputed embeddings and write a timestamped dataset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from email_cls_with_clustering.embeddings import (
    EMBED_BACKEND,
    ST_MAX_SEQ_LENGTH,
    active_embed_model,
    embedding_cache_path,
    load_or_embed,
)


@dataclass
class TopicHyperparams:
    """Notebook defaults for UMAP, HDBSCAN, BERTopic, and embeddings."""

    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    umap_metric: str = "cosine"
    random_state: int = 42
    min_cluster_size: int = 50
    hdbscan_metric: str = "euclidean"
    prediction_data: bool = True
    calculate_probabilities: bool = True
    embed_backend: str = EMBED_BACKEND
    embed_model: str | None = None
    batch_size: int | None = None
    max_seq_length: int = ST_MAX_SEQ_LENGTH

    def resolved_embed_model(self) -> str:
        return active_embed_model(self.embed_backend, self.embed_model)


def params_from_mapping(raw: dict[str, Any]) -> TopicHyperparams:
    """Build params from a dict, ignoring unknown keys."""
    known = {item.name for item in fields(TopicHyperparams)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown hyperparameter(s): {', '.join(unknown)}")
    return TopicHyperparams(**raw)


def merge_params(
    config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> TopicHyperparams:
    """Defaults, then JSON config, then explicit overrides (``None`` skipped)."""
    merged: dict[str, Any] = {}
    if config:
        merged.update(config)
    if overrides:
        merged.update({key: value for key, value in overrides.items() if value is not None})
    return params_from_mapping(merged)


def timestamped_output(path: Path, *, when: datetime | None = None) -> Path:
    """Insert ``YYYYMMDDTHHMMSS`` before the suffix so runs do not overwrite."""
    stamp = (when or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def fit_topics(
    frame: pd.DataFrame,
    params: TopicHyperparams,
    *,
    cache_dir: Path,
    env_file: Path | None = None,
):
    """Embed ``full_message`` (cached) and fit BERTopic. Returns ``(frame, model, cache)``."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
    from spacy.lang.es.stop_words import STOP_WORDS as ES_STOP
    from umap import UMAP

    if "full_message" not in frame.columns:
        raise KeyError("frame must include a 'full_message' column")

    texts = frame["full_message"].fillna("").astype(str).tolist()
    model_name = params.resolved_embed_model()
    cache_path = embedding_cache_path(cache_dir, params.embed_backend, model_name)
    embeddings = load_or_embed(
        texts,
        cache_path,
        backend=params.embed_backend,
        model=model_name,
        batch_size=params.batch_size,
        max_seq_length=params.max_seq_length,
        env_file=env_file,
    )

    umap_model = UMAP(
        n_neighbors=params.n_neighbors,
        n_components=params.n_components,
        min_dist=params.min_dist,
        metric=params.umap_metric,
        random_state=params.random_state,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        metric=params.hdbscan_metric,
        prediction_data=params.prediction_data,
    )
    vectorizer_model = CountVectorizer(stop_words=list(EN_STOP | ES_STOP))
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=params.calculate_probabilities,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(texts, embeddings)

    labeled = frame.copy()
    labeled["topic"] = topics
    if probs is not None:
        labeled["prob_vector"] = list(probs)
    return labeled, topic_model, cache_path


def write_topic_dataset(
    frame: pd.DataFrame,
    output: Path,
    params: TopicHyperparams,
    *,
    source: Path,
    cache_path: Path,
    when: datetime | None = None,
) -> Path:
    """Write ``output`` with a datetime suffix and a sibling JSON of the run."""
    moment = when or datetime.now()
    stamped = timestamped_output(output, when=moment)
    stamped.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stamped, index=False)

    sidecar = stamped.with_suffix(".json")
    payload = {
        "created_at": moment.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": str(source),
        "output": str(stamped),
        "embedding_cache": str(cache_path),
        "n_documents": int(len(frame)),
        "hyperparameters": asdict(params),
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {stamped}")
    print(f"wrote {sidecar}")
    return stamped
