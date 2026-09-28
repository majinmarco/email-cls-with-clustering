"""Fit BERTopic on precomputed embeddings and write a timestamped dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from email_cls_with_clustering.embeddings import (
    EMBED_BACKEND,
    ST_MAX_SEQ_LENGTH,
    active_embed_model,
    apply_prompt,
    embedding_cache_path,
    load_or_embed,
    texts_fingerprint,
)
from email_cls_with_clustering.representation import prepare_representation


@dataclass
class TopicHyperparams:
    """Defaults for UMAP, HDBSCAN, BERTopic, and embeddings."""

    n_neighbors: int = 15
    n_components: int = 10
    min_dist: float = 0.0
    umap_metric: str = "cosine"
    random_state: int | None = None
    n_jobs: int = -1
    min_cluster_size: int = 100
    min_samples: int = 10
    hdbscan_metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    cluster_selection_epsilon: float = 0.0
    prediction_data: bool = True
    gen_min_span_tree: bool = True
    calculate_probabilities: bool = True
    embed_backend: str = EMBED_BACKEND
    embed_model: str | None = None
    embed_prompt: str = ""
    batch_size: int | None = None
    max_seq_length: int = ST_MAX_SEQ_LENGTH
    post_processing: str = "none"
    top_k: int | None = None
    ngram_range: tuple[int, int] = (1, 1)

    def resolved_embed_model(self) -> str:
        return active_embed_model(self.embed_backend, self.embed_model)


def params_from_mapping(raw: dict[str, Any]) -> TopicHyperparams:
    """Build params from a dict. Rejects unknown keys; coerces ngram_range."""
    known = {item.name for item in fields(TopicHyperparams)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown hyperparameter(s): {', '.join(unknown)}")
    data = dict(raw)
    if "ngram_range" in data and isinstance(data["ngram_range"], list):
        data["ngram_range"] = tuple(data["ngram_range"])
    return TopicHyperparams(**data)


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


def params_for_json(params: TopicHyperparams) -> dict[str, Any]:
    """Hyperparameters for a JSON sidecar (tuples become lists)."""
    payload = asdict(params)
    if isinstance(payload.get("ngram_range"), tuple):
        payload["ngram_range"] = list(payload["ngram_range"])
    return payload


def _mlflow_value(value: Any) -> Any:
    """A value MLflow will show. ``None`` is the string ``null`` so it is not dropped."""
    if value is None:
        return "null"
    if isinstance(value, (tuple, list)):
        return json.dumps(list(value))
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def params_for_mlflow(
    params: TopicHyperparams,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every hyperparameter, including nulls, plus fixed vectorizer and embedding knobs."""
    logged = {key: _mlflow_value(value) for key, value in asdict(params).items()}
    logged["embed_model_resolved"] = params.resolved_embed_model()
    logged["normalize_embeddings"] = True
    logged["stop_words"] = "EN_STOP | ES_STOP"
    if extra:
        for key, value in extra.items():
            logged[key] = _mlflow_value(value)
    return logged


def umap_kwargs(params: TopicHyperparams) -> dict[str, Any]:
    """Keyword arguments for ``UMAP``."""
    return {
        "n_neighbors": params.n_neighbors,
        "n_components": params.n_components,
        "min_dist": params.min_dist,
        "metric": params.umap_metric,
        "random_state": params.random_state,
        "n_jobs": params.n_jobs,
    }


def hdbscan_kwargs(params: TopicHyperparams) -> dict[str, Any]:
    """Keyword arguments for ``HDBSCAN`` (``min_samples`` always explicit)."""
    return {
        "min_cluster_size": params.min_cluster_size,
        "min_samples": params.min_samples,
        "metric": params.hdbscan_metric,
        "cluster_selection_method": params.cluster_selection_method,
        "cluster_selection_epsilon": params.cluster_selection_epsilon,
        "prediction_data": params.prediction_data,
        "gen_min_span_tree": params.gen_min_span_tree,
    }


def projection_cache_path(
    directory: Path,
    params: TopicHyperparams,
    fingerprint: str,
) -> Path:
    """Cache path for a UMAP projection. Independent of HDBSCAN settings."""
    material = {
        "backend": params.embed_backend,
        "model": params.resolved_embed_model(),
        "prompt": params.embed_prompt,
        "post_processing": params.post_processing,
        "top_k": params.top_k,
        "n_neighbors": params.n_neighbors,
        "n_components": params.n_components,
        "min_dist": params.min_dist,
        "umap_metric": params.umap_metric,
        "random_state": params.random_state,
        "n_jobs": params.n_jobs,
        "fingerprint": fingerprint,
    }
    digest = hashlib.blake2b(
        json.dumps(material, sort_keys=True, default=str).encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    return directory / f"umap_projection_{digest}.npy"


def embeddings_for_metrics(high_dim: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """Return the high-dim matrix after checking row counts match the projection."""
    if len(high_dim) != len(projection):
        raise ValueError(
            f"row count mismatch: high_dim={len(high_dim)} projection={len(projection)}"
        )
    return high_dim


def timestamped_output(path: Path, *, when: datetime | None = None) -> Path:
    """Insert ``YYYYMMDDTHHMMSS`` before the suffix so runs do not overwrite."""
    stamp = (when or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def load_or_project(
    embeddings: np.ndarray,
    params: TopicHyperparams,
    cache_path: Path,
) -> np.ndarray:
    """Load a cached UMAP projection or fit and save one."""
    from umap import UMAP

    if cache_path.exists():
        projection = np.load(cache_path)
        if len(projection) == len(embeddings):
            print(f"loaded projection from {cache_path} → shape={projection.shape}")
            return projection
        print(f"invalidated stale projection cache at {cache_path}")

    umap_model = UMAP(**umap_kwargs(params))
    projection = umap_model.fit_transform(embeddings).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, projection)
    print(f"saved projection → {cache_path} shape={projection.shape}")
    return projection


def fit_hdbscan_on_projection(
    docs: list[str],
    projection: np.ndarray,
    params: TopicHyperparams,
    *,
    calculate_probabilities: bool,
):
    """Fit BERTopic with a frozen projection (no UMAP re-fit)."""
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
    from spacy.lang.es.stop_words import STOP_WORDS as ES_STOP

    vectorizer_model = CountVectorizer(
        stop_words=list(EN_STOP | ES_STOP),
        ngram_range=params.ngram_range,
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=HDBSCAN(**hdbscan_kwargs(params)),
        vectorizer_model=vectorizer_model,
        calculate_probabilities=calculate_probabilities,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(docs, projection)
    return topic_model, np.asarray(topics), probs


def reduced_labels(
    topic_model,
    docs: list[str],
    topics,
    *,
    strategy: str,
    threshold: float,
    embeddings=None,
) -> np.ndarray:
    """Return reduced labels without mutating the model via ``update_topics``."""
    reduced = topic_model.reduce_outliers(
        docs,
        list(topics),
        strategy=strategy,
        threshold=threshold,
        embeddings=embeddings,
    )
    return np.asarray(reduced, dtype=int)


def apply_topic_labels(topic_model, docs: list[str], topics) -> None:
    """Update c-TF-IDF with ``topics`` while keeping the fitted vectorizer."""
    topic_model.update_topics(
        docs,
        topics=list(map(int, topics)),
        vectorizer_model=topic_model.vectorizer_model,
        ctfidf_model=topic_model.ctfidf_model,
    )


def fit_topics(
    frame: pd.DataFrame,
    params: TopicHyperparams,
    *,
    cache_dir: Path,
    env_file: Path | None = None,
):
    """Embed ``embed_text`` (cached), post-process, and fit BERTopic.

    Returns ``(frame, model, cache, probabilities, embeddings)``.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
    from spacy.lang.es.stop_words import STOP_WORDS as ES_STOP
    from umap import UMAP

    if "embed_text" not in frame.columns or "ctfidf_text" not in frame.columns:
        raise KeyError("frame must include 'embed_text' and 'ctfidf_text' columns")

    embed_texts_col = frame["embed_text"].fillna("").astype(str).tolist()
    ctfidf_texts = frame["ctfidf_text"].fillna("").astype(str).tolist()
    model_name = params.resolved_embed_model()
    cache_path = embedding_cache_path(
        cache_dir, params.embed_backend, model_name, prompt=params.embed_prompt
    )
    embeddings = load_or_embed(
        embed_texts_col,
        cache_path,
        backend=params.embed_backend,
        model=model_name,
        batch_size=params.batch_size,
        max_seq_length=params.max_seq_length,
        env_file=env_file,
        prompt=params.embed_prompt,
    )
    embeddings = prepare_representation(embeddings, params)

    umap_model = UMAP(**umap_kwargs(params))
    hdbscan_model = HDBSCAN(**hdbscan_kwargs(params))
    vectorizer_model = CountVectorizer(
        stop_words=list(EN_STOP | ES_STOP),
        ngram_range=params.ngram_range,
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=params.calculate_probabilities,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(ctfidf_texts, embeddings)

    fingerprint = texts_fingerprint(apply_prompt(embed_texts_col, params.embed_prompt))
    proj_path = projection_cache_path(cache_dir, params, fingerprint)
    projection = np.asarray(topic_model.umap_model.embedding_, dtype=np.float32)
    proj_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(proj_path, projection)
    print(f"saved projection → {proj_path} shape={projection.shape}")

    labeled = frame.copy()
    labeled["topic"] = topics
    return labeled, topic_model, cache_path, probs, embeddings


def save_topic_model(topic_model, path: Path) -> Path:
    """Persist a fitted BERTopic model (pickle) for later load / transform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pickle keeps UMAP/HDBSCAN so ``transform`` works on new embeddings.
    # Embedding model is already None (embeddings are precomputed).
    topic_model.save(str(path), serialization="pickle", save_embedding_model=False)
    return path


def write_topic_dataset(
    frame: pd.DataFrame,
    output: Path,
    params: TopicHyperparams,
    *,
    source: Path,
    cache_path: Path,
    topic_model=None,
    probabilities=None,
    when: datetime | None = None,
) -> Path:
    """Write ``output`` with a datetime suffix, model file, and sibling JSON."""
    moment = when or datetime.now()
    stamped = timestamped_output(output, when=moment)
    stamped.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(stamped, index=False)
    if probabilities is not None:
        np.save(stamped.with_name(f"{stamped.stem}_probs.npy"), np.asarray(probabilities))

    model_path = None
    if topic_model is not None:
        model_path = stamped.with_name(f"{stamped.stem}_model.pkl")
        save_topic_model(topic_model, model_path)
        print(f"wrote {model_path}")

    sidecar = stamped.with_suffix(".json")
    payload = {
        "created_at": moment.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": str(source),
        "output": str(stamped),
        "topic_model": str(model_path) if model_path else None,
        "embedding_cache": str(cache_path),
        "n_documents": int(len(frame)),
        "hyperparameters": params_for_json(params),
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {stamped}")
    print(f"wrote {sidecar}")
    return stamped
