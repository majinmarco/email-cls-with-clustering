"""Scoring for a fitted BERTopic run. Nothing here refits anything."""

from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np


def _word_pairs(words, topk: int) -> list:
    """Top pairs for one topic. Empty or non-sequence input yields no pairs."""
    if not words:
        return []
    try:
        return list(words)[:topk]
    except TypeError:
        return []


def top_words(topic_model, topk: int = 10) -> dict[int, list[str]]:
    """Top-k words per topic, outlier topic -1 excluded."""
    return {
        t: [w for w, _ in _word_pairs(words, topk)]
        for t, words in topic_model.get_topics().items()
        if t != -1
    }


def topic_diversity(topic_model, topk: int = 10) -> float:
    """Share of distinct words across all topics' top-k lists. 1.0 = no repeats."""
    lists = top_words(topic_model, topk)
    total = sum(len(w) for w in lists.values())
    if total == 0:
        return float("nan")
    unique = len({w for words in lists.values() for w in words})
    return unique / total


def peakedness(topic_model, topk: int = 10) -> dict[int, float]:
    """Max c-TF-IDF weight per topic. Below ~0.05 means the cluster is filler.

    Topics with no top words are skipped. ``max`` on an empty list would raise.
    """
    scores: dict[int, float] = {}
    for topic_id, words in topic_model.get_topics().items():
        if topic_id == -1:
            continue
        pairs = _word_pairs(words, topk)
        if not pairs:
            continue
        scores[topic_id] = max(score for _, score in pairs)
    return scores


def coherence(topic_model, texts, measure: str = "c_npmi", topk: int = 10) -> float:
    """Gensim coherence. Needs `uv add gensim`. Use c_npmi, not c_v."""
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    tokens = [doc.split() for doc in texts]
    dictionary = Dictionary(tokens)
    vocab = dictionary.token2id
    topics = [[w for w in words if w in vocab]
              for words in top_words(topic_model, topk).values()]
    topics = [words for words in topics if len(words) >= 2]
    if not topics:
        return float("nan")
    try:
        model = CoherenceModel(
            topics=topics, texts=tokens, dictionary=dictionary,
            coherence=measure, topn=topk, processes=1,
        )
        return float(model.get_coherence())
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _stable_core_distance(distance_matrix, d=2.0):
    """All-points core distance computed in log space.

    ``hdbscan.validity.all_points_core_distance`` raises ``1 / dist`` to the
    ``d``-th power. On 384-dim embeddings that overflows to ``inf`` and the core
    distance collapses to 0, which silently biases DBCV. This returns
    ``(mean_j dist_ij ** -d) ** (-1 / d)`` via logsumexp, same as the paper.
    Zero distances (the point itself, duplicates) are skipped as upstream does.
    """
    from scipy.special import logsumexp

    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = distance_matrix.shape[0]
    if n < 2:
        return np.zeros(n)
    positive = distance_matrix > 0
    with np.errstate(divide="ignore"):
        log_inv = np.where(positive, -d * np.log(distance_matrix), -np.inf)
    log_mean = logsumexp(log_inv, axis=1) - np.log(n - 1)
    result = np.exp(-log_mean / d)
    # Rows with no positive distance have log_mean = -inf; upstream returns 0.
    result[~positive.any(axis=1)] = 0.0
    return result


def dbcv(embeddings, labels, sample: int | None = 20_000, seed: int = 42) -> float:
    """DBCV on the ORIGINAL embedding space. Noise points are dropped.

    Returns NaN when the labeling is too degenerate for the MST (fewer than
    two non-noise points, or any remaining cluster with size < 2), or when
    ``validity_index`` raises ``ValueError``. Core distances use
    :func:`_stable_core_distance` so high-dimensional inputs do not overflow.
    """
    from unittest import mock

    from hdbscan import validity

    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    keep = labels >= 0
    X, y = embeddings[keep], labels[keep]
    if sample is not None and len(X) > sample:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(X), size=sample, replace=False)
        X, y = X[pick], y[pick]
    if len(X) < 2:
        return float("nan")
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return float("nan")
    try:
        with mock.patch.object(validity, "all_points_core_distance", _stable_core_distance):
            return float(validity.validity_index(X, y))
    except ValueError:
        return float("nan")


def anisotropy(
    embeddings,
    sample: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Mean pairwise cosine, norm of mean vector, and top singular energy share."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return {
            "mean_pairwise_cosine": float("nan"),
            "norm_of_mean_vector": float("nan"),
            "top_singular_energy_share": float("nan"),
        }
    n = matrix.shape[0]
    if n > sample:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(n, size=sample, replace=False))
        matrix = matrix[pick]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    gram = normalized @ normalized.T
    n_sample = normalized.shape[0]
    if n_sample < 2:
        mean_pairwise = float("nan")
    else:
        iu = np.triu_indices(n_sample, k=1)
        mean_pairwise = float(gram[iu].mean())
    mean_vec = normalized.mean(axis=0)
    norm_of_mean = float(np.linalg.norm(mean_vec))
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(n_sample, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    energy = float(np.maximum(eigenvalues, 0.0).sum())
    if energy == 0.0:
        top_share = float("nan")
    else:
        top_share = float(np.maximum(eigenvalues, 0.0).max() / energy)
    return {
        "mean_pairwise_cosine": mean_pairwise,
        "norm_of_mean_vector": norm_of_mean,
        "top_singular_energy_share": top_share,
    }


def label_scores(y_true, y_pred) -> dict[str, float]:
    """Against real ground truth. ARI and AMI are chance-corrected, V-measure is not."""
    from sklearn.metrics import (
        adjusted_mutual_info_score, adjusted_rand_score, v_measure_score,
    )
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "ami": float(adjusted_mutual_info_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
    }


FLAT_PEAK = 0.05

DEFAULT_GATES = {
    "largest_topic_share_max": 0.10,
    "noise_share_max": 0.50,
    "flat_topic_share_max": 0.50,
    "n_topics_min": 30,
    "n_topics_max": 200,
}

METRIC_ALIASES = {
    "coherence": "coherence_c_npmi",
}

LOWER_IS_BETTER = {
    "noise_share",
    "largest_topic_share",
}


def _n_topics(labels: np.ndarray) -> int:
    """Count of topic ids that are zero or positive."""
    return int(np.unique(labels[labels >= 0]).size)


def _noise_share(labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    return float(np.mean(labels == -1))


def _largest_topic_share(labels: np.ndarray) -> float:
    """Biggest non-outlier topic, as a share of every document."""
    if labels.size == 0:
        return float("nan")
    kept = labels[labels >= 0]
    if kept.size == 0:
        return float("nan")
    _, counts = np.unique(kept, return_counts=True)
    return float(counts.max() / labels.size)


def _topic_size_entropy(labels: np.ndarray) -> float:
    """Shannon entropy of non-outlier sizes, divided by log(n_topics).

    1.0 is a perfectly even split. Undefined for fewer than two topics.
    """
    kept = labels[labels >= 0]
    if kept.size == 0:
        return float("nan")
    _, counts = np.unique(kept, return_counts=True)
    n_topics = counts.size
    if n_topics < 2:
        return float("nan")
    shares = counts / counts.sum()
    entropy = -np.sum(shares * np.log(shares))
    return float(entropy / np.log(n_topics))


def _assignment_metrics(probabilities) -> dict[str, float]:
    """Mean max probability and mean entropy for a document-by-topic matrix."""
    if probabilities is None:
        return {}
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] == 0:
        return {}
    finite_rows = np.isfinite(probs).all(axis=1)
    if not finite_rows.any():
        return {}
    probs = np.clip(probs[finite_rows], 0, None)
    totals = probs.sum(axis=1, keepdims=True)
    safe = np.divide(probs, totals, out=np.zeros_like(probs), where=totals > 0)
    log_p = np.zeros_like(safe)
    np.log(safe, out=log_p, where=safe > 0)
    entropy = -(safe * log_p).sum(axis=1)
    return {
        "mean_max_probability": float(probs.max(axis=1).mean()),
        "mean_assignment_entropy": float(entropy.mean()),
    }


def _finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if np.isfinite(value)}


def _resolve_metric_name(name: str) -> str:
    return METRIC_ALIASES.get(name, name)


def passes_hard_gates(
    metrics: dict[str, float],
    gates: dict[str, float] | None = None,
) -> bool:
    """Return True when every gate is satisfied. Missing metrics fail closed."""
    active = gates if gates is not None else DEFAULT_GATES
    for key, bound in active.items():
        if key.endswith("_max"):
            metric_name = key[: -len("_max")]
            value = metrics.get(metric_name)
            if value is None or not np.isfinite(value) or value > bound:
                return False
        elif key.endswith("_min"):
            metric_name = key[: -len("_min")]
            value = metrics.get(metric_name)
            if value is None or not np.isfinite(value) or value < bound:
                return False
        else:
            raise ValueError(f"Unrecognized gate key: {key}")
    return True


def gate_violation(
    metrics: dict[str, float],
    gates: dict[str, float] | None = None,
) -> float:
    """Sum of relative gate misses; 0.0 when every gate passes.

    Each miss is ``|value - bound| / max(|bound|, 1e-9)`` on the failing side.
    A missing or non-finite metric counts as a full miss of 1.0.
    """
    active = gates if gates is not None else DEFAULT_GATES
    total = 0.0
    for key, bound in active.items():
        if key.endswith("_max"):
            metric_name, sign = key[: -len("_max")], 1.0
        elif key.endswith("_min"):
            metric_name, sign = key[: -len("_min")], -1.0
        else:
            raise ValueError(f"Unrecognized gate key: {key}")
        value = metrics.get(metric_name)
        if value is None or not np.isfinite(value):
            total += 1.0
            continue
        miss = sign * (value - bound)
        if miss > 0:
            total += miss / max(abs(bound), 1e-9)
    return total


def rank_key(metrics: dict[str, float], order: Sequence[str]) -> tuple:
    """Sort key: better values first; non-finite / missing sort last."""
    key: list[Any] = []
    for name in order:
        resolved = _resolve_metric_name(name)
        value = metrics.get(resolved)
        if value is None or not np.isfinite(value):
            key.append((1, 0.0))
            continue
        if resolved in LOWER_IS_BETTER:
            key.append((0, float(value)))
        else:
            key.append((0, -float(value)))
    return tuple(key)


def select_advancing(
    records: Sequence[dict[str, Any]],
    order: Sequence[str],
    n: int,
    gates: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Filter by gates, rank, and return up to ``n`` passers. No backfill."""
    passed = [
        record for record in records
        if passes_hard_gates(record.get("metrics", {}), gates)
    ]
    passed.sort(key=lambda record: rank_key(record.get("metrics", {}), order))
    advanced = passed[:n]
    return advanced, len(passed) >= n


def _optional_metric(fn):
    """Run one metric. A degenerate input returns None instead of aborting the trial."""
    try:
        return fn()
    except (
        TypeError,
        ValueError,
        ZeroDivisionError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ):
        return None


def score_run(
    topic_model,
    labels,
    *,
    texts=None,
    embeddings=None,
    probabilities=None,
    ground_truth=None,
    include_coherence: bool = True,
    include_dbcv: bool = True,
    topk: int = 10,
    flat_threshold: float = FLAT_PEAK,
) -> dict[str, float]:
    """Scalar metrics for one fitted run.

    ``word_intrusion`` stays manual. Non-finite values are dropped so MLflow
    will accept the dict. Coherence and DBCV run only when their inputs are
    present and the fit has enough topics to score. Anisotropy is merged
    whenever ``embeddings`` is provided.
    """
    labels = np.asarray(labels)
    metrics: dict[str, float] = {
        "n_topics": float(_n_topics(labels)),
        "noise_share": _noise_share(labels),
        "largest_topic_share": _largest_topic_share(labels),
        "topic_size_entropy": _topic_size_entropy(labels),
    }
    diversity = _optional_metric(lambda: topic_diversity(topic_model, topk))
    if diversity is not None:
        metrics["topic_diversity"] = diversity
    peaks = _optional_metric(lambda: list(peakedness(topic_model, topk).values()))
    if peaks:
        peak_values = np.asarray(peaks, dtype=float)
        metrics["min_peakedness"] = float(peak_values.min())
        metrics["peakedness_mean"] = float(peak_values.mean())
        metrics["flat_topic_share"] = float(np.mean(peak_values < flat_threshold))
    metrics.update(_assignment_metrics(probabilities))
    if include_coherence and texts is not None:
        usable = [
            words for words in top_words(topic_model, topk).values() if len(words) >= 2
        ]
        if usable:
            coherence_score = _optional_metric(
                lambda: coherence(topic_model, texts, topk=topk)
            )
            if coherence_score is not None:
                metrics["coherence_c_npmi"] = coherence_score
    if include_dbcv and embeddings is not None and _n_topics(labels) >= 2:
        dbcv_score = _optional_metric(lambda: dbcv(embeddings, labels))
        if dbcv_score is not None:
            metrics["dbcv"] = dbcv_score
    if embeddings is not None:
        anisotropy_scores = _optional_metric(lambda: anisotropy(embeddings))
        if anisotropy_scores:
            metrics.update(anisotropy_scores)
    if ground_truth is not None:
        scored = label_scores(ground_truth, labels)
        metrics["ami"] = scored["ami"]
        metrics["ari"] = scored["ari"]
    return _finite_metrics(metrics)


def word_intrusion(topic_model, topic_id: int, seed: int = 0, topk: int = 5):
    """Print 5 topic words plus 1 intruder. You guess. Returns the answer."""
    rng = random.Random(seed)
    lists = top_words(topic_model, topk=25)   # BERTopic stores 10 per topic by default
    words = lists[topic_id][:topk]
    pool = [w for t, ws in lists.items() if t != topic_id
            for w in ws[topk:] if w not in words]
    if not pool:                              # tiny models: fall back to any other word
        pool = [w for t, ws in lists.items() if t != topic_id
                for w in ws if w not in words]
    intruder = rng.choice(pool)
    shown = words + [intruder]
    rng.shuffle(shown)
    print(f"topic {topic_id}: " + "  ".join(f"{i}. {w}" for i, w in enumerate(shown, 1)))
    return intruder
