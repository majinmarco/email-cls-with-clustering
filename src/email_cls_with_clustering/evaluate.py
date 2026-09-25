"""Scoring for a fitted BERTopic run. Nothing here refits anything."""

from __future__ import annotations

import random
import numpy as np


def top_words(topic_model, topk: int = 10) -> dict[int, list[str]]:
    """Top-k words per topic, outlier topic -1 excluded."""
    return {
        t: [w for w, _ in words[:topk]]
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
    """Max c-TF-IDF weight per topic. Below ~0.05 means the cluster is filler."""
    return {
        t: max(s for _, s in words[:topk])
        for t, words in topic_model.get_topics().items()
        if t != -1
    }


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
    model = CoherenceModel(
        topics=topics, texts=tokens, dictionary=dictionary,
        coherence=measure, topn=topk, processes=1,
    )
    return float(model.get_coherence())


def dbcv(embeddings, labels, sample: int | None = 20_000, seed: int = 42) -> float:
    """DBCV on the ORIGINAL embedding space. Noise points are dropped."""
    from hdbscan.validity import validity_index

    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    keep = labels >= 0
    X, y = embeddings[keep], labels[keep]
    if sample is not None and len(X) > sample:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(X), size=sample, replace=False)
        X, y = X[pick], y[pick]
    return float(validity_index(X, y))


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


def _n_topics(labels: np.ndarray) -> int:
    """Count of topic ids that are zero or positive."""
    return int(np.unique(labels[labels >= 0]).size)


def _outlier_share(labels: np.ndarray) -> float:
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


def score_run(
    topic_model,
    labels,
    *,
    texts=None,
    embeddings=None,
    probabilities=None,
    include_coherence: bool = True,
    include_dbcv: bool = True,
    topk: int = 10,
    flat_threshold: float = FLAT_PEAK,
) -> dict[str, float]:
    """Scalar metrics for one fitted run.

    ``label_scores`` and ``word_intrusion`` stay manual. Non-finite values
    are dropped so MLflow will accept the dict. Coherence and DBCV run only
    when their inputs are present and the fit has enough topics to score.
    """
    labels = np.asarray(labels)
    metrics: dict[str, float] = {
        "n_topics": float(_n_topics(labels)),
        "outlier_share": _outlier_share(labels),
        "largest_topic_share": _largest_topic_share(labels),
        "topic_size_entropy": _topic_size_entropy(labels),
        "topic_diversity": topic_diversity(topic_model, topk),
    }
    peaks = list(peakedness(topic_model, topk).values())
    if peaks:
        peak_values = np.asarray(peaks, dtype=float)
        metrics["peakedness_min"] = float(peak_values.min())
        metrics["peakedness_mean"] = float(peak_values.mean())
        metrics["flat_topic_share"] = float(np.mean(peak_values < flat_threshold))
    metrics.update(_assignment_metrics(probabilities))
    if include_coherence and texts is not None:
        usable = [words for words in top_words(topic_model, topk).values() if len(words) >= 2]
        if usable:
            metrics["coherence_c_npmi"] = coherence(topic_model, texts, topk=topk)
    if include_dbcv and embeddings is not None and _n_topics(labels) >= 2:
        metrics["dbcv"] = dbcv(embeddings, labels)
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