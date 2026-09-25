import numpy as np
import pandas as pd
import pytest

from email_cls_with_clustering.commands.topic_model import params_for_mlflow, topic_info_frame
from email_cls_with_clustering.evaluate import score_run
from email_cls_with_clustering.topics import TopicHyperparams


class _Stub:
    def __init__(self, topics):
        self._topics = topics

    def get_topics(self):
        return self._topics


def test_score_run_aggregates_size_peakedness_and_probabilities():
    model = _Stub({
        -1: [("noise", 0.01)],
        0: [("alpha", 0.20), ("beta", 0.10)],
        1: [("alpha", 0.90), ("gamma", 0.04)],
        2: [("delta", 0.01), ("epsilon", 0.005)],
    })
    labels = np.array([0, 0, 0, 1, 2, -1])
    probs = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.6, 0.3, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.2, 0.2, 0.6],
    ])
    metrics = score_run(
        model,
        labels,
        probabilities=probs,
        include_coherence=False,
        include_dbcv=False,
    )

    assert metrics["n_topics"] == 3
    assert metrics["outlier_share"] == pytest.approx(1 / 6)
    assert metrics["largest_topic_share"] == pytest.approx(0.5)
    shares = np.array([3, 1, 1], dtype=float) / 5
    expected_entropy = float(-np.sum(shares * np.log(shares)) / np.log(3))
    assert metrics["topic_size_entropy"] == pytest.approx(expected_entropy)
    assert metrics["topic_diversity"] == pytest.approx(5 / 6)
    assert metrics["peakedness_min"] == pytest.approx(0.01)
    assert metrics["peakedness_mean"] == pytest.approx((0.20 + 0.90 + 0.01) / 3)
    assert metrics["flat_topic_share"] == pytest.approx(1 / 3)
    assert metrics["mean_max_probability"] == pytest.approx(probs.max(axis=1).mean())
    safe = probs / probs.sum(axis=1, keepdims=True)
    log_p = np.log(safe, where=safe > 0, out=np.zeros_like(safe))
    expected_entropy = float((-(safe * log_p).sum(axis=1)).mean())
    assert metrics["mean_assignment_entropy"] == pytest.approx(expected_entropy)
    assert "coherence_c_npmi" not in metrics
    assert "dbcv" not in metrics


def test_n_topics_does_not_assume_an_outlier():
    model = _Stub({
        0: [("alpha", 0.4), ("beta", 0.2)],
        1: [("gamma", 0.3), ("delta", 0.1)],
    })
    metrics = score_run(
        model,
        np.array([0, 1, 0, 1]),
        include_coherence=False,
        include_dbcv=False,
    )
    assert metrics["n_topics"] == 2
    assert metrics["outlier_share"] == 0.0


def test_single_topic_omits_size_entropy():
    model = _Stub({0: [("alpha", 0.4), ("beta", 0.2)]})
    metrics = score_run(
        model,
        np.array([0, 0, 0]),
        probabilities=np.array([0.9, 0.8, 0.7]),
        include_coherence=False,
        include_dbcv=False,
    )
    assert metrics["n_topics"] == 1
    assert metrics["largest_topic_share"] == 1.0
    assert "topic_size_entropy" not in metrics
    assert "mean_max_probability" not in metrics
    assert "mean_assignment_entropy" not in metrics


def test_topic_info_frame_serializes_word_lists():
    class _Info:
        def get_topic_info(self):
            return pd.DataFrame({
                "Topic": [0],
                "Name": ["alpha"],
                "Representation": [["alpha", "beta"]],
            })

    frame = topic_info_frame(_Info())
    assert frame["Name"].tolist() == ["alpha"]
    assert frame["Representation"].tolist() == ['["alpha", "beta"]']


def test_params_for_mlflow_records_resolved_embed_model():
    params = TopicHyperparams(embed_model=None)
    logged = params_for_mlflow(params)
    assert "embed_model" not in logged
    assert logged["embed_model_resolved"] == params.resolved_embed_model()
    assert logged["calculate_probabilities"] is True
