# tests/test_evaluate.py
import numpy as np
import pandas as pd
import pytest

from email_cls_with_clustering.commands.topic_model import params_for_mlflow, topic_info_frame
from email_cls_with_clustering.evaluate import (
    DEFAULT_GATES,
    anisotropy,
    coherence,
    dbcv,
    peakedness,
    rank_key,
    score_run,
    select_advancing,
)
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
    assert metrics["noise_share"] == pytest.approx(1 / 6)
    assert metrics["largest_topic_share"] == pytest.approx(0.5)
    shares = np.array([3, 1, 1], dtype=float) / 5
    expected_entropy = float(-np.sum(shares * np.log(shares)) / np.log(3))
    assert metrics["topic_size_entropy"] == pytest.approx(expected_entropy)
    assert metrics["topic_diversity"] == pytest.approx(5 / 6)
    assert metrics["min_peakedness"] == pytest.approx(0.01)
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
    assert metrics["noise_share"] == 0.0


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
    assert logged["embed_model"] == "null"
    assert logged["random_state"] == "null"
    assert logged["top_k"] == "null"
    assert logged["batch_size"] == "null"
    assert logged["embed_model_resolved"] == params.resolved_embed_model()
    assert logged["calculate_probabilities"] is True
    assert logged["normalize_embeddings"] is True
    assert logged["stop_words"] == "EN_STOP | email | pii | calendar"


def test_score_run_ground_truth_logs_ami_ari():
    model = _Stub({
        0: [("alpha", 0.4), ("beta", 0.2)],
        1: [("gamma", 0.3), ("delta", 0.1)],
    })
    labels = np.array([0, 0, 1, 1])
    metrics = score_run(
        model,
        labels,
        ground_truth=labels,
        include_coherence=False,
        include_dbcv=False,
    )
    assert metrics["ami"] == pytest.approx(1.0)
    assert metrics["ari"] == pytest.approx(1.0)
    assert "v_measure" not in metrics


def test_rank_lower_noise_share_wins_when_others_tie():
    order = ["coherence", "topic_diversity", "noise_share", "largest_topic_share"]
    better = {
        "coherence_c_npmi": 0.5,
        "topic_diversity": 0.8,
        "noise_share": 0.1,
        "largest_topic_share": 0.2,
    }
    worse = {
        "coherence_c_npmi": 0.5,
        "topic_diversity": 0.8,
        "noise_share": 0.3,
        "largest_topic_share": 0.2,
    }
    assert rank_key(better, order) < rank_key(worse, order)


def test_select_advancing_drops_gate_failures_and_reports_quota():
    gates = {
        "largest_topic_share_max": 0.10,
        "noise_share_max": 0.50,
        "min_peakedness_min": 0.05,
        "n_topics_min": 30,
    }
    order = ["coherence", "topic_diversity", "noise_share"]
    records = [
        {
            "trial_id": "pass-a",
            "metrics": {
                "largest_topic_share": 0.05,
                "noise_share": 0.2,
                "min_peakedness": 0.1,
                "n_topics": 40,
                "coherence_c_npmi": 0.4,
                "topic_diversity": 0.7,
            },
        },
        {
            "trial_id": "fail-gates",
            "metrics": {
                "largest_topic_share": 0.5,
                "noise_share": 0.2,
                "min_peakedness": 0.1,
                "n_topics": 40,
                "coherence_c_npmi": 0.9,
                "topic_diversity": 0.9,
            },
        },
        {
            "trial_id": "pass-b",
            "metrics": {
                "largest_topic_share": 0.05,
                "noise_share": 0.1,
                "min_peakedness": 0.1,
                "n_topics": 40,
                "coherence_c_npmi": 0.3,
                "topic_diversity": 0.7,
            },
        },
    ]
    advanced, quota_met = select_advancing(records, order, n=3, gates=gates)
    assert quota_met is False
    assert [r["trial_id"] for r in advanced] == ["pass-a", "pass-b"]
    assert all(r["trial_id"] != "fail-gates" for r in advanced)


def test_anisotropy_keys():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(100, 8)).astype(np.float32)
    metrics = anisotropy(matrix, sample=50, seed=42)
    assert set(metrics) == {
        "mean_pairwise_cosine",
        "norm_of_mean_vector",
        "top_singular_energy_share",
    }
    assert all(np.isfinite(v) for v in metrics.values())


def test_dbcv_returns_nan_for_singleton_cluster_after_noise_drop():
    """A size-1 cluster after dropping noise must not abort scoring."""
    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [5.0, 5.0],
        [10.0, 0.0],  # noise
    ], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, -1])  # cluster 1 has one point after noise drop
    score = dbcv(embeddings, labels, sample=None)
    assert np.isnan(score)


def test_dbcv_returns_nan_when_fewer_than_two_non_noise_points():
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    labels = np.array([0, -1])
    assert np.isnan(dbcv(embeddings, labels, sample=None))


def test_rank_key_sorts_missing_dbcv_last():
    order = ["dbcv", "coherence"]
    with_dbcv = {"dbcv": 0.1, "coherence_c_npmi": 0.2}
    without_dbcv = {"coherence_c_npmi": 0.9}  # nan dropped by score_run
    assert rank_key(with_dbcv, order) < rank_key(without_dbcv, order)


def test_peakedness_skips_topics_with_no_words():
    model = _Stub({
        0: [],
        1: [("alpha", 0.4), ("beta", 0.2)],
    })
    assert peakedness(model) == {1: 0.4}
    metrics = score_run(
        model,
        np.array([0, 1, 1, -1]),
        include_coherence=False,
        include_dbcv=False,
    )
    assert metrics["noise_share"] == pytest.approx(0.25)
    assert metrics["min_peakedness"] == pytest.approx(0.4)


def test_coherence_returns_nan_when_no_topic_has_two_in_vocab_words():
    model = _Stub({0: [("zzzznotinvocab", 0.5), ("yyyynotinvocab", 0.4)]})
    score = coherence(model, ["hello world", "foo bar"])
    assert np.isnan(score)


def test_score_run_keeps_size_metrics_when_coherence_raises(monkeypatch):
    def boom(*_args, **_kwargs):
        raise ValueError("gensim")

    monkeypatch.setattr(
        "email_cls_with_clustering.evaluate.coherence", boom
    )
    model = _Stub({
        0: [("alpha", 0.5), ("beta", 0.4)],
        1: [("gamma", 0.3), ("delta", 0.2)],
    })
    metrics = score_run(
        model,
        np.array([0, 0, 1, -1]),
        texts=["alpha beta", "gamma delta"],
        include_dbcv=False,
    )
    assert metrics["n_topics"] == 2
    assert metrics["noise_share"] == pytest.approx(0.25)
    assert metrics["largest_topic_share"] == pytest.approx(0.5)
    assert "coherence_c_npmi" not in metrics


def test_default_gates_match_expected():
    assert DEFAULT_GATES == {
        "largest_topic_share_max": 0.10,
        "noise_share_max": 0.50,
        "flat_topic_share_max": 0.50,
        "n_topics_min": 30,
        "n_topics_max": 200,
    }


def test_stable_core_distance_matches_hdbscan_in_low_dim():
    from hdbscan.validity import all_points_core_distance

    from email_cls_with_clustering.evaluate import _stable_core_distance
    from sklearn.metrics import pairwise_distances

    rng = np.random.default_rng(0)
    dist = pairwise_distances(rng.normal(size=(50, 3)))
    np.testing.assert_allclose(
        _stable_core_distance(dist, d=3), all_points_core_distance(dist.copy(), d=3)
    )


def test_dbcv_is_finite_and_scale_invariant_in_high_dim():
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(2, 384))
    X = np.vstack([c + 0.3 * rng.normal(size=(100, 384)) for c in centers])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.repeat([0, 1], 100)
    score = dbcv(X, y, sample=None)
    assert np.isfinite(score)
    assert dbcv(X * 1000, y, sample=None) == pytest.approx(score)


def test_gate_violation_is_zero_when_passing_and_grows_with_the_miss():
    from email_cls_with_clustering.evaluate import gate_violation

    gates = {"noise_share_max": 0.5, "n_topics_min": 30}
    assert gate_violation({"noise_share": 0.4, "n_topics": 40}, gates) == 0.0
    near = gate_violation({"noise_share": 0.55, "n_topics": 40}, gates)
    far = gate_violation({"noise_share": 0.9, "n_topics": 10}, gates)
    assert 0 < near < far
    assert gate_violation({}, gates) == 2.0
