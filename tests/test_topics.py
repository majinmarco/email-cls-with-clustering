# tests/test_topics.py
import json
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from email_cls_with_clustering.embeddings import ST_MODEL
from email_cls_with_clustering.evaluate import DEFAULT_GATES, passes_hard_gates
from email_cls_with_clustering.hpo import (
    baseline_from_spec,
    expand_hdbscan,
    expand_outlier,
    expand_representation,
    expand_umap,
    load_spec,
    load_trial,
    record_failed_trial,
)
from email_cls_with_clustering.topics import (
    TopicHyperparams,
    embeddings_for_metrics,
    hdbscan_kwargs,
    params_for_mlflow,
    params_from_mapping,
    projection_cache_path,
    umap_kwargs,
    write_topic_dataset,
)


def test_failed_trial_is_recorded_and_fails_gates(tmp_path):
    path = tmp_path / "boom.json"
    record_failed_trial(
        path,
        trial_id="boom",
        stage=3,
        representation_id="rep",
        params={"min_cluster_size": 25},
        exc=ValueError("Invalid shape in axis 0: 0."),
    )
    loaded = load_trial(path)
    assert loaded is not None
    assert loaded["trial_id"] == "boom"
    assert loaded["metrics"] == {}
    assert loaded["passed_gates"] is False
    assert loaded["error"].startswith("ValueError:")
    assert passes_hard_gates(loaded["metrics"], DEFAULT_GATES) is False


def test_probabilities_go_to_npy_not_csv(tmp_path):
    frame = pd.DataFrame({"full_message": ["a b", "c d"], "topic": [0, -1]})
    probs = np.array([[0.9, 0.1], [0.4, 0.6]], dtype=np.float32)
    out = write_topic_dataset(
        frame, tmp_path / "clustered.csv", TopicHyperparams(),
        source=tmp_path / "in.csv", cache_path=tmp_path / "emb.npy",
        probabilities=probs,
    )
    assert "prob_vector" not in pd.read_csv(out).columns
    reloaded = np.load(out.with_name(f"{out.stem}_probs.npy"))
    assert reloaded.shape == (2, 2)
    assert reloaded.dtype.kind == "f"   # numbers, not strings
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["hyperparameters"]["ngram_range"] == [1, 1]


def test_umap_and_hdbscan_kwargs_match_baseline_defaults():
    params = TopicHyperparams()
    umap = umap_kwargs(params)
    hdb = hdbscan_kwargs(params)
    assert umap["n_components"] == 10
    assert umap["n_neighbors"] == 15
    assert umap["random_state"] is None
    assert umap["n_jobs"] == -1
    assert hdb["min_cluster_size"] == 100
    assert hdb["min_samples"] == 10
    assert hdb["gen_min_span_tree"] is True


def test_params_from_mapping_rejects_nested_hpo_spec():
    spec = load_spec()
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        params_from_mapping(spec)


def test_embeddings_for_metrics_returns_high_dim():
    high = np.ones((4, 8), dtype=np.float32)
    proj = np.zeros((4, 2), dtype=np.float32)
    assert embeddings_for_metrics(high, proj) is high


def test_projection_cache_path_ignores_hdbscan_includes_umap(tmp_path):
    base = TopicHyperparams()
    other_mcs = TopicHyperparams(min_cluster_size=400)
    other_nc = TopicHyperparams(n_components=25)
    fp = "abc"
    assert projection_cache_path(tmp_path, base, fp) == projection_cache_path(
        tmp_path, other_mcs, fp
    )
    assert projection_cache_path(tmp_path, base, fp) != projection_cache_path(
        tmp_path, other_nc, fp
    )


def test_defaults_match_baseline_from_spec():
    spec = load_spec()
    baseline = baseline_from_spec(spec)
    defaults = TopicHyperparams()
    for item in fields(TopicHyperparams):
        if item.name == "embed_model":
            assert getattr(defaults, item.name) is None
            assert baseline.embed_model == ST_MODEL
            continue
        assert getattr(defaults, item.name) == getattr(baseline, item.name), item.name
    assert defaults.resolved_embed_model() == baseline.resolved_embed_model() == ST_MODEL


def test_default_gates_match_spec():
    spec = load_spec()
    assert DEFAULT_GATES == spec["evaluation"]["hard_gates"]
    assert spec["evaluation"]["metrics"]["dbcv"]["sample"] == 20000
    assert spec["evaluation"]["metrics"]["dbcv"]["seed"] == 42
    assert spec["evaluation"]["metrics"]["anisotropy"]["sample"] == 5000


def test_grid_sizes_match_spec():
    spec = load_spec()
    assert len(expand_representation(spec)) == 15
    assert len(expand_umap(spec)) == 12
    assert len(expand_hdbscan(spec)) == 40
    assert len(expand_outlier(spec)) == 8


def test_params_for_mlflow_stringifies_ngram_range():
    logged = params_for_mlflow(TopicHyperparams())
    assert logged["ngram_range"] == "[1, 1]"
    assert logged["embed_model"] == "null"
    assert logged["min_samples"] == 10
    assert logged["n_jobs"] == -1
    assert logged["post_processing"] == "none"
    assert logged["calculate_probabilities"] is True


def test_params_for_mlflow_includes_outlier_reduction_settings():
    logged = params_for_mlflow(
        TopicHyperparams(),
        {"outlier_strategy": "c-tf-idf", "outlier_threshold": 0.1},
    )
    assert logged["outlier_strategy"] == "c-tf-idf"
    assert logged["outlier_threshold"] == 0.1
