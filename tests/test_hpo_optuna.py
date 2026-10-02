# tests/test_hpo_optuna.py
import optuna
import pytest

from email_cls_with_clustering import hpo
from email_cls_with_clustering.hpo import (
    hdbscan_objective,
    load_spec,
    load_trial,
    optuna_hdbscan_config,
    suggest_hdbscan,
    trial_path,
)
from email_cls_with_clustering.topics import TopicHyperparams

GATES = {"noise_share_max": 0.5, "n_topics_min": 30}
WEIGHTS = {"coherence_c_npmi": 1.0, "topic_diversity": 0.5, "noise_share": -0.5}


def _metrics(**overrides):
    base = {"noise_share": 0.3, "n_topics": 50, "coherence_c_npmi": 0.1, "topic_diversity": 0.8}
    return {**base, **overrides}


def test_objective_ranks_passing_above_failing_and_grades_failures():
    passing_worst = hdbscan_objective(
        _metrics(coherence_c_npmi=-1.0, topic_diversity=0.0, noise_share=0.5), GATES, WEIGHTS
    )
    near_miss = hdbscan_objective(_metrics(noise_share=0.52), GATES, WEIGHTS)
    far_miss = hdbscan_objective(_metrics(noise_share=0.9, n_topics=5), GATES, WEIGHTS)
    crashed = hdbscan_objective({}, GATES, WEIGHTS)
    assert passing_worst > near_miss > far_miss > crashed


def test_objective_treats_missing_weighted_metric_as_failure():
    metrics = _metrics()
    del metrics["coherence_c_npmi"]
    assert hdbscan_objective(metrics, GATES, WEIGHTS) < -sum(abs(w) for w in WEIGHTS.values())


def test_spec_optuna_space_is_numeric_and_objective_has_no_dbcv():
    config = optuna_hdbscan_config(load_spec())
    assert config["space"]["min_cluster_size"]["type"] == "int"
    assert config["space"]["cluster_selection_epsilon"]["type"] == "float"
    assert "dbcv" not in config["objective"]
    assert config["n_trials"] == 40


def test_config_falls_back_to_grid_categories_without_space():
    spec = load_spec()
    stage = hpo.stage_by_name(spec, "hdbscan")
    stage["fallback_search"].pop("space")
    space = optuna_hdbscan_config(spec)["space"]
    assert space["min_samples"] == {"type": "categorical", "choices": [1, 5, 15, 50]}


def test_suggest_caps_min_samples_at_min_cluster_size():
    space = {
        "min_cluster_size": {"type": "int", "low": 15, "high": 20},
        "min_samples": {"type": "int", "low": 50, "high": 60},
        "cluster_selection_method": {"type": "categorical", "choices": ["eom"]},
        "cluster_selection_epsilon": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.01},
    }
    study = optuna.create_study()
    item = suggest_hdbscan(study.ask(), space)
    assert item["min_samples"] == item["min_cluster_size"]


def test_optuna_study_resumes_and_returns_saved_records(tmp_path, monkeypatch):
    fits = []

    def fake_trial(item, *, root, **_):
        trial_id = f"mcs{item['min_cluster_size']}_ms{item['min_samples']}"
        fits.append(trial_id)
        record = {"trial_id": trial_id, "metrics": _metrics(), "params": item}
        hpo.write_trial(trial_path(3, trial_id, root), record)
        return record

    monkeypatch.setattr(hpo, "_run_stage3_trial", fake_trial)
    spec = load_spec()
    hpo.stage_by_name(spec, "hdbscan")["fallback_search"]["n_trials_per_projection"] = 4
    kwargs = dict(
        spec=spec,
        gates=GATES,
        root=tmp_path,
        stop_words=["x"],
        base_params=TopicHyperparams(),
        fingerprint="fp",
    )
    first = hpo._run_hdbscan_optuna_on_projection(**kwargs)
    assert len(fits) == 4
    second = hpo._run_hdbscan_optuna_on_projection(**kwargs)
    assert len(fits) == 4, "resume must not refit finished trials"
    assert {r["trial_id"] for r in second} == {r["trial_id"] for r in first}
    assert all(load_trial(trial_path(3, r["trial_id"], tmp_path)) for r in second)
