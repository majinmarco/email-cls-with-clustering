"""Staged hyperparameter search for the Enron topic-model pipeline."""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from email_cls_with_clustering.embeddings import (
    apply_prompt,
    embedding_cache_path,
    load_or_embed,
    texts_fingerprint,
)
from email_cls_with_clustering.evaluate import (
    DEFAULT_GATES,
    passes_hard_gates,
    score_run,
    select_advancing,
)
from email_cls_with_clustering.paths import PROJECT_ROOT, notebooks_dir
from email_cls_with_clustering.representation import prepare_representation
from email_cls_with_clustering.topics import (
    TopicHyperparams,
    apply_topic_labels,
    embeddings_for_metrics,
    fit_hdbscan_on_projection,
    hdbscan_kwargs,
    load_or_project,
    params_for_json,
    params_for_mlflow,
    params_from_mapping,
    projection_cache_path,
    reduced_labels,
    umap_kwargs,
    write_topic_dataset,
)
from email_cls_with_clustering.tracking import setup_tracking

DEFAULT_SPEC = PROJECT_ROOT / "configs" / "enron-email-topic-model-hpo.json"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def load_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the nested HPO JSON spec."""
    return json.loads((path or DEFAULT_SPEC).read_text())


def stage_by_name(spec: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the stage object whose ``name`` matches."""
    for stage in spec["stages"]:
        if stage["name"] == name:
            return stage
    raise KeyError(f"No stage named {name!r}")


def stage_by_id(spec: dict[str, Any], stage_id: int) -> dict[str, Any]:
    for stage in spec["stages"]:
        if stage["id"] == stage_id:
            return stage
    raise KeyError(f"No stage id={stage_id}")


def baseline_from_spec(spec: dict[str, Any]) -> TopicHyperparams:
    """TopicHyperparams from prerequisites + stage-1 fixed downstream + baseline embed."""
    stage1 = stage_by_name(spec, "representation")
    stage3 = stage_by_name(spec, "hdbscan")
    prereq = spec["prerequisites"]
    fixed = stage1["fixed_downstream"]
    baseline_model = next(
        item for item in stage1["space"]["embed_model"] if item.get("baseline")
    )
    epsilon = stage3["space"]["cluster_selection_epsilon"][0]
    return TopicHyperparams(
        n_neighbors=fixed["umap"]["n_neighbors"],
        n_components=fixed["umap"]["n_components"],
        min_dist=fixed["umap"]["min_dist"],
        umap_metric=fixed["umap"]["metric"],
        random_state=prereq["umap"]["random_state"],
        n_jobs=prereq["umap"]["n_jobs"],
        min_cluster_size=fixed["hdbscan"]["min_cluster_size"],
        min_samples=fixed["hdbscan"]["min_samples"],
        hdbscan_metric=prereq["hdbscan"]["metric"],
        cluster_selection_method=fixed["hdbscan"]["cluster_selection_method"],
        cluster_selection_epsilon=float(epsilon),
        prediction_data=prereq["hdbscan"]["prediction_data"],
        gen_min_span_tree=prereq["hdbscan"]["gen_min_span_tree"],
        embed_model=baseline_model["name"],
        embed_prompt=baseline_model.get("prompt", ""),
        max_seq_length=prereq["embedding"]["max_seq_length"],
        ngram_range=tuple(prereq["vectorizer"]["ngram_range"]),
        post_processing="none",
        top_k=None,
    )


def expand_representation(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """``embed_model`` × ``post_processing``. ``top_k`` is not a grid axis."""
    stage = stage_by_name(spec, "representation")
    items = []
    for model in stage["space"]["embed_model"]:
        for post in stage["space"]["post_processing"]:
            items.append({
                "embed_model": model["name"],
                "embed_prompt": model.get("prompt", ""),
                "post_processing": post,
            })
    expected = stage["n_trials"]
    if len(items) != expected:
        raise ValueError(
            f"representation grid size {len(items)} != spec n_trials {expected}"
        )
    return items


def expand_umap(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """UMAP grid per representation (metric renamed to umap_metric)."""
    stage = stage_by_name(spec, "umap")
    space = stage["space"]
    items = []
    for n_components, n_neighbors, min_dist, metric in itertools.product(
        space["n_components"],
        space["n_neighbors"],
        space["min_dist"],
        space["metric"],
    ):
        items.append({
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "umap_metric": metric,
        })
    expected = stage["n_trials_per_representation"]
    if len(items) != expected:
        raise ValueError(
            f"umap grid size {len(items)} != spec n_trials_per_representation {expected}"
        )
    return items


def expand_hdbscan(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Full HDBSCAN grid including cluster_selection_epsilon."""
    stage = stage_by_name(spec, "hdbscan")
    space = stage["space"]
    items = []
    for mcs, ms, method, eps in itertools.product(
        space["min_cluster_size"],
        space["min_samples"],
        space["cluster_selection_method"],
        space["cluster_selection_epsilon"],
    ):
        items.append({
            "min_cluster_size": mcs,
            "min_samples": ms,
            "cluster_selection_method": method,
            "cluster_selection_epsilon": eps,
        })
    expected = stage["n_trials_per_projection"]
    if len(items) != expected:
        raise ValueError(
            f"hdbscan grid size {len(items)} != spec n_trials_per_projection {expected}"
        )
    return items


def expand_outlier(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Outlier-reduction strategy × threshold grid."""
    stage = stage_by_name(spec, "outlier_reduction")
    space = stage["space"]
    items = []
    for strategy, threshold in itertools.product(space["strategy"], space["threshold"]):
        items.append({"strategy": strategy, "threshold": threshold})
    expected = stage["n_trials_per_finalist"]
    if len(items) != expected:
        raise ValueError(
            f"outlier grid size {len(items)} != spec n_trials_per_finalist {expected}"
        )
    return items


def safe_token(value: str) -> str:
    return _SAFE.sub("_", value).strip("_") or "x"


def representation_id(params: TopicHyperparams) -> str:
    """Filesystem-safe id: ``model__post`` or ``model__mean_removal_plus_top_k__k{k}``."""
    model = safe_token(params.resolved_embed_model().replace("/", "_"))
    if params.post_processing == "mean_removal_plus_top_k":
        return f"{model}__mean_removal_plus_top_k__k{params.top_k}"
    return f"{model}__{safe_token(params.post_processing)}"


def hpo_dir(root: Path | None = None) -> Path:
    return (root or notebooks_dir()) / "hpo"


def stage_dir(stage_id: int, root: Path | None = None) -> Path:
    path = hpo_dir(root) / f"stage{stage_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def trial_path(stage_id: int, trial_id: str, root: Path | None = None) -> Path:
    return stage_dir(stage_id, root) / f"{safe_token(trial_id)}.json"


def load_trial(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_trial(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")


def record_failed_trial(
    path: Path,
    *,
    trial_id: str,
    stage: int,
    representation_id: str,
    params: dict[str, Any],
    exc: BaseException,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a crashed trial so resume skips it and the stage continues.

    Empty metrics fail hard gates, so the trial cannot advance.
    """
    message = f"{type(exc).__name__}: {exc}"
    if len(message) > 500:
        message = message[:500]
    record: dict[str, Any] = {
        "trial_id": trial_id,
        "stage": stage,
        "representation_id": representation_id,
        "params": params,
        "metrics": {},
        "passed_gates": False,
        "error": message,
    }
    if extra:
        record.update(extra)
    print(f"trial {trial_id} failed: {message}")
    write_trial(path, record)
    return record


def _stage4_failed(
    path: Path,
    *,
    trial_id: str,
    base_params: TopicHyperparams,
    parent_trial_id: str,
    item: dict[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    return record_failed_trial(
        path,
        trial_id=trial_id,
        stage=4,
        representation_id=representation_id(base_params),
        params={**params_for_json(base_params), **item},
        exc=exc,
        extra={
            "parent_trial_id": parent_trial_id,
            "skipped": False,
            "accepted": False,
            "strategy": item["strategy"],
            "threshold": item["threshold"],
        },
    )


def write_advanced(
    stage_id: int,
    *,
    quota_met: bool,
    order: list[str],
    trials: list[dict[str, Any]],
    root: Path | None = None,
) -> Path:
    path = stage_dir(stage_id, root) / "advanced.json"
    payload = {"quota_met": quota_met, "order": order, "trials": trials}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def load_advanced(stage_id: int, root: Path | None = None) -> dict[str, Any]:
    path = stage_dir(stage_id, root) / "advanced.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run earlier stages first or pass --from-stage "
            f"starting at 1."
        )
    return json.loads(path.read_text())


def ranking_order(spec: dict[str, Any], key: str, *, has_ground_truth: bool) -> list[str]:
    ranking = spec["evaluation"]["ranking"]
    order = list(ranking[key])
    if has_ground_truth:
        order = list(ranking["if_ground_truth_available"]) + order
    return order


def _gates(spec: dict[str, Any]) -> dict[str, float]:
    return dict(spec["evaluation"].get("hard_gates", DEFAULT_GATES))


def _anisotropy_sample(spec: dict[str, Any]) -> int:
    return int(spec["evaluation"]["metrics"]["anisotropy"]["sample"])


def _top_k_candidates(spec: dict[str, Any]) -> list[int]:
    return list(stage_by_name(spec, "representation")["space"]["top_k_candidates"])


def _log_trial_mlflow(
    params: TopicHyperparams,
    metrics: dict[str, float],
    *,
    run_name: str,
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(params_for_mlflow(params, extra))
            mlflow.log_param("passed_gates", passes_hard_gates(metrics, DEFAULT_GATES))
            finite = {k: v for k, v in metrics.items() if np.isfinite(v)}
            if finite:
                mlflow.log_metrics(finite)
    except Exception as exc:
        print(f"MLflow log failed for {run_name}: {type(exc).__name__}: {exc}")


def _fit_bertopic_with_embeddings(
    docs: list[str],
    embeddings: np.ndarray,
    params: TopicHyperparams,
    *,
    cache_dir: Path,
    fingerprint: str,
):
    """Fit BERTopic with a live UMAP and save the projection npy."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
    from spacy.lang.es.stop_words import STOP_WORDS as ES_STOP
    from umap import UMAP

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(**umap_kwargs(params)),
        hdbscan_model=HDBSCAN(**hdbscan_kwargs(params)),
        vectorizer_model=CountVectorizer(
            stop_words=list(EN_STOP | ES_STOP),
            ngram_range=params.ngram_range,
        ),
        calculate_probabilities=params.calculate_probabilities,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
    proj_path = projection_cache_path(cache_dir, params, fingerprint)
    projection = np.asarray(topic_model.umap_model.embedding_, dtype=np.float32)
    proj_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(proj_path, projection)
    return topic_model, np.asarray(topics), probs, proj_path


def _score(
    topic_model,
    labels,
    *,
    docs: list[str],
    embeddings: np.ndarray | None,
    ground_truth,
    include_dbcv: bool,
) -> dict[str, float]:
    return score_run(
        topic_model,
        labels,
        texts=docs,
        embeddings=embeddings,
        ground_truth=ground_truth,
        include_coherence=True,
        include_dbcv=include_dbcv,
    )


def run_stage1(
    spec: dict[str, Any],
    frame: pd.DataFrame,
    *,
    cache_dir: Path,
    env_file: Path | None,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Representation screen: 15 trials, advance 2."""
    stage = stage_by_name(spec, "representation")
    baseline = baseline_from_spec(spec)
    docs = frame["ctfidf_text"].fillna("").astype(str).tolist()
    embed_texts = frame["embed_text"].fillna("").astype(str).tolist()
    ground_truth = (
        frame["ground_truth"].to_numpy() if "ground_truth" in frame.columns else None
    )
    has_gt = ground_truth is not None
    order = ranking_order(spec, "across_representations", has_ground_truth=has_gt)
    gates = _gates(spec)
    candidates = _top_k_candidates(spec)
    sample = _anisotropy_sample(spec)
    records: list[dict[str, Any]] = []

    for item in expand_representation(spec):
        params = replace(
            baseline,
            embed_model=item["embed_model"],
            embed_prompt=item["embed_prompt"],
            post_processing=item["post_processing"],
            top_k=None,
            calculate_probabilities=False,
        )
        # representation_id needs top_k for mean_removal_plus_top_k; filled after prepare.
        model_name = params.resolved_embed_model()
        emb_path = embedding_cache_path(
            cache_dir, params.embed_backend, model_name, prompt=params.embed_prompt
        )
        high_dim = load_or_embed(
            embed_texts,
            emb_path,
            backend=params.embed_backend,
            model=model_name,
            batch_size=params.batch_size,
            max_seq_length=params.max_seq_length,
            env_file=env_file,
            prompt=params.embed_prompt,
        )
        processed = prepare_representation(
            high_dim, params, candidates=candidates, sample=sample, seed=42
        )
        rep_id = representation_id(params)
        trial_id = f"rep__{rep_id}"
        path = trial_path(1, trial_id, root)
        existing = load_trial(path)
        if existing is not None:
            records.append(existing)
            continue
        fingerprint = texts_fingerprint(apply_prompt(embed_texts, params.embed_prompt))
        topic_model, topics, probs, proj_path = _fit_bertopic_with_embeddings(
            docs, processed, params, cache_dir=cache_dir, fingerprint=fingerprint
        )
        metrics = _score(
            topic_model,
            topics,
            docs=docs,
            embeddings=processed,
            ground_truth=ground_truth,
            include_dbcv=True,
        )
        record = {
            "trial_id": trial_id,
            "stage": 1,
            "representation_id": rep_id,
            "params": params_for_json(params),
            "metrics": metrics,
            "passed_gates": passes_hard_gates(metrics, gates),
            "embedding_cache": str(emb_path),
            "projection_cache": str(proj_path),
            "fingerprint": fingerprint,
        }
        write_trial(path, record)
        _log_trial_mlflow(params, metrics, run_name=trial_id)
        records.append(record)

    advanced, quota_met = select_advancing(records, order, stage["advance"], gates)
    if not advanced:
        raise RuntimeError("Stage 1: zero trials passed hard gates")
    if not quota_met:
        msg = (
            f"Stage 1: quota missed — advanced {len(advanced)} of "
            f"{stage['advance']} requested"
        )
        print(msg)
        mlflow.log_param("stage1_quota_met", False)
    else:
        mlflow.log_param("stage1_quota_met", True)
    write_advanced(1, quota_met=quota_met, order=order, trials=advanced, root=root)
    return advanced


def run_stage2(
    spec: dict[str, Any],
    frame: pd.DataFrame,
    advanced_reps: list[dict[str, Any]],
    *,
    cache_dir: Path,
    env_file: Path | None,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """UMAP screen on advanced representations: 12 settings each, advance 3."""
    stage = stage_by_name(spec, "umap")
    baseline = baseline_from_spec(spec)
    docs = frame["ctfidf_text"].fillna("").astype(str).tolist()
    embed_texts = frame["embed_text"].fillna("").astype(str).tolist()
    ground_truth = (
        frame["ground_truth"].to_numpy() if "ground_truth" in frame.columns else None
    )
    has_gt = ground_truth is not None
    order = ranking_order(spec, "across_representations", has_ground_truth=has_gt)
    gates = _gates(spec)
    candidates = _top_k_candidates(spec)
    sample = _anisotropy_sample(spec)
    umap_grid = expand_umap(spec)
    records: list[dict[str, Any]] = []

    for rep in advanced_reps:
        rep_params = replace(
            params_from_mapping(rep["params"]),
            calculate_probabilities=False,
        )
        model_name = rep_params.resolved_embed_model()
        emb_path = embedding_cache_path(
            cache_dir, rep_params.embed_backend, model_name, prompt=rep_params.embed_prompt
        )
        high_dim = load_or_embed(
            embed_texts,
            emb_path,
            backend=rep_params.embed_backend,
            model=model_name,
            max_seq_length=rep_params.max_seq_length,
            env_file=env_file,
            prompt=rep_params.embed_prompt,
        )
        # Restore selected top_k; do not re-select.
        processed = prepare_representation(
            high_dim, rep_params, candidates=candidates, sample=sample, seed=42
        )
        fingerprint = texts_fingerprint(
            apply_prompt(embed_texts, rep_params.embed_prompt)
        )
        for umap_item in umap_grid:
            params = replace(
                rep_params,
                n_components=umap_item["n_components"],
                n_neighbors=umap_item["n_neighbors"],
                min_dist=umap_item["min_dist"],
                umap_metric=umap_item["umap_metric"],
                # Keep stage-1 HDBSCAN for screening.
                min_cluster_size=baseline.min_cluster_size,
                min_samples=baseline.min_samples,
                cluster_selection_method=baseline.cluster_selection_method,
                cluster_selection_epsilon=baseline.cluster_selection_epsilon,
                calculate_probabilities=False,
            )
            trial_id = (
                f"{representation_id(params)}__umap_nc{params.n_components}"
                f"_nn{params.n_neighbors}_md{params.min_dist}"
            )
            path = trial_path(2, trial_id, root)
            existing = load_trial(path)
            if existing is not None:
                records.append(existing)
                continue
            proj_path = projection_cache_path(cache_dir, params, fingerprint)
            projection = load_or_project(processed, params, proj_path)
            topic_model, topics, probs = fit_hdbscan_on_projection(
                docs, projection, params, calculate_probabilities=False
            )
            metrics = _score(
                topic_model,
                topics,
                docs=docs,
                embeddings=embeddings_for_metrics(processed, projection),
                ground_truth=ground_truth,
                include_dbcv=True,
            )
            record = {
                "trial_id": trial_id,
                "stage": 2,
                "representation_id": representation_id(params),
                "params": params_for_json(params),
                "metrics": metrics,
                "passed_gates": passes_hard_gates(metrics, gates),
                "embedding_cache": str(emb_path),
                "projection_cache": str(proj_path),
                "fingerprint": fingerprint,
            }
            write_trial(path, record)
            _log_trial_mlflow(params, metrics, run_name=trial_id)
            records.append(record)

    advanced, quota_met = select_advancing(records, order, stage["advance"], gates)
    if not advanced:
        raise RuntimeError("Stage 2: zero trials passed hard gates")
    if not quota_met:
        print(
            f"Stage 2: quota missed — advanced {len(advanced)} of "
            f"{stage['advance']} requested"
        )
        mlflow.log_param("stage2_quota_met", False)
    else:
        mlflow.log_param("stage2_quota_met", True)
    write_advanced(2, quota_met=quota_met, order=order, trials=advanced, root=root)
    return advanced


def _hdbscan_objective(metrics: dict[str, float], gates: dict[str, float]) -> float:
    if not passes_hard_gates(metrics, gates):
        return 1e6
    required = ("dbcv", "coherence_c_npmi", "topic_diversity", "noise_share")
    if any(metrics.get(name) is None or not np.isfinite(metrics.get(name, float("nan")))
           for name in required):
        return 1e6
    return (
        -1000.0 * metrics["dbcv"]
        - 100.0 * metrics["coherence_c_npmi"]
        - 10.0 * metrics["topic_diversity"]
        + metrics["noise_share"]
    )


def _run_hdbscan_grid_on_projection(
    *,
    spec: dict[str, Any],
    docs: list[str],
    processed: np.ndarray,
    projection: np.ndarray,
    proj_path: Path,
    emb_path: Path,
    fingerprint: str,
    base_params: TopicHyperparams,
    ground_truth,
    gates: dict[str, float],
    root: Path | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in expand_hdbscan(spec):
        params = replace(
            base_params,
            min_cluster_size=item["min_cluster_size"],
            min_samples=item["min_samples"],
            cluster_selection_method=item["cluster_selection_method"],
            cluster_selection_epsilon=item["cluster_selection_epsilon"],
            calculate_probabilities=False,
        )
        trial_id = (
            f"{representation_id(params)}__umap_nc{params.n_components}"
            f"_nn{params.n_neighbors}"
            f"__mcs{params.min_cluster_size}_ms{params.min_samples}"
            f"_{params.cluster_selection_method}_eps{params.cluster_selection_epsilon}"
        )
        path = trial_path(3, trial_id, root)
        existing = load_trial(path)
        if existing is not None:
            records.append(existing)
            continue
        try:
            topic_model, topics, probs = fit_hdbscan_on_projection(
                docs, projection, params, calculate_probabilities=False
            )
            metrics = _score(
                topic_model,
                topics,
                docs=docs,
                embeddings=embeddings_for_metrics(processed, projection),
                ground_truth=ground_truth,
                include_dbcv=True,
            )
        except Exception as exc:
            records.append(
                record_failed_trial(
                    path,
                    trial_id=trial_id,
                    stage=3,
                    representation_id=representation_id(params),
                    params=params_for_json(params),
                    exc=exc,
                    extra={
                        "embedding_cache": str(emb_path),
                        "projection_cache": str(proj_path),
                        "fingerprint": fingerprint,
                    },
                )
            )
            continue
        record = {
            "trial_id": trial_id,
            "stage": 3,
            "representation_id": representation_id(params),
            "params": params_for_json(params),
            "metrics": metrics,
            "passed_gates": passes_hard_gates(metrics, gates),
            "embedding_cache": str(emb_path),
            "projection_cache": str(proj_path),
            "fingerprint": fingerprint,
        }
        write_trial(path, record)
        _log_trial_mlflow(params, metrics, run_name=trial_id)
        records.append(record)
    return records


def _run_hdbscan_optuna_on_projection(
    *,
    spec: dict[str, Any],
    docs: list[str],
    processed: np.ndarray,
    projection: np.ndarray,
    proj_path: Path,
    emb_path: Path,
    fingerprint: str,
    base_params: TopicHyperparams,
    ground_truth,
    gates: dict[str, float],
    root: Path | None,
) -> list[dict[str, Any]]:
    import optuna

    stage = stage_by_name(spec, "hdbscan")
    space = stage["space"]
    n_trials = stage["fallback_search"]["n_trials_per_projection"]
    records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        item = {
            "min_cluster_size": trial.suggest_categorical(
                "min_cluster_size", space["min_cluster_size"]
            ),
            "min_samples": trial.suggest_categorical("min_samples", space["min_samples"]),
            "cluster_selection_method": trial.suggest_categorical(
                "cluster_selection_method", space["cluster_selection_method"]
            ),
            "cluster_selection_epsilon": trial.suggest_categorical(
                "cluster_selection_epsilon", space["cluster_selection_epsilon"]
            ),
        }
        params = replace(
            base_params,
            min_cluster_size=item["min_cluster_size"],
            min_samples=item["min_samples"],
            cluster_selection_method=item["cluster_selection_method"],
            cluster_selection_epsilon=item["cluster_selection_epsilon"],
            calculate_probabilities=False,
        )
        trial_id = (
            f"{representation_id(params)}__umap_nc{params.n_components}"
            f"_nn{params.n_neighbors}__optuna{trial.number}"
        )
        path = trial_path(3, trial_id, root)
        existing = load_trial(path)
        if existing is not None:
            records.append(existing)
            return _hdbscan_objective(existing["metrics"], gates)
        try:
            topic_model, topics, probs = fit_hdbscan_on_projection(
                docs, projection, params, calculate_probabilities=False
            )
            metrics = _score(
                topic_model,
                topics,
                docs=docs,
                embeddings=embeddings_for_metrics(processed, projection),
                ground_truth=ground_truth,
                include_dbcv=True,
            )
        except Exception as exc:
            record_failed_trial(
                path,
                trial_id=trial_id,
                stage=3,
                representation_id=representation_id(params),
                params=params_for_json(params),
                exc=exc,
                extra={
                    "embedding_cache": str(emb_path),
                    "projection_cache": str(proj_path),
                    "fingerprint": fingerprint,
                    "optuna_number": trial.number,
                },
            )
            failed = load_trial(path)
            if failed is not None:
                records.append(failed)
            return 1e6
        record = {
            "trial_id": trial_id,
            "stage": 3,
            "representation_id": representation_id(params),
            "params": params_for_json(params),
            "metrics": metrics,
            "passed_gates": passes_hard_gates(metrics, gates),
            "embedding_cache": str(emb_path),
            "projection_cache": str(proj_path),
            "fingerprint": fingerprint,
            "optuna_number": trial.number,
        }
        write_trial(path, record)
        _log_trial_mlflow(params, metrics, run_name=trial_id)
        records.append(record)
        return _hdbscan_objective(metrics, gates)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials)
    return records


def run_stage3(
    spec: dict[str, Any],
    frame: pd.DataFrame,
    advanced_projections: list[dict[str, Any]],
    *,
    cache_dir: Path,
    env_file: Path | None,
    hdbscan_search: str = "grid",
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """HDBSCAN search on cached projections."""
    stage = stage_by_name(spec, "hdbscan")
    docs = frame["ctfidf_text"].fillna("").astype(str).tolist()
    embed_texts = frame["embed_text"].fillna("").astype(str).tolist()
    ground_truth = (
        frame["ground_truth"].to_numpy() if "ground_truth" in frame.columns else None
    )
    has_gt = ground_truth is not None
    gates = _gates(spec)
    candidates = _top_k_candidates(spec)
    sample = _anisotropy_sample(spec)
    records: list[dict[str, Any]] = []

    for proj in advanced_projections:
        base_params = replace(
            params_from_mapping(proj["params"]),
            calculate_probabilities=False,
        )
        model_name = base_params.resolved_embed_model()
        emb_path = Path(proj.get("embedding_cache") or embedding_cache_path(
            cache_dir, base_params.embed_backend, model_name, prompt=base_params.embed_prompt
        ))
        high_dim = load_or_embed(
            embed_texts,
            emb_path,
            backend=base_params.embed_backend,
            model=model_name,
            max_seq_length=base_params.max_seq_length,
            env_file=env_file,
            prompt=base_params.embed_prompt,
        )
        processed = prepare_representation(
            high_dim, base_params, candidates=candidates, sample=sample, seed=42
        )
        fingerprint = proj.get("fingerprint") or texts_fingerprint(
            apply_prompt(embed_texts, base_params.embed_prompt)
        )
        proj_path = Path(
            proj.get("projection_cache")
            or projection_cache_path(cache_dir, base_params, fingerprint)
        )
        projection = load_or_project(processed, base_params, proj_path)
        if hdbscan_search == "optuna":
            batch = _run_hdbscan_optuna_on_projection(
                spec=spec,
                docs=docs,
                processed=processed,
                projection=projection,
                proj_path=proj_path,
                emb_path=emb_path,
                fingerprint=fingerprint,
                base_params=base_params,
                ground_truth=ground_truth,
                gates=gates,
                root=root,
            )
        else:
            batch = _run_hdbscan_grid_on_projection(
                spec=spec,
                docs=docs,
                processed=processed,
                projection=projection,
                proj_path=proj_path,
                emb_path=emb_path,
                fingerprint=fingerprint,
                base_params=base_params,
                ground_truth=ground_truth,
                gates=gates,
                root=root,
            )
        records.extend(batch)

    rep_ids = {record["representation_id"] for record in records}
    rank_key_name = (
        "within_representation" if len(rep_ids) == 1 else "across_representations"
    )
    order = ranking_order(spec, rank_key_name, has_ground_truth=has_gt)
    advanced, quota_met = select_advancing(records, order, stage["advance"], gates)
    if not advanced:
        raise RuntimeError("Stage 3: zero trials passed hard gates")
    if not quota_met:
        print(
            f"Stage 3: quota missed — advanced {len(advanced)} of "
            f"{stage['advance']} requested"
        )
        mlflow.log_param("stage3_quota_met", False)
    else:
        mlflow.log_param("stage3_quota_met", True)
    write_advanced(3, quota_met=quota_met, order=order, trials=advanced, root=root)
    return advanced


def run_stage4(
    spec: dict[str, Any],
    frame: pd.DataFrame,
    finalists: list[dict[str, Any]],
    *,
    cache_dir: Path,
    env_file: Path | None,
    source: Path,
    output: Path | None = None,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Outlier reduction on stage-3 finalists. Writes full CSV/pkl per winner."""
    stage = stage_by_name(spec, "outlier_reduction")
    docs = frame["ctfidf_text"].fillna("").astype(str).tolist()
    embed_texts = frame["embed_text"].fillna("").astype(str).tolist()
    ground_truth = (
        frame["ground_truth"].to_numpy() if "ground_truth" in frame.columns else None
    )
    gates = _gates(spec)
    candidates = _top_k_candidates(spec)
    sample = _anisotropy_sample(spec)
    outlier_grid = expand_outlier(spec)
    out_dir = (root or notebooks_dir())
    default_output = out_dir / "emails_clustered.csv"
    records: list[dict[str, Any]] = []

    for finalist in finalists:
        base_params = replace(
            params_from_mapping(finalist["params"]),
            calculate_probabilities=False,
        )
        model_name = base_params.resolved_embed_model()
        emb_path = Path(finalist.get("embedding_cache") or embedding_cache_path(
            cache_dir, base_params.embed_backend, model_name, prompt=base_params.embed_prompt
        ))
        high_dim = load_or_embed(
            embed_texts,
            emb_path,
            backend=base_params.embed_backend,
            model=model_name,
            max_seq_length=base_params.max_seq_length,
            env_file=env_file,
            prompt=base_params.embed_prompt,
        )
        processed = prepare_representation(
            high_dim, base_params, candidates=candidates, sample=sample, seed=42
        )
        fingerprint = finalist.get("fingerprint") or texts_fingerprint(
            apply_prompt(embed_texts, base_params.embed_prompt)
        )
        proj_path = Path(
            finalist.get("projection_cache")
            or projection_cache_path(cache_dir, base_params, fingerprint)
        )
        projection = load_or_project(processed, base_params, proj_path)
        try:
            topic_model, topics, probs = fit_hdbscan_on_projection(
                docs, projection, base_params, calculate_probabilities=False
            )
            baseline_metrics = _score(
                topic_model,
                topics,
                docs=docs,
                embeddings=embeddings_for_metrics(processed, projection),
                ground_truth=ground_truth,
                include_dbcv=True,
            )
        except Exception as exc:
            refit_id = f"{finalist['trial_id']}__refit"
            records.append(
                record_failed_trial(
                    trial_path(4, refit_id, root),
                    trial_id=refit_id,
                    stage=4,
                    representation_id=representation_id(base_params),
                    params=params_for_json(base_params),
                    exc=exc,
                    extra={
                        "parent_trial_id": finalist["trial_id"],
                        "skipped": False,
                        "accepted": False,
                    },
                )
            )
            continue
        baseline_noise = baseline_metrics.get("noise_share", float("nan"))
        baseline_coherence = baseline_metrics.get("coherence_c_npmi")

        # Compute every reduction on the pristine model before update_topics.
        reductions: list[tuple[dict[str, Any], np.ndarray | None]] = []
        if stage.get("guard_zero_outliers") and baseline_noise == 0.0:
            for item in outlier_grid:
                trial_id = (
                    f"{finalist['trial_id']}__{safe_token(item['strategy'])}"
                    f"__thr{item['threshold']}"
                )
                record = {
                    "trial_id": trial_id,
                    "stage": 4,
                    "representation_id": representation_id(base_params),
                    "parent_trial_id": finalist["trial_id"],
                    "params": {**params_for_json(base_params), **item},
                    "metrics": baseline_metrics,
                    "passed_gates": passes_hard_gates(baseline_metrics, gates),
                    "skipped": True,
                    "accepted": False,
                    "strategy": item["strategy"],
                    "threshold": item["threshold"],
                }
                path = trial_path(4, trial_id, root)
                if load_trial(path) is None:
                    write_trial(path, record)
                    _log_trial_mlflow(
                        base_params,
                        baseline_metrics,
                        run_name=trial_id,
                        extra={
                            "outlier_strategy": item["strategy"],
                            "outlier_threshold": item["threshold"],
                            "outlier_skipped": True,
                        },
                    )
                records.append(load_trial(path) or record)
            winning_labels = topics
            winning_params = base_params
            winning_id = f"{finalist['trial_id']}__unreduced"
        else:
            for item in outlier_grid:
                trial_id = (
                    f"{finalist['trial_id']}__{safe_token(item['strategy'])}"
                    f"__thr{item['threshold']}"
                )
                path = trial_path(4, trial_id, root)
                if load_trial(path) is not None:
                    reductions.append((item, None))
                    continue
                emb_arg = projection if item["strategy"] == "embeddings" else None
                try:
                    labels = reduced_labels(
                        topic_model,
                        docs,
                        topics,
                        strategy=item["strategy"],
                        threshold=item["threshold"],
                        embeddings=emb_arg,
                    )
                except Exception as exc:
                    records.append(
                        _stage4_failed(
                            path,
                            trial_id=trial_id,
                            base_params=base_params,
                            parent_trial_id=finalist["trial_id"],
                            item=item,
                            exc=exc,
                        )
                    )
                    continue
                reductions.append((item, labels))

            accepted: list[dict[str, Any]] = []
            for item, labels in reductions:
                trial_id = (
                    f"{finalist['trial_id']}__{safe_token(item['strategy'])}"
                    f"__thr{item['threshold']}"
                )
                path = trial_path(4, trial_id, root)
                existing = load_trial(path)
                if existing is not None:
                    records.append(existing)
                    if existing.get("accepted"):
                        accepted.append(existing)
                    continue
                try:
                    # Score on a copy of topic words after applying labels.
                    apply_topic_labels(topic_model, docs, labels)
                    metrics = _score(
                        topic_model,
                        labels,
                        docs=docs,
                        embeddings=embeddings_for_metrics(processed, projection),
                        ground_truth=ground_truth,
                        include_dbcv=True,
                    )
                except Exception as exc:
                    records.append(
                        _stage4_failed(
                            path,
                            trial_id=trial_id,
                            base_params=base_params,
                            parent_trial_id=finalist["trial_id"],
                            item=item,
                            exc=exc,
                        )
                    )
                    continue
                finally:
                    try:
                        apply_topic_labels(topic_model, docs, topics)
                    except Exception as restore_exc:
                        print(
                            f"trial {trial_id} restore failed: "
                            f"{type(restore_exc).__name__}: {restore_exc}"
                        )
                noise = metrics.get("noise_share", float("nan"))
                coherence = metrics.get("coherence_c_npmi")
                is_accepted = (
                    np.isfinite(noise)
                    and noise < baseline_noise
                    and coherence is not None
                    and np.isfinite(coherence)
                    and baseline_coherence is not None
                    and coherence >= baseline_coherence
                )
                record = {
                    "trial_id": trial_id,
                    "stage": 4,
                    "representation_id": representation_id(base_params),
                    "parent_trial_id": finalist["trial_id"],
                    "params": {**params_for_json(base_params), **item},
                    "metrics": metrics,
                    "passed_gates": passes_hard_gates(metrics, gates),
                    "skipped": False,
                    "accepted": bool(is_accepted),
                    "strategy": item["strategy"],
                    "threshold": item["threshold"],
                    "labels": labels.tolist(),
                }
                write_trial(path, record)
                _log_trial_mlflow(
                    base_params,
                    metrics,
                    run_name=trial_id,
                    extra={
                        "outlier_strategy": item["strategy"],
                        "outlier_threshold": item["threshold"],
                        "outlier_skipped": False,
                    },
                )
                records.append(record)
                if is_accepted:
                    accepted.append(record)

            if accepted:
                accepted.sort(
                    key=lambda rec: (
                        rec["metrics"].get("noise_share", 1.0),
                        -rec["metrics"].get("coherence_c_npmi", float("-inf")),
                    )
                )
                winner = accepted[0]
                winning_labels = np.asarray(winner["labels"], dtype=int)
                winning_params = replace(
                    base_params,
                    # Keep HDBSCAN params; outlier choice is recorded in trial id.
                )
                winning_id = winner["trial_id"]
            else:
                winning_labels = topics
                winning_params = base_params
                winning_id = f"{finalist['trial_id']}__unreduced"

        try:
            apply_topic_labels(topic_model, docs, winning_labels)
            labeled = frame.copy()
            labeled["topic"] = winning_labels
            stem_output = Path(output or default_output)
            stem_output = stem_output.with_name(
                f"{stem_output.stem}_{safe_token(winning_id)}{stem_output.suffix}"
            )
            write_topic_dataset(
                labeled,
                stem_output,
                winning_params,
                source=source,
                cache_path=emb_path,
                topic_model=topic_model,
                probabilities=None,
            )
        except Exception as exc:
            print(
                f"trial {winning_id} export failed: {type(exc).__name__}: {exc}"
            )

    write_advanced(
        4,
        quota_met=True,
        order=["noise_share", "coherence"],
        trials=[{k: v for k, v in rec.items() if k != "labels"} for rec in records],
        root=root,
    )
    return records


def run_hpo(
    *,
    spec_path: Path | None = None,
    input_path: Path | None = None,
    from_stage: int = 1,
    hdbscan_search: str = "grid",
    output: Path | None = None,
    env_file: Path | None = None,
) -> None:
    """Run stages 1–4 of the HPO spec (stage 5 is not executed)."""
    spec = load_spec(spec_path)
    data_dir = notebooks_dir()
    source = input_path or (data_dir / "emails_preprocessed.csv")
    env = env_file or (data_dir / ".env")
    frame = pd.read_csv(source)
    print(f"loaded {len(frame)} rows from {source}")
    setup_tracking()
    with mlflow.start_run(run_name=spec["job"]):
        mlflow.log_param("from_stage", from_stage)
        mlflow.log_param("hdbscan_search", hdbscan_search)
        mlflow.log_param("spec", str(spec_path or DEFAULT_SPEC))

        advanced1: list[dict[str, Any]] | None = None
        advanced2: list[dict[str, Any]] | None = None
        advanced3: list[dict[str, Any]] | None = None

        if from_stage <= 1:
            advanced1 = run_stage1(
                spec, frame, cache_dir=source.parent, env_file=env
            )
        if from_stage <= 2:
            if advanced1 is None:
                advanced1 = load_advanced(1)["trials"]
            advanced2 = run_stage2(
                spec, frame, advanced1, cache_dir=source.parent, env_file=env
            )
        if from_stage <= 3:
            if advanced2 is None:
                advanced2 = load_advanced(2)["trials"]
            advanced3 = run_stage3(
                spec,
                frame,
                advanced2,
                cache_dir=source.parent,
                env_file=env,
                hdbscan_search=hdbscan_search,
            )
        if from_stage <= 4:
            if advanced3 is None:
                advanced3 = load_advanced(3)["trials"]
            run_stage4(
                spec,
                frame,
                advanced3,
                cache_dir=source.parent,
                env_file=env,
                source=source,
                output=output,
            )
