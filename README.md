# email-cls-with-clustering

Preprocessing and BERTopic each have their own command. A third command runs both in order. A fourth runs staged hyperparameter search. Analysis stays in `notebooks/email-analysis.ipynb`; load whichever `emails_clustered_*.csv` you want there.

Data and the embedding cache live under `notebooks/`. From the repo root:

```bash
uv sync
```

## Preprocess

Reads `notebooks/emails.csv`, parses each message, builds `embed_text` from subject and body (raw case, after stripping forwarded-message banners and `[IMAGE]` placeholders), drops empty rows, builds scrubbed `ctfidf_text`, dedupes on `ctfidf_text`, and writes `notebooks/emails_preprocessed.csv`. `full_message` is an alias of `embed_text`.

```bash
uv run preprocess-emails
```

If `notebooks/emails_expanded.csv` already exists, skip MIME parsing:

```bash
uv run preprocess-emails --from-expanded
```

`--input` and `--output` override the default paths. `--no-progress` hides the progress bars.

## Topic modeling

Reads `notebooks/emails_preprocessed.csv`, reuses `notebooks/embeddings_<backend>_<model>.npy` when the fingerprint matches, fits BERTopic, and writes a new file each run:

```text
notebooks/emails_clustered_YYYYMMDDTHHMMSS.csv
notebooks/emails_clustered_YYYYMMDDTHHMMSS.json
notebooks/emails_clustered_YYYYMMDDTHHMMSS_model.pkl
```

The JSON file records the hyperparameters and paths for that run. The `.pkl` is a full BERTopic pickle (UMAP/HDBSCAN included) — reload with `BERTopic.load(...)`. Same Python/BERTopic versions when loading. The datetime suffix is added even if you pass `--output`.

```bash
uv run topic-model-emails
```

Defaults match the stage-1 baseline in `configs/enron-email-topic-model-hpo.json`: UMAP `n_neighbors=15`, `n_components=10`, `min_dist=0.0`, cosine metric, `random_state` unset, `n_jobs=-1`; HDBSCAN `min_cluster_size=100`, `min_samples=10`, euclidean metric, `cluster_selection_method=eom`, `prediction_data` and `gen_min_span_tree` on; BERTopic `calculate_probabilities` on; sentence-transformer model `ibm-granite/granite-embedding-97m-multilingual-r2`.

Change one run from the command line. Flags override a flat `--config` JSON file whose keys are the same names (`min_cluster_size`, `min_samples`, `n_neighbors`, `n_components`, `min_dist`, `umap_metric`, `random_state`, `n_jobs` / `--umap-n-jobs`, `hdbscan_metric`, `cluster_selection_method`, `cluster_selection_epsilon`, `prediction_data`, `gen_min_span_tree`, `calculate_probabilities`, `embed_backend`, `embed_model`, `embed_prompt`, `batch_size`, `max_seq_length`, `post_processing`, `top_k`, `ngram_range`). A JSON list of objects is one topic-model run per object. `configs/min-cluster-size.json` is a flat `min_cluster_size` sweep (50 through 500) and must not be confused with the nested HPO spec.

```bash
uv run topic-model-emails --min-cluster-size 80 --n-neighbors 20
uv run topic-model-emails --config configs/min-cluster-size.json
uv run topic-model-emails --config topic-params.json --min-dist 0.1
uv run topic-model-emails --no-calculate-probabilities
uv run topic-model-emails --embed-prompt "clustering: " --post-processing mean_removal
uv run topic-model-emails --min-samples 15 --umap-n-jobs -1 --ngram-range 1 2
```

Embedding uses a progress bar. BERTopic prints its own progress. The OpenAI backend reads `notebooks/.env` (`--env-file` overrides that).

## Evaluation

Each topic-model run logs hyperparameters and scores to MLflow. The local store is `sqlite:///mlflow.db` at the repo root. `MLFLOW_TRACKING_URI` overrides it.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`coherence_c_npmi` and `dbcv` run unless you pass `--no-coherence` or `--no-dbcv`. If the input CSV has a `ground_truth` column, `ami` and `ari` are logged as well. The same flags work on `run-email-pipeline`.

A config file that is a list of objects is one parent run named `sweep`, with a nested run per object named `mcs-<min_cluster_size>`. Per-topic names and sizes are the `topic_info.json` artifact on each child run.

### Metrics

**Look at these three first.** They cover mostly orthogonal failure modes, so they stand in for the rest when picking a model:

1. **`coherence_c_npmi`** — Are the topics real themes? (also tends to move with peakedness / diversity.)
2. **`noise_share`** — How much of the corpus is usable vs left as `-1`?
3. **`largest_topic_share`** — Did clustering collapse into one mega-topic? (coherence can look fine on that blob.)

If a `ground_truth` column exists, prefer **`ami` / `ari`** over all three. Everything else below is diagnostic: use it when one of the top three is bad and you need to know *why*.

Hard gates (before HPO ranking) from `configs/enron-email-topic-model-hpo.json`: `largest_topic_share ≤ 0.10`, `noise_share ≤ 0.50`, `min_peakedness ≥ 0.05`, `n_topics ≥ 30`. Across representations, rank by coherence → topic diversity → noise share → largest-topic share. Within one representation, prefer DBCV first (not comparable across embedding spaces). HPO stage 4 (outlier reduction) lowers noise without dropping coherence — see [Hyperparameter search](#hyperparameter-search).

| Priority | Metric | Meaning | Better | How to optimize |
| --- | --- | --- | --- | --- |
| **1 — primary** | `coherence_c_npmi` | Gensim NPMI coherence on top-10 words vs `ctfidf_text` tokens. Roughly in `[-1, 1]`. | Higher | Representation quality, UMAP geometry, and HDBSCAN that keeps thematically tight clusters. Do not accept outlier reduction that lowers this. |
| **2 — primary** | `noise_share` | Share of documents labelled `-1` (before outlier reduction). | Lower (gate: ≤ 0.50) | HPO stage 4 outlier reduction (`c-tf-idf` / `embeddings` + threshold). Softer HDBSCAN: lower `min_samples`, raise `cluster_selection_epsilon`, or slightly lower `min_cluster_size`. Do not chase zero — forced assignment often hurts coherence. |
| **3 — primary** | `largest_topic_share` | Share of all documents in the biggest non-noise topic. | Lower (gate: ≤ 0.10) | Raise `min_cluster_size` or `min_samples`. Representation post-processing (`mean_removal`, `mean_removal_plus_top_k`) can also split a dominant hub. |
| diagnostic | `n_topics` | Count of non-noise topics (`label ≥ 0`). | Usually higher (gate: ≥ 30) | Lower `min_cluster_size` / `min_samples`, or use `cluster_selection_method=leaf`. Too high often means fragmented noise; raise MCS if topics look like near-duplicates. |
| diagnostic | `topic_size_entropy` | Shannon entropy of non-outlier topic sizes, divided by `log(n_topics)`. `1.0` = even split. Undefined for fewer than two topics. | Higher (balanced) | Same levers as `largest_topic_share`. Many tiny equal topics also score high. |
| diagnostic | `topic_diversity` | Share of distinct words across all topics' top-10 lists. `1.0` = no repeated words. | Higher | Better separation and less boilerplate in `ctfidf_text`. Often moves with coherence; check when topics share the same vocabulary. |
| diagnostic | `min_peakedness` | Minimum over topics of the max top-10 c-TF-IDF weight. | Higher (gate: ≥ 0.05) | Under-separated clusters or leftover boilerplate. Raise `min_cluster_size`, improve cleaning, or change representation. |
| diagnostic | `peakedness_mean` | Mean of those per-topic max c-TF-IDF weights. | Higher | Same as `min_peakedness`; look at the min when a few dead topics drag the floor. |
| diagnostic | `flat_topic_share` | Share of topics whose max top-10 c-TF-IDF is below `0.05`. | Lower | Same as peakedness. High share → reject even if average peakedness looks fine. |
| diagnostic | `dbcv` | Density-Based Cluster Validity on the **original** (post-processed) embedding space; noise points dropped; sampled to 20k. Degenerate labelings (e.g. a cluster with fewer than 2 points after noise drop) yield NaN instead of aborting the trial. | Higher | Only compare within one representation. Tune UMAP / HDBSCAN on a fixed projection. Never compare across embed models or post-processing. |
| diagnostic | `mean_pairwise_cosine` | Mean cosine between pairs of embedding vectors (5k sample). | Lower (less collapsed) | HPO stage 1: `mean_removal` / `mean_removal_plus_top_k`; `top_k` minimizes this. Or change `embed_model` / `embed_prompt`. |
| diagnostic | `norm_of_mean_vector` | L2 norm of the mean of L2-normalized embeddings. | Lower | Same post-processing; high values mean a strong shared direction (anisotropy). |
| diagnostic | `top_singular_energy_share` | Share of covariance energy in the top principal component. | Lower | `mean_removal_plus_top_k` removes leading PCs. |
| diagnostic | `mean_max_probability` | Mean over documents of `max(topic probabilities)`. Needs `calculate_probabilities`. | Higher (sharper assignment) | Clearer clusters; Soft-HDBSCAN memberships get peaker near one cluster. |
| diagnostic | `mean_assignment_entropy` | Mean Shannon entropy of each document's topic distribution. | Lower (less ambiguous) | Same as `mean_max_probability`. High entropy with low noise → overlapping topics. |
| overrides top 3 | `ami` / `ari` | Adjusted Mutual Information / Adjusted Rand Index vs a `ground_truth` column (optional). Chance-corrected; ~0 = random, 1 = match. | Higher | When labels exist and match your goal taxonomy, trust these over the unsupervised top 3. |

`word_intrusion` is manual (HPO stage 5): mix one outsider into a topic's top words and see if a human spots it. Not logged to MLflow.

## Hyperparameter search

`configs/enron-email-topic-model-hpo.json` is the config of record. A normal `topic-model-emails` / `run-email-pipeline` run stays one BERTopic fit. Staged search is a separate command:

```bash
uv run topic-model-hpo
uv run topic-model-hpo --spec configs/enron-email-topic-model-hpo.json --from-stage 1
uv run topic-model-hpo --input notebooks/emails_preprocessed.csv --hdbscan-search grid
uv run topic-model-hpo --hdbscan-search optuna --from-stage 3
```

It runs representation → UMAP → HDBSCAN on cached projections → outlier reduction (stages 1–4), writes trial JSON under `notebooks/hpo/stage{n}/`, and writes full clustered CSV / model pickle / JSON sidecar only for stage-4 finalists. A failed fit or metric is recorded on that trial and skipped on resume; it does not abort the stage. Stage 5 (seed stability and word intrusion) is recorded in the spec and is not run.

The screen keeps the trial budget near ~200 fits instead of a full ~960: fixed HDBSCAN picks 3 of 24 UMAP projections, then 40 HDBSCAN trials on each. Pass `--from-stage N` to resume from a previous stage's `advanced.json`. `--hdbscan-search optuna` swaps the stage-3 grid for a TPE search with the same categorical space.

Each stage writes one JSON file per trial under `notebooks/hpo/stage{n}/` and, when it advances, `advanced.json` in that directory. `--from-stage N` reads `stage{N-1}/advanced.json` and does not re-rank earlier stages. Stages 1–3 drop trials that miss the hard gates and stop if none pass. A trial that errors while fitting or scoring is written with empty metrics and skipped on the next resume.

**Stage 1 — representation.** Grid over embedding model × post-processing: five models (`granite-97m-multilingual`, `BAAI/bge-small-en-v1.5`, `granite-embedding-english-r2`, `Qwen3-Embedding-0.6B`, `nomic-embed-text-v1.5` with prompt `clustering: `) and `none` / `mean_removal` / `mean_removal_plus_top_k` (15 trials). For `+top_k`, `k` is chosen from `{1, 2, 3}` as the value with the lowest mean pairwise cosine on a 5k sample. UMAP and HDBSCAN stay fixed (`n_neighbors=15`, `n_components=10`, `min_dist=0.0`, cosine; `min_cluster_size=100`, `min_samples=10`, `eom`). Rank across representations by coherence, then topic diversity, then noise share, then largest-topic share. Advance 2.

**Stage 2 — UMAP.** Grid on each advanced representation: `n_components` ∈ {5, 10, 25, 50} × `n_neighbors` ∈ {15, 50, 100} (12 projections each; `min_dist=0.0`, cosine). HDBSCAN stays at the stage-1 settings so the screen is only the projection. Projections are cached as `.npy`. Same across-representation ranking as stage 1. Advance 3.

**Stage 3 — HDBSCAN.** Grid on each cached projection: `min_cluster_size` ∈ {25, 50, 100, 200, 400} × `min_samples` ∈ {1, 5, 15, 50} × `cluster_selection_method` ∈ {eom, leaf} × `cluster_selection_epsilon` ∈ {0.0} (40 trials per projection). When every survivor shares one representation, rank by DBCV, then coherence, topic diversity, and noise share. When more than one representation remains, use the stage-1 order instead. Advance 3. `--hdbscan-search optuna` replaces the grid with 40 TPE trials per projection over the same categories.

**Stage 4 — outlier reduction.** No refit. On each stage-3 finalist, try `c-tf-idf` and `embeddings` at thresholds {0.0, 0.05, 0.1, 0.2} (8 trials). A reduction is kept only when `noise_share` falls and `coherence_c_npmi` does not. If nothing qualifies, the unreduced labeling is the result. If the fit already has zero noise, every reduction is recorded as skipped. The winning labeling is written as a clustered CSV, model pickle, and JSON sidecar.

**Stage 5 — finalist validation.** Listed in the spec and not executed. It would refit UMAP seeds `{0, 1, 2}` and report pairwise ARI, then a manual word-intrusion check (30 topics × 3 finalists). Vectorizer `ngram_range` and `representation_model` are explicitly not tuned.

## Both steps

Runs preprocessing, then topic modeling, and prints the timestamped CSV path.

```bash
uv run run-email-pipeline --from-expanded --min-cluster-size 80
```

`--raw-input` is the preprocess input. `--preprocessed-output` is the cleaned CSV. `--output` is the clustered path before the datetime suffix. The topic-model flags above work here too.

The same entry points are `scripts/preprocess_emails.py`, `scripts/topic_model_emails.py`, `scripts/topic_model_hpo.py`, and `scripts/run_pipeline.py`.
