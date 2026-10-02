# email-cls-with-clustering

End-to-end analysis of the Enron corpus lives in [`notebooks/enron-email-analysis-e2e.ipynb`](notebooks/enron-email-analysis-e2e.ipynb). The notebook compares LDA and NMF with BERTopic, then carries one BERTopic model through labels, sentiment, the mail graph, time, and a topic classifier. Staged search for that model is `topic-model-hpo`. A single fit, including a refit of one HPO setting, is `topic-model-emails`.

From the repo root:

```bash
uv sync
```

Run the CLIs from the repo root. Open the notebook with its working directory set to `notebooks/`, where its CSV and pickle paths are relative. LLM topic labels read `OPENAI_API_KEY` from `notebooks/.env`.

## What the notebook does

The notebook loads `emails_preprocessed_no_spam.csv` and collapses each `thread_id` with `join_by_thread`, the same join as `topic-model-hpo --join-threads`. One row is one thread. `From`, `To`, and `Date` come from the earliest message in that thread.

### EDA

Weekly thread volume, length of the joined text, and the top senders and receivers. The stopword review tokenizes `ctfidf_text` with the same pattern and stop list as HPO (`TOKEN_PATTERN`, `default_stop_words`): letters only, no lemma or POS filter.

### Linguistic features

Gensim phrases, a TF-IDF view of those phrases, and a log-odds comparison of tokens before and after 2001-08-01. BERTopic's own c-TF-IDF uses a `CountVectorizer` with `ngram_range` `[1, 2]`. This section writes `notebooks/emails_with_tokens.csv`, which later cells reload.

### Topic modeling

LDA (`c_v`, `c_npmi`, `u_mass` for k in 10, 15, 20, 30, 40) and a 20-topic NMF are the baselines. BERTopic starts from the stage-4 pickle written by `topic-model-hpo` (the notebook loads the unreduced BGE mean-removal finalist, `min_cluster_size=200`, `min_samples=50`, UMAP `n_components=50`, `n_neighbors=100`). After `visualize_topics`, small topics are merged by hand with `merge_topics`. BAAI embeddings are loaded from the embedding cache, then KeyBERT and a gpt-4o labeler (`notebooks/.env`) fill `topic_representations_`. The merged model is saved as `notebooks/optimal_enron_topic_model_merged.pkl`. Later sections reload that file.

### Sentiment

VADER compound scores on `embed_text`, compared across merged topic names and as a weekly median over time.

### Network

A directed graph of `From` → `To`, keeping nodes of degree at least 3. PageRank, betweenness, and Louvain communities on the undirected graph. A community-by-topic share table drops BERTopic's outlier bin (`topic == -1`) before the columns are renamed to LLM labels.

### Time

Monthly share of non-outlier threads, inside the date interquartile range and only for months with at least 500 threads. Surges are the largest rise in a 3-month average share versus six months earlier. A second plot is the median hours from the first message in a thread to the second, by the quarter the thread opened. Same-timestamp pairs are mailbox copies, so only positive gaps count.

### Classification

TF-IDF (`min_df=5`, `max_df=0.5`, 20,000 features) and a class-weighted logistic regression predict `topic_name` on a stratified holdout. Rows with `topic == -1` stay out of the label set. The notebook prints a classification report and per-class F1.

## Produce the model

`preprocess-emails` reads `notebooks/emails.csv` and writes `notebooks/emails_preprocessed.csv`: parsed messages, `embed_text` and scrubbed `ctfidf_text`, Presidio redaction, exact and MinHash dedupe, and `thread_id`. The first run needs `python -m spacy download en_core_web_lg`. [`notebooks/email_preprocessed_spam_reduction.ipynb`](notebooks/email_preprocessed_spam_reduction.ipynb) labels that CSV and writes `notebooks/emails_preprocessed_no_spam.csv`, which is the HPO default and the notebook input.

`run-email-pipeline` runs preprocess and then one BERTopic fit. The analysis notebook starts from the no-spam CSV and an HPO pickle.

### Staged search

`topic-model-hpo` runs the search in [`configs/enron-email-topic-model-hpo.json`](configs/enron-email-topic-model-hpo.json). `--join-threads` collapses each thread before embedding, matching the notebook. Saved trials are reused when the text fingerprint and vectorizer match. A failed fit or metric is recorded on that trial and skipped on resume.

```bash
uv run topic-model-hpo --join-threads --from-stage 1
uv run topic-model-hpo --join-threads --hdbscan-search optuna --from-stage 3
```

`--spec` overrides the config. `--input` overrides the CSV. `--from-stage N` reads `notebooks/hpo/stage{N-1}/advanced.json` and does not re-rank earlier stages. `--hdbscan-search` is `grid` (default) or `optuna`. Each stage writes one JSON file per trial under `notebooks/hpo/stage{n}/`, plus `advanced.json` when it advances. Stages 1–3 drop trials that miss the hard gates.

**Stage 1 — representation.** `BAAI/bge-small-en-v1.5` crossed with `none`, `mean_removal`, and `mean_removal_plus_top_k` (3 trials). For `+top_k`, `k` is the value in `{1, 2, 3}` with the lowest mean pairwise cosine on a 5k sample. UMAP and HDBSCAN stay fixed (`n_neighbors=15`, `n_components=10`, `min_dist=0.0`, cosine; `min_cluster_size=100`, `min_samples=10`, `eom`). Rank across representations. Advance 2.

**Stage 2 — UMAP.** On each advanced representation: `n_components` in {5, 10, 25, 50} × `n_neighbors` in {15, 50, 100} (12 projections; `min_dist=0.0`, cosine). HDBSCAN stays at the stage-1 settings. Projections are cached as `.npy`. Same ranking as stage 1. Advance 3.

**Stage 3 — HDBSCAN.** On each cached projection: `min_cluster_size` in {25, 50, 100, 200, 400} × `min_samples` in {1, 5, 15, 50} × `cluster_selection_method` in {eom, leaf} × `cluster_selection_epsilon` in {0.0} (40 trials per projection). When every survivor shares one representation, rank by DBCV, then coherence, topic diversity, and noise share. When more than one representation remains, use the stage-1 order. Advance 3.

`--hdbscan-search optuna` replaces that grid with 40 TPE trials per projection: `min_cluster_size` in [15, 500] and `min_samples` in [1, 100] (log-scale integers, `min_samples` capped at `min_cluster_size`), `cluster_selection_method` in {eom, leaf}, `cluster_selection_epsilon` in [0, 0.5] in steps of 0.01. TPE maximizes `coherence_c_npmi` + 0.5·`topic_diversity` − 0.5·`noise_share`. Trials that miss a hard gate score below every passing trial. Studies live in `notebooks/hpo/stage3/optuna.db`, one per projection and search setup, so a rerun continues. Changing the space, objective, or gates starts a new study. Advancement still uses the ranking above.

**Stage 4 — outlier reduction.** No refit. On each stage-3 finalist, try `c-tf-idf` and `embeddings` at thresholds {0.0, 0.05, 0.1, 0.2}. A reduction is kept only when `noise_share` falls and `coherence_c_npmi` does not. If nothing qualifies, the unreduced labeling is the result. The winning labeling is written as a clustered CSV, a BERTopic pickle, and a JSON sidecar:

```text
notebooks/emails_clustered_<trial_id>_<YYYYMMDDTHHMMSS>.csv
notebooks/emails_clustered_<trial_id>_<YYYYMMDDTHHMMSS>.json
notebooks/emails_clustered_<trial_id>_<YYYYMMDDTHHMMSS>_model.pkl
```

The notebook's BERTopic section loads that `_model.pkl`. Reload with `BERTopic.load(...)` on the same Python and BERTopic versions.

**Stage 5 — finalist validation.** Listed in the spec: refit UMAP seeds `{0, 1, 2}` and report pairwise ARI, then a word-intrusion check. The command does not run it. Inspection, the manual topic merge, and the gpt-4o labels happen in the notebook.

### One BERTopic fit

`topic-model-emails` fits one model (or one parent MLflow run per object when `--config` is a JSON list). Use it to refit a chosen HPO setting without rerunning the search. The default input is `notebooks/emails_preprocessed.csv`; pass the no-spam CSV to match the notebook. `--join-threads` uses the same thread join. Flags override a flat `--config` JSON object whose keys match `TopicHyperparams` (`min_cluster_size`, `min_samples`, `n_neighbors`, `n_components`, `min_dist`, `umap_metric`, `hdbscan_metric`, `cluster_selection_method`, `cluster_selection_epsilon`, `embed_model`, `embed_prompt`, `post_processing`, `top_k`, `ngram_range`, and the rest of the CLI flags).

This refits the unreduced finalist the notebook loads:

```bash
uv run topic-model-emails \
  --input notebooks/emails_preprocessed_no_spam.csv \
  --join-threads \
  --embed-model BAAI/bge-small-en-v1.5 \
  --post-processing mean_removal \
  --umap-components 50 \
  --n-neighbors 100 \
  --min-cluster-size 200 \
  --min-samples 50 \
  --cluster-selection-method eom \
  --cluster-selection-epsilon 0.0
```

Each run writes a new timestamped set, even if you pass `--output`:

```text
notebooks/emails_clustered_YYYYMMDDTHHMMSS.csv
notebooks/emails_clustered_YYYYMMDDTHHMMSS.json
notebooks/emails_clustered_YYYYMMDDTHHMMSS_model.pkl
```

`coherence_c_npmi` and `dbcv` run unless you pass `--no-coherence` or `--no-dbcv`. Both this command and `topic-model-hpo` log hyperparameters and scores to MLflow. The local store is `sqlite:///mlflow.db` at the repo root. `MLFLOW_TRACKING_URI` overrides it.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Reading a trial

Look at these three first. They cover mostly separate failure modes:

1. `coherence_c_npmi` — Gensim NPMI on the top-10 words against `ctfidf_text`. Higher means the topic words hang together.
2. `noise_share` — Share of documents labelled `-1` before outlier reduction. Lower means more of the corpus is assigned. The gate is 0.65.
3. `largest_topic_share` — Share of documents in the biggest non-noise topic. Lower means clustering did not collapse into one hub. The gate is 0.10.

Hard gates in the spec, applied before ranking: `largest_topic_share` ≤ 0.10, `noise_share` ≤ 0.65, `flat_topic_share` ≤ 0.50, and `30 ≤ n_topics ≤ 300`. Trials that miss a gate do not advance.

When every trial under comparison shares one representation, rank by DBCV, then coherence, topic diversity, and noise share. DBCV is computed in that representation's embedding space, so it ranks trials inside one embedding model and post-processing. When more than one representation is in the set, rank by coherence, topic diversity, noise share, then largest-topic share. If the CSV has a `ground_truth` column, `ami` and `ari` are prepended to that order.
