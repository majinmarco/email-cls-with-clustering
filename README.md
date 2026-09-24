# email-cls-with-clustering

Preprocessing and BERTopic each have their own command. A third command runs both in order. Analysis stays in `notebooks/email-analysis.ipynb`; load whichever `emails_clustered_*.csv` you want there.

Data and the embedding cache live under `notebooks/`. From the repo root:

```bash
uv sync
```

## Preprocess

Reads `notebooks/emails.csv`, parses each message, builds `full_message` from subject and body, drops duplicate messages, cleans the text, and writes `notebooks/emails_preprocessed.csv`.

```bash
uv run preprocess-emails
```

If `notebooks/emails_expanded.csv` already exists, skip MIME parsing:

```bash
uv run preprocess-emails --from-expanded
```

`--input` and `--output` override the default paths. `--no-progress` hides the progress bars.

## Topic modeling

Reads `notebooks/emails_preprocessed.csv`, reuses `notebooks/embeddings_<backend>_<model>.npy` when the row count matches, fits BERTopic, and writes a new file each run:

```text
notebooks/emails_clustered_YYYYMMDDTHHMMSS.csv
notebooks/emails_clustered_YYYYMMDDTHHMMSS.json
```

The JSON file records the hyperparameters for that run. The datetime suffix is added even if you pass `--output`.

```bash
uv run topic-model-emails
```

Defaults match the notebook: UMAP `n_neighbors=15`, `n_components=5`, `min_dist=0.0`, cosine metric, `random_state=42`; HDBSCAN `min_cluster_size=50`, euclidean metric; BERTopic `calculate_probabilities` on; sentence-transformer model `ibm-granite/granite-embedding-97m-multilingual-r2`.

Change one run from the command line. Flags override a `--config` JSON file whose keys are the same names (`min_cluster_size`, `n_neighbors`, `n_components`, `min_dist`, `umap_metric`, `random_state`, `hdbscan_metric`, `prediction_data`, `calculate_probabilities`, `embed_backend`, `embed_model`, `batch_size`, `max_seq_length`). A JSON list of objects is one topic-model run per object. `configs/min-cluster-size.json` runs `min_cluster_size` 100 and 200 and leaves the rest at the defaults.

```bash
uv run topic-model-emails --min-cluster-size 80 --n-neighbors 20
uv run topic-model-emails --config configs/min-cluster-size.json
uv run topic-model-emails --config topic-params.json --min-dist 0.1
uv run topic-model-emails --no-calculate-probabilities
```

Embedding uses a progress bar. BERTopic prints its own progress. The OpenAI backend reads `notebooks/.env` (`--env-file` overrides that).

## Both steps

Runs preprocessing, then topic modeling, and prints the timestamped CSV path.

```bash
uv run run-email-pipeline --from-expanded --min-cluster-size 80
```

`--raw-input` is the preprocess input. `--preprocessed-output` is the cleaned CSV. `--output` is the clustered path before the datetime suffix. The topic-model flags above work here too.

The same entry points are `scripts/preprocess_emails.py`, `scripts/topic_model_emails.py`, and `scripts/run_pipeline.py`.
