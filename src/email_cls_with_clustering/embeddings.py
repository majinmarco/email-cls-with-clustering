"""Sentence-transformer and OpenAI embeddings, with an on-disk cache."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import hashlib

EMBED_BACKEND = "sentence_transformer"

ST_MODEL = "BAAI/bge-small-en-v1.5"
ST_MAX_SEQ_LENGTH = 512
ST_BATCH_SIZE = 32

OPENAI_MODEL = "text-embedding-3-small"
OPENAI_BATCH_SIZE = 64
OPENAI_MAX_TOKENS = 8191

_st_model = None
_openai_client = None
_tiktoken_enc = None


def active_embed_model(backend: str | None = None, model: str | None = None) -> str:
    """Model id for ``backend``. ``model`` overrides the backend default."""
    if model:
        return model
    backend = backend or EMBED_BACKEND
    if backend == "sentence_transformer":
        return ST_MODEL
    if backend == "openai":
        return OPENAI_MODEL
    raise ValueError(
        f"Unknown backend={backend!r}; use 'sentence_transformer' or 'openai'"
    )

def texts_fingerprint(texts: list[str]) -> str:
    """Stable content hash of the exact texts that produced a cache."""
    digest = hashlib.blake2b(digest_size=16)
    for text in texts:
        digest.update(text.encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()

def embedding_cache_path(directory: Path, backend: str, model: str) -> Path:
    """Cache file keyed by backend and model so dimensions are not mixed."""
    safe = model.replace("/", "_")
    return directory / f"embeddings_{backend}_{safe}.npy"


def embed_texts(
    texts: list[str],
    backend: str | None = None,
    batch_size: int | None = None,
    *,
    model: str | None = None,
    max_seq_length: int = ST_MAX_SEQ_LENGTH,
    env_file: Path | None = None,
) -> np.ndarray:
    """Embed texts with sentence-transformers or OpenAI.

    ``backend`` defaults to :data:`EMBED_BACKEND`.
    """
    backend = backend or EMBED_BACKEND

    cleaned = _clean_texts(texts)

    if backend == "sentence_transformer":
        st_model = _ensure_sentence_transformer(model, max_seq_length)
        bs = batch_size or ST_BATCH_SIZE
        return (
            st_model.encode(
                cleaned,
                batch_size=bs,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
        )

    if backend == "openai":
        client, enc = _ensure_openai(env_file)
        embed_model = model or OPENAI_MODEL
        bs = batch_size or OPENAI_BATCH_SIZE
        truncated = [_truncate_openai(text, enc) for text in cleaned]
        vectors: list[list[float]] = []
        steps = range(0, len(truncated), bs)
        for start in tqdm(steps, desc=f"embed ({embed_model})"):
            batch = truncated[start : start + bs]
            response = client.embeddings.create(model=embed_model, input=batch)
            ordered = sorted(response.data, key=lambda row: row.index)
            vectors.extend(row.embedding for row in ordered)
        return np.asarray(vectors, dtype=np.float32)

    raise ValueError(
        f"Unknown backend={backend!r}; use 'sentence_transformer' or 'openai'"
    )


def load_or_embed(
    texts: list[str],
    cache_path: Path,
    *,
    backend: str | None = None,
    model: str | None = None,
    batch_size: int | None = None,
    max_seq_length: int = ST_MAX_SEQ_LENGTH,
    env_file: Path | None = None,
) -> np.ndarray:
    """Load ``cache_path`` when its row count matches ``texts``, else embed and save."""
    backend = backend or EMBED_BACKEND
    stamp_path = cache_path.with_suffix(".fingerprint")
    fingerprint = texts_fingerprint(texts)
    if cache_path.exists():
        embeddings = np.load(cache_path)
        stored = stamp_path.read_text().strip() if stamp_path.exists() else None
        if len(embeddings) != len(texts) or stored != fingerprint:
            raise ValueError(
                f"Cache at {cache_path} does not match texts — "
                f"Delete and re-run"
            )
        print(f"loaded embeddings from {cache_path} → shape={embeddings.shape}")
        return embeddings

    if backend == "sentence_transformer" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    embeddings = embed_texts(
        texts,
        backend=backend,
        batch_size=batch_size,
        model=model,
        max_seq_length=max_seq_length,
        env_file=env_file,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    stamp_path.write_text(fingerprint)
    print(f"saved embeddings → {cache_path} shape={embeddings.shape}")
    return embeddings


def _ensure_sentence_transformer(model: str | None, max_seq_length: int):
    global _st_model
    model_name = model or ST_MODEL
    if _st_model is None or getattr(_st_model, "_email_cls_model_name", None) != model_name:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        _st_model._email_cls_model_name = model_name  # type: ignore[attr-defined]
    _st_model.max_seq_length = max_seq_length
    device = next(_st_model.parameters()).device
    print(
        f"loaded sentence_transformer model={model_name} "
        f"dim={_st_model.get_embedding_dimension()} "
        f"device={device} max_seq={_st_model.max_seq_length}"
    )
    return _st_model


def _ensure_openai(env_file: Path | None):
    global _openai_client, _tiktoken_enc
    if _openai_client is None:
        import tiktoken
        from dotenv import load_dotenv
        from openai import OpenAI

        if env_file is not None:
            load_dotenv(env_file)
        else:
            load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            where = env_file or "the environment"
            raise RuntimeError(f"OPENAI_API_KEY missing — set it in {where}")

        _openai_client = OpenAI()
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        print(f"loaded openai client model={OPENAI_MODEL}")
    return _openai_client, _tiktoken_enc


def _clean_texts(texts: list[str]) -> list[str]:
    return [(text or "").strip() or " " for text in texts]


def _truncate_openai(text: str, enc, max_tokens: int = OPENAI_MAX_TOKENS) -> str:
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
