"""Post-process embedding matrices before UMAP / BERTopic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from email_cls_with_clustering.topics import TopicHyperparams


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def mean_removal(embeddings: np.ndarray) -> np.ndarray:
    """Subtract the corpus mean and L2-renormalize. Returns float32."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    return _l2_normalize(centered).astype(np.float32)


def mean_removal_plus_top_k(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Mean-center, remove the top-k principal components, L2-renormalize."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    # Exact PCA via column covariance (chunk-friendly eigh on d×d).
    cov = centered.T @ centered / max(centered.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:k]]
    residual = centered - (centered @ components) @ components.T
    return _l2_normalize(residual).astype(np.float32)


def select_top_k(
    embeddings: np.ndarray,
    candidates: Sequence[int] = (1, 2, 3),
    sample: int = 5000,
    seed: int = 42,
) -> int:
    """Pick the candidate k with the lowest mean pairwise cosine on a sample."""
    from email_cls_with_clustering.evaluate import anisotropy

    best_k = int(candidates[0])
    best_score = float("inf")
    for k in candidates:
        processed = mean_removal_plus_top_k(embeddings, int(k))
        score = anisotropy(processed, sample=sample, seed=seed)["mean_pairwise_cosine"]
        if score < best_score:
            best_score = score
            best_k = int(k)
    return best_k


def prepare_representation(
    embeddings: np.ndarray,
    params: TopicHyperparams,
    candidates: Sequence[int] = (1, 2, 3),
    sample: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """Apply ``params.post_processing``. May set ``params.top_k`` when selecting."""
    method = params.post_processing
    if method == "none":
        return embeddings
    if method == "mean_removal":
        return mean_removal(embeddings)
    if method == "mean_removal_plus_top_k":
        if params.top_k is None:
            params.top_k = select_top_k(
                embeddings, candidates=candidates, sample=sample, seed=seed
            )
        return mean_removal_plus_top_k(embeddings, params.top_k)
    raise ValueError(f"Unknown post_processing={method!r}")
