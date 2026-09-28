# tests/test_representation.py
import numpy as np

from email_cls_with_clustering.evaluate import anisotropy
from email_cls_with_clustering.representation import (
    mean_removal,
    mean_removal_plus_top_k,
    select_top_k,
)


def test_mean_removal_lowers_anisotropy():
    rng = np.random.default_rng(0)
    # Cone: shared drift plus small noise → anisotropic before mean removal.
    drift = np.ones(16)
    matrix = drift + 0.05 * rng.normal(size=(200, 16))
    raw = anisotropy(matrix, sample=200, seed=42)
    cleaned = mean_removal(matrix)
    after = anisotropy(cleaned, sample=200, seed=42)
    assert after["mean_pairwise_cosine"] < raw["mean_pairwise_cosine"]
    assert after["norm_of_mean_vector"] < raw["norm_of_mean_vector"]


def test_select_top_k_picks_lowest_mean_pairwise_cosine():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(300, 12))
    # Inject a strong first PC and a weaker second so k=1 vs k=2 differ.
    direction = np.zeros(12)
    direction[0] = 1.0
    matrix = base + 3.0 * rng.normal(size=(300, 1)) * direction
    chosen = select_top_k(matrix, candidates=(1, 2, 3), sample=300, seed=42)
    scores = {
        k: anisotropy(mean_removal_plus_top_k(matrix, k), sample=300, seed=42)[
            "mean_pairwise_cosine"
        ]
        for k in (1, 2, 3)
    }
    assert chosen == min(scores, key=scores.get)
