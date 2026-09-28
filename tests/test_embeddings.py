# tests/test_embeddings.py
import numpy as np

from email_cls_with_clustering import embeddings as emb
from email_cls_with_clustering.embeddings import (
    embedding_cache_path,
    load_or_embed,
    texts_fingerprint,
)


def test_fingerprint_is_stable():
    assert texts_fingerprint(["a", "b"]) == texts_fingerprint(["a", "b"])


def test_fingerprint_changes_with_content_and_with_order():
    assert texts_fingerprint(["a", "b"]) != texts_fingerprint(["a", "c"])
    assert texts_fingerprint(["a", "b"]) != texts_fingerprint(["b", "a"])
    assert texts_fingerprint(["ab", ""]) != texts_fingerprint(["a", "b"])  # the \0 separator


def test_matching_cache_is_reused(tmp_path, monkeypatch):
    cache = tmp_path / "embeddings.npy"
    vectors = np.ones((2, 3), dtype=np.float32)
    np.save(cache, vectors)
    cache.with_suffix(".fingerprint").write_text(texts_fingerprint(["a", "b"]))

    def boom(*_args, **_kwargs):
        raise AssertionError("embed_texts should not run on a matching cache")

    monkeypatch.setattr(emb, "embed_texts", boom)
    loaded = load_or_embed(["a", "b"], cache)
    assert np.array_equal(loaded, vectors)


def test_mismatch_deletes_and_reembeds(tmp_path, monkeypatch):
    cache = tmp_path / "embeddings.npy"
    np.save(cache, np.ones((2, 3), dtype=np.float32))
    stamp = cache.with_suffix(".fingerprint")
    stamp.write_text("stale")
    fresh = np.arange(6, dtype=np.float32).reshape(2, 3)

    monkeypatch.setattr(emb, "embed_texts", lambda *a, **k: fresh)
    loaded = load_or_embed(["a", "b"], cache, prompt="")
    assert np.array_equal(loaded, fresh)
    assert stamp.read_text() == texts_fingerprint(["a", "b"])
    assert np.array_equal(np.load(cache), fresh)


def test_prompt_is_part_of_fingerprint_and_cache_path(tmp_path, monkeypatch):
    empty_path = embedding_cache_path(tmp_path, "sentence_transformer", "model")
    assert empty_path.name == "embeddings_sentence_transformer_model.npy"
    prompted_path = embedding_cache_path(
        tmp_path, "sentence_transformer", "model", prompt="clustering: "
    )
    assert prompted_path != empty_path

    fresh = np.ones((1, 2), dtype=np.float32)
    monkeypatch.setattr(emb, "embed_texts", lambda *a, **k: fresh)
    load_or_embed(["a"], prompted_path, prompt="clustering: ")
    assert prompted_path.with_suffix(".fingerprint").read_text() == texts_fingerprint(
        ["clustering: a"]
    )


def test_sentence_transformer_trust_remote_code(monkeypatch):
    calls = {}

    class FakeST:
        def __init__(self, name, device=None, trust_remote_code=False):
            calls["trust_remote_code"] = trust_remote_code
            calls["name"] = name
            self.max_seq_length = 0

        def parameters(self):
            import torch

            yield torch.nn.Parameter(torch.zeros(1))

        def get_embedding_dimension(self):
            return 1

        def encode(self, *args, **kwargs):
            return np.ones((1, 1), dtype=np.float32)

    import sys
    import types

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    emb._st_model = None
    try:
        emb._ensure_sentence_transformer("fake-model", 512)
        assert calls["trust_remote_code"] is True
    finally:
        emb._st_model = None
