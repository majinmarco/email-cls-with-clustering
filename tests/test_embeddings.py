# tests/test_embeddings.py
from email_cls_with_clustering.embeddings import texts_fingerprint


def test_fingerprint_is_stable():
    assert texts_fingerprint(["a", "b"]) == texts_fingerprint(["a", "b"])


def test_fingerprint_changes_with_content_and_with_order():
    assert texts_fingerprint(["a", "b"]) != texts_fingerprint(["a", "c"])
    assert texts_fingerprint(["a", "b"]) != texts_fingerprint(["b", "a"])
    assert texts_fingerprint(["ab", ""]) != texts_fingerprint(["a", "b"])  # the \0 separator