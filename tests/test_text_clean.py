# tests/test_text_clean.py
import pandas as pd

from email_cls_with_clustering.text_clean import prepare_messages


def test_embed_text_stays_raw_case_ctfidf_is_lower():
    frame = pd.DataFrame(
        {
            "Subject": ["Gas nomination", None, None, None],
            "body": [
                "confirm volume",
                "call me about the deal",
                "lunch at noon",
                "the contract is signed",
            ],
        }
    )
    out = prepare_messages(frame)
    assert len(out) == 4
    assert "Gas nomination confirm volume" in set(out["embed_text"])
    assert "Gas nomination confirm volume" in set(out["full_message"])
    assert "gas nomination confirm volume" in set(out["ctfidf_text"])


def test_fully_empty_messages_are_dropped():
    frame = pd.DataFrame({"Subject": ["Gas nomination", None], "body": ["confirm volume", None]})
    out = prepare_messages(frame)
    assert list(out["embed_text"]) == ["Gas nomination confirm volume"]
    assert list(out["ctfidf_text"]) == ["gas nomination confirm volume"]


def test_boilerplate_stripped_case_preserved():
    frame = pd.DataFrame(
        {
            "Subject": ["Hello"],
            "body": ["--- Forwarded by Alice ---\nWorld [IMAGE] today"],
        }
    )
    out = prepare_messages(frame)
    assert len(out) == 1
    assert "Forwarded by" not in out.loc[0, "embed_text"]
    assert "[IMAGE]" not in out.loc[0, "embed_text"]
    assert "IMAGE" not in out.loc[0, "embed_text"]
    assert "Hello" in out.loc[0, "embed_text"]
    assert "World" in out.loc[0, "embed_text"]
    assert "today" in out.loc[0, "embed_text"]


def test_dedupe_after_clean_keeps_first_raw_embed_text():
    frame = pd.DataFrame(
        {
            "Subject": ["Hello!", "Hello"],
            "body": ["World.", "World"],
        }
    )
    out = prepare_messages(frame)
    assert len(out) == 1
    assert out.loc[0, "embed_text"] == "Hello! World."
    assert out.loc[0, "ctfidf_text"] == "hello world"
