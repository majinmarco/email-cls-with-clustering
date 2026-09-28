# tests/test_text_clean.py
import pandas as pd

from email_cls_with_clustering.text_clean import (
    drop_near_duplicates,
    prepare_messages,
    word_shingles,
)


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
    out = prepare_messages(frame, redact_pii_enabled=False)
    assert len(out) == 4
    assert "Gas nomination confirm volume" in set(out["embed_text"])
    assert "Gas nomination confirm volume" in set(out["full_message"])
    assert "gas nomination confirm volume" in set(out["ctfidf_text"])


def test_fully_empty_messages_are_dropped():
    frame = pd.DataFrame({"Subject": ["Gas nomination", None], "body": ["confirm volume", None]})
    out = prepare_messages(frame, redact_pii_enabled=False)
    assert list(out["embed_text"]) == ["Gas nomination confirm volume"]
    assert list(out["ctfidf_text"]) == ["gas nomination confirm volume"]


def test_boilerplate_stripped_case_preserved():
    frame = pd.DataFrame(
        {
            "Subject": ["Hello"],
            "body": ["--- Forwarded by Alice ---\nWorld [IMAGE] today"],
        }
    )
    out = prepare_messages(frame, redact_pii_enabled=False)
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
    out = prepare_messages(frame, redact_pii_enabled=False)
    assert len(out) == 1
    assert out.loc[0, "embed_text"] == "Hello! World."
    assert out.loc[0, "ctfidf_text"] == "hello world"


def test_exact_dedupe_prefers_earliest_date():
    frame = pd.DataFrame(
        {
            "Subject": ["Hello", "Hello!"],
            "body": ["World", "World."],
            "date_parsed": pd.to_datetime(["2001-02-01", "2001-01-01"]),
        }
    )
    out = prepare_messages(frame, redact_pii_enabled=False)
    assert len(out) == 1
    assert out.loc[0, "embed_text"] == "Hello! World."
    assert str(out.loc[0, "date_parsed"].date()) == "2001-01-01"


def test_word_shingles_short_and_overlapping():
    assert word_shingles("one two", k=5) == {"one two"}
    assert word_shingles("a b c d e f", k=5) == {"a b c d e", "b c d e f"}


def test_near_dup_keeps_first_of_lightly_edited_copy():
    # Shared 40-token prefix → Jaccard on 5-grams is well above 0.85.
    prefix = " ".join(f"token{i}" for i in range(40))
    base = f"{prefix} closing thanks team"
    near = f"{prefix} closing regards team"
    unrelated = (
        "lunch is at noon in the conference room please bring your laptop and "
        "the quarterly budget slides for review today"
    )
    frame = pd.DataFrame(
        {
            "ctfidf_text": [base, near, unrelated],
            "embed_text": ["A", "B", "C"],
        }
    )
    out = drop_near_duplicates(frame, threshold=0.85, shingle_size=5)
    assert list(out["embed_text"]) == ["A", "C"]


def test_prepare_messages_near_dup_collapses_edited_body():
    prefix = " ".join(f"token{i}" for i in range(40))
    frame = pd.DataFrame(
        {
            "Subject": ["Nom", "Nom"],
            "body": [f"{prefix} closing thanks", f"{prefix} closing regards"],
        }
    )
    out = prepare_messages(frame, redact_pii_enabled=False)
    assert len(out) == 1
    assert "token0" in out.loc[0, "embed_text"]
