# tests/test_text_clean.py
import pandas as pd

from email_cls_with_clustering.text_clean import prepare_messages


def test_missing_subject_does_not_collapse_rows():
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
    assert set(out["full_message"]) == {
        "gas nomination confirm volume",
        "call me about the deal",
        "lunch at noon",
        "the contract is signed",
    }


def test_fully_empty_messages_are_dropped():
    frame = pd.DataFrame({"Subject": ["Gas nomination", None], "body": ["confirm volume", None]})
    assert list(prepare_messages(frame)["full_message"]) == ["gas nomination confirm volume"]