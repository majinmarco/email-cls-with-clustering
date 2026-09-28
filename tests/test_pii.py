# tests/test_pii.py
from unittest.mock import MagicMock, patch

import pandas as pd

from email_cls_with_clustering import pii
from email_cls_with_clustering.text_clean import prepare_messages


def test_redact_empty_and_whitespace():
    assert pii.redact("") == ""
    assert pii.redact("   ") == "   "


@patch("email_cls_with_clustering.pii._engines")
def test_redact_calls_presidio(mock_engines):
    analyzer = MagicMock()
    anonymizer = MagicMock()
    mock_engines.return_value = (analyzer, anonymizer)
    analyzer.analyze.return_value = [{"entity": "PERSON"}]
    anonymizer.anonymize.return_value = MagicMock(text="Call <PERSON> at <PHONE>")

    out = pii.redact("Call Jeff at 713-555-0100", lang="en")

    analyzer.analyze.assert_called_once()
    call_kw = analyzer.analyze.call_args.kwargs
    assert call_kw["text"] == "Call Jeff at 713-555-0100"
    assert call_kw["language"] == "en"
    assert "PERSON" in call_kw["entities"]
    anonymizer.anonymize.assert_called_once()
    assert out == "Call <PERSON> at <PHONE>"


def test_prepare_messages_redacts_when_enabled():
    frame = pd.DataFrame(
        {
            "Subject": ["Contact"],
            "body": ["jeff.skilling@enron.com"],
        }
    )
    with patch(
        "email_cls_with_clustering.text_clean.redact_pii",
        side_effect=lambda msg, lang="en": msg.replace("jeff.skilling@enron.com", "<EMAIL>"),
    ) as mock_redact:
        out = prepare_messages(frame, redact_pii_enabled=True, pii_lang="en")
    mock_redact.assert_called()
    assert "<EMAIL>" in out.loc[0, "embed_text"]
    assert "jeff.skilling@enron.com" not in out.loc[0, "embed_text"]


def test_prepare_messages_skips_redaction_when_disabled():
    frame = pd.DataFrame(
        {
            "Subject": ["Contact"],
            "body": ["jeff.skilling@enron.com"],
        }
    )
    with patch("email_cls_with_clustering.text_clean.redact_pii") as mock_redact:
        out = prepare_messages(frame, redact_pii_enabled=False)
    mock_redact.assert_not_called()
    assert "jeff.skilling@enron.com" in out.loc[0, "embed_text"]
