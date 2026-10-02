# tests/test_text_clean.py
import pandas as pd

from email_cls_with_clustering.text_clean import (
    drop_aadvantage_promos,
    drop_near_duplicates,
    prepare_messages,
    preprocess_full_message,
    word_shingles,
)


def test_prepare_messages_strips_html_from_full_message_and_ctfidf():
    frame = pd.DataFrame(
        {
            "Subject": ["(None)", None],
            "body": [
                "<html><head><title>Untitled Document</title></head>"
                "<body><p>Hello team</p></body></html>",
                "Ask <user@enron.com> if price < 5. See <Review of Enron>.",
            ],
        }
    )
    out = prepare_messages(frame, redact_pii_enabled=False)
    html_row = out.loc[out["full_message"].str.contains("Untitled")].iloc[0]
    plain_row = out.loc[out["full_message"].str.contains("price")].iloc[0]
    words = html_row["ctfidf_text"].split()
    assert "<" not in html_row["full_message"]
    assert "<" not in html_row["body"]
    assert "Untitled Document" in html_row["full_message"]
    assert "Hello team" in html_row["full_message"]
    for tag in ("html", "head", "body", "title", "p"):
        assert tag not in words
    assert "untitled" in words
    assert "hello" in words
    assert "<user@enron.com>" in plain_row["full_message"]
    assert "price < 5" in plain_row["full_message"]
    assert "<Review of Enron>" in plain_row["full_message"]
    assert pd.isna(plain_row["Subject"])


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


def test_mixed_tz_aware_and_naive_dates_sort():
    """Date headers with/without offsets must not break earliest-wins dedupe."""
    from datetime import datetime, timezone

    frame = pd.DataFrame(
        {
            "Subject": ["Hello", "Hello!"],
            "body": ["World", "World."],
            "date_parsed": [
                datetime(2001, 2, 1, tzinfo=timezone.utc),
                datetime(2001, 1, 1),  # naive — as from parsedate without offset
            ],
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


def test_preprocess_strips_sponsored_phrases():
    msg = (
        "Click Baker Hughes AADE Global Completion Service today. "
        "Earn 20000 AAdvantage bonus miles and AAdvantagec bonus miles now."
    )
    cleaned = preprocess_full_message(msg)
    assert "aade global completion" not in cleaned
    assert "aadvantage" not in cleaned
    assert "baker hughes" in cleaned
    assert "service today" in cleaned
    assert "miles" in cleaned


def test_prepare_messages_drops_aadvantage_promos_and_keeps_business_mail():
    frame = pd.DataFrame(
        {
            "From": [
                "aairmail@info.aa.com",
                "feedback@travelocity.m0.net",
                "lsutaylor@hotmail.com",
                "marie.heard@enron.com",
                "newsletter@rigzone.com",
                "someone@usaa.com",
            ],
            "Subject": [
                "American Airlines AAirmail",
                "Real Deals from Travelocity.com",
                "Fwd: American Airlines Net SAAver Fares",
                "RE: American Airlines ISDA",
                "RIGZONE Industry News",
                "Insurance documents",
            ],
            "body": [
                "Welcome to the AAdvantage program.",
                "Book now and earn 20000 AAdvantage bonus miles.",
                "Domestic weekend getaway fares.",
                "Please review the American Airlines ISDA draft.",
                "Baker Hughes AADE Global Completion Service conference listing.",
                "Your USAA documents are ready.",
            ],
        }
    )
    out = prepare_messages(frame, redact_pii_enabled=False)
    subjects = set(out["Subject"])
    assert "American Airlines AAirmail" not in subjects
    assert "Real Deals from Travelocity.com" not in subjects
    assert "Fwd: American Airlines Net SAAver Fares" not in subjects
    assert "RE: American Airlines ISDA" in subjects
    assert "Insurance documents" in subjects
    assert "RIGZONE Industry News" in subjects
    rigzone = out.loc[out["Subject"] == "RIGZONE Industry News"].iloc[0]
    assert "aade global completion" not in rigzone["ctfidf_text"]
    assert "baker hughes" in rigzone["ctfidf_text"]
    assert "conference listing" in rigzone["ctfidf_text"]


def test_drop_aadvantage_promos_matches_amr_mailbox():
    frame = pd.DataFrame(
        {
            "From": ["aadvantage_customer_svc@amrcorp.com", "james.derrick@enron.com"],
            "Subject": ["Welcome to the AAdvantage Program", "Citibank AAdvantage Visa"],
            "body": ["Dear Susan, thank you for joining.", "customer service"],
        }
    )
    out = drop_aadvantage_promos(frame)
    assert out.empty
