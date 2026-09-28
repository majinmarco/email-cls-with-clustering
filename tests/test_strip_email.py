# tests/test_strip_email.py
from email_cls_with_clustering.preprocess import strip_email


def test_strip_email_cuts_original_message_block():
    body = "Deal is done.\n\n-----Original Message-----\nFrom: bob@x.com\nSent: Monday\nOld thread"
    assert strip_email(body) == "Deal is done."


def test_strip_email_drops_quote_lines_and_disclaimer():
    body = (
        "See below.\n"
        "> prior reply\n"
        "This email is confidential and intended only for the recipient.*****"
    )
    cleaned = strip_email(body)
    assert "prior reply" not in cleaned
    assert "confidential" not in cleaned.lower()
    assert "See below." in cleaned


def test_strip_email_soft_signature():
    body = "Approved.\n\nThanks,\nAlice\nTrader"
    assert strip_email(body) == "Approved."
