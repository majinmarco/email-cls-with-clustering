from datetime import datetime

import pandas as pd

from email_cls_with_clustering.threads import (
    assign_thread_ids,
    normalize_subject,
    parse_address_list,
    participants_for_row,
)


def test_normalize_subject_strips_stacked_prefixes():
    assert normalize_subject("Re: Re: Budget") == "budget"
    assert normalize_subject("FW: Fwd: RV: RES: Hello") == "hello"
    assert normalize_subject("  re: fw: Topic  ") == "topic"


def test_parse_address_list_display_and_bare():
    assert parse_address_list("Alice <Alice@Example.COM>, bob@corp.com") == [
        "alice@example.com",
        "bob@corp.com",
    ]
    assert parse_address_list(None) == []
    assert parse_address_list("") == []


def test_participants_for_row_union():
    fs = participants_for_row(
        "Alice <a@b.com>",
        "bob@corp.com, Carol <c@d.com>",
        None,
    )
    assert fs == frozenset({"a@b.com", "bob@corp.com", "c@d.com"})


def _frame(rows):
    return pd.DataFrame(rows)


def test_same_subject_overlap_within_window_same_thread():
    base = datetime(2001, 1, 1)
    frame = _frame(
        {
            "Subject": ["Budget Q1", "Re: Budget Q1"],
            "From": ["alice@x.com", "bob@x.com"],
            "To": ["bob@x.com", "alice@x.com"],
            "Cc": [None, None],
            "date_parsed": [base, base.replace(day=10)],
        }
    )
    out = assign_thread_ids(frame)
    assert out.loc[0, "thread_id"] == out.loc[1, "thread_id"]
    assert out.loc[0, "subj_norm"] == "budget q1"
    assert out.loc[1, "subj_norm"] == "budget q1"


def test_same_subject_beyond_window_different_threads():
    frame = _frame(
        {
            "Subject": ["Budget", "Re: Budget"],
            "From": ["a@x.com", "b@x.com"],
            "To": ["b@x.com", "a@x.com"],
            "date_parsed": pd.to_datetime(["2001-01-01", "2001-02-01"]),
        }
    )
    out = assign_thread_ids(frame)
    assert out.loc[0, "thread_id"] != out.loc[1, "thread_id"]


def test_same_subject_no_participant_overlap_different_threads():
    frame = _frame(
        {
            "Subject": ["Hello", "Re: Hello"],
            "From": ["a@x.com", "c@x.com"],
            "To": ["b@x.com", "d@x.com"],
            "date_parsed": pd.to_datetime(["2001-01-01", "2001-01-05"]),
        }
    )
    out = assign_thread_ids(frame)
    assert out.loc[0, "thread_id"] != out.loc[1, "thread_id"]


def test_empty_subject_singleton_threads():
    frame = _frame(
        {
            "Subject": ["", ""],
            "From": ["a@x.com", "a@x.com"],
            "To": ["b@x.com", "b@x.com"],
            "date_parsed": pd.to_datetime(["2001-01-01", "2001-01-02"]),
        }
    )
    out = assign_thread_ids(frame)
    assert out.loc[0, "thread_id"] != out.loc[1, "thread_id"]


def test_assign_thread_ids_preserves_input_row_order():
    frame = _frame(
        {
            "Subject": ["Later", "Earlier"],
            "From": ["a@x.com", "a@x.com"],
            "To": ["b@x.com", "b@x.com"],
            "date_parsed": pd.to_datetime(["2001-01-10", "2001-01-01"]),
        }
    )
    out = assign_thread_ids(frame)
    assert list(out["Subject"]) == ["Later", "Earlier"]


def test_in_reply_to_links_over_subject_heuristic():
    frame = _frame(
        {
            "Subject": ["Topic A", "Unrelated subject"],
            "From": ["a@x.com", "z@y.com"],
            "To": ["b@x.com", "w@y.com"],
            "date_parsed": pd.to_datetime(["2001-01-01", "2001-01-02"]),
            "Message-ID": ["<msg-1>", "<msg-2>"],
            "In-Reply-To": [None, "<msg-1>"],
        }
    )
    out = assign_thread_ids(frame)
    assert out.loc[1, "thread_id"] == out.loc[0, "thread_id"]


def test_skip_threading_without_from_or_date():
    frame = pd.DataFrame({"Subject": ["Hi"], "body": ["there"]})
    out = assign_thread_ids(frame)
    assert "thread_id" not in out.columns

    frame2 = pd.DataFrame(
        {"Subject": ["Hi"], "From": ["a@x.com"], "body": ["there"]}
    )
    out2 = assign_thread_ids(frame2)
    assert "thread_id" not in out2.columns


def test_self_mail_blast_stays_singleton():
    frame = _frame(
        {
            "Subject": ["Schedule Crawler: HourAhead Failure"] * 3,
            "From": ["pete.davis@enron.com"] * 3,
            "To": ["pete.davis@enron.com"] * 3,
            "date_parsed": pd.to_datetime(
                ["2001-01-01", "2001-01-02", "2001-01-03"]
            ),
        }
    )
    out = assign_thread_ids(frame)
    assert out["thread_id"].nunique() == 3


def test_inbound_petition_blast_caps_unique_from():
    # Many strangers → one inbox (klay-style petition storm).
    n = 10
    frame = _frame(
        {
            "Subject": ["Demand Ken Lay Donate Bonus"] * n,
            "From": [f"user{i}@example.com" for i in range(n)],
            "To": ["klay@enron.com"] * n,
            "date_parsed": pd.to_datetime(
                [f"2001-01-{(i % 28) + 1:02d}" for i in range(n)]
            ),
        }
    )
    out = assign_thread_ids(frame, max_unique_from=3)
    sizes = out.groupby("thread_id").size()
    assert sizes.max() <= 3
    assert out["thread_id"].nunique() >= 4


def test_outbound_newsletter_blast_caps_unique_to():
    n = 12
    frame = _frame(
        {
            "Subject": ["Williams Energy News Live"] * n,
            "From": ["news@williams.com"] * n,
            "To": [f"reader{i}@enron.com" for i in range(n)],
            "date_parsed": pd.to_datetime(
                [f"2001-01-{(i % 28) + 1:02d}" for i in range(n)]
            ),
        }
    )
    out = assign_thread_ids(frame, max_unique_to=4)
    sizes = out.groupby("thread_id").size()
    assert sizes.max() <= 4
    assert out["thread_id"].nunique() >= 3


def test_fixed_list_digest_caps_single_sender_size():
    # Same From → same list address repeatedly (Williams-style digest).
    n = 20
    frame = _frame(
        {
            "Subject": ["Williams Energy News Live"] * n,
            "From": ["news@williams.com"] * n,
            "To": ["enl-members@williams.com"] * n,
            "date_parsed": pd.to_datetime(
                [f"2001-01-{(i % 28) + 1:02d}" for i in range(n)]
            ),
        }
    )
    out = assign_thread_ids(frame, max_single_sender_size=5)
    sizes = out.groupby("thread_id").size()
    assert sizes.max() <= 5
    assert out["thread_id"].nunique() >= 4


def test_normal_reply_chain_still_merges_with_blast_guards():
    base = datetime(2001, 1, 1)
    frame = _frame(
        {
            "Subject": ["Budget Q1", "Re: Budget Q1", "Re: Budget Q1"],
            "From": ["alice@x.com", "bob@x.com", "alice@x.com"],
            "To": ["bob@x.com", "alice@x.com", "bob@x.com"],
            "date_parsed": [
                base,
                base.replace(day=2),
                base.replace(day=3),
            ],
        }
    )
    out = assign_thread_ids(frame)
    assert out["thread_id"].nunique() == 1
