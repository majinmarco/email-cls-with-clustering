from datetime import datetime, timezone

import pandas as pd
import pytest

from email_cls_with_clustering.topics import THREAD_JOIN_SEP, join_by_thread


def _msgs(**kwargs):
    return pd.DataFrame(kwargs)


def test_join_order_is_chronological_by_date():
    base = datetime(2001, 1, 1, tzinfo=timezone.utc)
    frame = _msgs(
        thread_id=[1, 1, 1],
        date_parsed=[base.replace(day=3), base.replace(day=1), base.replace(day=2)],
        embed_text=["third", "first", "second"],
        ctfidf_text=["c3", "c1", "c2"],
    )
    out = join_by_thread(frame)
    assert len(out) == 1
    assert out.loc[0, "n_messages"] == 3
    assert out.loc[0, "embed_text"] == THREAD_JOIN_SEP.join(["first", "second", "third"])
    assert out.loc[0, "ctfidf_text"] == THREAD_JOIN_SEP.join(["c1", "c2", "c3"])


def test_singleton_thread_unchanged():
    frame = _msgs(
        thread_id=[7],
        date_parsed=[datetime(2001, 1, 1, tzinfo=timezone.utc)],
        embed_text=["only body"],
        ctfidf_text=["only clean"],
        subj_norm=["budget"],
    )
    out = join_by_thread(frame)
    assert len(out) == 1
    assert out.loc[0, "thread_id"] == 7
    assert out.loc[0, "n_messages"] == 1
    assert out.loc[0, "embed_text"] == "only body"
    assert out.loc[0, "ctfidf_text"] == "only clean"
    assert out.loc[0, "subj_norm"] == "budget"


def test_multi_message_concatenates_both_text_columns():
    frame = _msgs(
        thread_id=[2, 2],
        date_parsed=[
            datetime(2001, 1, 1, tzinfo=timezone.utc),
            datetime(2001, 1, 2, tzinfo=timezone.utc),
        ],
        embed_text=["Subject A body a", "Subject A body b"],
        ctfidf_text=["clean a", "clean b"],
    )
    out = join_by_thread(frame)
    assert out.loc[0, "embed_text"] == f"Subject A body a{THREAD_JOIN_SEP}Subject A body b"
    assert out.loc[0, "ctfidf_text"] == f"clean a{THREAD_JOIN_SEP}clean b"
    # Columns stay separate: embed join must not appear in ctfidf and vice versa.
    assert "Subject A" not in out.loc[0, "ctfidf_text"]
    assert "clean" not in out.loc[0, "embed_text"]


def test_default_path_does_not_join():
    """Without calling join_by_thread, rows stay one document each (fit_topics input)."""
    frame = _msgs(
        thread_id=[1, 1],
        embed_text=["a", "b"],
        ctfidf_text=["x", "y"],
    )
    assert len(frame) == 2
    assert list(frame["embed_text"]) == ["a", "b"]


def test_missing_thread_id_raises():
    frame = _msgs(embed_text=["a"], ctfidf_text=["b"])
    with pytest.raises(KeyError, match="thread_id"):
        join_by_thread(frame)


def test_null_thread_id_raises():
    frame = _msgs(
        thread_id=[1, None],
        embed_text=["a", "b"],
        ctfidf_text=["x", "y"],
    )
    with pytest.raises(ValueError, match="missing values"):
        join_by_thread(frame)


def test_join_keeps_earliest_message_metadata():
    early = datetime(2001, 1, 1, tzinfo=timezone.utc)
    late = datetime(2001, 1, 3, tzinfo=timezone.utc)
    frame = _msgs(
        thread_id=[1, 1],
        date_parsed=[late, early],
        embed_text=["later", "earlier"],
        ctfidf_text=["c2", "c1"],
        **{
            "From": ["second@enron.com", "first@enron.com"],
            "To": ["b@enron.com", "a@enron.com"],
            "Date": ["Tue, 3 Jan 2001", "Mon, 1 Jan 2001"],
        },
    )
    out = join_by_thread(frame)
    assert out.loc[0, "date_parsed"] == early
    assert out.loc[0, "Date"] == "Mon, 1 Jan 2001"
    assert out.loc[0, "From"] == "first@enron.com"
    assert out.loc[0, "To"] == "a@enron.com"


def test_stable_order_without_date_parsed():
    frame = _msgs(
        thread_id=[9, 9, 9],
        embed_text=["first", "second", "third"],
        ctfidf_text=["c1", "c2", "c3"],
    )
    out = join_by_thread(frame)
    assert out.loc[0, "embed_text"] == THREAD_JOIN_SEP.join(["first", "second", "third"])
