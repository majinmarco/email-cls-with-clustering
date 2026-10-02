from email_cls_with_clustering.preprocess import parse_email, strip_html


def test_strip_html_leaves_plain_text_unchanged():
    text = "Hello  team\n\nprice < 5 and volume > 10"
    assert strip_html(text) == text


def test_strip_html_keeps_angle_bracket_titles_and_addresses():
    text = "See <user@enron.com> re <Review of Enron North America>"
    assert strip_html(text) == text


def test_strip_html_document_drops_tags_and_style_keeps_visible_text():
    raw = (
        "<HTML>\n<HEAD>\n<TITLE>Untitled Document</TITLE>\n"
        "<style>body {color:red}</style>\n</HEAD>\n"
        "<BODY><font color=blue>Hello team</font></BODY></HTML>"
    )
    cleaned = strip_html(raw)
    assert "<" not in cleaned
    assert "color:red" not in cleaned
    assert "Untitled Document" in cleaned
    assert "Hello team" in cleaned
    for tag in ("html", "head", "title", "font", "body"):
        assert tag not in cleaned.lower().split()


def test_strip_html_inline_and_multiline_tags():
    raw = (
        "Dear Valued Subscriber,\n\n"
        "SIGN UP:\n"
        "<A\n"
        'HREF="https://example.com/trial">click here</A>\n'
        "Thanks"
    )
    cleaned = strip_html(raw)
    assert "<" not in cleaned
    assert "click here" in cleaned
    assert "Dear Valued Subscriber," in cleaned
    assert "Thanks" in cleaned


def test_strip_html_unescapes_then_strips_revealed_tags():
    cleaned = strip_html("Before &lt;b&gt;bold&lt;/b&gt; after")
    assert "<" not in cleaned
    assert "bold" in cleaned
    assert "Before" in cleaned and "after" in cleaned


def test_plain_part_that_is_html_does_not_leak_tags():
    raw = """From: a@enron.com
To: b@enron.com
Subject: error
MIME-Version: 1.0
Content-Type: text/plain; charset="us-ascii"

<HTML><HEAD><TITLE>Server Error</TITLE></HEAD>
<BODY><H1>Server Error</H1>
This server has encountered an internal error.
</BODY></HTML>
"""
    record = parse_email(raw)
    assert "<" not in record.body
    assert "<" not in record.body_raw
    assert "Server Error" in record.body
    assert "internal error" in record.body
