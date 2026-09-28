"""Text cleaning that feeds topic modeling.

``embed_text`` keeps subject plus body, with forwarded-message banners and
image placeholders removed and without lowercasing or punctuation stripping.
``ctfidf_text`` is the scrubbed form of that string. Empty rows are dropped
before dedupe, and dedupe runs on the scrubbed text.
"""

from __future__ import annotations

import re

import pandas as pd
from tqdm import tqdm

_BOILERPLATE = re.compile(r"(?i)-+\s*forwarded by .*?-+|\[?IMAGE\]?")
_GAP = re.compile(r"[ \t]{2,}")
_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+")
_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}|\b\d{10,}\b"
)
_NOISE = re.compile(r"[-_]{2,}")
_SYMBOLS = re.compile(r"[^a-z0-9\s]")
_SPACE = re.compile(r"\s+")


def strip_boilerplate(msg: str) -> str:
    """Remove forwarded-message banners and image placeholders."""
    if not isinstance(msg, str):
        return ""
    cleaned = _GAP.sub(" ", _BOILERPLATE.sub(" ", msg))
    return cleaned.strip()


def preprocess_full_message(msg: str) -> str:
    """Lowercase and strip links, addresses, phones, and noisy symbols."""
    if not isinstance(msg, str):
        return ""
    msg = msg.lower()
    msg = _URL.sub("", msg)
    msg = _EMAIL.sub("", msg)
    msg = _PHONE.sub("", msg)
    msg = _NOISE.sub("", msg)
    msg = _SYMBOLS.sub("", msg)
    return _SPACE.sub(" ", msg).strip()


def prepare_messages(frame: pd.DataFrame, *, progress: bool = False) -> pd.DataFrame:
    """Build ``embed_text`` (raw) and ``ctfidf_text`` (scrubbed). Returns a new frame."""
    if "Subject" not in frame.columns or "body" not in frame.columns:
        raise KeyError("frame must include 'Subject' and 'body' columns")

    prepared = frame.copy()
    subject = prepared["Subject"].fillna("").astype(str).str.strip()
    body = prepared["body"].fillna("").astype(str).str.strip()
    combined = (subject + " " + body).map(strip_boilerplate)
    prepared["embed_text"] = combined

    # Drop genuinely empty rows first, so they do not all dedupe into one.
    prepared = prepared[prepared["embed_text"].str.len() > 0]

    messages = prepared["embed_text"].tolist()
    iterator = tqdm(messages, desc="clean ctfidf_text") if progress else messages
    prepared["ctfidf_text"] = [preprocess_full_message(msg) for msg in iterator]
    prepared = prepared.drop_duplicates(subset=["ctfidf_text"])
    # Keep the old name as an alias of the raw embedding text.
    prepared["full_message"] = prepared["embed_text"]
    return prepared.reset_index(drop=True)
