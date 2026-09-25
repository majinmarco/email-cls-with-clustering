"""Text cleaning that feeds topic modeling.

Matches the notebook's ``full_message`` construction: subject plus body,
exact-row dedupe, then the regex cleaner.
"""

from __future__ import annotations

import re

import pandas as pd
from tqdm import tqdm

_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+")
_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}|\b\d{10,}\b"
)
_NOISE = re.compile(r"[-_]{2,}")
_SYMBOLS = re.compile(r"[^a-z0-9\s]")
_SPACE = re.compile(r"\s+")


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
    prepared["embed_text"] = (subject + " " + body).str.strip()

    # Drop genuinely empty rows first, so they do not all dedupe into one.
    prepared = prepared[prepared["embed_text"].str.len() > 0]
    prepared = prepared.drop_duplicates(subset=["embed_text"])

    messages = prepared["embed_text"].tolist()
    iterator = tqdm(messages, desc="clean ctfidf_text") if progress else messages
    prepared["ctfidf_text"] = [preprocess_full_message(msg) for msg in iterator]
    # Keep the old name as an alias so nothing downstream breaks yet.
    prepared["full_message"] = prepared["embed_text"]
    return prepared.reset_index(drop=True)
