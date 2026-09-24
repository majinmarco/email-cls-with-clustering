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
    """Build, dedupe, and clean ``full_message``. Returns a new frame."""
    if "Subject" not in frame.columns or "body" not in frame.columns:
        raise KeyError("frame must include 'Subject' and 'body' columns")

    prepared = frame.copy()
    prepared["full_message"] = prepared["Subject"] + " " + prepared["body"]
    prepared = prepared.drop_duplicates(subset=["full_message"])

    messages = prepared["full_message"].tolist()
    iterator = tqdm(messages, desc="clean full_message") if progress else messages
    prepared["full_message"] = [preprocess_full_message(msg) for msg in iterator]
    return prepared.reset_index(drop=True)
