"""Text cleaning that feeds topic modeling.

``embed_text`` keeps subject plus body, with forwarded-message banners and
image placeholders removed and without lowercasing or punctuation stripping.
Optional Presidio PII redaction (see ``pii``) runs on that string before
dedupe and embeddings. ``ctfidf_text`` is the scrubbed form of the same
string. Empty rows are dropped before dedupe. Exact duplicates are removed on
scrubbed text, then MinHash LSH drops near-duplicates (word shingles).
"""

from __future__ import annotations

import re

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from email_cls_with_clustering.pii import redact as redact_pii
from email_cls_with_clustering.threads import assign_thread_ids

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

DEFAULT_NEAR_DUP_THRESHOLD = 0.85
DEFAULT_NUM_PERM = 128
DEFAULT_SHINGLE_SIZE = 5
# Favor recall over precision so Jaccard≈0.9 pairs are not missed at 0.85.
DEFAULT_LSH_WEIGHTS = (0.2, 0.8)


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


def word_shingles(text: str, k: int = DEFAULT_SHINGLE_SIZE) -> set[str]:
    """Return overlapping k-word shingles; short texts become a single shingle."""
    words = text.split()
    if not words:
        return set()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def drop_near_duplicates(
    frame: pd.DataFrame,
    *,
    text_col: str = "ctfidf_text",
    threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    weights: tuple[float, float] = DEFAULT_LSH_WEIGHTS,
    progress: bool = False,
) -> pd.DataFrame:
    """Keep the first row of each near-duplicate cluster (MinHash LSH).

    Rows should already be ordered so the preferred survivor comes first
    (e.g. earliest ``date_parsed``). Jaccard similarity is estimated on
    ``shingle_size``-word shingles of ``text_col``.
    """
    if text_col not in frame.columns:
        raise KeyError(f"frame must include '{text_col}' column")
    if frame.empty:
        return frame.reset_index(drop=True)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, weights=weights)
    keep: list[int] = []
    texts = frame[text_col].fillna("").astype(str).tolist()
    positions = range(len(texts))
    iterator = tqdm(positions, desc="near-dup dedupe") if progress else positions

    for pos in iterator:
        m = MinHash(num_perm=num_perm)
        shingles = word_shingles(texts[pos], shingle_size)
        if shingles:
            m.update_batch([s.encode("utf-8") for s in shingles])
        if lsh.query(m):
            continue
        lsh.insert(str(pos), m)
        keep.append(pos)

    return frame.iloc[keep].reset_index(drop=True)


def prepare_messages(
    frame: pd.DataFrame,
    *,
    progress: bool = False,
    redact_pii_enabled: bool = True,
    pii_lang: str = "en",
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    lsh_weights: tuple[float, float] = DEFAULT_LSH_WEIGHTS,
) -> pd.DataFrame:
    """Build ``embed_text`` (raw) and ``ctfidf_text`` (scrubbed).

    When ``redact_pii_enabled`` is true, Presidio redaction runs on the
    subject+body string after boilerplate stripping and before dedupe/embeddings.
    When ``date_parsed`` and ``From`` exist, adds ``subj_norm`` and ``thread_id``.
    Returns a new frame.
    """
    if "Subject" not in frame.columns or "body" not in frame.columns:
        raise KeyError("frame must include 'Subject' and 'body' columns")

    prepared = frame.copy()
    subject = prepared["Subject"].fillna("").astype(str).str.strip()
    body = prepared["body"].fillna("").astype(str).str.strip()
    combined = (subject + " " + body).map(strip_boilerplate)
    if redact_pii_enabled:
        iterator_pii = tqdm(combined, desc="PII redaction") if progress else combined
        combined = pd.Series(
            [redact_pii(msg, lang=pii_lang) for msg in iterator_pii],
            index=combined.index,
        )
    prepared["embed_text"] = combined

    # Drop genuinely empty rows first, so they do not all dedupe into one.
    prepared = prepared[prepared["embed_text"].str.len() > 0]

    messages = prepared["embed_text"].tolist()
    iterator = tqdm(messages, desc="clean ctfidf_text") if progress else messages
    prepared["ctfidf_text"] = [preprocess_full_message(msg) for msg in iterator]

    # Prefer the earliest copy when dates exist (Enron-style folder duplicates).
    if "date_parsed" in prepared.columns:
        prepared = prepared.sort_values(
            "date_parsed", kind="mergesort", na_position="last"
        )

    prepared = prepared.drop_duplicates(subset=["ctfidf_text"])
    prepared = drop_near_duplicates(
        prepared,
        text_col="ctfidf_text",
        threshold=near_dup_threshold,
        num_perm=num_perm,
        shingle_size=shingle_size,
        weights=lsh_weights,
        progress=progress,
    )
    # Keep the old name as an alias of the raw embedding text.
    prepared["full_message"] = prepared["embed_text"]
    prepared = assign_thread_ids(prepared)
    return prepared.reset_index(drop=True)
