"""Enron-style email thread reconstruction (subject + participants + window)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from email.utils import getaddresses

import pandas as pd

_SUBJECT_PREFIX = re.compile(
    r"^\s*((re|fw|fwd|rv|res)\s*:\s*)+",
    re.IGNORECASE,
)

# Caps that trip on the known Enron mega-threads (petitions, digests, admin blasts).
DEFAULT_MAX_UNIQUE_FROM = 3
DEFAULT_MAX_UNIQUE_TO = 8
# Same sender (≤2 Froms) repeating a subject to a fixed list — Williams digests, etc.
DEFAULT_MAX_SINGLE_SENDER_SIZE = 10


def normalize_subject(subject: str) -> str:
    """Strip reply/forward prefixes and lower-case for thread matching."""
    if subject is None or (isinstance(subject, float) and pd.isna(subject)):
        return ""
    text = str(subject)
    text = _SUBJECT_PREFIX.sub("", text)
    return text.strip().lower()


def parse_address_list(value) -> list[str]:
    """Parse a header value into normalized address tokens."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    out: list[str] = []
    for _name, addr in getaddresses([raw]):
        if addr:
            out.append(addr.lower())
        else:
            token = (_name or "").strip().lower()
            if token:
                out.append(token)
    return out


def participants_for_row(from_value, to_value, cc_value) -> frozenset[str]:
    """Build the participant set for one message row."""
    parts: list[str] = []
    parts.extend(parse_address_list(from_value))
    parts.extend(parse_address_list(to_value))
    parts.extend(parse_address_list(cc_value))
    return frozenset(parts)


def is_self_mail(from_value, to_value) -> bool:
    """True when a single From address mails only itself (crawler / bounce loops)."""
    senders = parse_address_list(from_value)
    recipients = parse_address_list(to_value)
    if len(senders) != 1 or not recipients:
        return False
    return set(senders) == set(recipients)


def _normalize_message_id(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().lower()


@dataclass
class _ThreadState:
    thread_id: int
    last_date: object
    participants: frozenset[str]
    froms: set[str] = field(default_factory=set)
    tos: set[str] = field(default_factory=set)
    self_mail: bool = False
    size: int = 1


def _blast_blocks_merge(
    state: _ThreadState,
    *,
    new_froms: set[str],
    new_tos: set[str],
    new_is_self_mail: bool,
    max_unique_from: int,
    max_unique_to: int,
    max_single_sender_size: int,
    split_self_mail: bool,
) -> bool:
    """Return True when joining this message would look like a mass blast."""
    if split_self_mail and (state.self_mail or new_is_self_mail):
        return True
    # Inbound fan-in: many distinct senders, typical of petition storms.
    if len(state.froms) >= max_unique_from and (new_froms - state.froms):
        return True
    # Outbound fan-out: one sender (or few) to many distinct recipients.
    if len(state.tos) >= max_unique_to and (new_tos - state.tos):
        return True
    # Fixed-list digest: same 1–2 senders keep posting the same subject.
    if (
        len(state.froms) <= 2
        and new_froms
        and new_froms <= state.froms
        and state.size >= max_single_sender_size
    ):
        return True
    return False


def assign_thread_ids(
    frame: pd.DataFrame,
    *,
    window_days: int = 14,
    date_col: str = "date_parsed",
    subject_col: str = "Subject",
    from_col: str = "From",
    to_col: str = "To",
    cc_col: str = "Cc",
    in_reply_col: str = "In-Reply-To",
    message_id_col: str = "Message-ID",
    max_unique_from: int = DEFAULT_MAX_UNIQUE_FROM,
    max_unique_to: int = DEFAULT_MAX_UNIQUE_TO,
    max_single_sender_size: int = DEFAULT_MAX_SINGLE_SENDER_SIZE,
    split_self_mail: bool = True,
) -> pd.DataFrame:
    """Assign ``subj_norm`` and ``thread_id``; preserve input row order.

    Subject + participant overlap + time window, with blast guards:
    - self-mail (From == To) never joins via the subject heuristic
    - inbound fan-in stops once a thread already has ``max_unique_from`` senders
    - outbound fan-out stops once a thread already has ``max_unique_to`` recipients
    - same 1–2 senders repeating a subject stop after ``max_single_sender_size``
    Subjects that trip a guard are locked so later blasts stay singletons.
    ``In-Reply-To`` links still win when present.
    """
    out = frame.copy()
    out["_row"] = range(len(out))

    if date_col not in out.columns or from_col not in out.columns:
        return out.drop(columns=["_row"])

    subjects = (
        out[subject_col].fillna("").astype(str)
        if subject_col in out.columns
        else pd.Series([""] * len(out), index=out.index)
    )
    out["subj_norm"] = subjects.map(normalize_subject)

    from_sets: dict[object, set[str]] = {}
    to_sets: dict[object, set[str]] = {}
    participant_sets: dict[object, frozenset[str]] = {}
    self_mail_flags: dict[object, bool] = {}
    for idx in out.index:
        from_val = out.at[idx, from_col]
        to_val = out.at[idx, to_col] if to_col in out.columns else None
        cc_val = out.at[idx, cc_col] if cc_col in out.columns else None
        from_sets[idx] = set(parse_address_list(from_val))
        to_sets[idx] = set(parse_address_list(to_val)) | set(parse_address_list(cc_val))
        participant_sets[idx] = participants_for_row(from_val, to_val, cc_val)
        self_mail_flags[idx] = is_self_mail(from_val, to_val)

    sorted_idx = out.sort_values(date_col, kind="mergesort", na_position="last").index
    thread_by_index: dict[object, int] = {}
    last_seen: dict[str, _ThreadState] = {}
    blast_locked: set[str] = set()
    thread_state: dict[int, _ThreadState] = {}
    msg_id_to_thread: dict[str, int] = {}

    has_reply_headers = (
        in_reply_col in out.columns and message_id_col in out.columns
    )

    for idx in sorted_idx:
        row = out.loc[idx]
        key = row["subj_norm"]
        participants = participant_sets[idx]
        new_froms = from_sets[idx]
        new_tos = to_sets[idx]
        new_self = self_mail_flags[idx]
        date = row[date_col]
        assigned: int | None = None

        if has_reply_headers:
            in_reply = _normalize_message_id(row.get(in_reply_col))
            if in_reply and in_reply in msg_id_to_thread:
                assigned = msg_id_to_thread[in_reply]

        if assigned is None and key and key not in blast_locked:
            prev = last_seen.get(key)
            if (
                prev is not None
                and not pd.isna(date)
                and not pd.isna(prev.last_date)
                and (date - prev.last_date).days <= window_days
                and bool(participants & prev.participants)
            ):
                if _blast_blocks_merge(
                    prev,
                    new_froms=new_froms,
                    new_tos=new_tos,
                    new_is_self_mail=new_self,
                    max_unique_from=max_unique_from,
                    max_unique_to=max_unique_to,
                    max_single_sender_size=max_single_sender_size,
                    split_self_mail=split_self_mail,
                ):
                    blast_locked.add(key)
                else:
                    assigned = prev.thread_id

        if assigned is None:
            assigned = int(idx)

        thread_by_index[idx] = assigned

        if assigned in thread_state:
            state = thread_state[assigned]
            state.last_date = date
            state.participants = participants
            state.froms |= new_froms
            state.tos |= new_tos
            state.self_mail = state.self_mail and new_self
            state.size += 1
        else:
            state = _ThreadState(
                thread_id=assigned,
                last_date=date,
                participants=participants,
                froms=set(new_froms),
                tos=set(new_tos),
                self_mail=new_self,
                size=1,
            )
            thread_state[assigned] = state

        if key:
            last_seen[key] = state

        if has_reply_headers:
            mid = _normalize_message_id(row.get(message_id_col))
            if mid:
                msg_id_to_thread[mid] = assigned

    out["thread_id"] = out.index.map(thread_by_index)
    return out.sort_values("_row", kind="mergesort").drop(columns=["_row"])
