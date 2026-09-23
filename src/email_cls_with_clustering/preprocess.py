"""Parse raw RFC822 messages into a frame the analysis notebook can use.

Adapted from the messy-email preprocess lambda's MIME parsing. S3, metrics,
and tracing stay out. Body text keeps newlines so Talon can strip quoted
replies and signatures.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime
from html import unescape
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

_TALON: tuple[Any, Any] | None = None

_BLOCK_END = re.compile(
    r"(?i)<\s*br\s*/?\s*>|</\s*(?:p|div|tr|li|h[1-6]|blockquote|table)\s*>"
)
_TAG = re.compile(r"<[^>]+>")
_SPACE_RUN = re.compile(r"[^\S\n]+")
_BLANK_RUN = re.compile(r"\n{3,}")


# Header spellings that show up in this corpus. Any other header is kept
# under the name the message actually uses.
_HEADER_ORDER = (
    "Message-ID",
    "Date",
    "From",
    "To",
    "Cc",
    "Bcc",
    "Subject",
    "Reply-To",
    "In-Reply-To",
    "Mime-Version",
    "Content-Type",
    "Content-Transfer-Encoding",
    "X-From",
    "X-To",
    "X-cc",
    "X-bcc",
    "X-Folder",
    "X-Origin",
    "X-FileName",
)
_CANONICAL_HEADERS = {name.casefold(): name for name in _HEADER_ORDER}
_DERIVED_COLUMNS = (
    "date_parsed",
    "body_raw",
    "body",
    "signature",
    "attachment_count",
    "attachment_names",
    "parse_error",
)


@dataclass
class ParsedEmail:
    file: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    date_parsed: datetime | None = None
    body_raw: str = ""
    body: str = ""
    signature: str = ""
    attachment_count: int = 0
    attachment_names: list[str] = field(default_factory=list)
    parse_error: str = ""


def parse_email(
    raw: str | bytes,
    *,
    file: str = "",
    save_dir: Path | None = None,
) -> ParsedEmail:
    """Parse one message. Failures stay on the row as ``parse_error``."""
    record = ParsedEmail(file=file)
    if not isinstance(raw, (str, bytes)) or not raw:
        record.parse_error = "empty message"
        return record

    try:
        msg = BytesParser(policy=policy.default).parse(BytesIO(_as_bytes(raw)))
        record.headers = _headers(msg)
        record.date_parsed = _parse_date(record.headers.get("Date", ""))
        record.body_raw = extract_body_text(msg)
        record.body, record.signature = clean_body(
            record.body_raw, _sender(record.headers.get("From", ""))
        )
        names = extract_attachments(msg, mail_key=_mail_key(file), save_dir=save_dir)
        record.attachment_names = names
        record.attachment_count = len(names)
    except Exception as exc:
        record.parse_error = f"{type(exc).__name__}: {exc}"
    return record


def expand_emails(frame: pd.DataFrame, *, save_attachments: bool = False) -> pd.DataFrame:
    """Expand a ``file`` / ``message`` frame into one row per parsed email.

    Every header on a message becomes its own column (``From``, ``To``,
    ``X-Folder``, ...). A header that is absent on a row is left empty.
    Attachment bytes are written under ``./attachments`` only when
    ``save_attachments`` is true.
    """
    if "message" not in frame.columns:
        raise KeyError("frame must include a 'message' column")

    save_dir = None
    if save_attachments:
        save_dir = Path("attachments")
        save_dir.mkdir(parents=True, exist_ok=True)

    files = frame["file"] if "file" in frame.columns else [""] * len(frame)
    records = [
        _flat_record(
            parse_email(
                message,
                file="" if file is None else str(file),
                save_dir=save_dir,
            )
        )
        for file, message in zip(files, frame["message"], strict=True)
    ]
    expanded = pd.DataFrame.from_records(records)
    return expanded.reindex(columns=_column_order(expanded.columns))


def extract_body_text(msg) -> str:
    """Plain text wins. HTML is converted only when the text part is empty."""
    text, html = _bodies_via_policy(msg)
    if text is None and html is None:
        text, html = _bodies_via_walk(msg)

    plain = _as_str(text).strip()
    if plain:
        return _as_str(text).strip()
    if html:
        return _html_to_text(_as_str(html))
    return ""


def clean_body(text: str, sender: str) -> tuple[str, str]:
    """Drop quoted replies, then the signature. Returns ``(body, signature)``."""
    if not text or not text.strip():
        return "", ""

    try:
        signature_mod, quotations = _talon()
        reply = quotations.extract_from(text, "text/plain")
        if not isinstance(reply, str) or not reply.strip():
            reply = text

        cleaned, sig = signature_mod.extract(reply, sender or "")
        if not isinstance(cleaned, str):
            cleaned = reply
        if sig:
            return cleaned.strip(), _as_str(sig).strip()

        from talon.signature.bruteforce import extract_signature

        bf_body, bf_sig = extract_signature(cleaned)
        if bf_sig:
            body = bf_body if isinstance(bf_body, str) else cleaned
            return body.strip(), _as_str(bf_sig).strip()
        return cleaned.strip(), ""
    except Exception:
        return text.strip(), ""


def extract_attachments(msg, *, mail_key: str, save_dir: Path | None) -> list[str]:
    """Keep real attachments. Inline parts stay out except ``.rpmsg``."""
    names: list[str] = []
    counts: dict[str, int] = {}
    index = 0

    for part in msg.walk():
        if not _is_attachment(part):
            continue
        index += 1
        filename = part.get_filename()
        if not filename:
            filename = f"attachment_{index}{_extension(part)}"
        filename = _unique_name(sanitize_filename(filename), counts)
        names.append(filename)

        if save_dir is None:
            continue
        content = _part_bytes(part)
        if content is None:
            continue
        target = save_dir / f"{mail_key}_{index}_{filename}"
        target.write_bytes(content)

    return names


def sanitize_filename(filename: str) -> str:
    """Drop path components and characters that are unsafe on disk."""
    if not filename:
        return "unnamed_attachment"

    name = Path(filename).name
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\.{2,}", ".", name).strip(". ")
    ascii_name = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    )
    safe = "".join(c for c in ascii_name if c.isalnum() or c in "._- ")
    safe = re.sub(r"\s+", "_", safe).strip("._")[:100] or "unnamed_attachment"

    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{n}" for n in range(1, 10)),
        *(f"LPT{n}" for n in range(1, 10)),
    }
    if safe.split(".", 1)[0].upper() in reserved:
        safe = f"file_{safe}"
    return safe


def _talon() -> tuple[Any, Any]:
    global _TALON
    if _TALON is None:
        from email_cls_with_clustering.talon_compat import init_talon

        _TALON = init_talon()
    return _TALON


def _bodies_via_policy(msg) -> tuple[str | None, str | None]:
    try:
        text_part = msg.get_body(preferencelist=("plain",))
        html_part = msg.get_body(preferencelist=("html",))
        text = text_part.get_content() if text_part is not None else None
        html = html_part.get_content() if html_part is not None else None
        return text, html
    except Exception:
        return None, None


def _bodies_via_walk(msg) -> tuple[str | None, str | None]:
    text = None
    html = None
    parts = msg.walk() if msg.is_multipart() else (msg,)
    for part in parts:
        if part.get_content_disposition() == "attachment":
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain" and text is None:
            text = _part_text(part)
        elif ctype == "text/html" and html is None:
            html = _part_text(part)
    return text, html


def _part_text(part) -> str | None:
    try:
        content = part.get_content()
    except Exception:
        payload = part.get_payload(decode=True)
        if not isinstance(payload, (bytes, str)):
            return None
        content = payload
    text = _as_str(content)
    return text or None


def _html_to_text(html: str) -> str:
    if not html.strip():
        return ""
    try:
        from talon.utils import html_to_text

        converted = html_to_text(html)
    except Exception:
        converted = None

    if isinstance(converted, bytes):
        converted = converted.decode("utf-8", errors="replace")
    if isinstance(converted, str) and converted.strip():
        return _tidy_lines(converted)
    return _html_fallback(html)


def _html_fallback(html: str) -> str:
    text = _BLOCK_END.sub("\n", html)
    text = _TAG.sub("", text)
    text = unescape(text)
    text = _SPACE_RUN.sub(" ", text)
    return _tidy_lines(text)


def _tidy_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()


def _is_attachment(part) -> bool:
    if part.is_multipart():
        return False
    disposition = (part.get_content_disposition() or "").lower()
    filename = part.get_filename()
    if disposition == "attachment":
        return True
    if disposition == "inline":
        return bool(filename) and Path(filename).suffix.lower() == ".rpmsg"
    if not filename:
        return False
    return part.get_content_type() not in {"text/plain", "text/html"}


def _extension(part) -> str:
    import mimetypes

    return mimetypes.guess_extension(part.get_content_type()) or ""


def _unique_name(filename: str, counts: dict[str, int]) -> str:
    seen = counts.get(filename, 0) + 1
    counts[filename] = seen
    if seen == 1:
        return filename
    stem, dot, ext = filename.rpartition(".")
    if dot:
        return f"{stem}__{seen}.{ext}"
    return f"{filename}__{seen}"


def _part_bytes(part) -> bytes | None:
    try:
        content = part.get_content()
    except Exception:
        content = part.get_payload(decode=True)
    if isinstance(content, str):
        return content.encode("utf-8", errors="replace")
    if isinstance(content, bytes) and content:
        return content
    return None


def _sender(from_header: str) -> str:
    _name, addr = parseaddr(from_header)
    return addr or from_header


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _headers(msg) -> dict[str, str]:
    """Every header on this message, including ones the first row never has."""
    found: dict[str, str] = {}
    for key in msg.keys():
        name = _CANONICAL_HEADERS.get(str(key).casefold(), str(key))
        values = msg.get_all(key) or []
        parts = [str(value).strip() for value in values if value is not None and str(value).strip()]
        if name in found:
            if parts:
                found[name] = ", ".join(p for p in (found[name], *parts) if p)
            continue
        found[name] = ", ".join(parts)
    return found


def _flat_record(record: ParsedEmail) -> dict[str, Any]:
    row: dict[str, Any] = {"file": record.file}
    row.update(record.headers)
    row["date_parsed"] = record.date_parsed
    row["body_raw"] = record.body_raw
    row["body"] = record.body
    row["signature"] = record.signature
    row["attachment_count"] = record.attachment_count
    row["attachment_names"] = record.attachment_names
    row["parse_error"] = record.parse_error
    return row


def _column_order(columns) -> list[str]:
    present = list(columns)
    known = set(present)
    ordered = [name for name in ("file", *_HEADER_ORDER) if name in known]
    rest = [
        name
        for name in present
        if name not in ordered and name not in _DERIVED_COLUMNS
    ]
    derived = [name for name in _DERIVED_COLUMNS if name in known]
    return ordered + rest + derived


def _as_bytes(raw: str | bytes) -> bytes:
    if isinstance(raw, bytes):
        return raw
    return raw.encode("utf-8", errors="surrogateescape")


def _as_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _mail_key(file: str) -> str:
    if not file:
        return "mail"
    return sanitize_filename(file.replace("/", "_"))
